"""
Realistic VoIP/RTC channel simulation for training-time augmentation.

Passes clean audio through an actual lossy codec (Opus/G.711/MP3/AAC via
ffmpeg subprocess round-trips) and a Gilbert-Elliott bursty packet-loss model
with fade-based concealment, approximating what audio looks like after a real
VoIP call (WhatsApp/Signal-style) or a compressed/forwarded voice note.
"""
from __future__ import annotations

import random
import shutil
import subprocess
import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio

from datasets.channel_profiles import ChannelProfile, build_default_profiles

_ffmpeg_availability_cache: dict = {}
_warned_messages: set = set()


def _warn_once(msg: str) -> None:
    if msg not in _warned_messages:
        _warned_messages.add(msg)
        warnings.warn(msg)


def _check_ffmpeg(ffmpeg_bin: str) -> bool:
    if ffmpeg_bin not in _ffmpeg_availability_cache:
        available = shutil.which(ffmpeg_bin) is not None
        _ffmpeg_availability_cache[ffmpeg_bin] = available
        if not available:
            _warn_once(
                f"ffmpeg binary '{ffmpeg_bin}' not found; falling back to "
                "DSP-only channel approximation for all RTC augmentation."
            )
    return _ffmpeg_availability_cache[ffmpeg_bin]


def _run_ffmpeg(args: list, input_bytes: bytes, timeout: float) -> bytes:
    proc = subprocess.run(
        args,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=True,
    )
    return proc.stdout


def _encode_args(ffmpeg_bin: str, codec: str, in_sr: int, op_sr: int, bitrate_kbps: Optional[float]) -> list:
    base = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error",
        "-f", "f32le", "-ar", str(in_sr), "-ac", "1", "-i", "pipe:0",
        "-ar", str(op_sr), "-ac", "1", "-threads", "1",
    ]
    if codec == "opus":
        return base + [
            "-c:a", "libopus", "-b:a", f"{bitrate_kbps:.0f}k",
            "-application", "voip", "-frame_duration", "20",
            "-f", "opus", "pipe:1",
        ]
    if codec == "mp3":
        return base + ["-c:a", "libmp3lame", "-b:a", f"{bitrate_kbps:.0f}k", "-f", "mp3", "pipe:1"]
    if codec == "aac":
        return base + ["-c:a", "aac", "-b:a", f"{bitrate_kbps:.0f}k", "-f", "adts", "pipe:1"]
    if codec == "g711_alaw":
        return base + ["-c:a", "pcm_alaw", "-f", "alaw", "pipe:1"]
    raise ValueError(f"Unsupported codec: {codec}")


def _decode_args(ffmpeg_bin: str, codec: str, op_sr: int, out_sr: int) -> list:
    if codec == "opus":
        in_fmt = ["-f", "ogg"]
    elif codec == "mp3":
        in_fmt = ["-f", "mp3"]
    elif codec == "aac":
        in_fmt = ["-f", "aac"]
    elif codec == "g711_alaw":
        # Raw alaw has no container header -- rate/channels must be told to the demuxer.
        in_fmt = ["-f", "alaw", "-ar", str(op_sr), "-ac", "1"]
    else:
        raise ValueError(f"Unsupported codec: {codec}")
    return [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error",
        *in_fmt, "-i", "pipe:0",
        "-ar", str(out_sr), "-ac", "1", "-threads", "1",
        "-f", "f32le", "pipe:1",
    ]


def _conform_length(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    """Codec encoder priming delay (mp3/aac) shifts sample count slightly.
    Truncate/cyclic-pad back to the original length -- exact alignment doesn't
    matter for utterance-level spoof/bonafide labels, only a consistent shape."""
    cur_len = waveform.shape[-1]
    if cur_len == target_len:
        return waveform
    if cur_len > target_len:
        return waveform[..., :target_len]
    reps = (target_len // max(1, cur_len)) + 1
    return waveform.repeat(1, reps)[..., :target_len]


def _fallback_dsp(waveform: torch.Tensor) -> torch.Tensor:
    """Lightweight DSP approximation used when ffmpeg is unavailable/fails."""
    quantized = torch.round(waveform * 127.0) / 127.0
    attenuated = quantized * random.uniform(0.7, 1.0)
    return attenuated


def ffmpeg_codec_round_trip(
    waveform: torch.Tensor,
    in_sample_rate: int,
    profile: ChannelProfile,
    ffmpeg_bin: str = "ffmpeg",
    timeout: float = 8.0,
) -> Tuple[torch.Tensor, bool]:
    """Encodes `waveform` ([1, T] mono float32) through the profile's real
    codec and decodes it back. Returns (output_with_same_shape, used_ffmpeg).
    Never raises -- falls back to a DSP approximation on any failure."""
    if not _check_ffmpeg(ffmpeg_bin):
        return _fallback_dsp(waveform), False

    target_len = waveform.shape[-1]
    pcm_in = waveform.squeeze(0).contiguous().to(torch.float32).numpy().tobytes()
    op_sr = profile.operating_sample_rate
    bitrate = profile.sample_bitrate_kbps()

    try:
        encoded = _run_ffmpeg(
            _encode_args(ffmpeg_bin, profile.codec, in_sample_rate, op_sr, bitrate),
            pcm_in,
            timeout,
        )
        decoded = _run_ffmpeg(
            _decode_args(ffmpeg_bin, profile.codec, op_sr, in_sample_rate),
            encoded,
            timeout,
        )
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        _warn_once(f"ffmpeg codec round-trip failed for profile '{profile.name}': {e}")
        return _fallback_dsp(waveform), False

    pcm_out = np.frombuffer(decoded, dtype=np.float32)
    if pcm_out.size == 0:
        _warn_once(f"ffmpeg produced empty output for profile '{profile.name}'")
        return _fallback_dsp(waveform), False

    out = torch.from_numpy(pcm_out.copy()).unsqueeze(0)
    out = _conform_length(out, target_len)
    return out, True


class GilbertElliottLossSimulator:
    """2-state Markov bursty packet-loss model with repeat+fade concealment,
    operating at ~20ms packet granularity (WebRTC/Opus's real frame size)."""

    def __init__(self, params, sample_rate: int, decay_factor: float = 0.6, crossfade_ms: float = 2.0):
        self.params = params
        self.sample_rate = sample_rate
        self.decay_factor = decay_factor
        self.packet_len = max(1, int(round(params.packet_ms / 1000.0 * sample_rate)))
        self.crossfade_len = max(2, int(round(crossfade_ms / 1000.0 * sample_rate)))

    def _smooth_boundary(self, out: torch.Tensor, boundary_idx: int) -> None:
        n = self.crossfade_len
        lo = max(0, boundary_idx - n // 2)
        hi = min(out.shape[-1], boundary_idx + n // 2)
        if hi - lo <= 2:
            return
        segment = out[:, lo:hi]
        start_val = segment[:, :1]
        end_val = segment[:, -1:]
        ramp = torch.linspace(0.0, 1.0, hi - lo, device=out.device).unsqueeze(0)
        interp = start_val + (end_val - start_val) * ramp
        out[:, lo:hi] = 0.5 * segment + 0.5 * interp

    def apply(self, waveform: torch.Tensor) -> torch.Tensor:
        total_len = waveform.shape[-1]
        out = waveform.clone()
        packet_len = self.packet_len
        num_packets = (total_len + packet_len - 1) // packet_len

        state_bad = random.random() < self.params.steady_state_bad
        last_good_packet: Optional[torch.Tensor] = None
        consecutive_losses = 0
        prev_lost = False

        for i in range(num_packets):
            start = i * packet_len
            end = min(start + packet_len, total_len)
            seg_len = end - start

            loss_prob = self.params.loss_prob_bad if state_bad else self.params.loss_prob_good
            lost = random.random() < loss_prob

            if lost:
                consecutive_losses += 1
                if last_good_packet is not None:
                    gain = self.decay_factor ** consecutive_losses
                    concealment = last_good_packet
                    if concealment.shape[-1] < seg_len:
                        reps = (seg_len // concealment.shape[-1]) + 1
                        concealment = concealment.repeat(1, reps)
                    out[:, start:end] = concealment[:, :seg_len] * gain
                else:
                    out[:, start:end] = 0.0
            else:
                consecutive_losses = 0
                last_good_packet = out[:, start:end].clone()

            if lost != prev_lost:
                self._smooth_boundary(out, start)
            prev_lost = lost

            # Transition state for the next packet.
            if state_bad:
                state_bad = random.random() >= self.params.p_bg
            else:
                state_bad = random.random() < self.params.p_gb

        return out


def _agc_normalize(waveform: torch.Tensor, target_rms_dbfs: float = -20.0, max_gain_db: float = 12.0) -> torch.Tensor:
    rms = waveform.pow(2).mean().sqrt().clamp_min(1e-8)
    target_rms = 10 ** (target_rms_dbfs / 20.0)
    max_gain = 10 ** (max_gain_db / 20.0)
    gain = (target_rms / rms).clamp(max=max_gain)
    return (waveform * gain).clamp(-1.0, 1.0)


def _highpass_dc_removal(waveform: torch.Tensor, sample_rate: int, cutoff_hz: float = 80.0) -> torch.Tensor:
    try:
        return torchaudio.functional.highpass_biquad(waveform, sample_rate, cutoff_hz)
    except Exception:
        return waveform - waveform.mean(dim=-1, keepdim=True)


class RTCAudioSimulator:
    """Picks clean vs. a random realistic VoIP/RTC channel profile per call."""

    def __init__(
        self,
        sample_rate: int = 16000,
        profiles: Optional[dict] = None,
        clean_prob: float = 0.18,
        enabled: bool = True,
        degrade_eval: bool = False,
        ffmpeg_bin: str = "ffmpeg",
        ffmpeg_timeout: float = 8.0,
    ):
        self.sample_rate = sample_rate
        self.profiles = profiles or build_default_profiles()
        self.profile_names = list(self.profiles)
        self.clean_prob = clean_prob
        self.enabled = enabled
        self.degrade_eval = degrade_eval
        self.ffmpeg_bin = ffmpeg_bin
        self.ffmpeg_timeout = ffmpeg_timeout
        # Deliberately no private seeded RNG here: this object is constructed once in
        # the main process before DataLoader forks workers. A private random.Random(seed)
        # would give every worker identical state -> identical augmentation choices for
        # the whole run. The module-level `random` is already reseeded per-worker via
        # baselines/common.py's seed_worker/build_loader wiring.

    def process(
        self, waveform: torch.Tensor, sample_rate: Optional[int] = None, is_training: bool = True
    ) -> Tuple[torch.Tensor, str]:
        sr = sample_rate or self.sample_rate

        if not self.enabled or not self.profile_names or (not is_training and not self.degrade_eval):
            return waveform, "clean"
        if random.random() < self.clean_prob:
            return waveform, "clean"

        profile = self.profiles[random.choice(self.profile_names)]
        out, _ = ffmpeg_codec_round_trip(waveform, sr, profile, self.ffmpeg_bin, self.ffmpeg_timeout)

        if profile.loss_model is not None:
            out = GilbertElliottLossSimulator(profile.loss_model, sr).apply(out)
        if profile.apply_agc:
            out = _agc_normalize(out)
        if profile.apply_highpass:
            out = _highpass_dc_removal(out, sr)

        return out, profile.name
