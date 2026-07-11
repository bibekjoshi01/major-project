"""
Channel condition definitions for RTC/VoIP-style training augmentation.

Pure data layer: no torch/subprocess here so profiles stay trivially
picklable and importable from anywhere (including DataLoader workers).
"""

from __future__ import annotations

import csv
import os
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# Fallback bitrates (kbps) if codec.config.csv is missing/unparseable, e.g. on
# a dev box without the ASVspoof5 dataset downloaded. Mirrors the real CSV values.
_FALLBACK_BITRATES = {
    "opus_wb": [6.0, 12.0, 18.0, 24.0, 30.0],
    "mp3_wb": [65.0, 100.0, 135.0, 190.0, 240.0],
    "m4a_wb": [16.0, 32.0, 64.0, 96.0, 128.0],
}


@dataclass(frozen=True)
class GilbertElliottParams:
    """2-state Markov bursty packet-loss model, parameterized per ~20ms packet."""

    p_gb: float  # P(Good -> Bad) transition probability per packet
    p_bg: float  # P(Bad -> Good) transition probability per packet
    loss_prob_good: float = 0.0
    loss_prob_bad: float = 1.0
    packet_ms: float = 20.0

    @property
    def steady_state_bad(self) -> float:
        return self.p_gb / (self.p_gb + self.p_bg)

    @property
    def expected_loss_rate(self) -> float:
        bad = self.steady_state_bad
        return bad * self.loss_prob_bad + (1.0 - bad) * self.loss_prob_good

    @property
    def mean_burst_packets(self) -> float:
        return 1.0 / self.p_bg


@dataclass(frozen=True)
class ChannelProfile:
    """A realistic VoIP/RTC channel condition to pass audio through."""

    name: str
    codec: str  # "opus" | "mp3" | "aac" | "g711_alaw"
    operating_sample_rate: int  # 16000 for wideband opus/mp3/aac, 8000 for g711
    bitrate_choices_kbps: Sequence[float] = field(default_factory=tuple)
    loss_model: Optional[GilbertElliottParams] = None
    apply_agc: bool = True
    apply_highpass: bool = True
    description: str = ""

    def sample_bitrate_kbps(self) -> Optional[float]:
        if not self.bitrate_choices_kbps:
            return None
        return random.choice(list(self.bitrate_choices_kbps))


def _parse_bitrate_cell(cell: str) -> Optional[float]:
    cell = cell.strip()
    if not cell or cell.lower().startswith("see"):
        return None
    if "-" in cell:
        lo, hi = cell.split("-", 1)
        try:
            return (float(lo) + float(hi)) / 2.0
        except ValueError:
            return None
    try:
        return float(cell)
    except ValueError:
        return None


def load_codec_bitrates(csv_path: str) -> Dict[str, List[float]]:
    """Parse ASVspoof5's codec.config.csv into {codec_name: [bitrate_kbps, ...]}.

    Never raises -- returns {} and warns once on any I/O/parse failure so a
    missing dataset checkout doesn't crash dataset construction.
    """
    bitrates: Dict[str, List[float]] = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                codec = (row.get("CODEC") or "").strip()
                rate_cell = row.get("Bitrate(kbps)")
                if not codec or rate_cell is None:
                    continue
                rate = _parse_bitrate_cell(rate_cell)
                if rate is None:
                    continue
                bitrates.setdefault(codec, []).append(rate)
    except (OSError, csv.Error) as e:
        warnings.warn(f"Could not load codec bitrates from {csv_path}: {e}")
        return {}
    return bitrates


def build_default_profiles(
    database_path: Optional[str] = None,
) -> Dict[str, ChannelProfile]:
    """Builds the default set of realistic VoIP/RTC channel profiles.

    Bitrates are sourced from ASVspoof5's own codec.config.csv when available
    (falls back to hardcoded values mirroring that file otherwise).
    """
    csv_bitrates: Dict[str, List[float]] = {}
    if database_path:
        csv_path = os.path.join(database_path, "protocols", "codec.config.csv")
        csv_bitrates = load_codec_bitrates(csv_path)

    def bitrates_for(codec_key: str) -> List[float]:
        return csv_bitrates.get(codec_key) or _FALLBACK_BITRATES[codec_key]

    opus_wb_bitrates = bitrates_for("opus_wb")
    mp3_wb_bitrates = bitrates_for("mp3_wb")
    m4a_wb_bitrates = bitrates_for("m4a_wb")

    profiles = {
        "whatsapp_opus_wb": ChannelProfile(
            name="whatsapp_opus_wb",
            codec="opus",
            operating_sample_rate=16000,
            bitrate_choices_kbps=tuple(opus_wb_bitrates),
            loss_model=GilbertElliottParams(p_gb=0.01, p_bg=0.40),
            description="Healthy WhatsApp/Signal-style Opus wideband call.",
        ),
        "mobile_lossy_opus": ChannelProfile(
            name="mobile_lossy_opus",
            codec="opus",
            operating_sample_rate=16000,
            bitrate_choices_kbps=tuple(opus_wb_bitrates[:2]),
            loss_model=GilbertElliottParams(p_gb=0.05, p_bg=0.20),
            description="Poor mobile network: low-bitrate Opus with heavy bursty loss.",
        ),
        "pstn_g711": ChannelProfile(
            name="pstn_g711",
            codec="g711_alaw",
            operating_sample_rate=8000,
            bitrate_choices_kbps=(),
            loss_model=GilbertElliottParams(p_gb=0.005, p_bg=0.50),
            description="VoIP-to-PSTN gateway: G.711 A-law, forced narrowband.",
        ),
        "voice_note_mp3": ChannelProfile(
            name="voice_note_mp3",
            codec="mp3",
            operating_sample_rate=16000,
            bitrate_choices_kbps=tuple(mp3_wb_bitrates),
            loss_model=None,
            description="Stored/forwarded voice note (mp3), no packet loss.",
        ),
        "voice_note_aac": ChannelProfile(
            name="voice_note_aac",
            codec="aac",
            operating_sample_rate=16000,
            bitrate_choices_kbps=tuple(m4a_wb_bitrates),
            loss_model=None,
            description="Stored/forwarded voice note (m4a/aac), no packet loss.",
        ),
    }

    return profiles
