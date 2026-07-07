import torch
import random


class RTCAudioSimulator:
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate

    def apply_packet_loss(self, waveform, drop_probability=0.05, max_burst_len=800):
        """
        Simulates network packet drops by zeroing out contiguous chunks of audio.

        Args:
            waveform (Tensor): Shape (64000,)
            drop_probability (float): Probability that a frame starts a burst drop.
            max_burst_len (int): Maximum samples to drop at once (800 samples = 50ms at 16kHz).
        """
        degraded = waveform.clone()
        num_samples = degraded.shape[0]
        i = 0

        while i < num_samples:
            if random.random() < drop_probability:
                # Determine how long the network dropped packets (burst length)
                burst_len = random.randint(160, max_burst_len)  # 10ms to 50ms drop
                end_idx = min(i + burst_len, num_samples)

                # Zero out the dropped packet space
                degraded[i:end_idx] = 0.0
                i = end_idx
            else:
                i += 1

        return degraded

    def apply_codec_distortion(self, waveform):
        """
        Simulates telecom bandpass degradation and quantization noise
        (similar to G.711 / AMR constraints) using tensor math.
        """
        # 1. Step down bit-depth to simulate telephone quantization noise (e.g., A-law approximation)
        # Quantize down to 8-bit equivalent values, then rescale back
        quantized = torch.round(waveform * 127.0) / 127.0

        # 2. Emulate an unstable VoIP connection by scaling down signal amplitude slightly
        attenuated = quantized * random.uniform(0.7, 1.0)

        return attenuated

    def process(self, waveform, is_training=True):
        """
        Pipeline entry point. Applies mutations based on current mode configuration.
        """
        if not is_training:
            return waveform

        # Apply network noise step-by-step randomly during training runs
        processed_wave = waveform

        if random.random() < 0.5:
            processed_wave = self.apply_packet_loss(processed_wave)
        if random.random() < 0.5:
            processed_wave = self.apply_codec_distortion(processed_wave)

        return processed_wave
