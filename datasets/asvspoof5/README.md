LINK: https://huggingface.co/datasets/jungjee/asvspoof5/tree/main

# ASVspoof5 Dataset Pipeline

This repository provides a clean, structured pipeline for working with the ASVspoof5 dataset. It handles dataset downloading, extraction, protocol parsing, and creation of ML-ready manifests for spoof detection and speaker verification tasks.

---

# 1. Overview

ASVspoof5 is a protocol-driven speech dataset for:

- Speech spoofing detection (Track 1)
- Speaker verification under spoofing conditions (Track 2)

It consists of:

- Large FLAC audio files (T / D / E splits)
- Protocol TSV files defining labels and evaluation structure

---

# 2. Dataset Structure

## Raw data (after download)

    datasets/asvspoof5/
        raw/
        flac_T/
        flac_D/
        flac_E/

### Splits

| Split | Meaning                  |
| ----- | ------------------------ |
| T     | Training                 |
| D     | Development (validation) |
| E     | Evaluation (test)        |

---

## Protocol files

    datasets/asvspoof5/
        protocols/
            ASVspoof5.train.tsv
            ASVspoof5.dev.track_1.tsv
            ASVspoof5.dev.track_2.enroll.tsv
            ASVspoof5.dev.track_2.trial.tsv
            ASVspoof5.eval.track_1.tsv
            ASVspoof5.eval.track_2.enroll.tsv
            ASVspoof5.eval.track_2.trial.tsv
            ASVspoof5.codec.config.csv

---

## ML-ready manifests

    datasets/asvspoof5/
        manifests/
            train.jsonl
            dev.jsonl
            eval.jsonl


