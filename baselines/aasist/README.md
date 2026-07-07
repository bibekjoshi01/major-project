# AASIST Baseline

This baseline uses the project `DatasetBase` data pipeline and reads all runtime
settings from `baselines/aasist/aasist.conf`.

Install dependencies before training:

```bash
pip install -r requirements.txt
pip install -r baselines/aasist/requirements.txt
```

## Data Layout

Set `data.database_path` in the config to the dataset root. The default expects:

```text
data/asvspoof5/
  manifests/
    train.jsonl
    dev.jsonl
    eval.jsonl
  audio/
    ...
```

Each manifest row must contain the fields used by `DatasetBase`: `audio_path`,
`label`, `utt_id`, and `attack_label`.

## Train

```bash
python -m baselines.aasist.train --config baselines/aasist/aasist.conf
```

Important config keys:

- `data.database_path`: dataset root.
- `data.max_len`: waveform length after repeat-pad/truncate.
- `training.batch_size`, `training.num_epochs`, `training.num_workers`.
- `training.freq_aug`: AASIST frequency masking toggle.
- `optimizer`: optimizer and scheduler settings.
- `loss.class_weights`: binary cross-entropy class weights for labels `0` and `1`.

Training writes checkpoints and logs under `output_dir/experiment_name`, including:

```text
checkpoints/best.pth
checkpoints/last.pth
train.log
config.conf
```

## Evaluate

```bash
python -m baselines.aasist.evaluate \
  --config baselines/aasist/aasist.conf \
  --checkpoint baselines/runs/aasist/aasist_baseline/checkpoints/best.pth \
  --split eval \
  --output baselines/runs/aasist/eval_scores.txt
```

`--split` can be `train`, `dev`, or `eval`. The score file contains:

```text
utt_id score label prediction
```
