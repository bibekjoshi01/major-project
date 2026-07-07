# RawNet2 Baseline

This baseline uses the project `DatasetBase` data pipeline and reads all runtime
settings from `baselines/rawnet2/rawnet2.conf`.

Install dependencies before training:

```bash
pip install -r requirements.txt
pip install -r baselines/rawnet2/requirements.txt
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
python -m baselines.rawnet2.train --config baselines/rawnet2/rawnet2.conf
```

Important config keys:

- `data.database_path`: dataset root.
- `data.max_len`: waveform length after repeat-pad/truncate.
- `training.batch_size`, `training.num_epochs`, `training.num_workers`.
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
python -m baselines.rawnet2.evaluate \
  --config baselines/rawnet2/rawnet2.conf \
  --checkpoint baselines/runs/rawnet2/rawnet2_baseline/checkpoints/best.pth \
  --split eval \
  --output baselines/runs/rawnet2/eval_scores.txt
```

`--split` can be `train`, `dev`, or `eval`. The score file contains:

```text
utt_id score label prediction
```
