# CHDL

[English](README.md) | [中文 (简体)](README.zh-CN.md)

CHDL is an implementation for video moment retrieval that supports joint retrieval using video, subtitles, and textual descriptions.
It targets datasets such as TVR, ActivityNet, Charades, and DiDeMo, and provides training, inference, and evaluation pipelines. The core model predicts temporal moment boundaries and supports multiple retrieval tasks:

- VCMR: Video Corpus Moment Retrieval
- SVMR: Single Video Moment Retrieval
- VR: Video Retrieval

## Table of Contents

- `chdl/`
  - `train.py`: Training entry point with training, validation and automatic evaluation workflows.
  - `inference.py`: Inference entry point to load a checkpoint and produce submission files.
  - `config.py`: Training/testing option definitions, including data, model, optimization and post-processing settings.
  - `dataset.py`: Dataset loading and preprocessing logic. Supports JSONL and HDF5 feature files.
  - `model.py`: CHDL model definition and forward pass.
  - `optimization.py`: Optimizers, LR schedules and a custom Adam implementation.
  - `components.py`: Model components such as attention, positional encodings, conv heads and temporal localization modules.
  - `utils.py`: Model helper utilities.
- `eval/`
  - `eval.py`: Evaluation utilities for retrieval metrics (R@K, IoU, etc.).
- `utils/`
  - `basic_utils.py`, `tensor_utils.py`, `model_utils.py` and other reusable helpers.
  - `video_feature/`: Video feature extraction and conversion scripts.
  - `text_feature/`: Text feature preprocessing and export scripts.
- `setup.sh`: Adds the project root to `PYTHONPATH` for convenient execution.

## Requirements

Create a Python environment and install the main dependencies:

```bash
python -m pip install torch torchvision torchaudio
python -m pip install numpy tqdm easydict tensorboard h5py
```

Additional packages (e.g. `scipy`, `pandas`) may be required depending on your workflow.

## Quick Start

1. Change into the repository root:

```bash
cd /home/jynp/CHDL
```

2. Enable the project path:

```bash
source setup.sh
```

3. Prepare dataset JSONL files and feature HDF5 files before training.

## Training

Training entrypoint is `chdl/train.py`.

Example:

```bash
python chdl/train.py \
  --exp_id my_experiment \
  --dset_name tvr \
  --ctx_mode video_sub \
  --train_path data/tvr_train_release.jsonl \
  --eval_path data/tvr_val_release.jsonl \
  --eval_split_name val \
  --vid_feat_path data/tvr_feature_release/video_feature/tvr_resnet152_rgb_max_i3d_rgb600_avg_cat_cl-1.5.h5 \
  --desc_bert_path data/tvr_feature_release/bert_feature/query_only/tvr_query_pretrained_w_query.h5 \
  --sub_bert_path data/tvr_feature_release/bert_feature/sub_query/tvr_sub_pretrained_w_sub_query_max_cl-1.5.h5 \
  --bsz 128 \
  --lr 1e-4 \
  --n_epoch 100
```

### Training arguments (high level)

- `--exp_id`: Experiment id for this run (required).
- `--dset_name`: Dataset name, one of `tvr`, `activitynet`, `charades`, `didemo`.
- `--ctx_mode`: Context mode. Options: `video`, `sub`, `tef`, `video_sub`, `video_tef`, `sub_tef`, `video_sub_tef`.
- `--train_path`: Path to the training JSONL file.
- `--eval_path`: Path to the validation JSONL file (used for validation/early stop during training).
- `--vid_feat_path`: Path to video feature HDF5.
- `--desc_bert_path`: Path to description BERT/Roberta features HDF5.
- `--sub_bert_path`: Path to subtitle BERT features HDF5 (required when `ctx_mode` includes `sub`).
- `--bsz`: Training batch size.
- `--lr`: Learning rate.
- `--n_epoch`: Number of training epochs.
- `--results_root`: Root folder for results (default: `results`).

### Training outputs

Training creates a results directory under `results/`. Example:

```
results/tvr-video_sub-my_experiment-YYYY_MM_DD_hh_mm_ss
```

Typical contents:

- `model.ckpt`: Saved model checkpoint
- `train.log.txt`: Training loss logs
- `eval.log.txt`: Validation metrics logs
- `tensorboard_log/`: TensorBoard logs
- `code.zip`: Code snapshot saved at training time

## Inference & Evaluation

Inference entrypoint is `chdl/inference.py`.

Example:

```bash
python chdl/inference.py \
  --model_dir my_experiment_dir_name \
  --eval_id test_run_001 \
  --dset_name tvr \
  --ctx_mode video_sub \
  --eval_split_name val \
  --eval_path data/tvr_val_release.jsonl \
  --vid_feat_path data/tvr_feature_release/video_feature/tvr_resnet152_rgb_max_i3d_rgb600_avg_cat_cl-1.5.h5 \
  --desc_bert_path data/tvr_feature_release/bert_feature/query_only/tvr_query_pretrained_w_query.h5 \
  --sub_bert_path data/tvr_feature_release/bert_feature/sub_query/tvr_sub_pretrained_w_sub_query_max_cl-1.5.h5 \
  --tasks VCMR SVMR VR
```

> Note: `--model_dir` should point to the model results directory name under `chdl/results/`, not an absolute path.

### Inference arguments

- `--model_dir`: Name of the results folder created during training (relative to `chdl/results/`).
- `--eval_id`: Identifier for this inference run (used in output filenames).
- `--tasks`: Tasks to run: `VCMR`, `SVMR`, `VR`.
- `--nms_thd`: If set to a value other than `-1`, NMS will be applied to predictions before evaluation.

### Inference outputs

Predictions are saved as JSON files under the model results folder, for example:

```
chdl/results/<model_dir>/inference_<dset_name>_<split>_<eval_id>_predictions_<tasks>.json
```

If ground-truth annotations are available (e.g., `val` split), evaluation metrics are computed and written to a `_metrics.json` alongside predictions.

## Data format & preprocessing

### JSONL schema

`chdl/dataset.py` accepts JSONL files. Typical fields supported include:

- `desc_id` / `id`
- `desc` / `fig_desc` / `cog_desc` / `text`
- `vid_name` / `video` / `video_id`
- `duration`
- `ts` / `time` (timestamp span)

The loader is compatible with multiple field name variants across datasets.

### Feature files

The project relies on HDF5 feature files:

- Video features: ResNet, I3D, or concatenated ResNet+I3D
- Text features: BERT/Roberta embeddings for descriptions
- Subtitle features: used when `sub` mode is enabled

Feature conversion and extraction scripts are available under `utils/video_feature` and `utils/text_feature`.

## Evaluation metrics

Evaluation utilities are in `eval/eval.py` and support:

- R@K (Recall@1, @5, @10, @100)
- IoU thresholds: `0.5`, `0.7`
- Task types: `VCMR`, `SVMR`, `VR`

If `eval_split_name` is `val`, inference runs will automatically compute and save metrics.

## Debugging & quick checks

- `--debug`: Run in debug (fast) mode with reduced data/loops.
- `--data_ratio`: Use a fraction of the data, e.g. `--data_ratio 0.1`.
- `--no_core_driver`: Disable HDF5 `core` driver to avoid loading entire files into RAM.

## Notes

- Run `source setup.sh` before executing Python scripts to add the project root to `PYTHONPATH`.
- For multi-GPU training use `--device_ids 0 1`.
- After training, `chdl/train.py` triggers a single inference/evaluation on the best saved checkpoint.

## Extending & Reproducing

See `chdl/scripts` for shell examples that demonstrate dataset-specific commands and parameter settings.

---

The original Chinese localization has been preserved in `README.zh-CN.md`.

If you want, I can also add a dedicated "Data preparation & feature extraction" section with concrete commands for the scripts under `utils/video_feature` and `utils/text_feature`.