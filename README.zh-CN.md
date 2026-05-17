# CHDL

[English](README.md) | [中文 (简体)](README.zh-CN.md)

CHDL 是一个视频时序检索（Video Moment Retrieval）项目实现，支持基于视频、字幕和文本描述的联合检索。
它主要面向 TVR、ActivityNet、Charades 和 DiDeMo 等数据集，提供训练、推理和评估流程。项目核心实现了带时序边界预测的多模态检索模型，并支持多种任务类型：

- VCMR：Video Corpus Moment Retrieval
- SVMR：Single Video Moment Retrieval
- VR：Video Retrieval

## 目录结构

- `chdl/`
  - `train.py`：训练入口，包含模型训练、验证与自动评估流程。
  - `inference.py`：推理入口，用于加载 checkpoint 并生成提交结果。
  - `config.py`：训练/测试选项定义，包含数据、模型、优化和后处理参数。
  - `dataset.py`：数据集加载与预处理逻辑，支持 `jsonl` 数据格式和 HDF5 特征。
  - `model.py`：CHDL 模型定义及前向计算。
  - `optimization.py`：优化器、学习率调度和自定义 Adam 实现。
  - `components.py`：模型组件，包括注意力、位置编码、卷积和时序定位模块。
  - `utils.py`：模型辅助工具函数。
- `eval/`
  - `eval.py`：检索结果评估工具，支持 R@K、IoU 等指标。
- `utils/`
  - `basic_utils.py`、`tensor_utils.py`、`model_utils.py` 等通用工具。
  - `video_feature/`：视频特征提取与处理脚本。
  - `text_feature/`：文本特征预处理与导出脚本。
- `setup.sh`：将当前仓库根目录加入 `PYTHONPATH`，方便直接运行 Python 脚本。

## 依赖环境


```bash
python -m pip install torch torchvision torchaudio
python -m pip install numpy tqdm easydict tensorboard h5py
```

## 快速开始

1. 进入仓库根目录：

```bash
cd /home/jynp/CHDL
```

2. 启用项目路径：

```bash
source setup.sh
```

3. 运行训练命令前，请准备好数据集 JSONL 文件和特征文件。

## 训练

训练入口为 `chdl/train.py`。

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

### 训练参数说明

- `--exp_id`：本次训练 experiment id，必须指定。
- `--dset_name`：数据集名称，可选 `tvr`, `activitynet`, `charades`, `didemo`。
- `--ctx_mode`：上下文模式，可选 `video`, `sub`, `tef`, `video_sub`, `video_tef`, `sub_tef`, `video_sub_tef`。
- `--train_path`：训练集 JSONL 文件路径。
- `--eval_path`：验证集 JSONL 文件路径，训练过程中用于验证与早停。
- `--vid_feat_path`：视频特征 HDF5 路径。
- `--desc_bert_path`：描述文本特征 HDF5 路径。
- `--sub_bert_path`：字幕文本特征 HDF5 路径，仅当 `ctx_mode` 包含 `sub` 时需要。
- `--bsz`：训练批次大小。
- `--lr`：优化器学习率。
- `--n_epoch`：训练轮数。
- `--results_root`：结果目录根路径，默认 `results`。

### 训练后输出

训练时会在 `results/` 下创建目录，目录名格式类似：

```
results/tvr-video_sub-my_experiment-YYYY_MM_DD_hh_mm_ss
```

目录内通常包含：

- `model.ckpt`：保存的模型 checkpoint
- `train.log.txt`：训练损失日志
- `eval.log.txt`：验证指标日志
- `tensorboard_log/`：TensorBoard 日志
- `code.zip`：训练时保存的代码快照

## 推理与评估

推理入口为 `chdl/inference.py`。

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

> 注意：`--model_dir` 需指定模型所在结果目录在 `chdl/results/` 下的目录名，而不是完整路径。

### 推理参数说明

- `--model_dir`：训练产生的结果目录名（相对于 `chdl/results/`）。
- `--eval_id`：本次推理 ID，用于生成输出文件名。
- `--tasks`：要运行的任务，可选 `VCMR`, `SVMR`, `VR`。
- `--nms_thd`：如果设置为非 `-1`，会对预测结果进行 NMS 后再评估。

### 推理输出

推理结果会保存为 JSON 文件，目录通常在：

```
chdl/results/<model_dir>/inference_<dset_name>_<split>_<eval_id>_predictions_<tasks>.json
```

如果验证集带有标注，代码会自动计算评估指标，并将结果写入 `_metrics.json`。

## 数据格式与预处理

### JSONL 数据格式

`chdl/dataset.py` 支持 JSONL 输入，基本字段包括：

- `desc_id` / `id`
- `desc` / `fig_desc` / `cog_desc` / `text`
- `vid_name` / `video` / `video_id`
- `duration`
- `ts` / `time`（时间戳边界）

对于不同数据集，代码已兼容多种字段命名。

### 特征数据

项目依赖 HDF5 特征文件：

- 视频特征：如 ResNet、I3D、ResNet+I3D 拼接特征
- 文本描述特征：BERT/Roberta 提取的描述特征
- 字幕特征：仅当使用 `sub` 模式时需要

特征文件通常为 `.h5` 格式，可使用 `utils/video_feature` 与 `utils/text_feature` 下的脚本进行转换与提取。

## 评估指标

评估工具位于 `eval/eval.py`，支持计算：

- R@K（Recall@1, @5, @10, @100）
- IoU 阈值：`0.5`, `0.7`
- 任务类型：`VCMR`, `SVMR`, `VR`

如果 `eval_split_name` 为 `val`，推理时会自动对生成结果执行评估。

## 调试与快速验证

- `--debug`：进入快速调试模式，降低数据加载和训练规模。
- `--data_ratio`：只使用部分数据进行训练/验证，例如 `--data_ratio 0.1`。
- `--no_core_driver`：禁用 HDF5 `core` driver，适用于不希望全部映射到内存的情况。

## 备注

- `setup.sh` 会将仓库根目录加入 `PYTHONPATH`，建议在执行 Python 脚本前先运行 `source setup.sh`。
- 如果使用多 GPU，可通过 `--device_ids 0 1` 指定多个 GPU。
- 模型训练结束后，`chdl/train.py` 会自动对当前最佳模型执行一次推理评估。

## 扩展

如果你希望在已有数据集上复现实验，可参考 `chdl/scripts` 中的 shell 示例，按数据集名称和特征路径调整参数。

---

如果你需要，我也可以进一步补充一份“数据准备与特征提取”章节，包含 `utils/video_feature` / `utils/text_feature` 的常用命令。
