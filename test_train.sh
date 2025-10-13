#!/bin/bash

# Simple training test to verify NaN fixes
# Run just a few iterations to check stability

echo "Testing training stability with fixed code..."

python method_tvr/train.py \
--dset_name=activitynet \
--eval_split_name=val \
--nms_thd=0.5 \
--results_root=results \
--debug \
--max_epochs=1 \
--max_n_examples_per_epoch=10 \
--train_path=data/activitynet/activitynet_train_release.jsonl \
--desc_bert_path=data/activitynet/activitynet_train_query_bert_base_uncased.h5 \
--video_duration_idx_path=data/activitynet/activitynet_duration_idx_release.csv \
--vid_feat_path=data/activitynet/activitynet_c3d_feat.h5 \
--clip_length=1.5 \
--vid_feat_size=500 \
--ctx_mode=video_tef