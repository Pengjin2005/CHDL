python method_tvr/check2.py \
  --dset_name activitynet \
  --train_path data/VERIFIED/ActivityNet/activitynet_fig_train.jsonl \
  --vid_feat_path data/VERIFIED/ActivityNet/video_feature/anet_resnet152_4fps_max_1fps.h5 \
  --desc_bert_path data/VERIFIED/ActivityNet/new_desc_feature/vcmr_roberta_base_anet_embed.h5 \
  --max_desc_l 64 --max_ctx_l 128 \
  --clip_length 2.0 --ctx_mode video_tef \
  --bsz 8 --num_workers 4 \
  --max_batches 50 \
  --save_report /tmp/zero_row_report.json
