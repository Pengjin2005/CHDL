# method_tvr/check2.py
import argparse, json, os, traceback
import torch
from torch.utils.data import DataLoader

# --- 你工程里的数据集与collate ---
from method_tvr.start_end_verified_dataset import StartEndDataset, start_end_collate

def _check_tensor_pair(name, pair, meta, batch_index):
    padded, mask = pair
    X = padded.detach().to("cpu", non_blocking=True)
    M = mask.detach().to("cpu", non_blocking=True).bool()

    row_sum = X.abs().sum(dim=-1)          # (B, L)
    zero_rows = (row_sum == 0) & M         # 只统计被认为有效的位置

    has_nan = torch.isnan(X) & M.unsqueeze(-1)
    has_inf = torch.isinf(X) & M.unsqueeze(-1)

    bad_list = []
    if zero_rows.any():
        b_idx, _ = zero_rows.nonzero(as_tuple=True)
        for b in b_idx.unique().tolist():
            vid = meta[b].get("vid_name", f"unknown@batch{batch_index}")
            cnt = (b_idx == b).sum().item()
            bad_list.append({"type": f"{name}_valid_zero_rows",
                             "batch": batch_index, "sample_index": b,
                             "vid_name": vid, "count": cnt})
    if has_nan.any():
        b_idx = has_nan.any(dim=-1).any(dim=-1).nonzero(as_tuple=True)[0]
        for b in b_idx.tolist():
            vid = meta[b].get("vid_name", f"unknown@batch{batch_index}")
            bad_list.append({"type": f"{name}_nan",
                             "batch": batch_index, "sample_index": b, "vid_name": vid})
    if has_inf.any():
        b_idx = has_inf.any(dim=-1).any(dim=-1).nonzero(as_tuple=True)[0]
        for b in b_idx.tolist():
            vid = meta[b].get("vid_name", f"unknown@batch{batch_index}")
            bad_list.append({"type": f"{name}_inf",
                             "batch": batch_index, "sample_index": b, "vid_name": vid})

    # 打点统计
    if M.any():
        vals = X[M]
        print(f"[STAT][{name}] batch={batch_index} valid={int(M.sum())} "
              f"zero_rows={int(zero_rows.sum())} "
              f"val[min={vals.min().item():.4g}, max={vals.max().item():.4g}, mean={vals.mean().item():.4g}]")
    return bad_list

def scan_loader_zero_rows(loader, max_batches=50, save_report_path="/tmp/zero_row_report.json"):
    all_bad, total_batches = [], 0
    for bi, batch in enumerate(loader):
        if batch is None:
            print(f"[WARN] batch {bi} is None (filtered).")
            continue
        meta, bd = batch
        if "video_feat" in bd and isinstance(bd["video_feat"], (tuple, list)):
            all_bad += _check_tensor_pair("video_feat", bd["video_feat"], meta, bi)
        if "sub_feat" in bd and isinstance(bd["sub_feat"], (tuple, list)):
            all_bad += _check_tensor_pair("sub_feat", bd["sub_feat"], meta, bi)
        if "query_feat" in bd and isinstance(bd["query_feat"], (tuple, list)):
            all_bad += _check_tensor_pair("query_feat", bd["query_feat"], meta, bi)
        total_batches += 1
        if total_batches >= max_batches:
            break

    print("\n=== Scan Summary ===")
    print(f"batches_scanned = {total_batches}")
    if not all_bad:
        print("No issues found 🎉")
    else:
        from collections import Counter
        cnt = Counter([x["type"] for x in all_bad])
        for k, v in cnt.items():
            print(f"{k}: {v} case(s)")
        try:
            os.makedirs(os.path.dirname(save_report_path), exist_ok=True)
            with open(save_report_path, "w", encoding="utf-8") as f:
                json.dump(all_bad, f, ensure_ascii=False, indent=2)
            print(f"Report saved to: {save_report_path}")
        except Exception as e:
            print(f"[WARN] failed to save report: {e}")

def build_loader(args):
    dset = StartEndDataset(
        dset_name=args.dset_name,
        data_path=args.train_path,
        desc_bert_path_or_handler=args.desc_bert_path,
        sub_bert_path_or_handler=(args.sub_bert_path if args.sub_bert_path else None),
        max_desc_len=args.max_desc_l,
        max_ctx_len=args.max_ctx_l,
        vid_feat_path_or_handler=args.vid_feat_path,
        clip_length=args.clip_length,
        ctx_mode=args.ctx_mode,
        normalize_vfeat=not args.no_norm_vfeat,
        normalize_tfeat=not args.no_norm_tfeat,
        h5driver=None,
        data_ratio=1.0,
    )
    loader = DataLoader(
        dset,
        batch_size=int(args.bsz),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=start_end_collate,
        drop_last=True,
        pin_memory=True,
    )
    return loader

def parse_args():
    p = argparse.ArgumentParser("Scan zero-rows / NaN / Inf in features (no training, no model).")
    p.add_argument("--dset_name", required=True)
    p.add_argument("--train_path", required=True)
    p.add_argument("--vid_feat_path", required=True)
    p.add_argument("--desc_bert_path", required=True)
    p.add_argument("--sub_bert_path", default=None)
    p.add_argument("--max_desc_l", type=int, required=True)
    p.add_argument("--max_ctx_l", type=int, required=True)
    p.add_argument("--clip_length", type=float, required=True)
    p.add_argument("--ctx_mode", required=True)  # e.g., video_tef / video_sub_tef
    p.add_argument("--bsz", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_norm_vfeat", action="store_true", help="disable video feature L2 normalization")
    p.add_argument("--no_norm_tfeat", action="store_true", help="disable text feature L2 normalization")
    p.add_argument("--max_batches", type=int, default=50)
    p.add_argument("--save_report", default="/tmp/zero_row_report.json")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        loader = build_loader(args)
        scan_loader_zero_rows(loader, max_batches=args.max_batches, save_report_path=args.save_report)
    except Exception as e:
        print("[ERROR] failed to run scanner:")
        print(repr(e))
        traceback.print_exc()
