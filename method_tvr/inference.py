import os
import pprint
import logging
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from method_tvr.config import TestOptions
from method_tvr.model import ReLoCLNet
from method_tvr.start_end_dataset import start_end_collate, StartEndEvalDataset, prepare_batch_inputs
from utils.basic_utils import save_json, load_json
from utils.temporal_nms import temporal_non_maximum_suppression
from utils.tensor_utils import find_max_triples_from_upper_triangle_product
from standalone_eval.eval import eval_retrieval

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def filter_vcmr_by_nms(all_video_predictions, nms_threshold=0.6, max_before_nms=1000, max_after_nms=100,
                       score_col_idx=3):
    """ Apply non-maximum suppression for all the predictions for each video.
    1) group predictions by video index
    2) apply nms individually for each video index group
    3) combine and sort the predictions
    Args:
        all_video_predictions: list(sublist),
            Each sublist is [video_idx (int), st (float), ed(float), score (float)]
            Note the scores are negative distances.
        nms_threshold: float
        max_before_nms: int
        max_after_nms: int
        score_col_idx: int
    """
    predictions_neg_by_video_group = defaultdict(list)
    for pred in all_video_predictions[:max_before_nms]:
        predictions_neg_by_video_group[pred[0]].append(pred[1:])  # [st (float), ed(float), score (float)]
    predictions_by_video_group_neg_after_nms = dict()
    for video_idx, grouped_preds in predictions_neg_by_video_group.items():
        predictions_by_video_group_neg_after_nms[video_idx] = temporal_non_maximum_suppression(
            grouped_preds, nms_threshold=nms_threshold)
    predictions_after_nms = []
    for video_idx, grouped_preds in predictions_by_video_group_neg_after_nms.items():
        for pred in grouped_preds:
            pred = [video_idx] + pred  # [video_idx (int), st (float), ed(float), score (float)]
            predictions_after_nms.append(pred)
    # ranking happens across videos, descending order
    predictions_after_nms = sorted(predictions_after_nms, key=lambda x: x[score_col_idx], reverse=True)[:max_after_nms]
    return predictions_after_nms


def post_processing_vcmr_nms(vcmr_res, nms_thd=0.6, max_before_nms=1000, max_after_nms=100):
    """
    vcmr_res: list(dict), each dict is
        {
            "desc": str,
            "desc_id": int,
            "predictions": list(sublist)  # each sublist is
                [video_idx (int), st (float), ed(float), score (float)], video_idx could be different
        }
    """
    processed_vcmr_res = []
    for e in vcmr_res:
        e["predictions"] = filter_vcmr_by_nms(e["predictions"], nms_threshold=nms_thd, max_before_nms=max_before_nms,
                                              max_after_nms=max_after_nms)
        processed_vcmr_res.append(e)
    return processed_vcmr_res


def post_processing_svmr_nms(svmr_res, nms_thd=0.6, max_before_nms=1000, max_after_nms=100):
    """
    svmr_res: list(dict), each dict is
        {
            "desc": str,
            "desc_id": int,
            "predictions": list(sublist)  # each sublist is
                [video_idx (int), st (float), ed(float), score (float)], video_idx is the same.
         }
    """
    processed_svmr_res = []
    for e in svmr_res:
        # the predictions are sorted inside the nms func.
        e["predictions"] = temporal_non_maximum_suppression(e["predictions"][:max_before_nms],
                                                            nms_threshold=nms_thd)[:max_after_nms]
        processed_svmr_res.append(e)
    return processed_svmr_res


def get_submission_top_n(submission, top_n=100):
    def get_prediction_top_n(list_dict_predictions, top_n_):
        top_n_res = []
        for e in list_dict_predictions:
            e["predictions"] = e["predictions"][:top_n_]
            top_n_res.append(e)
        return top_n_res

    top_n_submission = dict(video2idx=submission["video2idx"], )
    for k in submission:
        if k != "video2idx":
            top_n_submission[k] = get_prediction_top_n(submission[k], top_n)
    return top_n_submission

def _masked_mean_pool(feat, mask):
    """
    feat: (N, L, H), mask: (N, L) with 1 for valid
    return: (N, H)
    """
    if feat is None or mask is None:
        return None
    m = mask.unsqueeze(-1).float()              # (N, L, 1)
    s = (feat * m).sum(dim=1)                   # (N, H)
    d = m.sum(dim=1).clamp_min(1e-6)            # (N, 1)
    return s / d


def compute_context_info(model, eval_dataset, opt):
    """Use val set to do evaluation, remember to run with torch.no_grad().
    estimated 2200 (videos) * 100 (frm) * 500 (hsz) * 4 (B) * 2 (video/sub) * 2 (layers) / (1024 ** 2) = 1.76 GB
    max_n_videos: only consider max_n_videos videos for each query to return st_ed scores.
    """
    model.eval()
    eval_dataset.set_data_mode("context")
    context_dataloader = DataLoader(eval_dataset, collate_fn=start_end_collate, batch_size=opt.eval_context_bsz,
                                    num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)
    metas = []  # list(dicts)
    video_feat, video_mask = [], []
    sub_feat, sub_mask = [], []
    for idx, batch in tqdm(enumerate(context_dataloader), desc="Computing query2video scores",
                           total=len(context_dataloader)):
        metas.extend(batch[0])
        model_inputs = prepare_batch_inputs(batch[1], device=opt.device, non_blocking=opt.pin_memory)
        _video_feat, _sub_feat = model.encode_context(model_inputs["video_feat"], model_inputs["video_mask"],
                                                      model_inputs["sub_feat"], model_inputs["sub_mask"])
        if "video" in opt.ctx_mode:
            video_feat.append(_video_feat)
            video_mask.append(model_inputs["video_mask"])
        if "sub" in opt.ctx_mode:
            sub_feat.append(_sub_feat)
            sub_mask.append(model_inputs["sub_mask"])

    def cat_tensor(tensor_list):
        if len(tensor_list) == 0:
            return None
        else:
            seq_l = [e.shape[1] for e in tensor_list]
            b_sizes = [e.shape[0] for e in tensor_list]
            b_sizes_cumsum = np.cumsum([0] + b_sizes)
            if len(tensor_list[0].shape) == 3:
                hsz = tensor_list[0].shape[2]
                res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l), hsz)
            elif len(tensor_list[0].shape) == 2:
                res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l))
            else:
                raise ValueError("Only support 2/3 dimensional tensors")
            for i, e in enumerate(tensor_list):
                res_tensor[b_sizes_cumsum[i]:b_sizes_cumsum[i+1], :seq_l[i]] = e
            return res_tensor

    return dict(
        video_metas=metas,  # list(dict) (N_videos)
        video_feat=cat_tensor(video_feat),  # (N_videos, L, hsz),
        video_mask=cat_tensor(video_mask),  # (N_videos, L)
        sub_feat=cat_tensor(sub_feat),
        sub_mask=cat_tensor(sub_mask))


def index_if_not_none(input_tensor, indices):
    if input_tensor is None:
        return input_tensor
    else:
        return input_tensor[indices]


def compute_query2ctx_info_svmr_only(model, eval_dataset, opt, ctx_info, max_before_nms=1000):
    """Use val set to do evaluation, remember to run with torch.no_grad().
    estimated size 20,000 (query) * 500 (hsz) * 4 / (1024**2) = 38.15 MB
    max_n_videos: int, use max_n_videos videos for computing VCMR results
    """
    model.eval()
    eval_dataset.set_data_mode("query")
    eval_dataset.load_gt_vid_name_for_query(True)
    query_eval_loader = DataLoader(eval_dataset, collate_fn=start_end_collate, batch_size=opt.eval_query_bsz,
                                   num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)
    video2idx = eval_dataset.video2idx
    video_metas = ctx_info["video_metas"]
    n_total_query = len(eval_dataset)
    bsz = opt.eval_query_bsz
    ctx_len = eval_dataset.max_ctx_len  # all pad to this length

    svmr_video2meta_idx = {e["vid_name"]: idx for idx, e in enumerate(video_metas)}
    svmr_gt_st_probs = np.zeros((n_total_query, ctx_len), dtype=np.float32)
    svmr_gt_ed_probs = np.zeros((n_total_query, ctx_len), dtype=np.float32)

    query_metas = []
    with torch.inference_mode():
        for idx, batch in tqdm(
                enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):
            _query_metas = batch[0]
            query_metas.extend(batch[0])
            model_inputs = prepare_batch_inputs(batch[1], device=opt.device, non_blocking=opt.pin_memory)
            # query_context_scores (_N_q, N_videos), st_prob, ed_prob (_N_q, L)
            query2video_meta_indices = torch.tensor([svmr_video2meta_idx[e["vid_name"]] for e in _query_metas], dtype=torch.long, requires_grad=False)
            _query_context_scores, _st_probs, _ed_probs = \
                model.get_pred_from_raw_query(model_inputs["query_feat"], model_inputs["query_mask"], index_if_not_none(ctx_info["video_feat"], query2video_meta_indices), index_if_not_none(ctx_info["video_mask"], query2video_meta_indices), index_if_not_none(ctx_info["sub_feat"], query2video_meta_indices), index_if_not_none(ctx_info["sub_mask"], query2video_meta_indices), cross=False)
            _query_context_scores = _query_context_scores + 1  # move cosine similarity to [0, 2]

            # normalize to get true probabilities!!!
            # the probabilities here are already (pad) masked, so only need to do softmax
            _st_probs = F.softmax(_st_probs, dim=-1)  # (_N_q, L)
            _ed_probs = F.softmax(_ed_probs, dim=-1)

            # svmr_gt_st_probs[idx * bsz:(idx + 1) * bsz, :_st_probs.shape[1]] = _st_probs.cpu().numpy()
            # svmr_gt_ed_probs[idx * bsz:(idx + 1) * bsz, :_ed_probs.shape[1]] = _ed_probs.cpu().numpy()

            bsz = _st_probs.size(0)
            start = idx * bsz
            end = start + bsz
            Ls = _st_probs.size(1)
            Le = _ed_probs.size(1)

            # —— 关键：异步 D2H，不再 .cpu().numpy() 触发同步，每步都留在 Torch 里 ——
            svmr_gt_st_probs[start:end, :Ls].copy_(_st_probs, non_blocking=True)
            svmr_gt_ed_probs[start:end, :Le].copy_(_ed_probs, non_blocking=True)


            if opt.debug:
                break
        torch.cuda.synchronize()
        svmr_gt_st_probs = svmr_gt_st_probs.numpy()
        svmr_gt_ed_probs = svmr_gt_ed_probs.numpy()
        svmr_res = get_svmr_res_from_st_ed_probs(svmr_gt_st_probs, svmr_gt_ed_probs, query_metas, video2idx, clip_length=opt.clip_length, min_pred_l=opt.min_pred_l, max_pred_l=opt.max_pred_l, max_before_nms=max_before_nms)
    return dict(SVMR=svmr_res)


def generate_min_max_length_mask(array_shape, min_l, max_l):
    """ The last two dimension denotes matrix of upper-triangle with upper-right corner masked,
    below is the case for 4x4.
    [[0, 1, 1, 0],
     [0, 0, 1, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 0]]
    Args:
        array_shape: np.shape??? The last two dimensions should be the same
        min_l: int, minimum length of predicted span
        max_l: int, maximum length of predicted span
    Returns:
    """
    single_dims = (1, ) * (len(array_shape) - 2)
    mask_shape = single_dims + array_shape[-2:]
    extra_length_mask_array = np.ones(mask_shape, dtype=np.float32)  # (1, ..., 1, L, L)
    mask_triu = np.triu(extra_length_mask_array, k=min_l)
    mask_triu_reversed = 1 - np.triu(extra_length_mask_array, k=max_l)
    final_prob_mask = mask_triu * mask_triu_reversed
    return final_prob_mask  # with valid bit to be 1


def get_svmr_res_from_st_ed_probs(svmr_gt_st_probs, svmr_gt_ed_probs, query_metas, video2idx, clip_length, min_pred_l,
                                  max_pred_l, max_before_nms):
    """
    Args:
        svmr_gt_st_probs: np.ndarray (N_queries, L, L), value range [0, 1]
        svmr_gt_ed_probs:
        query_metas:
        video2idx:
        clip_length: float, how long each clip is in seconds
        min_pred_l: int, minimum number of clips
        max_pred_l: int, maximum number of clips
        max_before_nms: get top-max_before_nms predictions for each query
    Returns:
    """
    svmr_res = []
    query_vid_names = [e["vid_name"] for e in query_metas]
    # masking very long ones! Since most are relatively short.
    st_ed_prob_product = np.einsum("bm,bn->bmn", svmr_gt_st_probs, svmr_gt_ed_probs)  # (N, L, L)
    valid_prob_mask = generate_min_max_length_mask(st_ed_prob_product.shape, min_l=min_pred_l, max_l=max_pred_l)
    st_ed_prob_product *= valid_prob_mask  # invalid location will become zero!
    batched_sorted_triples = find_max_triples_from_upper_triangle_product(st_ed_prob_product, top_n=max_before_nms,
                                                                          prob_thd=None)
    for i, q_vid_name in tqdm(enumerate(query_vid_names), desc="[SVMR] Loop over queries to generate predictions",
                              total=len(query_vid_names)):  # i is query_id
        q_m = query_metas[i]
        video_idx = video2idx[q_vid_name]
        _sorted_triples = batched_sorted_triples[i]
        # _sorted_triples[:, 1] += 1  # as we redefined ed_idx, which is inside the moment.
        _sorted_triples[:, :2] = _sorted_triples[:, :2] * clip_length
        # [video_idx(int), st(float), ed(float), score(float)]
        cur_ranked_predictions = [[video_idx, ] + row for row in _sorted_triples.tolist()]
        cur_query_pred = dict(desc_id=q_m["desc_id"], desc=q_m["desc"], predictions=cur_ranked_predictions)
        svmr_res.append(cur_query_pred)
    return svmr_res


def load_external_vr_res2(external_vr_res_path, top_n_vr_videos=5):
    """return a mapping from desc_id to top retrieved video info"""
    external_vr_res = load_json(external_vr_res_path)
    external_vr_res = get_submission_top_n(external_vr_res, top_n=top_n_vr_videos)["VR"]
    query2video = {e["desc_id"]: e["predictions"] for e in external_vr_res}
    return query2video

import contextlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

@torch.no_grad()
def compute_query2ctx_info(model, eval_dataset, opt, ctx_info, max_before_nms=1000, max_n_videos=100, tasks=("SVMR",)):
    """
    兼容原始返回：{ "SVMR": [...], "VCMR": [...], "VR": [...] }（去掉空项）
    主要优化：
      - 预分配 pinned CPU tensor + 独立 CUDA stream 做异步 D2H
      - 先 topk 再对前 k 个做 exp（节省 O(N_videos) 指数）
      - 使用 gather 选取 SVMR GT 行，避免跨设备高级索引
      - offset 写位，避免 idx*bsz 对最后一批的假设
      - 缓存 VCMR 的长度掩码
    """
    # ---- 任务标志 ----
    is_svmr = "SVMR" in tasks
    is_vr = "VR" in tasks
    is_vcmr = "VCMR" in tasks

    # ---- 准备数据/loader ----
    video2idx = eval_dataset.video2idx
    video_metas = ctx_info["video_metas"]
    video_idx2meta_idx = None
    external_query2video = None

    model.eval()
    eval_dataset.set_data_mode("query")
    eval_dataset.load_gt_vid_name_for_query(is_svmr)

    query_eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_query_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory,
        persistent_workers=True if getattr(opt, "num_workers", 0) > 0 else False,
    )

    device = torch.device(opt.device)
    use_amp = bool(getattr(opt, "use_amp", True) and device.type == "cuda")

    n_total_query = len(eval_dataset)
    ctx_len = int(getattr(opt, "max_ctx_l", eval_dataset.max_ctx_len))

    # ---- 预分配 pinned CPU 缓冲（循环内异步写入，循环后一次性 numpy） ----
    if is_vcmr:
        flat_st_ed_scores_sorted_indices_t = torch.empty((n_total_query, max_before_nms), dtype=torch.int64, device="cpu", pin_memory=True)
        flat_st_ed_sorted_scores_t         = torch.empty((n_total_query, max_before_nms), dtype=torch.float32, device="cpu", pin_memory=True)
    else:
        flat_st_ed_scores_sorted_indices_t, flat_st_ed_sorted_scores_t = None, None

    if is_vr or is_vcmr:
        sorted_q2c_indices_t = torch.empty((n_total_query, max_n_videos), dtype=torch.int64, device="cpu", pin_memory=True)
        sorted_q2c_scores_t  = torch.empty((n_total_query, max_n_videos), dtype=torch.float32, device="cpu", pin_memory=True)
    else:
        sorted_q2c_indices_t, sorted_q2c_scores_t = None, None

    if is_svmr:
        svmr_video2meta_idx = {e["vid_name"]: idx for idx, e in enumerate(video_metas)}
        svmr_gt_st_probs_t = torch.zeros((n_total_query, ctx_len), dtype=torch.float32, device="cpu", pin_memory=True)
        svmr_gt_ed_probs_t = torch.zeros((n_total_query, ctx_len), dtype=torch.float32, device="cpu", pin_memory=True)
    else:
        svmr_video2meta_idx, svmr_gt_st_probs_t, svmr_gt_ed_probs_t = None, None, None

    # ---- 其它状态 ----
    query_metas = []
    offset = 0  # 已写入样本计数
    copy_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    mask_cache = {}  # {L: torch.BoolTensor on device}

    # ---- 主循环 ----
    for bi, batch in tqdm(enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):
        _query_metas = batch[0]
        query_metas.extend(_query_metas)
        model_inputs = prepare_batch_inputs(batch[1], device=device, non_blocking=opt.pin_memory)

        # 前向：(_N_q, N_videos), (_N_q, N_videos, L), (_N_q, N_videos, L)
        with torch.cuda.amp.autocast(device_type="cuda", enabled=use_amp):
            _q2c, _st_logits, _ed_logits = model.get_pred_from_raw_query(
                model_inputs["query_feat"], model_inputs["query_mask"],
                ctx_info["video_feat"], ctx_info["video_mask"],
                ctx_info["sub_feat"],  ctx_info["sub_mask"],
                cross=True
            )

            # 概率（时间维 softmax；logits 已带 pad mask）
            _st_probs = F.softmax(_st_logits, dim=-1)
            _ed_probs = F.softmax(_ed_logits, dim=-1)

        B, Nv, L = _st_probs.shape

        # ---------- SVMR：收集 GT 视频的一行 ----------
        if is_svmr:
            gt_meta_idx = torch.as_tensor(
                [svmr_video2meta_idx[e["vid_name"]] for e in _query_metas],
                dtype=torch.long, device=_st_probs.device
            )  # (B,)
            idx3 = gt_meta_idx.view(B, 1, 1).expand(-1, 1, L)        # (B,1,L)
            st_sel = _st_probs.gather(1, idx3).squeeze(1).contiguous()  # (B,L)
            ed_sel = _ed_probs.gather(1, idx3).squeeze(1).contiguous()  # (B,L)

            if device.type == "cuda":
                cur_stream = torch.cuda.current_stream(device=device)
                copy_stream.wait_stream(cur_stream)
                with torch.cuda.stream(copy_stream):
                    svmr_gt_st_probs_t.narrow(0, offset, B).narrow(1, 0, L).copy_(st_sel, non_blocking=True)
                    svmr_gt_ed_probs_t.narrow(0, offset, B).narrow(1, 0, L).copy_(ed_sel, non_blocking=True)
            else:
                svmr_gt_st_probs_t.narrow(0, offset, B).narrow(1, 0, L).copy_(st_sel)
                svmr_gt_ed_probs_t.narrow(0, offset, B).narrow(1, 0, L).copy_(ed_sel)

        # 若不需要 VR/VCMR，跳过后续
        if not (is_vr or is_vcmr):
            offset += B
            if getattr(opt, "debug", False): break
            continue

        # ---------- VR/VCMR：先 topk 再对 topk 做 exp ----------
        # 与原实现等价：原先是 exp(alpha*s) 后 topk；这里用 scaled=alpha*s 先 topk，再对 topk exp
        scaled = opt.q2c_alpha * _q2c  # (B, Nv)
        if external_query2video is None:
            _topk_scaled, _topk_idx = torch.topk(scaled, max_n_videos, dim=1, largest=True)  # (B, K)
            _topk_scores = torch.exp(_topk_scaled)  # 等价于对原实现的 top-k 值做 exp
        else:
            # 兼容外部提供的候选（通常在 CPU 列表上）
            relevant_video_info = [external_query2video[qm["desc_id"]] for qm in _query_metas]
            _topk_idx = torch.as_tensor(
                [[video_idx2meta_idx[sub_e[0]] for sub_e in e] for e in relevant_video_info],
                dtype=torch.long, device=scaled.device
            )
            raw_scores = torch.as_tensor(
                [[sub_e[3] for sub_e in e] for e in relevant_video_info],
                dtype=scaled.dtype, device=scaled.device
            )
            _topk_scores = torch.exp(opt.q2c_alpha * raw_scores)  # (B, K)

        K = _topk_idx.size(1)

        # 拷贝 topk 的索引/分数（异步 D2H）
        if device.type == "cuda":
            cur_stream = torch.cuda.current_stream(device=device)
            copy_stream.wait_stream(cur_stream)
            with torch.cuda.stream(copy_stream):
                sorted_q2c_indices_t.narrow(0, offset, B).narrow(1, 0, K).copy_(_topk_idx.to("cpu", non_blocking=True), non_blocking=True)
                sorted_q2c_scores_t.narrow(0, offset, B).narrow(1, 0, K).copy_(_topk_scores.to("cpu", non_blocking=True), non_blocking=True)
        else:
            sorted_q2c_indices_t.narrow(0, offset, B).narrow(1, 0, K).copy_(_topk_idx.cpu())
            sorted_q2c_scores_t.narrow(0, offset, B).narrow(1, 0, K).copy_(_topk_scores.cpu())

        # ---------- VCMR：只对 top-k 视频计算组合得分 ----------
        if is_vcmr:
            # 选出 top-k 上的 (B,K,L) 概率：用 gather 避免高级索引
            idx3k = _topk_idx.view(B, K, 1).expand(-1, -1, L)         # (B,K,L)
            st_topk = _st_probs.gather(1, idx3k)                      # (B,K,L)
            ed_topk = _ed_probs.gather(1, idx3k)                      # (B,K,L)

            # 组合：st * q2c * ed -> (B,K,L,L)
            _st_ed_scores = torch.einsum("bkl,bk,bkn->bkln", st_topk, _topk_scores, ed_topk)

            # 有效起止长度掩码（缓存按 L）
            if L not in mask_cache:
                valid_mask_np = generate_min_max_length_mask(_st_ed_scores.shape, min_l=opt.min_pred_l, max_l=opt.max_pred_l)
                mask_cache[L] = torch.from_numpy(valid_mask_np).to(_st_ed_scores.device, non_blocking=True)
            _st_ed_scores *= mask_cache[L]

            # 展平排序，取前 max_before_nms
            _flat = _st_ed_scores.reshape(B, -1)  # (B, K*L*L)
            _vals, _inds = torch.sort(_flat, dim=1, descending=True)
            _vals = _vals[:, :max_before_nms].contiguous()
            _inds = _inds[:, :max_before_nms].contiguous()

            # 异步拷贝
            if device.type == "cuda":
                cur_stream = torch.cuda.current_stream(device=device)
                copy_stream.wait_stream(cur_stream)
                with torch.cuda.stream(copy_stream):
                    flat_st_ed_sorted_scores_t.narrow(0, offset, B).copy_(_vals, non_blocking=True)
                    flat_st_ed_scores_sorted_indices_t.narrow(0, offset, B).copy_(_inds, non_blocking=True)
            else:
                flat_st_ed_sorted_scores_t.narrow(0, offset, B).copy_(_vals)
                flat_st_ed_scores_sorted_indices_t.narrow(0, offset, B).copy_(_inds)

        offset += B
        if getattr(opt, "debug", False):
            break

    # ---- 等待所有异步拷贝完成，再 numpy ----
    if device.type == "cuda":
        copy_stream.synchronize()

    if is_svmr:
        svmr_gt_st_probs = svmr_gt_st_probs_t.numpy()
        svmr_gt_ed_probs = svmr_gt_ed_probs_t.numpy()
    else:
        svmr_gt_st_probs = svmr_gt_ed_probs = None

    if is_vr or is_vcmr:
        sorted_q2c_indices = sorted_q2c_indices_t.numpy()
        sorted_q2c_scores  = sorted_q2c_scores_t.numpy()
    else:
        sorted_q2c_indices = sorted_q2c_scores = None

    if is_vcmr:
        flat_st_ed_scores_sorted_indices = flat_st_ed_scores_sorted_indices_t.numpy()
        flat_st_ed_sorted_scores = flat_st_ed_sorted_scores_t.numpy()
    else:
        flat_st_ed_scores_sorted_indices = flat_st_ed_sorted_scores = None

    # ---- 组装 SVMR 结果 ----
    svmr_res = []
    if is_svmr:
        svmr_res = get_svmr_res_from_st_ed_probs(
            svmr_gt_st_probs, svmr_gt_ed_probs, query_metas, video2idx,
            clip_length=opt.clip_length, min_pred_l=opt.min_pred_l,
            max_pred_l=opt.max_pred_l, max_before_nms=max_before_nms
        )

    # ---- 组装 VR 结果 ----
    vr_res = []
    if is_vr:
        for i, (_scores_row, _indices_row) in tqdm(
            enumerate(zip(sorted_q2c_scores[:, :100], sorted_q2c_indices[:, :100])),
            desc="[VR] Loop over queries to generate predictions", total=n_total_query
        ):
            cur_vr_predictions = []
            for v_score, v_meta_idx in zip(_scores_row, _indices_row):
                video_idx = video2idx[video_metas[int(v_meta_idx)]["vid_name"]]
                cur_vr_predictions.append([video_idx, 0, 0, float(v_score)])
            cur_query_pred = dict(desc_id=query_metas[i]['desc_id'], desc=query_metas[i]["desc"],
                                  predictions=cur_vr_predictions)
            vr_res.append(cur_query_pred)

    # ---- 组装 VCMR 结果 ----
    vcmr_res = []
    if is_vcmr:
        for i, (_inds_row, _vals_row) in tqdm(
            enumerate(zip(flat_st_ed_scores_sorted_indices, flat_st_ed_sorted_scores)),
            desc="[VCMR] Loop over queries to generate predictions", total=n_total_query
        ):
            # indices 是在 (K, L, L) 展平后的局部索引
            video_meta_indices_local, pred_st_indices, pred_ed_indices = np.unravel_index(
                _inds_row, shape=(max_n_videos, ctx_len, ctx_len)
            )
            # 本地 top-k -> 全局 meta idx
            video_meta_indices = sorted_q2c_indices[i, video_meta_indices_local]
            pred_st_in_seconds = pred_st_indices.astype(np.float32) * opt.clip_length
            pred_ed_in_seconds = pred_ed_indices.astype(np.float32) * opt.clip_length

            cur_vcmr_predictions = []
            for j, (v_meta_idx, v_score) in enumerate(zip(video_meta_indices, _vals_row)):
                video_idx = video2idx[video_metas[int(v_meta_idx)]["vid_name"]]
                cur_vcmr_predictions.append([video_idx, float(pred_st_in_seconds[j]), float(pred_ed_in_seconds[j]), float(v_score)])

            cur_query_pred = dict(desc_id=query_metas[i]["desc_id"], desc=query_metas[i]["desc"],
                                  predictions=cur_vcmr_predictions)
            vcmr_res.append(cur_query_pred)

    res = dict(SVMR=svmr_res, VCMR=vcmr_res, VR=vr_res)
    return {k: v for k, v in res.items() if len(v) != 0}


def get_eval_res(model, eval_dataset, opt, tasks):
    """compute and save query and video proposal embeddings"""
    context_info = compute_context_info(model, eval_dataset, opt)
    if "VCMR" in tasks or "VR" in tasks:
        logger.info("Inference with full-script.")
        eval_res = compute_query2ctx_info(model, eval_dataset, opt, context_info, max_before_nms=opt.max_before_nms,
                                          max_n_videos=opt.max_vcmr_video, tasks=tasks)
    else:
        logger.info("Inference at [SVMR only] mode. This script is different.")
        eval_res = compute_query2ctx_info_svmr_only(model, eval_dataset, opt, context_info,
                                                    max_before_nms=opt.max_before_nms)
    eval_res["video2idx"] = eval_dataset.video2idx
    return eval_res


POST_PROCESSING_MMS_FUNC = {"SVMR": post_processing_svmr_nms, "VCMR": post_processing_vcmr_nms}


def eval_epoch(model, eval_dataset, opt, save_submission_filename, tasks=("SVMR",), max_after_nms=100):
    """max_after_nms: always set to 100, since the eval script only evaluate top-100"""
    model.eval()
    logger.info("Computing scores")
    st_time = time.time()
    eval_submission_raw = get_eval_res(model, eval_dataset, opt, tasks)
    total_time = time.time() - st_time
    print("\n" + "\x1b[1;31m" + str(total_time) + "\x1b[0m", flush=True)

    IOU_THDS = (0.5, 0.7)  # (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    eval_submission = get_submission_top_n(eval_submission_raw, top_n=max_after_nms)
    save_json(eval_submission, submission_path)

    if opt.eval_split_name == "val":  # since test_public has no GT
        metrics = eval_retrieval(eval_submission, eval_dataset.query_data, iou_thds=IOU_THDS,
                                 match_number=not opt.debug, verbose=opt.debug, use_desc_type=opt.dset_name == "tvr")
        save_metrics_path = submission_path.replace(".json", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [submission_path, ]

    if opt.nms_thd != -1:
        logger.info("Performing nms with nms_thd {}".format(opt.nms_thd))
        eval_submission_after_nms = dict(video2idx=eval_submission_raw["video2idx"])
        for k, nms_func in POST_PROCESSING_MMS_FUNC.items():
            if k in eval_submission_raw:
                eval_submission_after_nms[k] = nms_func(eval_submission_raw[k], nms_thd=opt.nms_thd,
                                                        max_before_nms=opt.max_before_nms, max_after_nms=max_after_nms)
        logger.info("Saving/Evaluating nms results")
        submission_nms_path = submission_path.replace(".json", "_nms_thd_{}.json".format(opt.nms_thd))
        save_json(eval_submission_after_nms, submission_nms_path)
        if opt.eval_split_name == "val":
            metrics_nms = eval_retrieval(eval_submission_after_nms, eval_dataset.query_data, iou_thds=IOU_THDS,
                                         match_number=not opt.debug, verbose=opt.debug)
            save_metrics_nms_path = submission_nms_path.replace(".json", "_metrics.json")
            save_json(metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False)
            latest_file_paths += [submission_nms_path, save_metrics_nms_path]
        else:
            metrics_nms = None
            latest_file_paths = [submission_nms_path, ]
    else:
        metrics_nms = None
    return metrics, metrics_nms, latest_file_paths


def setup_model(opt):
    """Load model from checkpoint and move to specified device"""
    checkpoint = torch.load(opt.ckpt_filepath)
    loaded_model_cfg = checkpoint["model_cfg"]
    model = ReLoCLNet(loaded_model_cfg)
    model.load_state_dict(checkpoint["model"])
    logger.info("Loaded model saved at epoch {} from checkpoint: {}".format(checkpoint["epoch"], opt.ckpt_filepath))

    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        if len(opt.device_ids) > 1:
            logger.info("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU
    return model


def start_inference():
    logger.info("Setup config, data and model...")
    opt = TestOptions().parse()
    cudnn.benchmark = False
    cudnn.deterministic = True

    assert opt.eval_path is not None
    eval_dataset = StartEndEvalDataset(
        dset_name=opt.dset_name,
        eval_split_name=opt.eval_split_name,  # should only be val set
        data_path=opt.eval_path,
        desc_bert_path_or_handler=opt.desc_bert_path,
        sub_bert_path_or_handler=opt.sub_bert_path,
        max_desc_len=opt.max_desc_l,
        max_ctx_len=opt.max_ctx_l,
        video_duration_idx_path=opt.video_duration_idx_path,
        vid_feat_path_or_handler=opt.vid_feat_path,
        clip_length=opt.clip_length,
        ctx_mode=opt.ctx_mode,
        data_mode="query",
        h5driver=opt.h5driver,
        data_ratio=opt.data_ratio,
        normalize_vfeat=not opt.no_norm_vfeat,
        normalize_tfeat=not opt.no_norm_tfeat)

    model = setup_model(opt)
    save_submission_filename = "inference_{}_{}_{}_predictions_{}.json".format(
        opt.dset_name, opt.eval_split_name, opt.eval_id, "_".join(opt.tasks))
    logger.info("Starting inference...")
    with torch.no_grad():
        metrics_no_nms, metrics_nms, latest_file_paths = eval_epoch(model, eval_dataset, opt, save_submission_filename,
                                                                    tasks=opt.tasks, max_after_nms=100)
    logger.info("metrics_no_nms \n{}".format(pprint.pformat(metrics_no_nms, indent=4)))
    logger.info("metrics_nms \n{}".format(pprint.pformat(metrics_nms, indent=4)))


if __name__ == '__main__':
    start_inference()
