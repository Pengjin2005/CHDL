import logging
import os
import pprint
import time
from collections import defaultdict
from time import perf_counter

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from chdl.config import TestOptions
from chdl.dataset import (
    StartEndEvalDataset,
    prepare_batch_inputs,
    start_end_collate,
)
from chdl.model import CHDL
from eval.eval import eval_retrieval
from utils.basic_utils import load_json, save_json
from utils.temporal_nms import temporal_non_maximum_suppression
from utils.tensor_utils import find_max_triples_from_upper_triangle_product


def _sync_if_cuda(opt):
    if getattr(opt, "device", None) is not None and getattr(opt.device, "type", "") == "cuda":
        torch.cuda.synchronize()

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
    n_total_query = len(eval_dataset)
    timing = {
        "model_forward_s": 0.0,     # 前向 + 生成 st/ed 概率
        "svmr_decode_s": 0.0,       # 仅提取/写入 SVMR 概率矩阵部分
        "n_queries": n_total_query,
        "n_batches": 0,
    }

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
            
            timing["n_batches"] += 1

            _query_metas = batch[0]
            query_metas.extend(batch[0])
            model_inputs = prepare_batch_inputs(batch[1], device=opt.device, non_blocking=opt.pin_memory)
            # query_context_scores (_N_q, N_videos), st_prob, ed_prob (_N_q, L)
            query2video_meta_indices = torch.tensor([svmr_video2meta_idx[e["vid_name"]] for e in _query_metas],
                                                    dtype=torch.long, requires_grad=False)


            _sync_if_cuda(opt)
            t0 = perf_counter()
            _query_context_scores, _st_probs, _ed_probs = \
                model.get_pred_from_raw_query(model_inputs["query_feat"], model_inputs["query_mask"],
                                            index_if_not_none(ctx_info["video_feat"], query2video_meta_indices),
                                            index_if_not_none(ctx_info["video_mask"], query2video_meta_indices),
                                            index_if_not_none(ctx_info["sub_feat"], query2video_meta_indices),
                                            index_if_not_none(ctx_info["sub_mask"], query2video_meta_indices),
                                            cross=False)
            _sync_if_cuda(opt)
            timing["model_forward_s"] += (perf_counter() - t0)

            _query_context_scores = _query_context_scores + 1  # move cosine similarity to [0, 2]

            # normalize to get true probabilities!!!
            # the probabilities here are already (pad) masked, so only need to do softmax
            _st_probs = F.softmax(_st_probs, dim=-1)  # (_N_q, L)
            _ed_probs = F.softmax(_ed_probs, dim=-1)

            # svmr_gt_st_probs[idx * bsz:(idx + 1) * bsz, :_st_probs.shape[1]] = _st_probs.cpu().numpy()
            # svmr_gt_ed_probs[idx * bsz:(idx + 1) * bsz, :_ed_probs.shape[1]] = _ed_probs.cpu().numpy()

            # bsz = _st_probs.size(0)
            # start = idx * bsz
            # end = start + bsz
            # Ls = _st_probs.size(1)
            # Le = _ed_probs.size(1)

            t1 = perf_counter()
            bsz = _st_probs.size(0)
            start = idx * bsz
            end = start + bsz
            Ls = _st_probs.size(1)
            Le = _ed_probs.size(1)

            # —— 关键：异步 D2H，不再 .cpu().numpy() 触发同步，每步都留在 Torch 里 ——
            svmr_gt_st_probs[start:end, :Ls].copy_(_st_probs, non_blocking=True)
            svmr_gt_ed_probs[start:end, :Le].copy_(_ed_probs, non_blocking=True)
            _sync_if_cuda(opt)  # 保守同步，确保统计稳定
            timing["svmr_decode_s"] += (perf_counter() - t1)

            if opt.debug:
                break
        torch.cuda.synchronize()
        svmr_gt_st_probs = svmr_gt_st_probs.numpy()
        svmr_gt_ed_probs = svmr_gt_ed_probs.numpy()
        svmr_res = get_svmr_res_from_st_ed_probs(svmr_gt_st_probs, svmr_gt_ed_probs, query_metas, video2idx,
                                                clip_length=opt.clip_length, min_pred_l=opt.min_pred_l,
                                                max_pred_l=opt.max_pred_l, max_before_nms=max_before_nms)
    return dict(SVMR=svmr_res), timing


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


def compute_query2ctx_info(model, eval_dataset, opt, ctx_info, max_before_nms=1000, max_n_videos=100, tasks=("SVMR",)):
    """Use val set to do evaluation, remember to run with torch.no_grad().
    estimated size 20,000 (query) * 500 (hsz) * 4 / (1024**2) = 38.15 MB
    max_n_videos: int, use max_n_videos videos for computing VCMR/VR results
    """
    is_svmr = "SVMR" in tasks
    is_vr = "VR" in tasks
    is_vcmr = "VCMR" in tasks
    video2idx = eval_dataset.video2idx
    video_metas = ctx_info["video_metas"]
    video_idx2meta_idx = None
    external_query2video = None
    model.eval()
    eval_dataset.set_data_mode("query")
    eval_dataset.load_gt_vid_name_for_query(is_svmr)
    query_eval_loader = DataLoader(eval_dataset, collate_fn=start_end_collate, batch_size=opt.eval_query_bsz, num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)
    n_total_query = len(eval_dataset)
    bsz = opt.eval_query_bsz

    timing = {
        "model_forward_s": 0.0,  # 前向 + 生成 q2c 分数、st/ed 概率
        "svmr_decode_s": 0.0,    # 仅 SVMR 的提取/写入
        "vr_rank_s": 0.0,        # VR 的 topk 排序 + 收集
        "vcmr_decode_s": 0.0,    # VCMR 的时序打分组合/排序
        "n_batches": 0,
        "n_queries": n_total_query  # <--- 在这里添加这一行
    }

    if is_vcmr:
        flat_st_ed_scores_sorted_indices = np.empty((n_total_query, max_before_nms), dtype=np.int32)
        flat_st_ed_sorted_scores = np.zeros((n_total_query, max_before_nms), dtype=np.float32)
    else:
        flat_st_ed_scores_sorted_indices, flat_st_ed_sorted_scores = None, None

    if is_vr or is_vcmr:
        sorted_q2c_indices = np.empty((n_total_query, max_n_videos), dtype=np.int32)
        sorted_q2c_scores = np.empty((n_total_query, max_n_videos), dtype=np.float32)
    else:
        sorted_q2c_indices, sorted_q2c_scores = None, None

    if is_svmr:
        svmr_video2meta_idx = {e["vid_name"]: idx for idx, e in enumerate(video_metas)}
        svmr_gt_st_probs = np.zeros((n_total_query, opt.max_ctx_l), dtype=np.float32)
        svmr_gt_ed_probs = np.zeros((n_total_query, opt.max_ctx_l), dtype=np.float32)
    else:
        svmr_video2meta_idx, svmr_gt_st_probs, svmr_gt_ed_probs = None, None, None

    query_metas = []
    for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):
        timing["n_batches"] += 1
        
        _query_metas = batch[0]
        query_metas.extend(batch[0])
        model_inputs = prepare_batch_inputs(batch[1], device=opt.device, non_blocking=opt.pin_memory)
        # query_context_scores (_N_q, N_videos), st_prob, ed_prob (_N_q, N_videos, L)
        # _query_context_scores, _st_probs, _ed_probs = model.get_pred_from_raw_query(
        #     model_inputs["query_feat"], model_inputs["query_mask"], ctx_info["video_feat"], ctx_info["video_mask"],
        #     ctx_info["sub_feat"], ctx_info["sub_mask"], cross=True)
        # _query_context_scores = _query_context_scores + 1  # move cosine similarity to [0, 2]
        # To give more importance to top scores, the higher opt.alpha is the more importance will be given


        _query_context_scores, _st_probs, _ed_probs = model.get_pred_from_raw_query(
            model_inputs["query_feat"], model_inputs["query_mask"],
            ctx_info["video_feat"], ctx_info["video_mask"],
            ctx_info["sub_feat"], ctx_info["sub_mask"], cross=True, timing_dict=timing)

        _query_context_scores = torch.exp(opt.q2c_alpha * _query_context_scores)
        # normalize to get true probabilities!!!
        # the probabilities here are already (pad) masked, so only need to do softmax
        _st_probs = F.softmax(_st_probs, dim=-1)  # (_N_q, N_videos, L)
        _ed_probs = F.softmax(_ed_probs, dim=-1)

        if is_svmr:  # collect SVMR data
            # row_indices = torch.arange(0, len(_st_probs))
            # query2video_meta_indices = torch.tensor([svmr_video2meta_idx[e["vid_name"]] for e in _query_metas],
            #                                         dtype=torch.long)
            # svmr_gt_st_probs[idx * bsz:(idx + 1) * bsz, :_st_probs.shape[2]] = \
            #     _st_probs[row_indices, query2video_meta_indices].cpu().numpy()
            # svmr_gt_ed_probs[idx * bsz:(idx + 1) * bsz, :_ed_probs.shape[2]] = \
            #     _ed_probs[row_indices, query2video_meta_indices].cpu().numpy()
            t_s = perf_counter()
            row_indices = torch.arange(0, len(_st_probs))
            query2video_meta_indices = torch.tensor([svmr_video2meta_idx[e["vid_name"]] for e in _query_metas],
                                                    dtype=torch.long)
            svmr_gt_st_probs[idx * bsz:(idx + 1) * bsz, :_st_probs.shape[2]] = \
                _st_probs[row_indices, query2video_meta_indices].cpu().numpy()
            svmr_gt_ed_probs[idx * bsz:(idx + 1) * bsz, :_ed_probs.shape[2]] = \
                _ed_probs[row_indices, query2video_meta_indices].cpu().numpy()
            torch.cuda.synchronize()
            timing["svmr_decode_s"] += (perf_counter() - t_s)

        if not (is_vr or is_vcmr):
            continue

        # Get top-max_n_videos videos for each query
        t_vr = perf_counter()
        if external_query2video is None:
            _sorted_q2c_scores, _sorted_q2c_indices = torch.topk(_query_context_scores, max_n_videos, dim=1,
                                                                 largest=True)
        else:
            relevant_video_info = [external_query2video[qm["desc_id"]] for qm in _query_metas]
            _sorted_q2c_indices = _query_context_scores.new_tensor([[video_idx2meta_idx[sub_e[0]] for sub_e in e] for e in relevant_video_info], dtype=torch.long)
            _sorted_q2c_scores = _query_context_scores.new_tensor([[sub_e[3] for sub_e in e] for e in relevant_video_info])
            _sorted_q2c_scores = torch.exp(opt.q2c_alpha * _sorted_q2c_scores)
        torch.cuda.synchronize()
        timing["vr_rank_s"] += (perf_counter() - t_vr)

        # collect data for vr and backup_vcmr
        sorted_q2c_indices[idx * bsz:(idx + 1) * bsz] = _sorted_q2c_indices.cpu().numpy()
        sorted_q2c_scores[idx * bsz:(idx + 1) * bsz] = _sorted_q2c_scores.cpu().numpy()

        if not is_vcmr:
            continue
        # Get VCMR results
        # compute combined scores
        # row_indices = torch.arange(0, len(_st_probs), device=opt.device).unsqueeze(1)
        # _st_probs = _st_probs[row_indices, _sorted_q2c_indices]  # (_N_q, max_n_videos, L)
        # _ed_probs = _ed_probs[row_indices, _sorted_q2c_indices]
        # # (_N_q, max_n_videos, L, L)
        # _st_ed_scores = torch.einsum("qvm,qv,qvn->qvmn", _st_probs, _sorted_q2c_scores, _ed_probs)
        # valid_prob_mask = generate_min_max_length_mask(_st_ed_scores.shape, min_l=opt.min_pred_l, max_l=opt.max_pred_l)
        # _st_ed_scores *= torch.from_numpy(valid_prob_mask).to(_st_ed_scores.device)  # invalid location will become zero
        # # sort across the top-max_n_videos videos (by flatten from the 2nd dim)
        # # the indices here are local indices, not global indices
        # _n_q = _st_ed_scores.shape[0]
        # _flat_st_ed_scores = _st_ed_scores.reshape(_n_q, -1)  # (N_q, max_n_videos*L*L)
        # _flat_st_ed_sorted_scores, _flat_st_ed_scores_sorted_indices = torch.sort(_flat_st_ed_scores, dim=1,
        #                                                                           descending=True)
        
        t_vcmr = perf_counter()
        row_indices = torch.arange(0, len(_st_probs), device=opt.device).unsqueeze(1)
        _st_probs_sel = _st_probs[row_indices, _sorted_q2c_indices]  # (N_q, V, L)
        _ed_probs_sel = _ed_probs[row_indices, _sorted_q2c_indices]
        _st_ed_scores = torch.einsum("qvm,qv,qvn->qvmn", _st_probs_sel, _sorted_q2c_scores, _ed_probs_sel)
        valid_prob_mask = generate_min_max_length_mask(_st_ed_scores.shape, min_l=opt.min_pred_l, max_l=opt.max_pred_l)
        _st_ed_scores *= torch.from_numpy(valid_prob_mask).to(_st_ed_scores.device)
        _n_q = _st_ed_scores.shape[0]
        _flat = _st_ed_scores.reshape(_n_q, -1)
        _flat_st_ed_sorted_scores, _flat_st_ed_scores_sorted_indices = torch.sort(_flat, dim=1, descending=True)
        _sync_if_cuda(opt)
        timing["vcmr_decode_s"] += (perf_counter() - t_vcmr)
        
        # collect data
        flat_st_ed_sorted_scores[idx * bsz:(idx + 1)*bsz] = _flat_st_ed_sorted_scores[:, :max_before_nms].cpu().numpy()
        flat_st_ed_scores_sorted_indices[idx * bsz:(idx + 1) * bsz] = \
            _flat_st_ed_scores_sorted_indices[:, :max_before_nms].cpu().numpy()
        if opt.debug:
            break

    svmr_res = []
    if is_svmr:
        svmr_res = get_svmr_res_from_st_ed_probs(svmr_gt_st_probs, svmr_gt_ed_probs, query_metas, video2idx,
                                                 clip_length=opt.clip_length, min_pred_l=opt.min_pred_l,
                                                 max_pred_l=opt.max_pred_l, max_before_nms=max_before_nms)
    vr_res = []
    if is_vr:
        for i, (_sorted_q2c_scores_row, _sorted_q2c_indices_row) in tqdm(
                enumerate(zip(sorted_q2c_scores[:, :100], sorted_q2c_indices[:, :100])),
                desc="[VR] Loop over queries to generate predictions", total=n_total_query):
            cur_vr_redictions = []
            for j, (v_score, v_meta_idx) in enumerate(zip(_sorted_q2c_scores_row, _sorted_q2c_indices_row)):
                video_idx = video2idx[video_metas[v_meta_idx]["vid_name"]]
                cur_vr_redictions.append([video_idx, 0, 0, float(v_score)])
            cur_query_pred = dict(desc_id=query_metas[i]['desc_id'], desc=query_metas[i]["desc"],
                                  predictions=cur_vr_redictions)
            vr_res.append(cur_query_pred)

    vcmr_res = []
    if is_vcmr:
        for i, (_flat_st_ed_scores_sorted_indices, _flat_st_ed_sorted_scores) in tqdm(
                enumerate(zip(flat_st_ed_scores_sorted_indices, flat_st_ed_sorted_scores)),
                desc="[VCMR] Loop over queries to generate predictions", total=n_total_query):  # i is query_idx
            # list([video_idx(int), st(float), ed(float), score(float)])
            video_meta_indices_local, pred_st_indices, pred_ed_indices = np.unravel_index(
                _flat_st_ed_scores_sorted_indices, shape=(max_n_videos, opt.max_ctx_l, opt.max_ctx_l))
            # video_meta_indices_local refers to the indices among the top-max_n_videos
            # video_meta_indices refers to the indices in all the videos, which is the True indices
            video_meta_indices = sorted_q2c_indices[i, video_meta_indices_local]
            pred_st_in_seconds = pred_st_indices.astype(np.float32) * opt.clip_length
            pred_ed_in_seconds = pred_ed_indices.astype(np.float32) * opt.clip_length  # + opt.clip_length
            cur_vcmr_redictions = []
            for j, (v_meta_idx, v_score) in enumerate(zip(video_meta_indices, _flat_st_ed_sorted_scores)):  # videos
                video_idx = video2idx[video_metas[v_meta_idx]["vid_name"]]
                cur_vcmr_redictions.append([video_idx, float(pred_st_in_seconds[j]), float(pred_ed_in_seconds[j]),
                                            float(v_score)])
            cur_query_pred = dict(desc_id=query_metas[i]["desc_id"], desc=query_metas[i]["desc"],
                                  predictions=cur_vcmr_redictions)
            vcmr_res.append(cur_query_pred)

    res = dict(SVMR=svmr_res, VCMR=vcmr_res, VR=vr_res)
    return {k: v for k, v in res.items() if len(v) != 0}, timing


def get_eval_res(model, eval_dataset, opt, tasks):
    """compute and save query and video proposal embeddings"""
    context_info = compute_context_info(model, eval_dataset, opt)
    if "VCMR" in tasks or "VR" in tasks:
        logger.info("Inference with full-script.")
        eval_res, timing = compute_query2ctx_info(model, eval_dataset, opt, context_info, max_before_nms=opt.max_before_nms,
                                          max_n_videos=opt.max_vcmr_video, tasks=tasks)
    else:
        logger.info("Inference at [SVMR only] mode. This script is different.")
        eval_res, timing = compute_query2ctx_info_svmr_only(model, eval_dataset, opt, context_info,
                                                    max_before_nms=opt.max_before_nms)
    eval_res["video2idx"] = eval_dataset.video2idx
    return eval_res, timing


def create_offline_index(model, eval_dataset, opt):
    """
    [离线步骤] 遍历所有上下文数据，计算特征和哈希码，并保存到磁盘。
    """
    logger.info("Starting offline index creation...")
    model.eval()
    eval_dataset.set_data_mode("context")
    context_dataloader = DataLoader(eval_dataset, collate_fn=start_end_collate, batch_size=opt.eval_context_bsz,
                                  num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)
    
    metas = []
    # 收集所有批次的索引数据
    all_indexed_data_batches = {
        "vr_video_hash": [], "vr_sub_hash": [],
        "mr_video_feat": [], "mr_sub_feat": [],
        "video_mask": []
    }
    
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(context_dataloader), desc="Indexing context batches",
                               total=len(context_dataloader)):
            metas.extend(batch[0])
            model_inputs = prepare_batch_inputs(batch[1], device=opt.device, non_blocking=opt.pin_memory)
            
            # 使用新的模型函数获取此批次的索引数据
            indexed_batch = model.export_context_index(
                model_inputs["video_feat"], model_inputs["video_mask"],
                model_inputs["sub_feat"], model_inputs["sub_mask"]
            )
            
            # 收集数据块 (转移到CPU以节省VRAM)
            all_indexed_data_batches["vr_video_hash"].append(indexed_batch["vr_video_hash"].cpu())
            if indexed_batch["vr_sub_hash"] is not None:
                all_indexed_data_batches["vr_sub_hash"].append(indexed_batch["vr_sub_hash"].cpu())
            
            all_indexed_data_batches["mr_video_feat"].append(indexed_batch["mr_video_feat"].cpu())
            if indexed_batch["mr_sub_feat"] is not None:
                all_indexed_data_batches["mr_sub_feat"].append(indexed_batch["mr_sub_feat"].cpu())
                
            all_indexed_data_batches["video_mask"].append(indexed_batch["video_mask"].cpu())

    logger.info("Concatenating index tensors...")
    
    # 辅助函数：连接(B, L, D)或(B, L)张量，处理padding
    def cat_tensor_with_padding(tensor_list):
        if not tensor_list: 
            return None
        max_l = max(t.shape[1] for t in tensor_list)
        total_b = sum(t.shape[0] for t in tensor_list)
        
        if len(tensor_list[0].shape) == 3: # (B, L, D)
            hsz = tensor_list[0].shape[2]
            out_tensor = torch.zeros((total_b, max_l, hsz), dtype=tensor_list[0].dtype)
        else: # (B, L)
            out_tensor = torch.zeros((total_b, max_l), dtype=tensor_list[0].dtype)
        
        b_start = 0
        for t in tensor_list:
            b_end = b_start + t.shape[0]
            l_t = t.shape[1]
            if len(t.shape) == 3:
                out_tensor[b_start:b_end, :l_t, :] = t
            else:
                out_tensor[b_start:b_end, :l_t] = t
            b_start = b_end
        return out_tensor

    # 拼接所有数据块
    final_index = {
        "video_metas": metas,
        # 哈希码是扁平的 (Nc*L, nbytes)，直接cat
        "vr_video_hash": torch.cat(all_indexed_data_batches["vr_video_hash"], dim=0),
        "vr_sub_hash": torch.cat(all_indexed_data_batches["vr_sub_hash"], dim=0) if all_indexed_data_batches["vr_sub_hash"] else None,
        # 密集特征和掩码需要paddings
        "mr_video_feat": cat_tensor_with_padding(all_indexed_data_batches["mr_video_feat"]),
        "mr_sub_feat": cat_tensor_with_padding(all_indexed_data_batches["mr_sub_feat"]),
        "video_mask": cat_tensor_with_padding(all_indexed_data_batches["video_mask"]),
    }
    
    logger.info(f"VR Video Hash (Nc*L, nbytes): {final_index['vr_video_hash'].shape}")
    logger.info(f"MR Video Feat (Nc, L, D): {final_index['mr_video_feat'].shape}")
    
    # 保存索引到磁盘
    index_path = os.path.join(opt.results_dir, f"offline_index_{opt.dset_name}_{opt.eval_split_name}.pt")
    torch.save(final_index, index_path)
    logger.info(f"Offline index saved to {index_path}")
    return index_path

def online_query_loop(model, eval_dataset, opt, indexed_ctx, max_before_nms=1000, max_n_videos=100, tasks=("SVMR",)):
    """
    [在线步骤] 加载预计算的索引，遍历查询，执行快速匹配。
    此函数替换 compute_query2ctx_info。
    """
    # 1. 准备 (与 compute_query2ctx_info 相同)
    is_svmr = "SVMR" in tasks
    is_vr = "VR" in tasks
    is_vcmr = "VCMR" in tasks
    video2idx = eval_dataset.video2idx
    video_metas = indexed_ctx["video_metas"] # 从索引中加载
    
    model.eval()
    eval_dataset.set_data_mode("query")
    eval_dataset.load_gt_vid_name_for_query(is_svmr)
    query_eval_loader = DataLoader(eval_dataset, collate_fn=start_end_collate, batch_size=opt.eval_query_bsz,
                                 num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)
    n_total_query = len(eval_dataset)
    bsz = opt.eval_query_bsz

    timing = {
        "query_hash_s": 0.0,      # 新增：查询哈希时间
        "vr_score_calc_s": 0.0, # 新增：VR哈希匹配+池化时间
        "mr_prob_calc_s": 0.0,  # 新增：MR密集计算时间
        "svmr_decode_s": 0.0,
        "vr_rank_s": 0.0,
        "vcmr_decode_s": 0.0,
        "n_batches": 0,
        "n_queries": n_total_query
    }

    # 2. 将整个索引加载到GPU (关键一步)
    logger.info("Moving context index to GPU...")
    indexed_ctx_gpu = {}
    for k, v in indexed_ctx.items():
        if isinstance(v, torch.Tensor):
            indexed_ctx_gpu[k] = v.to(opt.device, non_blocking=opt.pin_memory)
    indexed_ctx_gpu["video_metas"] = indexed_ctx["video_metas"] # Metas 留在CPU
    logger.info("Index moved to GPU.")

    # 3. 准备结果缓冲区 (与 compute_query2ctx_info 相同)
    if is_vcmr:
        flat_st_ed_scores_sorted_indices = np.empty((n_total_query, max_before_nms), dtype=np.int32)
        flat_st_ed_sorted_scores = np.zeros((n_total_query, max_before_nms), dtype=np.float32)
    else:
        flat_st_ed_scores_sorted_indices, flat_st_ed_sorted_scores = None, None

    if is_vr or is_vcmr:
        sorted_q2c_indices = np.empty((n_total_query, max_n_videos), dtype=np.int32)
        sorted_q2c_scores = np.empty((n_total_query, max_n_videos), dtype=np.float32)
    else:
        sorted_q2c_indices, sorted_q2c_scores = None, None

    if is_svmr:
        svmr_video2meta_idx = {e["vid_name"]: idx for idx, e in enumerate(video_metas)}
        svmr_gt_st_probs = np.zeros((n_total_query, opt.max_ctx_l), dtype=np.float32)
        svmr_gt_ed_probs = np.zeros((n_total_query, opt.max_ctx_l), dtype=np.float32)
    else:
        svmr_video2meta_idx, svmr_gt_st_probs, svmr_gt_ed_probs = None, None, None

    # 4. 在线查询循环
    query_metas = []
    for idx, batch in tqdm(enumerate(query_eval_loader), desc="Online Query", total=len(query_eval_loader)):
        timing["n_batches"] += 1
        
        _query_metas = batch[0]
        query_metas.extend(batch[0])
        model_inputs = prepare_batch_inputs(batch[1], device=opt.device, non_blocking=opt.pin_memory)
        
        # === 核心变化：在线计算 ===
        
        # 4.1. 哈希查询 (快)
        t_qhash = perf_counter()
        packed_q_vid, packed_q_sub, video_query, sub_query = model.export_query_hash(
            model_inputs["query_feat"], model_inputs["query_mask"]
        )
        _sync_if_cuda(opt)
        timing["query_hash_s"] += (perf_counter() - t_qhash)

        # 4.2. 用索引计算分数 (VR快, MR慢)
        # timing 字典会被传入并在内部更新 "vr_score_calc_s" 和 "mr_prob_calc_s"
        _query_context_scores, _st_probs, _ed_probs = model.get_pred_from_indexed_query(
            packed_q_vid, packed_q_sub,
            video_query, sub_query,
            indexed_ctx_gpu, # 传入已在GPU上的索引
            opt, timing_dict=timing
        )
        # === 结束 ===

        # 5. 后处理 (与 compute_query2ctx_info 完全相同)
        if is_svmr: 
            t_s = perf_counter()
            row_indices = torch.arange(0, len(_st_probs))
            query2video_meta_indices = torch.tensor([svmr_video2meta_idx[e["vid_name"]] for e in _query_metas],
                                                    dtype=torch.long)
            svmr_gt_st_probs[idx * bsz:(idx + 1) * bsz, :_st_probs.shape[2]] = \
                _st_probs[row_indices, query2video_meta_indices].cpu().numpy()
            svmr_gt_ed_probs[idx * bsz:(idx + 1) * bsz, :_ed_probs.shape[2]] = \
                _ed_probs[row_indices, query2video_meta_indices].cpu().numpy()
            timing["svmr_decode_s"] += (perf_counter() - t_s)
        
        if not (is_vr or is_vcmr):
            continue

        t_vr = perf_counter()
        _sorted_q2c_scores, _sorted_q2c_indices = torch.topk(_query_context_scores, max_n_videos, dim=1,
                                                             largest=True)
        timing["vr_rank_s"] += (perf_counter() - t_vr)

        sorted_q2c_indices[idx * bsz:(idx + 1) * bsz] = _sorted_q2c_indices.cpu().numpy()
        sorted_q2c_scores[idx * bsz:(idx + 1) * bsz] = _sorted_q2c_scores.cpu().numpy()
        
        if not is_vcmr:
            continue
            
        t_vcmr = perf_counter()
        row_indices = torch.arange(0, len(_st_probs), device=opt.device).unsqueeze(1)
        _st_probs_sel = _st_probs[row_indices, _sorted_q2c_indices] 
        _ed_probs_sel = _ed_probs[row_indices, _sorted_q2c_indices]
        _st_ed_scores = torch.einsum("qvm,qv,qvn->qvmn", _st_probs_sel, _sorted_q2c_scores, _ed_probs_sel)
        valid_prob_mask = generate_min_max_length_mask(_st_ed_scores.shape, min_l=opt.min_pred_l, max_l=opt.max_pred_l)
        _st_ed_scores *= torch.from_numpy(valid_prob_mask).to(_st_ed_scores.device)
        _n_q = _st_ed_scores.shape[0]
        _flat = _st_ed_scores.reshape(_n_q, -1)
        _flat_st_ed_sorted_scores, _flat_st_ed_scores_sorted_indices = torch.sort(_flat, dim=1, descending=True)
        _sync_if_cuda(opt)
        timing["vcmr_decode_s"] += (perf_counter() - t_vcmr)
        
        flat_st_ed_sorted_scores[idx * bsz:(idx + 1)*bsz] = _flat_st_ed_sorted_scores[:, :max_before_nms].cpu().numpy()
        flat_st_ed_scores_sorted_indices[idx * bsz:(idx + 1) * bsz] = \
            _flat_st_ed_scores_sorted_indices[:, :max_before_nms].cpu().numpy()
        if opt.debug:
            break

    # 6. 组装结果 (与 compute_query2ctx_info 完全相同)
    svmr_res = []
    if is_svmr:
        svmr_res = get_svmr_res_from_st_ed_probs(svmr_gt_st_probs, svmr_gt_ed_probs, query_metas, video2idx,
                                                 clip_length=opt.clip_length, min_pred_l=opt.min_pred_l,
                                                 max_pred_l=opt.max_pred_l, max_before_nms=max_before_nms)
    vr_res = []
    if is_vr:
        for i, (_sorted_q2c_scores_row, _sorted_q2c_indices_row) in tqdm(
                enumerate(zip(sorted_q2c_scores[:, :100], sorted_q2c_indices[:, :100])),
                desc="[VR] Loop over queries to generate predictions", total=n_total_query):
            cur_vr_redictions = []
            for j, (v_score, v_meta_idx) in enumerate(zip(_sorted_q2c_scores_row, _sorted_q2c_indices_row)):
                video_idx = video2idx[video_metas[v_meta_idx]["vid_name"]]
                cur_vr_redictions.append([video_idx, 0, 0, float(v_score)])
            cur_query_pred = dict(desc_id=query_metas[i]['desc_id'], desc=query_metas[i]["desc"],
                                  predictions=cur_vr_redictions)
            vr_res.append(cur_query_pred)

    vcmr_res = []
    if is_vcmr:
        for i, (_flat_st_ed_scores_sorted_indices, _flat_st_ed_sorted_scores) in tqdm(
                enumerate(zip(flat_st_ed_scores_sorted_indices, flat_st_ed_sorted_scores)),
                desc="[VCMR] Loop over queries to generate predictions", total=n_total_query): 
            video_meta_indices_local, pred_st_indices, pred_ed_indices = np.unravel_index(
                _flat_st_ed_scores_sorted_indices, shape=(max_n_videos, opt.max_ctx_l, opt.max_ctx_l))
            video_meta_indices = sorted_q2c_indices[i, video_meta_indices_local]
            pred_st_in_seconds = pred_st_indices.astype(np.float32) * opt.clip_length
            pred_ed_in_seconds = pred_ed_indices.astype(np.float32) * opt.clip_length
            cur_vcmr_redictions = []
            for j, (v_meta_idx, v_score) in enumerate(zip(video_meta_indices, _flat_st_ed_sorted_scores)): 
                video_idx = video2idx[video_metas[v_meta_idx]["vid_name"]]
                cur_vcmr_redictions.append([video_idx, float(pred_st_in_seconds[j]), float(pred_ed_in_seconds[j]),
                                            float(v_score)])
            cur_query_pred = dict(desc_id=query_metas[i]["desc_id"], desc=query_metas[i]["desc"],
                                  predictions=cur_vcmr_redictions)
            vcmr_res.append(cur_query_pred)

    res = dict(SVMR=svmr_res, VCMR=vcmr_res, VR=vr_res)
    return {k: v for k, v in res.items() if len(v) != 0}, timing

def eval_epoch_onoff(model, eval_dataset, opt, save_submission_filename, tasks=("SVMR",), max_after_nms=100):
    """
    [在线步骤] 评估的主函数。
    它会加载或创建离线索引，然后执行在线查询。
    """
    model.eval()
    
    # 1. 检查或创建离线索引
    index_path = os.path.join(opt.results_dir, f"offline_index_{opt.dset_name}_{opt.eval_split_name}.pt")
    
    # 你可以添加一个
    # --force_reindex 
    # 启动参数来强制重新生成索引
    if not os.path.exists(index_path) or getattr(opt, "force_reindex", False):
        logger.info(f"Index not found or force_reindex=True. Creating offline index at {index_path}...")
        # 调用离线索引函数 (这会很慢，只做一次)
        create_offline_index(model, eval_dataset, opt)
    
    # 2. 加载索引
    logger.info(f"Loading pre-computed index from {index_path}...")
    # 将索引加载到 CPU 内存
    indexed_ctx = torch.load(index_path, map_location="cpu")
    
    # 3. 执行在线查询
    logger.info("Starting online query processing...")
    st_time = time.time()
    
    # 调用新的在线查询循环
    eval_submission_raw, timing = online_query_loop(
        model, eval_dataset, opt, indexed_ctx, 
        max_before_nms=opt.max_before_nms, 
        max_n_videos=opt.max_vcmr_video, 
        tasks=tasks
    )
    
    total_time = time.time() - st_time
    print("\n" + "\x1b[1;31m" + f"Total Online Query Time: {total_time:.2f}s" + "\x1b[0m", flush=True)

    # 4. 后处理 (与原始 eval_epoch 完全相同)
    eval_submission_raw["video2idx"] = eval_dataset.video2idx
    
    # ---- 计时 & 保存 (与原始 eval_epoch 相同) ----
    avg_ms = {}
    n_q = max(1, timing.get("n_queries", 1))
    for k, v in timing.items():
        if k.startswith("n_"):
            continue
        avg_ms[k.replace("_s", "_ms_per_query")] = (v / n_q) * 1000.0
    avg_ms["total_ms_per_query"] = (total_time / n_q) * 1000.0

    timing_out = dict(
        n_queries=n_q,
        n_batches=timing.get("n_batches", None),
        avg_ms_per_query=avg_ms
    )
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    timings_path = submission_path.replace(".json", "_timings.json")
    save_json(timing_out, timings_path)

    IOU_THDS = (0.5, 0.7) 
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    eval_submission = get_submission_top_n(eval_submission_raw, top_n=max_after_nms)
    save_json(eval_submission, submission_path)

    if opt.eval_split_name == "val": 
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


POST_PROCESSING_MMS_FUNC = {"SVMR": post_processing_svmr_nms, "VCMR": post_processing_vcmr_nms}


def eval_epoch(model, eval_dataset, opt, save_submission_filename, tasks=("SVMR",), max_after_nms=100):
    """max_after_nms: always set to 100, since the eval script only evaluate top-100"""
    model.eval()
    logger.info("Computing scores")
    st_time = time.time()
    eval_submission_raw, timing = get_eval_res(model, eval_dataset, opt, tasks)
    total_time = time.time() - st_time
    print("\n" + "\x1b[1;31m" + str(total_time) + "\x1b[0m", flush=True)

    # ---- 新增：写 timing JSON（按查询平均，毫秒）----
    avg_ms = {}
    n_q = max(1, timing.get("n_queries", 1))
    for k, v in timing.items():
        if k.startswith("n_"):
            continue
        avg_ms[k.replace("_s", "_ms_per_query")] = (v / n_q) * 1000.0
    avg_ms["total_ms_per_query"] = (total_time / n_q) * 1000.0

    timing_out = dict(
        n_queries=n_q,
        n_batches=timing.get("n_batches", None),
        avg_ms_per_query=avg_ms
    )
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    timings_path = submission_path.replace(".json", "_timings.json")
    save_json(timing_out, timings_path)  # <<<<<<<<<<<<<

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
    model = CHDL(loaded_model_cfg)
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
        metrics_no_nms, metrics_nms, latest_file_paths = eval_epoch(model, eval_dataset, opt, save_submission_filename, tasks=opt.tasks, max_after_nms=100)
    logger.info("metrics_no_nms \n{}".format(pprint.pformat(metrics_no_nms, indent=4)))
    logger.info("metrics_nms \n{}".format(pprint.pformat(metrics_nms, indent=4)))


if __name__ == '__main__':
    start_inference()
