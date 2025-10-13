import logging
import h5py
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.basic_utils import load_jsonl, load_json, l2_normalize_np_array, uniform_feature_sampling
from utils.tensor_utils import pad_sequences_1d
from method_tvr.config import BaseOptions

logger = logging.getLogger(__name__)

def _safe_np_2d(a: np.ndarray) -> np.ndarray:
    """确保 float32 + 无 NaN/Inf + 至少是二维"""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    np.nan_to_num(a, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return a

def _safe_l2_normalize(a: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """对每一行做 L2 归一化，带 eps，并再次清 NaN/Inf"""
    a = _safe_np_2d(a)
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    norms = np.maximum(norms, eps)
    a = a / norms
    np.nan_to_num(a, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def build_match_labels_from_indices(
    st_ed_indices: torch.Tensor,  # (B,2) 闭区间 [st, ed]
    video_mask: torch.Tensor,     # (B,Lv) 1/0
    ensure_min_len: int = 1,      # 至少包含几帧
    dilate: int = 0               # 可选：窗口左右扩张（帧）
) -> torch.Tensor:
    """
    返回 (B, Lv) 的 {0,1} 张量，保证：
      - 与 video_mask 对齐（padding 区为 0）
      - 索引逐样本 clamp 到 [0, 有效长度-1]
      - 至少 1 个正帧；可选 dilate
    """
    B, Lv = video_mask.shape
    dev = video_mask.device
    out = torch.zeros((B, Lv), dtype=torch.float32, device=dev)

    st = st_ed_indices[:, 0].long().clone()
    ed = st_ed_indices[:, 1].long().clone()

    # 纠正顺序 & 最小长度（闭区间）
    ed = torch.maximum(ed, st)
    if ensure_min_len > 1:
        ed = torch.maximum(ed, st + (ensure_min_len - 1))

    # 膨胀
    if dilate > 0:
        st -= dilate
        ed += dilate

    # 有效长度（逐样本）
    L_valid = video_mask.sum(dim=1).long().clamp(min=0)  # (B,)

    for b in range(B):
        Lb = int(L_valid[b].item())
        if Lb <= 0:
            continue
        stb = int(torch.clamp(st[b], 0, Lb - 1))
        edb = int(torch.clamp(ed[b], stb, Lb - 1))
        out[b, stb:edb + 1] = 1.0

        # 兜底：如果仍为 0（极端裁剪），给有效区中点置 1
        if out[b].sum() == 0:
            mid = (Lb - 1) // 2
            out[b, mid] = 1.0

    # 屏蔽 padding
    out *= video_mask.float()
    return out


def _to_tvr_schema(raw):
    """
    统一为 {desc_id:int, desc:str, vid_name:str, duration:float, ts:[st,ed]}
    兼容 VERIFIED/ActivityNet-FIG 的 fig_desc/cog_desc/text, video/time 等别名
    """
    if raw is None:
        return None

    desc_text = (raw.get("desc")
                 or raw.get("fig_desc")
                 or raw.get("cog_desc")
                 or raw.get("text")
                 or "")

    vid_name = (raw.get("vid_name")
                or raw.get("video")
                or raw.get("video_id")
                or "")

    ts = raw.get("ts", raw.get("time", [0.0, 0.0]))
    duration = float(raw.get("duration", 0.0))
    desc_id = raw.get("desc_id", raw.get("id"))

    return {
        "desc_id": desc_id,
        "desc": desc_text,      # 一定存在（可能是空串）
        "vid_name": vid_name,
        "duration": duration,
        "ts": ts,
    }



class StartEndDataset(Dataset):
    """
    Args:
        dset_name, str, ["tvr", "didemo-fig", ...]
        ctx_mode: str, "video", "sub", "tef" 的组合, 如 "video_tef" / "video_sub_tef"
    Return:
        {
          "meta": {desc_id, desc, vid_name, duration, ts},
          "model_inputs": {
              "query_feat": (L, D_q),
              "video_feat": (N, D_v[+2]),
              "sub_feat":   (N, D_s[+2]),
              "st_ed_indices": (2,)
          }
        }
    """
    def __init__(self, dset_name, data_path, desc_bert_path_or_handler, sub_bert_path_or_handler, max_desc_len,
                 max_ctx_len, vid_feat_path_or_handler, clip_length, ctx_mode="video", normalize_vfeat=True,
                 normalize_tfeat=True, h5driver=None, data_ratio=1.0):
        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio

        self.desc_bert_path_or_handler = desc_bert_path_or_handler
        self.max_desc_len = max_desc_len

        self.sub_bert_path_or_handler = sub_bert_path_or_handler
        self.max_ctx_len = max_ctx_len
        self.vid_feat_path_or_handler = vid_feat_path_or_handler
        self.clip_length = clip_length
        self.ctx_mode = ctx_mode

        # prepare desc data
        self.data = load_jsonl(data_path)
        if self.data_ratio != 1:
            n_examples = int(len(self.data) * data_ratio)
            self.data = self.data[:n_examples]
            logger.info("Using {}% of the data: {} examples".format(data_ratio * 100, n_examples))

        self.use_video = "video" in self.ctx_mode
        self.use_sub = "sub" in self.ctx_mode
        self.use_tef = "tef" in self.ctx_mode

        if self.use_video:
            if isinstance(vid_feat_path_or_handler, h5py.File):
                self.vid_feat_h5 = vid_feat_path_or_handler
            else:  # str path
                self.vid_feat_h5 = h5py.File(vid_feat_path_or_handler, "r", driver=h5driver)

        if isinstance(desc_bert_path_or_handler, h5py.File):
            self.desc_bert_h5 = desc_bert_path_or_handler
        else:
            self.desc_bert_h5 = h5py.File(desc_bert_path_or_handler, "r", driver=h5driver)

        if self.use_sub and sub_bert_path_or_handler is not None:
            if isinstance(sub_bert_path_or_handler, h5py.File):
                self.sub_bert_h5 = sub_bert_path_or_handler
            else:  # str path
                self.sub_bert_h5 = h5py.File(sub_bert_path_or_handler, "r", driver=h5driver)
        else:
            self.sub_bert_h5 = None

        self.normalize_vfeat = normalize_vfeat
        self.normalize_tfeat = normalize_tfeat

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raw = self.data[index]
        raw_data = _to_tvr_schema(raw)

        # 兜底日志（可选）
        if raw_data["desc_id"] is None or raw_data["vid_name"] == "":
            logger.warning(f"[dataset] bad record at idx={index}: {raw}")

        # initialize with basic data
        meta = dict(
            desc_id=raw_data["desc_id"],
            desc=raw_data["desc"],              # 这里不会再 KeyError
            vid_name=raw_data["vid_name"],
            duration=raw_data["duration"],
            ts=raw_data["ts"]
        )
        model_inputs = dict()

        # —— Query 特征 ——（带清洗+安全归一化）
        q = self.desc_bert_h5[str(meta["desc_id"])]
        q = q[:self.max_desc_len]
        q = _safe_np_2d(q)
        if self.normalize_tfeat:
            q = _safe_l2_normalize(q)
        model_inputs["query_feat"] = torch.from_numpy(q)

        ctx_l_v, ctx_l_s = 0, 0

        # —— Video 特征 ——（uniform 采样前先清洗；采样后再清洗一次）
        if self.use_video:
            vf = self.vid_feat_h5[meta['vid_name']][:]
            vf = _safe_np_2d(vf)
            vf = uniform_feature_sampling(vf, self.max_ctx_len)
            vf = _safe_np_2d(vf)
            if self.normalize_vfeat:
                vf = _safe_l2_normalize(vf)
            model_inputs["video_feat"] = torch.from_numpy(vf)
            ctx_l_v = len(vf)
        else:
            model_inputs["video_feat"] = torch.zeros((2, 2))
            ctx_l_v = 0

        # —— Sub 特征 ——（可选）
        if self.use_sub and self.sub_bert_h5 is not None:
            sf = self.sub_bert_h5[meta["vid_name"]][:]
            sf = _safe_np_2d(sf)
            sf = uniform_feature_sampling(sf, self.max_ctx_len)
            sf = _safe_np_2d(sf)
            if self.normalize_tfeat:
                sf = _safe_l2_normalize(sf)
            model_inputs["sub_feat"] = torch.from_numpy(sf)
            ctx_l_s = len(sf)
        else:
            model_inputs["sub_feat"] = None
            ctx_l_s = 0

        # —— TEF ——（长度与 context 对齐；若只有 video，则以 video 为准）
        if self.use_tef:
            ctx_l = ctx_l_v if ctx_l_v > 0 else ctx_l_s
            if ctx_l == 0:
                # 极端：没有任何上下文，兜底 1 段，避免后面除零
                ctx_l = int(meta["duration"] // self.clip_length + 1) if meta["duration"] > 0 else 1
            tef_st = torch.arange(0, ctx_l, 1.0) / max(ctx_l, 1)
            tef_ed = tef_st + 1.0 / max(ctx_l, 1)
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (ctx_l, 2)
        else:
            tef = None

        if self.use_video and self.use_tef:
            model_inputs["video_feat"] = torch.cat([model_inputs["video_feat"], tef], dim=1)
            ctx_l_v = ctx_l  # 以拼接后的长度为准
        if self.use_sub and self.use_tef and self.sub_bert_h5 is not None:
            model_inputs["sub_feat"] = torch.cat([model_inputs["sub_feat"], tef], dim=1)
            ctx_l_s = ctx_l

        # —— 计算 st/ed ——（以 video 长度为准；若没有 video，则以 sub 长度为准）
        ctx_l_for_label = ctx_l_v if ctx_l_v > 0 else ctx_l_s
        if ctx_l_for_label <= 0:
            # 仍然极端：给出 1 段兜底，防训练出错
            st_ed = torch.tensor([0, 0], dtype=torch.long)
        else:
            st_ed = self.get_st_ed_label(meta["ts"], max_idx=ctx_l_for_label - 1)
        model_inputs["st_ed_indices"] = st_ed
        return dict(meta=meta, model_inputs=model_inputs)


    def get_st_ed_label(self, ts, max_idx):
        """
        ts: [st_sec, ed_sec]，ed>st
        返回闭区间 [st_idx, ed_idx]，其中：
        st_idx = floor(st/clip_len)
        ed_idx = ceil(ed/clip_len) - 1
        然后 clamp 到 [0, max_idx]
        """
        st_idx = int(math.floor(ts[0] / self.clip_length))
        ed_idx = int(math.ceil (ts[1] / self.clip_length) - 1)
        st_idx = max(0, min(st_idx, max_idx))
        ed_idx = max(st_idx, min(ed_idx, max_idx))
        return torch.tensor([st_idx, ed_idx], dtype=torch.long)


    def get_query_feat_by_desc_id(self, desc_id):
        query_feat = self.desc_bert_h5[str(desc_id)][:self.max_desc_len]
        if self.normalize_tfeat:
            query_feat = l2_normalize_np_array(query_feat)
        return torch.from_numpy(query_feat)


class StartEndEvalDataset(Dataset):
    """
    init_data_mode: `video_query` or `video_only` or `query_only`,
    data_mode: `context` or `query`
    """
    def __init__(self, dset_name, eval_split_name, data_path=None, desc_bert_path_or_handler=None, max_desc_len=None,
                 max_ctx_len=None, sub_bert_path_or_handler=None, vid_feat_path_or_handler=None,
                 video_duration_idx_path=None, clip_length=None, ctx_mode="video", data_mode="context", h5driver=None,
                 data_ratio=1.0, normalize_vfeat=True, normalize_tfeat=True):
        self.dset_name = dset_name
        self.eval_split_name = eval_split_name
        self.ctx_mode = ctx_mode
        self.load_gt_video = False
        self.data_ratio = data_ratio  # only affect query data
        self.normalize_vfeat = normalize_vfeat
        self.normalize_tfeat = normalize_tfeat

        self.data_mode = None
        self.set_data_mode(data_mode)

        self.max_desc_len = max_desc_len
        self.max_ctx_len = max_ctx_len
        self.data_path = data_path
        if isinstance(desc_bert_path_or_handler, h5py.File):
            self.desc_bert_h5 = desc_bert_path_or_handler
        else:
            self.desc_bert_h5 = h5py.File(desc_bert_path_or_handler, "r", driver=h5driver)

        video_data = load_json(video_duration_idx_path)[self.eval_split_name]
        self.video_data = [{"vid_name": k, "duration": v[0]} for k, v in video_data.items()]
        self.video2idx = {k: v[1] for k, v in video_data.items()}
        self.clip_length = clip_length

        self.use_video = "video" in self.ctx_mode
        self.use_sub = "sub" in self.ctx_mode
        self.use_tef = "tef" in self.ctx_mode

        if self.use_video:
            if isinstance(vid_feat_path_or_handler, h5py.File):
                self.vid_feat_h5 = vid_feat_path_or_handler
            else:  # str path
                self.vid_feat_h5 = h5py.File(vid_feat_path_or_handler, "r", driver=h5driver)

        if self.use_sub and sub_bert_path_or_handler is not None:
            if isinstance(sub_bert_path_or_handler, h5py.File):
                self.sub_bert_h5 = sub_bert_path_or_handler
            else:  # str path
                self.sub_bert_h5 = h5py.File(sub_bert_path_or_handler, "r", driver=h5driver)
        else:
            self.sub_bert_h5 = None

        self.query_data = load_jsonl(data_path)
        if data_ratio != 1:
            n_examples = int(len(self.query_data) * data_ratio)
            self.query_data = self.query_data[:n_examples]
            logger.info("Using {}% of the query data: {} examples".format(data_ratio * 100, n_examples))

    def set_data_mode(self, data_mode):
        """context or query"""
        assert data_mode in ["context", "query"]
        self.data_mode = data_mode

    def load_gt_vid_name_for_query(self, load_gt_video):
        """load_gt_video: bool, affect the returned value of self._get_item_query"""
        if load_gt_video:
            # VERIFIED/TVR 的 query jsonl 都含 vid/video 字段
            assert "vid_name" in _to_tvr_schema(self.query_data[0]) or "video" in self.query_data[0]
        self.load_gt_video = load_gt_video

    def __len__(self):
        if self.data_mode == "context":
            return len(self.video_data)
        else:
            return len(self.query_data)
    
    def __getitem__(self, index):
        if self.data_mode == "context":
            return self._get_item_context(index)
        elif self.data_mode == "query":
            return self._get_item_query(index)
        else:
            raise ValueError(f"Unknown data_mode: {self.data_mode}")


    def get_query_feat_by_desc_id(self, desc_id):
        arr = self.desc_bert_h5.get(str(desc_id), None)
        if arr is None:
            raise KeyError(f"desc_bert_h5 missing key: {desc_id}")
        q = arr[:self.max_desc_len]
        q = _safe_np_2d(q)
        if self.normalize_tfeat:
            q = _safe_l2_normalize(q)
        return torch.from_numpy(q)


    def _get_item_query(self, index):
        raw = self.query_data[index]
        raw_data = _to_tvr_schema(raw)
        meta = dict(
            desc_id=raw_data["desc_id"],
            desc=raw_data["desc"],
            vid_name=raw_data["vid_name"] if self.load_gt_video else None
        )
        model_inputs = dict()
        model_inputs["query_feat"] = self.get_query_feat_by_desc_id(meta["desc_id"])        
        return dict(meta=meta, model_inputs=model_inputs)


    def get_st_ed_label(self, ts, max_idx):
        """
        ts: [st_sec, ed_sec]，ed>st
        返回闭区间 [st_idx, ed_idx]，其中：
        st_idx = floor(st/clip_len)
        ed_idx = ceil(ed/clip_len) - 1
        然后 clamp 到 [0, max_idx]
        """
        st_idx = int(math.floor(ts[0] / self.clip_length))
        ed_idx = int(math.ceil (ts[1] / self.clip_length) - 1)
        st_idx = max(0, min(st_idx, max_idx))
        ed_idx = max(st_idx, min(ed_idx, max_idx))
        return torch.tensor([st_idx, ed_idx], dtype=torch.long)


    def _get_item_context(self, index):
        """No need to batch, since it has already been batched here"""
        raw_data = self.video_data[index]
        meta = dict(vid_name=raw_data["vid_name"], duration=raw_data["duration"])
        model_inputs = dict()
        ctx_l_v, ctx_l_s = 0, 0

        # —— Video 特征：先清洗，再采样，再清洗/归一化
        if self.use_video:
            vf = self.vid_feat_h5[meta["vid_name"]][:]
            vf = _safe_np_2d(vf)
            vf = uniform_feature_sampling(vf, self.max_ctx_len)
            vf = _safe_np_2d(vf)
            if self.normalize_vfeat:
                vf = _safe_l2_normalize(vf)
            model_inputs["video_feat"] = torch.from_numpy(vf)
            ctx_l_v = len(vf)
        else:
            model_inputs["video_feat"] = torch.zeros((2, 2))
            ctx_l_v = 0

        # —— Sub 特征：同上（存在才读）
        if self.use_sub and self.sub_bert_h5 is not None:
            sf = self.sub_bert_h5[meta["vid_name"]][:]
            sf = _safe_np_2d(sf)
            sf = uniform_feature_sampling(sf, self.max_ctx_len)
            sf = _safe_np_2d(sf)
            if self.normalize_tfeat:
                sf = _safe_l2_normalize(sf)
            model_inputs["sub_feat"] = torch.from_numpy(sf)
            ctx_l_s = len(sf)
        else:
            model_inputs["sub_feat"] = None
            ctx_l_s = 0

        # —— TEF：与实际上下文长度对齐；都没有时按 duration/clip_length 兜底
        if self.use_tef:
            ctx_l = ctx_l_v if ctx_l_v > 0 else ctx_l_s
            if ctx_l == 0:
                ctx_l = int(meta["duration"] // self.clip_length + 1) if meta["duration"] > 0 else 1
            tef_st = torch.arange(0, ctx_l, 1.0) / max(ctx_l, 1)
            tef_ed = tef_st + 1.0 / max(ctx_l, 1)
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (ctx_l, 2)
        else:
            tef = None

        if self.use_video and self.use_tef:
            model_inputs["video_feat"] = torch.cat([model_inputs["video_feat"], tef], dim=1)
            ctx_l_v = ctx_l
        if self.use_sub and self.use_tef and self.sub_bert_h5 is not None:
            model_inputs["sub_feat"] = torch.cat([model_inputs["sub_feat"], tef], dim=1)
            ctx_l_s = ctx_l

        return dict(meta=meta, model_inputs=model_inputs)


def start_end_collate(batch):
    # —— 新增：过滤掉异常样本（极少见；一般不会触发）——
    batch = [e for e in batch if e is not None]
    if len(batch) == 0:
        raise ValueError("Empty batch after filtering.")

    batch_meta = [e["meta"] for e in batch]
    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()

    # 取消强行截断：让长度由数据集的 uniform_feature_sampling 决定
    FIXED_KEYS = set()   # 原来是 {'video_feat','sub_feat','tef_feat'}
    FIXED_LEN = None

    for k in model_inputs_keys:
        vals = [e["model_inputs"][k] for e in batch]
        if all(v is None for v in vals):
            continue

        if "feat" in k:
            fixed_length = (FIXED_LEN if k in FIXED_KEYS else None)
            padded, mask = pad_sequences_1d(
                [v for v in vals if v is not None],
                dtype=torch.float32,
                fixed_length=fixed_length
            )
            # —— 新增：再次清理 NaN/Inf —— 
            padded = torch.nan_to_num(padded, nan=0.0, posinf=0.0, neginf=0.0)
            batched_data[k] = (padded, mask)
        else:
            pass

    # st_ed_indices 原样堆叠
    if "st_ed_indices" in model_inputs_keys:
        st_ed_indices = torch.stack([e["model_inputs"]["st_ed_indices"] for e in batch], dim=0)  # (B,2)
        batched_data["st_ed_indices"] = st_ed_indices

        # 用 pad 后的 video_mask 构造 match_labels（内部会再按各自长度 clamp）
        assert "video_feat" in batched_data, "video_feat is required to build match_labels"
        video_mask = batched_data["video_feat"][1]  # (B, Lv)

        match_labels = build_match_labels_from_indices(
            st_ed_indices=st_ed_indices,
            video_mask=video_mask,
            ensure_min_len=1,
            dilate=0
        )
        batched_data["match_labels"] = match_labels

    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    model_inputs = {}
    for k, v in batched_model_inputs.items():
        if "feat" in k:
            if v is None:
                continue
            model_inputs[k] = v[0].to(device, non_blocking=non_blocking)
            model_inputs[k.replace("feat", "mask")] = v[1].to(device, non_blocking=non_blocking)
        else:
            model_inputs[k] = v.to(device, non_blocking=non_blocking)

    # —— 新增：若没有字幕分支，显式补 None —— 
    if "sub_feat" not in model_inputs:
        model_inputs["sub_feat"] = None
        model_inputs["sub_mask"] = None

    return model_inputs




if __name__ == '__main__':
    options = BaseOptions().parse()
