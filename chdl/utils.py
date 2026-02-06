import math

import torch
import torch.nn.functional as F


def log_sum_exp(x, axis=None):
    """
    Numerically stable log sum exp function
    Args:
        x: Input.
        axis: Axis over which to perform sum.
    Returns:
        torch.Tensor: log sum exp
    """
    x_max = torch.max(x, axis, keepdim=True)[0]
    # Clamp x_max to prevent extreme values
    x_max = torch.clamp(x_max, min=-50.0, max=50.0)
    y = torch.log((torch.exp(x - x_max)).sum(axis, keepdim=True)) + x_max
    if axis is not None:
        y = y.squeeze(axis)
    return torch.nan_to_num(y, nan=0.0, posinf=50.0, neginf=-50.0)


def get_positive_expectation(p_samples, measure="JSD", average=True):
    """
    Computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.0)
    if measure == "GAN":
        Ep = -F.softplus(-p_samples)
    elif measure == "JSD":
        Ep = log_2 - F.softplus(-p_samples)
    elif measure == "X2":
        Ep = p_samples**2
    elif measure == "KL":
        Ep = p_samples + 1.0
    elif measure == "RKL":
        # Clamp to prevent overflow in exp
        p_clamped = torch.clamp(-p_samples, max=50.0)
        Ep = -torch.exp(p_clamped)
    elif measure == "DV":
        Ep = p_samples
    elif measure == "H2":
        # Clamp to prevent overflow in exp
        p_clamped = torch.clamp(-p_samples, max=50.0)
        Ep = torch.ones_like(p_samples) - torch.exp(p_clamped)
    elif measure == "W1":
        Ep = p_samples
    else:
        raise ValueError("Unknown measurement {}".format(measure))
    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure="JSD", average=True):
    """
    Computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.0)
    if measure == "GAN":
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == "JSD":
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == "X2":
        Eq = -0.5 * ((torch.sqrt(q_samples**2) + 1.0) ** 2)
    elif measure == "KL":
        # Clamp to prevent overflow in exp
        q_clamped = torch.clamp(q_samples, max=50.0)
        Eq = torch.exp(q_clamped)
    elif measure == "RKL":
        Eq = q_samples - 1.0
    elif measure == "DV":
        Eq = log_sum_exp(q_samples, 0) - math.log(max(q_samples.size(0), 1))
    elif measure == "H2":
        # Clamp to prevent overflow in exp
        q_clamped = torch.clamp(q_samples, max=50.0)
        Eq = torch.exp(q_clamped) - 1.0
    elif measure == "W1":
        Eq = q_samples
    else:
        raise ValueError("Unknown measurement {}".format(measure))
    if average:
        return Eq.mean()
    else:
        return Eq


def _l2_normalize(x, dim=-1, eps=1e-6):
    y = x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)
    return torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def masked_logsumexp(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1):
    """
    对 mask 为 True 的位置计算 logsumexp；
    若某一行在该 dim 上全是 False，则该行直接返回 0，且不对该行调用 logsumexp，
    从而避免 LogsumexpBackward 在 all -inf 时产生 NaN 梯度。
    """
    if logits.dtype.is_floating_point is False:
        logits = logits.float()
    mask = mask.bool()

    # 统一把目标维挪到最后一维，便于做“按行”索引
    if dim < 0:
        dim = logits.dim() + dim
    if dim != logits.dim() - 1:
        perm = list(range(logits.dim()))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        logits = logits.permute(*perm)
        mask = mask.permute(*perm)
        need_unpermute = True
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
    else:
        need_unpermute = False

    *head, L = logits.shape
    logits_flat = logits.reshape(-1, L)
    mask_flat = mask.reshape(-1, L)

    out = logits_flat.new_zeros((logits_flat.shape[0],))
    valid_rows = mask_flat.any(dim=-1)

    if valid_rows.any():
        # 只对有至少一个有效元素的行做 logsumexp
        sel_logits = logits_flat[valid_rows].masked_fill(
            ~mask_flat[valid_rows], float("-inf")
        )
        out_valid = torch.logsumexp(sel_logits, dim=-1)
        out[valid_rows] = out_valid

    out = out.view(*head)
    if need_unpermute:
        out = out.permute(*inv_perm)
    return out


def local_token_frame_contrastive_loss(z, pos_mask, all_mask, temperature=0.07):
    """
    z:         [B, Lq, Lv]  任意最后一维为候选维
    pos_mask:  [B, Lq, Lv]  True 表示正样本位置
    all_mask:  [B, Lq, Lv]  True 表示有效位置
    """
    # 防止 tau 太小导致 logits 过大
    tau = max(float(temperature), 1e-3)

    # 关闭 AMP 到 fp32 计算 logsumexp，避免半精度下 exp 门限更低
    with torch.cuda.amp.autocast(enabled=False):
        z = z.float() / tau
        all_mask = all_mask.bool()
        pos_mask = pos_mask.bool()

        # 无效位置置 -inf，随后做稳定的 logsumexp
        logits = z.masked_fill(~all_mask, float("-inf"))

        log_den = masked_logsumexp(logits, all_mask, dim=-1)  # [B, Lq]
        log_num = masked_logsumexp(logits, pos_mask, dim=-1)  # [B, Lq]

        # 仅统计分子、分母都有效的行
        valid = all_mask.any(dim=-1) & pos_mask.any(dim=-1)  # [B, Lq]
        log_den = log_den[valid]
        log_num = log_num[valid]

        # 若全部无效，返回 0 以避免 nan
        if log_den.numel() == 0:
            return z.new_tensor(0.0)

        loss = -(log_num - log_den).mean()
    return loss


def batch_local_token_frame_loss(
    video_feat: torch.Tensor,  # [B, Lv, D]
    query_feat: torch.Tensor,  # [B, Lq, D]
    match_labels: torch.Tensor,  # [B, Lv] 或 [B, Lq, Lv]  (>=1 表示正)
    video_mask: torch.Tensor,  # [B, Lv]  (1/True 有效)
    query_mask: torch.Tensor,  # [B, Lq]  (1/True 有效)
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    计算 token–frame 局部对比损失。内部先构造 z=[B,Lq,Lv]、pos_mask、all_mask，
    再调用 local_token_frame_contrastive_loss。
    """
    # L2 归一化以稳定点积尺度
    q = _l2_normalize(query_feat, dim=-1)  # [B, Lq, D]
    v = _l2_normalize(video_feat, dim=-1)  # [B, Lv, D]

    # 构造 pairwise 相似度 z: [B, Lq, Lv]
    z = torch.einsum("bqd,bvd->bqv", q, v)

    # 有效位置 all_mask: [B, Lq, Lv]
    all_mask = query_mask.bool().unsqueeze(-1) & video_mask.bool().unsqueeze(1)

    # 正样本掩码 pos_mask: 支持两种标签形状
    if match_labels.dim() == 2:
        # [B, Lv] -> 广播到 [B, Lq, Lv]，仅对有效 query token 计正样本
        pos_mask = query_mask.bool().unsqueeze(-1) & (
            match_labels > 0
        ).bool().unsqueeze(1)
    else:
        # [B, Lq, Lv]
        pos_mask = (match_labels > 0).bool()

    # 将 pos_mask 也限制在有效位置内，避免“伪正样本”溢出
    pos_mask = pos_mask & all_mask

    return local_token_frame_contrastive_loss(
        z, pos_mask, all_mask, temperature=temperature
    )


# MIT License
#
# Copyright (c) 2018 Victor Escorcia Castillo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
"""
Group multiple methods to generate salient temporal windows in a video"""
import itertools

import numpy as np

PROPOSAL_SCHEMES = ["DidemoICCV17SS", "SlidingWindowMSRSS"]


class TemporalProposalsBase:
    """Base class (signature) to generate temporal candidate in video"""

    def __call__(self, video_id, metadata=None, feature_collection=None):
        raise NotImplementedError("Implement with the signature above")


class DidemoICCV17SS(TemporalProposalsBase):
    """Original search space of moments proposed in ICCV-2017

    Attributes:
        clip_length_min (float) : minimum length, in seconds, of a video clip.
        proposals (numpy array) : of shape [21, 2] representing all the
            possible temporal segments of valid annotations of DiDeMo dataset.
            It represents the search space of a temporal localization
            algorithm.

    Reference: Hendricks et al. Localizing Moments in Video with Natural
        Language. ICCV 2017.
    """

    clip_length_min = 5.0

    def __init__(self, *args, dtype=np.float32, **kwargs):
        clips_indices = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        for i in itertools.combinations(range(len(clips_indices)), 2):
            clips_indices.append(i)
        self.proposals = np.array(clips_indices, dtype=dtype)
        self.proposals *= self.clip_length_min
        self.proposals[:, 1] += self.clip_length_min

    def __call__(self, *args, **kwargs):
        return self.proposals


class SlidingWindowMSRSS(TemporalProposalsBase):
    """Multi-scale sliding window with relative stride within the same scale

    Attributes:
        length (float) : length of smallest window.
        scales (sequence of int) : duration of moments relative to
            `length`.
        stride (float) : relative stride between two windows with the same
            duration. We used different strides for each scale rounding it
            towards a multiple of `length`. Note that the minimum stride is
            `length` for any window will be the `length` itself.
        dtype (numpy.dtype) :
    """

    def __init__(self, length, scales, stride=0.5, round_base=0.5, dtype=np.float32):
        self.length = length
        self.scales = scales
        self.round_base = round_base
        self.relative_stride = stride
        # pick strides per scale that are multiples of length
        self.strides = [
            max(round(s * stride / round_base) * round_base, round_base) * length
            for s in scales
        ]
        self.dtype = dtype
        assert len(scales) > 0

    def sliding_windows(self, t_end, t_start=0):
        """sliding canonical windows over a given time interval"""
        windows_ = []
        for i, stride in enumerate(self.strides):
            num_i = np.ceil((t_end - t_start) / stride)
            windows_i = np.empty((int(num_i), 2), dtype=np.float32)
            windows_i[:, 0] = np.arange(t_start, t_end, stride)
            windows_i[:, 1] = windows_i[:, 0] + self.length * self.scales[i]
            windows_i[windows_i[:, 1] > t_end, 1] = t_end
            windows_.append(windows_i)
            # print("--------------------------------{}".format(i))
            # print(windows_i)
        # import sys
        # sys.exit(1)
        windows = np.concatenate(windows_, axis=0)
        # Hacky way to make windows fit inside video
        # It implies windows at the end may not belong to the set spanned by
        # length and scales.
        return np.unique(windows, axis=0)

    def __call__(self, video_id, metadata=None, feature_collection=None):
        """return: (N_window, 2), each row contains (start, end)"""
        duration = metadata.get("duration")
        assert duration is not None
        return self.sliding_windows(duration)


ProposalConfigs = {
    "didemo": {
        "proposal_interface": "DidemoICCV17SS",
        "clip_length": 2.5,
    },
    "tvr": {
        "length": 3,  # min proposal length
        "scales": [1, 2, 4, 8],
        "stride": 0.3,
        "round_base": 1,
        "min_proposal_length": 3,  # length * min(scales)
        "clip_length": 1.5,  # length should be divisible by clip_length
        "proposal_interface": "SlidingWindowMSRSS",
    },
    "anet_cap": {
        "length": 5,
        "scales": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26],
        "stride": 0.3,
        "round_base": 1,
        "min_proposal_length": 10,  # length * min(scales)
        "clip_length": 5,  # length * min(scales) / 2
        "proposal_interface": "SlidingWindowMSRSS",
    },
    "charades_sta": {
        "length": 3,
        "scales": [2, 3, 4, 5, 6, 7, 8],
        "stride": 0.3,
        "round_base": 1,
        "min_proposal_length": 6,  # length * min(scales)
        "clip_length": 3,  # length * min(scales) / 2
        "proposal_interface": "SlidingWindowMSRSS",
    },
    "profiling": {
        "length": 5,
        "scales": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "stride": 0.3,
        "round_base": 1,
        "clip_length": 5,  # length * min(scales) / 2
        "proposal_interface": "SlidingWindowMSRSS",
    },
}
"""
'min_clip_length' is used to uniformly segment the video into smaller clips, it is a half of
the 'min_proposal_length'. Thus we can enforce each moment has at least 2 clips.
"""


def get_proposal_interface(dset_name):
    """dset_name (str): one of ["tvr"]"""
    assert dset_name in ProposalConfigs
    if dset_name == "didemo":
        return DidemoICCV17SS()
    else:
        arg_names = ["length", "scales", "stride", "round_base"]
        func_args = {k: ProposalConfigs[dset_name][k] for k in arg_names}
        return SlidingWindowMSRSS(**func_args)


if __name__ == "__main__":
    test_fns_args = [
        (
            DidemoICCV17SS,
            (),
        ),
        (SlidingWindowMSRSS, (1.5, [2, 4, 6, 12])),
    ]
    for fn_i, args_i in test_fns_args:
        proposal_fn = fn_i(*args_i)
        x = proposal_fn("hola", {"duration": 15})
        if fn_i == DidemoICCV17SS:
            assert len(x) == 21
