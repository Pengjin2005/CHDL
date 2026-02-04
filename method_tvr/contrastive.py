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
