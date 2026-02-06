from dataclasses import dataclass
from typing import Tuple, List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from contextlib import nullcontext
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def onehot(indexes, N=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().long().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    return output


class SmoothedCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SmoothedCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, labels, smooth_eps=0.1, mask=None, from_logits=True):
        """
        Args:
            logits: (N, Lv), unnormalized probabilities, torch.float32
            labels: (N, Lv) or (N, ), one hot labels or indices labels, torch.float32 or torch.int64
            smooth_eps: float
            mask: (N, Lv)
            from_logits: bool
        """
        if from_logits:
            probs = F.log_softmax(logits, dim=-1)
        else:
            probs = logits
        num_classes = probs.size()[-1]
        if len(probs.size()) > len(labels.size()):
            labels = onehot(labels, num_classes).type(probs.dtype)
        if mask is None:
            labels = labels * (1 - smooth_eps) + smooth_eps / num_classes
        else:
            mask = mask.type(probs.dtype)
            valid_samples = torch.sum(mask, dim=-1, keepdim=True, dtype=probs.dtype)  # (N, 1)
            eps_per_sample = smooth_eps / valid_samples
            labels = (labels * (1 - smooth_eps) + eps_per_sample) * mask
        loss = -torch.sum(labels * probs, dim=-1)
        if self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss  # (N, )


class MILNCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MILNCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, q2ctx_scores=None, contexts=None, queries=None):
        if q2ctx_scores is None:
            assert contexts is not None and queries is not None
            x = torch.matmul(contexts, queries.t())
            device = contexts.device
            bsz = contexts.shape[0]
        else:
            x = q2ctx_scores
            device = q2ctx_scores.device
            bsz = q2ctx_scores.shape[0]
            
        # Clean input from NaN/Inf
        x = torch.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0)
        # Clamp to prevent extreme values
        x = torch.clamp(x, min=-50.0, max=50.0)
        
        x = x.view(bsz, bsz, -1)
        nominator = x * torch.eye(x.shape[0], dtype=torch.float32, device=device)[:, :, None]
        nominator = nominator.sum(dim=1)
        
        # Numerically stable logsumexp
        nominator = torch.logsumexp(nominator, dim=1)
        nominator = torch.nan_to_num(nominator, nan=0.0, posinf=50.0, neginf=-50.0)
        
        denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        denominator = torch.nan_to_num(denominator, nan=0.0, posinf=50.0, neginf=-50.0)
        
        result = denominator - nominator
        result = torch.nan_to_num(result, nan=0.0, posinf=50.0, neginf=-50.0)
        
        if self.reduction:
            return torch.mean(result)
        else:
            return result

class TrainablePositionalEncoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def add_position_emb(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return input_feat + position_embeddings


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [nn.Dropout(dropout), nn.Linear(in_hsz, out_hsz)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


class BertLayer(nn.Module):
    def __init__(self, config, use_self_attention=True):
        super(BertLayer, self).__init__()
        self.use_self_attention = use_self_attention
        if use_self_attention:
            self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states:  (N, L, D)
            attention_mask:  (N, L) with 1 indicate valid, 0 indicates invalid
        """
        if self.use_self_attention:
            attention_output = self.attention(hidden_states, attention_mask)
        else:
            attention_output = hidden_states
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, L)
        """
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Sequential(nn.Linear(config.hidden_size, config.intermediate_size), nn.ReLU(True))

    def forward(self, hidden_states):
        return self.dense(hidden_states)


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (
                config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states, attention_mask):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)
        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)
        # transpose
        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # compute output context
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
@dataclass
class HashedVector:
    recon: torch.Tensor
    code: torch.Tensor
    bin_like: torch.Tensor
    x_origin: torch.Tensor
    bits01: torch.Tensor = None        # (B, Dh) uint8 in {0,1}
    packed_bits: torch.Tensor = None   # (B, ceil(Dh/8)) uint8

class HashLayer(nn.Module):
    def __init__(self, input_output_size, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
        nn.Linear(input_output_size, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5), 
        nn.Linear(1024, hidden_size)
        )
        self.decoder = nn.Sequential(
        nn.Linear(hidden_size, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, input_output_size),
        )
        # 4096     

        lut = torch.tensor([int(bin(i).count("1")) for i in range(256)], dtype=torch.uint8)
        self.register_buffer("_pop_lut_256", lut, persistent=False)

    @staticmethod
    def _pack_bits(bits01: torch.Tensor) -> torch.Tensor:
        """
        bits01: (B, D) uint8 in {0,1}  ->  packed: (B, ceil(D/8)) uint8
        """
        B, D = bits01.shape
        pad = (-D) % 8
        if pad:
            bits01 = F.pad(bits01, (0, pad), value=0)
            D += pad
        bits01 = bits01.view(B, D // 8, 8)                        # (B, nbytes, 8)
        # 权重 [1,2,4,8,16,32,64,128]
        weights = (1 << torch.arange(8, device=bits01.device, dtype=torch.uint8))
        packed = (bits01 * weights).sum(dim=-1).to(torch.uint8)   # (B, nbytes)
        return packed

    @torch.no_grad()
    def export_bits(self, x: torch.Tensor, eta: float = 1.0, pack: bool = True):
        """
        仅推理使用：编码 -> sign -> {0,1} -> (可选)pack 成字节
        返回 (bits01, packed)；train/eval 无关
        """
        code = self.encoder(x)
        code = F.normalize(code, dim=1, eps=1e-6)
        bits01 = (code >= 0).to(torch.uint8)   # (B, Dh) 0/1
        if pack:
            packed = self._pack_bits(bits01)
            return bits01, packed
        else:
            return bits01, None
    
    def forward(self, x, eta=1.0, eval_skip_decoder = True, export_packed = False):
        self.x = x
        code = self.encoder(x)            # (B, hidden_size)
        code = F.normalize(code, dim=1, eps=1e-6)
        # Clamp eta*code to prevent extreme values in tanh
        if self.training:
            scaled_code = torch.clamp(eta * code, min=-10.0, max=10.0)
            bin_like = torch.tanh(scaled_code) if self.training else torch.sign(code)
            recon = self.decoder(bin_like)    # (B, input_output_size)
            return HashedVector(recon, code, bin_like, x)
        else:
            bin_like = torch.sign(code)
            recon = x
            if export_packed:
                bits01 = (bin_like  > 0).to(torch.uint8)   # (B, Dh) 0/1
                packed = self._pack_bits(bits01)
                return HashedVector(recon, code, bin_like, x, bits01=bits01, packed_bits=packed)
            else:
                return HashedVector(recon, code, bin_like, x)
    
    def xnor_popcount(self, A_packed: torch.Tensor, B_packed: torch.Tensor, tile_M: int = 16384) -> torch.Tensor:
        """
        A_packed: (N, nbytes) uint8
        B_packed: (M, nbytes) uint8
        return   : (N, M) int32 —— XNOR+popcount 匹配位数
        说明：
        - 分块沿 M 维计算，避免一次性构建 (N, M, nbytes)。
        - 每块内仍用广播，但中间张量规模从 N×M×nb 降到 N×tile_M×nb。
        """
        assert A_packed.dtype == torch.uint8 and B_packed.dtype == torch.uint8
        N, nb = A_packed.shape
        M, nb2 = B_packed.shape
        assert nb == nb2, "nbytes mismatch"

        # 结果缓冲
        # matches = A_packed.new_empty((N, M), dtype=torch.int32)

        # 让张量连续可提升访问效率
        A = A_packed.contiguous()
        B = B_packed.contiguous()

        # 分块循环（经验：4k~16k 之间找一个能跑满显存又不爆的块）
        
        # j1 = min(j0 + tile_M, M)
        # Bj = B[j0:j1]  # (t, nbytes)

        # XNOR = ~(A ^ Bj)
        x = torch.bitwise_not(A.unsqueeze(1) ^ B.unsqueeze(0))      # (N, t, nbytes) uint8

        # LUT popcount: uint8 -> [0..8]，再沿 nbytes 求和到 int32
        # 注：某些 PyTorch 版本对 uint8 索引支持不一，保险起见转 long。
        cnt = self._pop_lut_256[x.long()].to(torch.int32).sum(dim=-1)  # (N, t) int32

        

        return cnt

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        # Numerically stable version: log(cosh(x)) = |x| + log(1 + exp(-2|x|))
        # For large |x|, this approaches |x|
        abs_x = torch.abs(x)
        # Clamp to prevent overflow in exp(-2*abs_x)
        abs_x_clamped = torch.clamp(abs_x, max=20.0)  
        return abs_x_clamped + torch.log1p(torch.exp(-2.0 * abs_x_clamped))
    
    def regularizers(self, hv: HashedVector, use_smooth_abs: bool = True,
                     reduction: str = "mean"):
        """
        返回: L_q, L_b, L_r （3个标量）
        - 训练态: 基于 hv.bin_like
        - 评估态: 基于 sign(hv.code)
        """
        B = hv.bin_like if self.training else torch.sign(hv.code)  # (..., Dh)
        B = B.reshape(-1, B.size(-1))                              # (N, Dh)

        # L_q = sum || |B| - 1 ||_1  ，用 log(cosh) 平滑近似绝对值
        x = B.abs() - 1.0
        lq_map = self._log_cosh(x) if use_smooth_abs else x.abs()
        L_q = lq_map.mean() if reduction == "mean" else lq_map.sum()

        # L_b = (1/l) * sum_j ( (1/N) * sum_i b_ij )^2
        bit_means = B.mean(dim=0)          # (Dh,)
        L_b = (bit_means ** 2).mean()      
    
        # Reconstruction loss
        #L_r = (1/N) * sum_i || x_i - x'_i ||_2^2
        lr_map = (hv.recon - hv.x_origin).pow(2)
        L_r = lr_map.mean() if reduction == "mean" else lr_map.sum()  
        return L_q, L_b, L_r

class AdditiveAttention(nn.Module):
    def __init__ (self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
    
    def forward(self, input):
        attn_weights = F.softmax(self.W(input), dim=-1)
        return torch.sum(attn_weights * input, dim=1)


class MultiScaleDilatedConv(nn.Module):
    """
    H^c(σ) = ReLU(Conv1D_σ(H^c)), 公式(16)
    传入 (N, L, D)，返回按通道拼接的多尺度结果 (N, L, S*D)，S 为尺度数。
    """
    def __init__(self, d_model: int, kernel_size: int = 5, dilations=(1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList()
        padding_fn = lambda k, d: (k - 1) // 2 * d  # 保长 padding
        for d in dilations:
            self.branches.append(
                nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                          padding=padding_fn(kernel_size, d), dilation=d, bias=False)
            )

    def forward(self, x, mask=None):
        # x: (N, L, D)
        N, L, D = x.shape
        x_c = x.transpose(1, 2)  # (N, D, L)
        outs = []
        for conv in self.branches:
            y = conv(x_c)  # (N, D, L)
            y = F.relu(y)
            outs.append(y.transpose(1, 2))  # -> (N, L, D)
        out = torch.cat(outs, dim=-1)  # (N, L, S*D)
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out  # (N, L, S*D)


class ConvolutionalStartEndDetector(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1,
                 kernel_size: int = 5, dilations=(1, 2, 4), proj_out: int = None, lstm_hidden: int = None):
        super().__init__()
        self.query_proj_v = nn.Linear(d_model, d_model)
        self.query_proj_s = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        padding = kernel_size // 2
        self.conv_start = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=True)
        self.conv_end = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=True)

    def _pool_query(self, query_tokens, query_mask):
        if query_mask is None:
            return query_tokens.mean(dim=1)
        weights = query_mask.float()
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.sum(query_tokens * weights.unsqueeze(-1), dim=1) / denom

    def _predict_span(self, scores, mask):
        logits = scores.unsqueeze(1)  # (B, 1, L)
        start_logits = self.conv_start(logits).squeeze(1)
        end_logits = self.conv_end(logits).squeeze(1)
        if mask is not None:
            mask = mask.float()
            start_logits = start_logits + (1.0 - mask) * (-1e4)
            end_logits = end_logits + (1.0 - mask) * (-1e4)
        return start_logits, end_logits

    def forward(self,
                video_feat, video_mask,
                sub_feat,   sub_mask,
                query_tokens, query_mask,
                pairwise: bool = False,
                chunk_size: int = 64,
                use_autocast: bool = True):
        Nc, L, _ = video_feat.size()
        Nq = query_tokens.size(0)

        query_summary = self._pool_query(query_tokens, query_mask)
        qv = self.dropout(self.query_proj_v(query_summary))
        use_sub = sub_feat is not None and sub_mask is not None
        if use_sub:
            qs = self.dropout(self.query_proj_s(query_summary))

        if (not pairwise) and (Nc == Nq):
            S_v = (video_feat * qv.unsqueeze(1)).sum(dim=-1)
            if video_mask is not None:
                S_v = S_v + (1.0 - video_mask.float()) * (-1e4)
            if use_sub:
                S_s = (sub_feat * qs.unsqueeze(1)).sum(dim=-1)
                S_s = S_s + (1.0 - sub_mask.float()) * (-1e4)
                S = 0.5 * (S_v + S_s)
            else:
                S = S_v
            Ss, Se = self._predict_span(S, video_mask)
            return Ss, Se

        Ss_parts, Se_parts = [], []
        for c0 in range(0, Nc, chunk_size):
            c1 = min(c0 + chunk_size, Nc)
            Bc = c1 - c0
            v_blk = video_feat[c0:c1]                        # (Bc, L, D)
            vm_blk = video_mask[c0:c1] if video_mask is not None else None

            S_v = torch.einsum("bld,qd->qbl", v_blk, qv)      # (Nq, Bc, L)
            if vm_blk is not None:
                S_v = S_v + (1.0 - vm_blk.float().unsqueeze(0)) * (-1e4)

            if use_sub:
                s_blk = sub_feat[c0:c1]
                sm_blk = sub_mask[c0:c1]
                S_s = torch.einsum("bld,qd->qbl", s_blk, qs)
                S_s = S_s + (1.0 - sm_blk.float().unsqueeze(0)) * (-1e4)
                S = 0.5 * (S_v + S_s)
            else:
                S = S_v

            S_flat = S.reshape(-1, L)
            mask_flat = None
            if vm_blk is not None:
                mask_flat = vm_blk.unsqueeze(0).expand(Nq, -1, -1).reshape(-1, L)
            Ss_blk, Se_blk = self._predict_span(S_flat, mask_flat)
            Ss_parts.append(Ss_blk.view(Nq, Bc, L))
            Se_parts.append(Se_blk.view(Nq, Bc, L))

        Ss = torch.cat(Ss_parts, dim=1)
        Se = torch.cat(Se_parts, dim=1)
        return Ss, Se

NEG_INF = -1e4  # 与规范中建议保持一致


def _ensure_bool01_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
    """
    将掩码规范化为 float32 的 {0,1}，并做取值合法性检查。
    """
    if mask.dtype == torch.bool:
        mask01 = mask.to(dtype=torch.float32)
    else:
        mask01 = mask.to(dtype=torch.float32)
        ok = torch.all((mask01 == 0) | (mask01 == 1))
        if not ok:
            raise ValueError(
                f"[MaskIllegal] {name} 必须为布尔或{{0,1}}，但检测到其它取值；"
                f"实际 dtype={mask.dtype}, 取值范围约=({float(mask01.min())}, {float(mask01.max())})."
            )
    return mask01

def _check_finite(x: torch.Tensor, name: str):
    if not torch.isfinite(x).all():
        raise ValueError(f"[NonFinite] 输入 {name} 含 NaN/Inf，请检查上游预处理。")

def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    # mask: same shape as x without feature dim, broadcastable
    m = mask.to(x.dtype)
    num = (x * m).sum(dim=dim)
    den = m.sum(dim=dim).clamp_min(1e-6)
    return num / den

class ConditionalEndGivenStart(nn.Module):
    """
    给定起点锚点，使用单向 LSTM 仅在尾段 t >= t_s 上预测终点 logits。
    - Hq: (N, L, D)
    - anchor_idx: (N,) 每个样本的起点帧下标
    - mask: (N, L) 0/1（可为 None）。本模块不做 padding / 负无穷；交由外层统一处理。
    返回:
    - Se: (N, L)，仅 t >= anchor 位置写入预测，其余为 0（外层再置 -inf/做掩码）
    """
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.init_end = nn.Linear(d_model, 2 * hidden)   # -> (h0, c0)
        self.lstm_end = nn.LSTM(input_size=2 * d_model,  # concat(frame, anchor)
                                hidden_size=hidden, batch_first=True)
        self.fc_end = nn.Linear(hidden + d_model, 1)

    def forward(self, Hq, anchor_idx, mask=None):
        N, L, D = Hq.size()
        device = Hq.device

        # (0) indices must be long
        anchor_idx = anchor_idx.long()

        # (1) anchor vector & initial state
        a = Hq[torch.arange(N, device=device), anchor_idx]          # (N, D)
        h0, c0 = self.init_end(a).chunk(2, dim=-1)                  # (N, H), (N, H)
        h0 = h0.unsqueeze(0).contiguous()                           # (1, N, H)
        c0 = c0.unsqueeze(0).contiguous()                           # (1, N, H)

        # (2) tail lengths
        lengths = (L - anchor_idx).clamp_min(0)                     # (N,)
        T_max = int(lengths.max().item())

        # output buffer with same dtype/device as Hq
        Se = Hq.new_zeros(N, L)                                     # (N, L)

        if T_max == 0:
            return Se  # all anchors at last frame

        # (3) tail time indices
        t = torch.arange(T_max, device=device, dtype=anchor_idx.dtype).unsqueeze(0)  # (1, T_max)
        idx_time = anchor_idx.unsqueeze(1) + t                                       # (N, T_max)
        valid = idx_time < L                                                         # (N, T_max)
        idx_time = idx_time.clamp(max=L-1)                                          # keep in range for gather

        # (4) gather tails and build inputs
        tails = Hq.gather(1, idx_time.unsqueeze(-1).expand(-1, -1, D))               # (N, T_max, D)
        a_ctx = a.unsqueeze(1).expand(N, T_max, D)                                   # (N, T_max, D)
        x = torch.cat([tails, a_ctx], dim=-1)                                        # (N, T_max, 2D)

        # (5) pack with nonzero lengths for safety (older PyTorch)
        lengths_for_pack = torch.where(lengths == 0, torch.ones_like(lengths), lengths)
        packed = pack_padded_sequence(x, lengths=lengths_for_pack.cpu(),
                                    batch_first=True, enforce_sorted=False)
        packed_h, _ = self.lstm_end(packed, (h0, c0))
        h, _ = pad_packed_sequence(packed_h, batch_first=True, total_length=T_max)   # (N, T_max, H)

        # (6) logits on tail
        logits_tail = self.fc_end(torch.cat([h, tails], dim=-1)).squeeze(-1)         # (N, T_max)

        # (7) scatter back only at valid positions
        bidx = torch.arange(N, device=device, dtype=idx_time.dtype).unsqueeze(1).expand_as(idx_time)  # (N, T_max)

        # Ensure dtype match on write; advanced-index assignment is clearer than index_put_
        Se[bidx[valid].long(), idx_time[valid].long()] = logits_tail[valid].to(Se.dtype)

        return Se


class AtomicEventMomentLocalizationModule(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        # —— 算法相关可选参数（不改变对外 IO）——
        proj_out: Optional[int] = None,
        lstm_hidden: Optional[int] = None,
        # 原子事件分割
        latent_dim: int = 64,
        boundary_percentile: float = 95.0,   # τ = p-th 百分位
        # 概率边界生成器
        decoder: str = "pointer",            # ["kadane", "pointer"]
        smooth_kernel: int = 9,             # kadane 路径的 1D 平滑
        mask_value: float = NEG_INF,
        strict_checks: bool = True,         # 是否严格按规范抛错
        auto_switch_pairwise: bool = False  # Nc≠Nq 且 pairwise=False 时是否自动切换
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)
        self.latent_dim = int(latent_dim)
        self.boundary_percentile = float(boundary_percentile)
        self.decoder = decoder.lower()
        self.smooth_kernel = int(smooth_kernel)
        self.mask_value = float(mask_value)
        self.strict_checks = bool(strict_checks)
        self.auto_switch_pairwise = bool(auto_switch_pairwise)

        if self.decoder not in {"kadane", "pointer"}:
            raise ValueError(f"[Config] decoder 仅支持 'kadane' 或 'pointer'，收到: {decoder}")

        # 跨注意力：H_c ⟵ Attn(Q=H_c, K=Q~, V=Q~)
        self.attn_ctx = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        # 若存在辅助模态（aux_feat），为其也做一次跨注意力（与 PDF 的多模态融合一致）
        self.attn_aux = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        # 多模态融合（concat 后线性映射回 d_model）
        self.fuse = nn.Linear(2 * d_model, d_model)

        # 潜空间映射用于 L1 变化检测
        self.latent_proj = nn.Linear(d_model, self.latent_dim)

        # Pointer 路径：事件起点分类、条件终点指针
        self.fc_event_start = nn.Linear(d_model, 1)
        hid = lstm_hidden if lstm_hidden is not None else self.latent_dim // 2
        self.fc_pointer_end = nn.Sequential(
            nn.Linear(2 * d_model, hid),
            nn.GELU(),
            nn.Linear(hid, 1),
        )
        self.end_given_start = ConditionalEndGivenStart(d_model=self.d_model, hidden=hid)


        # 可选的输出降维（外部接口不变，仅内部特征可用）
        self.proj_out = None
        if proj_out is not None:
            self.proj_out = nn.Linear(d_model, int(proj_out))

    def _neg_fill_like(self, x: torch.Tensor, prefer: float) -> torch.Tensor:
        # fp16/bf16 用 -1e4，其他 dtype 用 prefer（通常是 -1e10）
        if x.dtype in (torch.float16, torch.bfloat16):
            val = -1e4
        else:
            val = prefer
        return torch.full_like(x, val)

    def _many_pairs_one_context(
        self,
        q_tok: torch.Tensor,  # (M, Lq, D)
        q_msk: torch.Tensor,  # (M, Lq)
        c_feat: torch.Tensor, # (1 or M, L, D) —— 若为 (1,...) 会在函数内 expand 到 (M,...)
        c_msk: torch.Tensor,  # (1 or M, L)
        a_feat: Optional[torch.Tensor],  # (1 or M, L, D) | None
        a_msk: Optional[torch.Tensor],   # (1 or M, L)   | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 统一维度
        M, Lq, D = q_tok.shape
        L = c_feat.size(1)
        if c_feat.size(0) == 1:
            c_feat = c_feat.expand(M, L, D).contiguous()
            c_msk  = c_msk.expand(M, L).contiguous()
            if a_feat is not None:
                a_feat = a_feat.expand(M, L, D).contiguous()
                a_msk  = a_msk.expand(M, L).contiguous()

        # 1) 跨注意力融合（批量）
        Hcq, _ = self.attn_ctx(query=c_feat, key=q_tok, value=q_tok, key_padding_mask=(q_msk == 0))
        if a_feat is not None:
            Haq, _ = self.attn_aux(query=a_feat, key=q_tok, value=q_tok, key_padding_mask=(q_msk == 0))
            H = self.fuse(torch.cat([Hcq, Haq], dim=-1))  # (M, L, D)
            H = H * a_msk.unsqueeze(-1)                   # 避免辅助模态 padding 污染
        else:
            H = Hcq
        # 上下文 padding 清零
        H = H * c_msk.unsqueeze(-1)

        # 2) 潜空间 & 事件边界（向量化）
        latent = F.relu(self.latent_proj(H))             # (M, L, Dl)
        diffs = torch.abs(latent[:, 1:, :] - latent[:, :-1, :]).sum(dim=-1)  # (M, L-1)
        qthr  = torch.quantile(diffs, self.boundary_percentile / 100.0, dim=1, keepdim=True)  # (M,1)
        boundary_bool = (diffs > qthr).to(torch.long)    # (M, L-1), {0,1}

        # 帧→事件段 ID：idx[:,0]=0；之后每遇到边界就 +1
        idx = torch.cumsum(
            torch.cat([torch.zeros((M, 1), dtype=torch.long, device=diffs.device), boundary_bool], dim=1),
            dim=1
        )  # (M, L)，取值范围 [0, Ne_m-1]

        Emax = int(idx.max().item()) + 1                # 所有样本的最大事件数（用于对齐）

        # 3) 事件池化（分段求和/计数，纯张量）
        X = H                                           # (M, L, D)
        w = c_msk.unsqueeze(-1).type_as(X)              # (M, L, 1)

        # 将 (batch, frame) 的二维索引映射为一维 offset 索引，便于 index_add_
        offset = (torch.arange(M, device=idx.device, dtype=torch.long) * Emax).unsqueeze(1)  # (M,1)
        flat_idx = (idx + offset).reshape(-1)           # (M·L,)

        sum_ev = X.new_zeros((M * Emax, D)).index_add_(0, flat_idx, (X * w).reshape(M * L, D)).view(M, Emax, D)
        cnt_ev = X.new_zeros((M * Emax, 1)).index_add_(0, flat_idx, w.reshape(M * L, 1)).view(M, Emax, 1)
        Ev = sum_ev / cnt_ev.clamp_min(1e-6)            # (M, Emax, D)

        # 4) 查询-事件相似度 & 帧级回填（无 for）
        q_vec = _masked_mean(q_tok, q_msk.unsqueeze(-1), dim=1)               # (M, D)
        Sev = F.cosine_similarity(Ev, q_vec.unsqueeze(1), dim=-1)             # (M, Emax)

        # 将没有任何帧的“空事件”置为极小值，避免被 Kadane/Pointer 误选
        invalid_ev = (cnt_ev.squeeze(-1) <= 0)                                 # (M, Emax)
        Sev = torch.where(invalid_ev, self._neg_fill_like(Sev, self.mask_value), Sev)

        # 帧级分数：按 idx 在维度 1 上 gather（每帧取其所属事件分数）
        S = torch.gather(Sev, dim=1, index=idx)                                # (M, L)

        # 事件起点 logits：选 anchor 事件
        start_ev_logits = self.fc_event_start(Ev).squeeze(-1)              # (M, Emax)
        anchor_ev = torch.argmax(start_ev_logits, dim=1)                   # (M,)

        # 事件→帧：找每个样本的 anchor 事件的“首帧位置”
        arange_L = torch.arange(L, device=idx.device).unsqueeze(0).expand(M, L)  # (M, L)
        eq = (idx == anchor_ev.unsqueeze(1))                               # (M, L)
        # 将不等处设为 L，取行最小值即为首个等于 anchor_ev 的位置
        anchor_frame = torch.min(torch.where(eq, arange_L, arange_L.new_full(arange_L.shape, L)), dim=1).values
        anchor_frame = anchor_frame.clamp(max=L-1)                          # 安全

        # 帧级 start_logits：把事件起点 logits 广播到帧
        start_logits = torch.gather(start_ev_logits, dim=1, index=idx)     # (M, L)


        end_logits = self.end_given_start(H, anchor_frame, mask=c_msk)
        
        before_anchor = arange_L < anchor_frame.unsqueeze(1)             # (M, L)
        end_logits = torch.where(
            before_anchor, self._neg_fill_like(end_logits, self.mask_value), end_logits
        )
        # # 条件终点：拼接 (H_t, H_anchor) 预测
        # anchor_feat = H[torch.arange(M, device=H.device), anchor_frame, :]  # (M, D)
        # tiled_anchor = anchor_feat.unsqueeze(1).expand(M, L, D)             # (M, L, D)
        # pair_feat = torch.cat([H, tiled_anchor], dim=-1)                    # (M, L, 2D)
        # end_logits = self.fc_pointer_end(pair_feat).squeeze(-1)             # (M, L)

        # # 物理约束：t < anchor 不可能为终点 → 屏蔽
        # before_anchor = arange_L < anchor_frame.unsqueeze(1)                # (M, L)
        # end_logits = torch.where(before_anchor, self._neg_fill_like(end_logits, self.mask_value), end_logits)

        # 5) 上下文 padding 屏蔽（fp16 安全）
        start_logits = torch.where(c_msk > 0, start_logits, self._neg_fill_like(start_logits, self.mask_value))
        end_logits   = torch.where(c_msk > 0, end_logits,   self._neg_fill_like(end_logits,   self.mask_value))
        return start_logits, end_logits



    # ---------------------------
    # 规范定义的前向接口
    # ---------------------------
    def forward(
        self,
        context_feat: torch.Tensor,   # (Nc, L, D)
        context_mask: Optional[torch.Tensor],  # (Nc, L) | None
        aux_feat: Optional[torch.Tensor],      # (Nc, L, D) | None
        aux_mask: Optional[torch.Tensor],      # (Nc, L) | None
        query_tokens: torch.Tensor,   # (Nq, Lq, D)
        query_mask: Optional[torch.Tensor],    # (Nq, Lq) | None
        pairwise: bool = False,
        chunk_size: int = 256,
        use_autocast: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
          - 若 pairwise=False 且 Nc==Nq: (start_logits, end_logits) 形状 (N, L)
          - 否则: (Nq, Nc, L)
        注意:
          - 输出 logits 的无效帧位置已按 context_mask 屏蔽为极小值 (mask_value)。
        """
        # ----- 基本形状/数值检查 -----
        if context_feat.ndim != 3:
            raise ValueError(f"[Shape] context_feat 期望 (Nc,L,D)，实际 {tuple(context_feat.shape)}")
        if query_tokens.ndim != 3:
            raise ValueError(f"[Shape] query_tokens 期望 (Nq,Lq,D)，实际 {tuple(query_tokens.shape)}")
        Nc, L, Dc = context_feat.shape
        Nq, Lq, Dq = query_tokens.shape
        if Dc != Dq:
            raise ValueError(f"[DimMismatch] D 不一致: context_feat.D={Dc}, query_tokens.D={Dq}")
        if aux_feat is not None:
            if aux_feat.shape != (Nc, L, Dc):
                raise ValueError(f"[Shape] aux_feat 期望 (Nc,L,D)=({Nc},{L},{Dc})，实际 {tuple(aux_feat.shape)}")
        if context_mask is None:
            context_mask = torch.ones((Nc, L), dtype=torch.float32, device=context_feat.device)
        else:
            if context_mask.shape != (Nc, L):
                raise ValueError(f"[Shape] context_mask 期望 (Nc,L)=({Nc},{L})，实际 {tuple(context_mask.shape)}")
            context_mask = _ensure_bool01_mask(context_mask, "context_mask").to(device=context_feat.device)
        if aux_feat is not None:
            if aux_mask is None:
                aux_mask = torch.ones((Nc, L), dtype=torch.float32, device=context_feat.device)
            else:
                if aux_mask.shape != (Nc, L):
                    raise ValueError(f"[Shape] aux_mask 期望 (Nc,L)=({Nc},{L})，实际 {tuple(aux_mask.shape)}")
                aux_mask = _ensure_bool01_mask(aux_mask, "aux_mask").to(device=context_feat.device)
        if query_mask is None:
            query_mask = torch.ones((Nq, Lq), dtype=torch.float32, device=query_tokens.device)
        else:
            if query_mask.shape != (Nq, Lq):
                raise ValueError(f"[Shape] query_mask 期望 (Nq,Lq)=({Nq},{Lq})，实际 {tuple(query_mask.shape)}")
            query_mask = _ensure_bool01_mask(query_mask, "query_mask").to(device=query_tokens.device)
        # 数值有限性
        _check_finite(context_feat, "context_feat")
        _check_finite(query_tokens, "query_tokens")
        if aux_feat is not None:
            _check_finite(aux_feat, "aux_feat")

        # ----- pairwise 语义确定 -----
        if (Nc != Nq) and (not pairwise):
            if self.strict_checks and (not self.auto_switch_pairwise):
                raise ValueError(
                    "[BatchMismatch] pairwise=False 但 Nc≠Nq；"
                    "该模式要求同批一一配对。若需两两匹配，请将 pairwise=True。"
                )
            pairwise = True

        device = context_feat.device
        dtype = context_feat.dtype
        device_type = "cuda" if context_feat.is_cuda else "cpu"
        autocast_ctx = (
            torch.autocast(device_type=device_type, enabled=use_autocast)
            if hasattr(torch, "autocast") else nullcontext()
        )

        # 输出缓冲区
        if pairwise:
            start_all = torch.empty((Nq, Nc, L), dtype=dtype, device=device)
            end_all   = torch.empty((Nq, Nc, L), dtype=dtype, device=device)
        else:
            N = Nc  # Nc == Nq
            start_all = torch.empty((N, L), dtype=dtype, device=device)
            end_all   = torch.empty((N, L), dtype=dtype, device=device)

        # ---------- 主循环（支持 pairwise 与分块） ----------
        if pairwise:
            # 用同一 chunk_size 同时控制 query 与 context 的块大小
            # 如需独立控制，可把 ctx_block 改成 self.ctx_block 或新增 forward 入参
            ctx_block = max(1, int(chunk_size))
            qi = 0
            while qi < Nq:
                qj = min(qi + max(1, int(chunk_size)), Nq)
                q_blk  = query_tokens[qi:qj, :, :]   # (M, Lq, D)
                qm_blk = query_mask[qi:qj, :]        # (M, Lq)
                M = qj - qi

                cj0 = 0
                while cj0 < Nc:
                    cj1 = min(cj0 + ctx_block, Nc)
                    K = cj1 - cj0

                    c_blk  = context_feat[cj0:cj1, :, :]    # (K, L, D)
                    cm_blk = context_mask[cj0:cj1, :]       # (K, L)
                    if aux_feat is not None:
                        a_blk  = aux_feat[cj0:cj1, :, :]    # (K, L, D)
                        am_blk = aux_mask[cj0:cj1, :]       # (K, L)
                    else:
                        a_blk = None
                        am_blk = None

                    # === 关键：块内做笛卡尔积，形成 (B=M*K, …) 的一一配对 batch ===
                    # queries: (M, Lq, D) -> (M, K, Lq, D) -> (M*K, Lq, D)
                    q_b  = q_blk[:, None, :, :].expand(M, K, Lq, Dc).reshape(M*K, Lq, Dc).contiguous()
                    qm_b = qm_blk[:, None, :].expand(M, K, Lq).reshape(M*K, Lq).contiguous()

                    # contexts: (K, L, D) -> (M, K, L, D) -> (M*K, L, D)
                    c_b  = c_blk[None, :, :, :].expand(M, K, L, Dc).reshape(M*K, L, Dc).contiguous()
                    cm_b = cm_blk[None, :, :].expand(M, K, L).reshape(M*K, L).contiguous()

                    # aux（可选）
                    if a_blk is not None:
                        a_b  = a_blk[None, :, :, :].expand(M, K, L, Dc).reshape(M*K, L, Dc).contiguous()
                        am_b = am_blk[None, :, :].expand(M, K, L).reshape(M*K, L).contiguous()
                    else:
                        a_b, am_b = None, None

                    # 一次性算完 (M*K) 对配对
                    with autocast_ctx:
                        s_b, e_b = self._many_pairs_one_context(
                            q_b, qm_b, c_b, cm_b, a_b, am_b
                        )  # -> (M*K, L)

                    # 还原成 (M, K, L) 并写回 (Nq, Nc, L)
                    s_blk = s_b.view(M, K, L)
                    e_blk = e_b.view(M, K, L)
                    start_all[qi:qj, cj0:cj1, :] = s_blk
                    end_all[qi:qj,   cj0:cj1, :] = e_blk

                    cj0 = cj1
                qi = qj

        else:
            # 同批一一配对：直接批处理 (N, ...)
            s_logits, e_logits = self._many_pairs_one_context(
                query_tokens, query_mask,
                context_feat, context_mask,
                aux_feat, aux_mask
            )
            start_all[:, :] = s_logits  # (N, L)
            end_all[:, :]   = e_logits
        return start_all, end_all


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (L,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 注册为buffer，自动随设备/精度移动
        self.register_buffer('pe', pe)  # (L, D)

    def forward(self, x: torch.Tensor):
        """
        x: (N, L, D)
        """
        L = x.size(1)
        x = x + self.pe[:L, :].unsqueeze(0).to(dtype=x.dtype)  # (1,L,D)
        return self.dropout(x)


class CNN2DFeatureExtractor(nn.Module):
    """
    将相似度矩阵 M (N, 1, L, Lq) 编码为多通道特征 (N, C, L, Lq)
    采用多层(3x3)卷积 + BN + ReLU，支持可配置的空洞率（统一用于两维）。
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 64, num_blocks: int = 3,
                 kernel_size: int = 3, dilations=(1, 2, 4)):
        super().__init__()
        assert num_blocks >= 1
        ks = kernel_size
        pad = lambda d: ((ks - 1) // 2) * d
        chs = [in_ch] + [base_ch] * num_blocks
        layers = []
        for i in range(num_blocks):
            d = dilations[i % len(dilations)]
            layers += [
                nn.Conv2d(chs[i], chs[i+1], kernel_size=ks,
                          padding=pad(d), dilation=d, bias=False),
                nn.BatchNorm2d(chs[i+1]),
                nn.ReLU(inplace=True),
            ]
        self.net = nn.Sequential(*layers)
        self.out_channels = base_ch

    def forward(self, x):
        return self.net(x)  # (N, C, L, Lq)

class TemporalAggregator(nn.Module):
    def __init__(self, in_ch: int, d_model: int):
        super().__init__()
        self.attn_conv = nn.Conv2d(in_ch, 1, kernel_size=1)   # (N,1,L,Lq)
        self.proj = nn.Conv1d(in_ch * 3, d_model, kernel_size=1)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, feat_2d, q_mask):
        """
        feat_2d: (N, C, L, Lq)
        q_mask : (N, Lq)  1/True 有效；0/False 无效；允许 float/bool/long
        return : (N, L, D)
        """
        assert feat_2d.dim() == 4, f"feat_2d must be (N,C,L,Lq), got {feat_2d.shape}"
        N, C, L, Lq = feat_2d.shape
        device, dtype = feat_2d.device, feat_2d.dtype

        # 标准化 mask
        if q_mask is None:
            q_mask = torch.ones(N, Lq, dtype=torch.bool, device=device)
        else:
            q_mask = q_mask.to(torch.bool).to(device)

        mq_bool = q_mask[:, None, None, :]                 # (N,1,1,Lq) bool
        mq_float = mq_bool.to(dtype)                       # (N,1,1,Lq) float
        neg_inf = torch.finfo(dtype).min

        # 样本级：是否至少有一个有效 token
        has_valid_q = q_mask.any(dim=-1)                   # (N,)
        has_valid_q_3d = has_valid_q[:, None, None]        # (N,1,1)
        has_valid_q_4d = has_valid_q[:, None, None, None]  # (N,1,1,1)

        # ---- masked max -> (N,C,L)
        x_masked = feat_2d.masked_fill(~mq_bool, neg_inf)  # (N,C,L,Lq)
        max_pool = torch.amax(x_masked, dim=-1)            # (N,C,L)
        # 无有效 token 的样本，置零，避免传播极小值
        max_pool = torch.where(has_valid_q_3d, max_pool, torch.zeros_like(max_pool))

        # ---- masked avg -> (N,C,L)
        sum_pool = (feat_2d * mq_float).sum(dim=-1)        # (N,C,L)
        valid_cnt = mq_float.sum(dim=-1).clamp(min=1e-6)   # (N,1,1)
        avg_pool = sum_pool / valid_cnt                    # (N,C,L)

        # ---- masked attention -> (N,C,L)
        energy = self.attn_conv(feat_2d)                   # (N,1,L,Lq)
        energy = energy.masked_fill(~mq_bool, neg_inf)
        attn = torch.softmax(energy, dim=-1)               # (N,1,L,Lq)
        # 全被 mask 时 softmax 得到 NaN；直接置 0
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = torch.where(has_valid_q_4d, attn, torch.zeros_like(attn))
        attn_pool = (feat_2d * attn).sum(dim=-1)           # (N,C,L)

        # —— 统一保证三者都是 (N,C,L) ——（防止意外多出一维）
        def to_NCL(x):
            if x.dim() == 4 and x.size(-1) == 1:
                return x.squeeze(-1)
            elif x.dim() == 3:
                return x
            elif x.dim() == 4:
                # 极端 fallback：平均掉最后一维
                return x.mean(dim=-1)
            else:
                raise RuntimeError(f"Unexpected tensor dim: {x.shape}")

        max_pool  = to_NCL(max_pool)
        avg_pool  = to_NCL(avg_pool)
        attn_pool = to_NCL(attn_pool)

        # 拼接并投影 -> (N,L,D)
        cat = torch.cat([max_pool, avg_pool, attn_pool], dim=1)  # (N,3C,L)
        out = self.proj(cat).transpose(1, 2).contiguous()        # (N,L,D)
        out = self.ln(out)
        return out

class MultiScaleDilatedHead(nn.Module):
    def __init__(self, ks=(3,5), dilations=(1,2,4)):
        super().__init__()
        banks = []
        for k in ks:
            for d in dilations:
                pad = (d * (k - 1)) // 2  # stride=1 等长
                banks.append(nn.Conv1d(1, 1, kernel_size=k, padding=pad, dilation=d, bias=False))
        self.banks = nn.ModuleList(banks)
        # 融合各分支：把通道数=分支数，1×1Conv 降到 1 通道
        self.fuse = nn.Conv1d(len(banks), 1, kernel_size=1, bias=False)

    def forward(self, sim):          # sim: (B, L)
        x = sim.unsqueeze(1)         # (B, 1, L)
        feats = [bank(x) for bank in self.banks]   # n * (B,1,L)
        x = torch.cat(feats, dim=1)  # (B, n, L)
        x = self.fuse(x)             # (B, 1, L)
        return x.squeeze(1)          # (B, L)