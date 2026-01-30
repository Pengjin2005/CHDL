import chunk
import copy
import math
from dataclasses import dataclass
from locale import normalize
from time import perf_counter  # <-- 导入 perf_counter
from tkinter import NO

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from method_tvr.contrastive import batch_local_token_frame_loss, batch_video_query_loss
from method_tvr.model_components import (
    AtomicEventMomentLocalizationModule,
    BertAttention,
    BertSelfAttention,
    Conv2DMomentLocalization,
    HashLayer,
    LinearLayer,
    MILNCELoss,
    MultiScaleDilatedHead,
    SpotlightMomentLocalization,
    TrainablePositionalEncoding,
)
from numpy import isin


def _sync_if_cuda(tensor):
    """如果张量在GPU上，则执行 CUDA 同步"""
    if tensor is not None and isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
        torch.cuda.synchronize()

class CHDL(nn.Module):
    def __init__(self, config):
        super(CHDL, self).__init__()
        self.config = config
        self._epoch = 0
        # self.config.lw_rec = 1.0
        # self.config.lw_q = 1e-2
        # self.config.lw_b = 1e-3
        # self.eta = nn.Parameter(torch.tensor(1.0))  # eta is a trainable parameter, initialized to 1.0
        self.eta = 1.0
        self.query_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_desc_l, 
            hidden_size=config.hidden_size, 
            dropout=config.input_drop
        )
        self.ctx_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_ctx_l, 
            hidden_size=config.hidden_size, 
            dropout=config.input_drop
        )

        self.query_input_proj = LinearLayer(
            config.query_input_size, 
            config.hidden_size, 
            layer_norm=True,
            dropout=config.input_drop, 
            relu=True
        )

        self.query_encoder = BertAttention(
            edict(hidden_size=config.hidden_size, 
                  intermediate_size=config.hidden_size,
                  hidden_dropout_prob=config.drop,
                  num_attention_heads=config.n_heads,
                  attention_probs_dropout_prob=config.drop)
            )
        self.query_encoder1 = copy.deepcopy(self.query_encoder)

        cross_att_cfg = edict(hidden_size=config.hidden_size, num_attention_heads=config.n_heads,
                              attention_probs_dropout_prob=config.drop)
        # use_video
        self.video_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.video_encoder1 = copy.deepcopy(self.query_encoder)
        self.video_encoder2 = copy.deepcopy(self.query_encoder)
        self.video_encoder3 = copy.deepcopy(self.query_encoder)
        self.video_cross_att = BertSelfAttention(cross_att_cfg)
        self.video_cross_layernorm = nn.LayerNorm(config.hidden_size)
        self.video_query_linear = nn.Linear(config.hidden_size, config.hidden_size)

        # use_sub
        self.sub_input_proj = LinearLayer(config.sub_input_size, config.hidden_size, layer_norm=True,
                                          dropout=config.input_drop, relu=True)
        self.sub_encoder1 = copy.deepcopy(self.query_encoder)
        self.sub_encoder2 = copy.deepcopy(self.query_encoder)
        self.sub_encoder3 = copy.deepcopy(self.query_encoder)
        self.sub_cross_att = BertSelfAttention(cross_att_cfg)
        self.sub_cross_layernorm = nn.LayerNorm(config.hidden_size)
        self.sub_query_linear = nn.Linear(config.hidden_size, config.hidden_size)

        self.modular_vector_mapping = nn.Linear(in_features=config.hidden_size, out_features=2, bias=False)

        # conv_cfg = dict(in_channels=1, out_channels=1, kernel_size=config.conv_kernel_size, stride=config.conv_stride, padding=config.conv_kernel_size // 2, bias=False)
        # self.merged_st_predictor = nn.Conv1d(**conv_cfg)
        # self.merged_ed_predictor = nn.Conv1d(**conv_cfg)

        ms_ks = getattr(config, "ms_kernel_sizes", (3,5,9,17))
        ms_dil = getattr(config, "ms_dilations", (1,))
        self.merged_st_predictor = MultiScaleDilatedHead(ms_ks, ms_dil)
        self.merged_ed_predictor = MultiScaleDilatedHead(ms_ks, ms_dil)

        self.temporal_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.nce_criterion = MILNCELoss(reduction='mean')

        self.hash1 = HashLayer(input_output_size=config.hidden_size, hidden_size=1024) 
        self.hash2 = HashLayer(input_output_size=config.hidden_size, hidden_size=1024)
        
        # self.video_aggregator = AdditiveAttention(config.hidden_size, config.hidden_size)
        # self.sub_aggregator = AdditiveAttention(config.hidden_size, config.hidden_size)

        self.moment = AtomicEventMomentLocalizationModule(
            d_model=config.hidden_size,
            n_heads=8,
            decoder="pointer",            # 或 "pointer"
            boundary_percentile=95.0,
            smooth_kernel=5,
            latent_dim=32
        )

        # self.moment = SpotlightMomentLocalization(
        #     d_model=config.hidden_size,
        #     n_heads=8,
        #     dropout=0.1,
        #     kernel_size=5,
        #     dilations=(1,2,4),
        #     proj_out=config.hidden_size,
        #     lstm_hidden=config.hidden_size,
        # )

        # self.moment = Conv2DMomentLocalization(
        #     d_model=config.hidden_size,
        #     n_heads=8,
        #     dropout=0.1,
        #     dilations=(1,2,4)
        # )

    def reset_parameters(self):
        """ Initialize the weights."""
        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size

    def set_train_st_ed(self, lw_st_ed):
        """pre-train video retrieval then span prediction"""
        self.config.lw_st_ed = lw_st_ed

    def forward(self, query_feat, query_mask, video_feat, video_mask, st_ed_indices, match_labels, sub_feat=None, sub_mask=None):
        """
        Args:
            query_feat: (N, Lq, Dq)
            query_mask: (N, Lq)
            video_feat: (N, Lv, Dv) or None
            video_mask: (N, Lv) or None
            sub_feat: (N, Lv, Ds) or None
            sub_mask: (N, Lv) or None
            st_ed_indices: (N, 2), torch.LongTensor, 1st, 2nd columns are st, ed labels respectively.
            match_labels: (N, Lv), torch.LongTensor, matching labels for detecting foreground and background (not used)
        """
            
        video_feat, sub_feat, mid_x_video_feat, mid_x_sub_feat, x_video_feat, x_sub_feat = self.encode_context(
            video_feat, video_mask, sub_feat, sub_mask, return_mid_output=True)

        # x_video_feat_hashed = self.hash1(x_video_feat, self.eta)
        # x_sub_feat_hashed = self.hash1(x_sub_feat, self.eta)
        video_query, sub_query, encoded_query, query_context_scores, st_prob, ed_prob, reg_loss = self.get_pred_from_raw_query(
            query_feat, query_mask, x_video_feat, video_mask, x_sub_feat, sub_mask, cross=False,
            return_query_feats=True)

        # frame level contrastive learning loss (FrameCL)
        loss_fcl = 0
        # if self.config.lw_fcl != 0:
        #     loss_fcl_vq = batch_video_query_loss(mid_x_video_feat, video_query, match_labels, video_mask, measure='JSD')
        #     loss_fcl_sq = batch_video_query_loss(mid_x_sub_feat, sub_query, match_labels, sub_mask, measure='JSD')
        #     loss_fcl = (loss_fcl_vq + loss_fcl_sq) / 2.0
        #     loss_fcl = self.config.lw_fcl * loss_fcl
        if self.config.lw_fcl != 0:
            # 获取查询的token表示（而不是单一的查询向量）
            # encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder, self.query_pos_embed)
            # encoded_query = self.query_encoder1(encoded_query, query_mask.unsqueeze(1))  # (N, Lq, D)
            
            # 视频模态的token-frame对比学习
            loss_fcl_vq = batch_local_token_frame_loss(
                x_video_feat, encoded_query, match_labels, video_mask, query_mask
            )
            
            # 字幕模态的token-frame对比学习 
            if x_sub_feat is not None:
                loss_fcl_sq = batch_local_token_frame_loss(
                    x_sub_feat, encoded_query, match_labels, sub_mask, query_mask
                )
                loss_fcl = (loss_fcl_vq + loss_fcl_sq) / 2.0
            else:
                loss_fcl = loss_fcl_vq
            loss_fcl = self.config.lw_fcl * loss_fcl
        
            
        # video level contrastive learning loss (VideoCL)
        loss_vcl = 0
        if self.config.lw_vcl != 0:
            mid_video_q2ctx_scores = self.get_unnormalized_video_level_scores(video_query, x_video_feat, video_mask)
            mid_video_q2ctx_scores, _ = torch.max(mid_video_q2ctx_scores, dim=1)
            if sub_feat is not None:
                mid_sub_q2ctx_scores = self.get_unnormalized_video_level_scores(sub_query, x_sub_feat, sub_mask)
                mid_sub_q2ctx_scores, _ = torch.max(mid_sub_q2ctx_scores, dim=1)
                mid_q2ctx_scores = (mid_video_q2ctx_scores + mid_sub_q2ctx_scores) / 2.0
            else:
                mid_q2ctx_scores = mid_video_q2ctx_scores
            loss_vcl = self.nce_criterion(mid_q2ctx_scores)
            loss_vcl = self.config.lw_vcl * loss_vcl
            
            
        # moment localization loss
        loss_st_ed = 0
        if self.config.lw_st_ed != 0:
            # Clean st_prob and ed_prob before CrossEntropyLoss
            st_prob_clean = torch.nan_to_num(st_prob, nan=0.0, posinf=50.0, neginf=-50.0)
            ed_prob_clean = torch.nan_to_num(ed_prob, nan=0.0, posinf=50.0, neginf=-50.0)
            st_prob_clean = torch.clamp(st_prob_clean, min=-50.0, max=50.0)
            ed_prob_clean = torch.clamp(ed_prob_clean, min=-50.0, max=50.0)
            
            loss_st = self.temporal_criterion(st_prob_clean, st_ed_indices[:, 0])
            loss_ed = self.temporal_criterion(ed_prob_clean, st_ed_indices[:, 1])
            loss_st_ed = loss_st + loss_ed
            loss_st_ed = self.config.lw_st_ed * loss_st_ed
            
            
        # video level retrieval loss
        loss_neg_ctx, loss_neg_q = 0, 0
        if self.config.lw_neg_ctx != 0 or self.config.lw_neg_q != 0:
            loss_neg_ctx, loss_neg_q = self.get_video_level_loss(query_context_scores)
            loss_neg_ctx = self.config.lw_neg_ctx * loss_neg_ctx
            loss_neg_q = self.config.lw_neg_q * loss_neg_q
        # sum loss
        loss_L_q = reg_loss["L_q"] * self.config.lw_q
        loss_L_b = reg_loss["L_b"] * self.config.lw_b
        loss_L_r = reg_loss["L_r"] * self.config.lw_rec
        
        # Clean all loss components before summation
        param = next(self.parameters())
        if not isinstance(loss_fcl, torch.Tensor):
            loss_fcl = torch.as_tensor(loss_fcl, device=param.device, dtype=param.dtype)
        if not isinstance(loss_vcl, torch.Tensor):
            loss_vcl = torch.as_tensor(loss_vcl, device=param.device, dtype=param.dtype)
        loss_fcl = torch.nan_to_num(loss_fcl, nan=0.0, posinf=1e6, neginf=-1e6)
        loss_vcl = torch.nan_to_num(loss_vcl, nan=0.0, posinf=1e6, neginf=-1e6)
        loss_st_ed = torch.nan_to_num(loss_st_ed, nan=0.0, posinf=1e6, neginf=-1e6)
        loss_neg_ctx = torch.nan_to_num(loss_neg_ctx, nan=0.0, posinf=1e6, neginf=-1e6)
        loss_neg_q = torch.nan_to_num(loss_neg_q, nan=0.0, posinf=1e6, neginf=-1e6)
        loss_L_b = torch.nan_to_num(loss_L_b, nan=0.0, posinf=1e6, neginf=-1e6)
        loss_L_q = torch.nan_to_num(loss_L_q, nan=0.0, posinf=1e6, neginf=-1e6)
        loss_L_r = torch.nan_to_num(loss_L_r, nan=0.0, posinf=1e6, neginf=-1e6)
        
        loss = loss_fcl + loss_vcl + loss_st_ed + loss_neg_ctx + loss_neg_q + loss_L_b + loss_L_q + loss_L_r
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)
        
        self._epoch += 1
        # Clamp eta to prevent it from growing too large
        self.eta = min(math.pow((1.0 * self._epoch + 1.0), 0.5), 10.0)
        
        return loss, {"loss_st_ed": float(loss_st_ed), "loss_fcl": float(loss_fcl), 
                      "loss_vcl": loss_vcl, "loss_neg_ctx": float(loss_neg_ctx), 
                      "loss_neg_q": float(loss_neg_q), "loss_L_b": float(loss_L_b),
                      "loss_L_q": float(loss_L_q), "loss_L_r": float(loss_L_r),
                      "loss_overall": float(loss)}

    def encode_query(self, query_feat, query_mask):
        encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder,
                                          self.query_pos_embed)  # (N, Lq, D)
        encoded_query = self.query_encoder1(encoded_query, query_mask.unsqueeze(1))
        video_query, sub_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 2
        return video_query, sub_query, encoded_query

    def encode_context(self, video_feat, video_mask, sub_feat, sub_mask, return_mid_output=False):
        '''
        Retunrns:
                encoded_video_feat: 添加位置编码后的特征
                ncoded_sub_feat: 添加位置编码后的特征
                x_encoded_video_feat_: H_v^'
                x_encoded_sub_feat_: H_s^'
                x_encoded_video_feat: H_v
                x_encoded_sub_feat): H_s
            
        '''
        # encoding video and subtitle features, respectively
        encoded_video_feat = self.encode_input(video_feat, video_mask, self.video_input_proj, self.video_encoder1,self.ctx_pos_embed)
        x_encoded_video_feat = encoded_video_feat
        if sub_feat is not None:
            encoded_sub_feat = self.encode_input(sub_feat, sub_mask, self.sub_input_proj, self.sub_encoder1, self.ctx_pos_embed)
            # cross encoding subtitle features
            x_encoded_video_feat = self.cross_context_encoder(encoded_video_feat, video_mask, encoded_sub_feat, sub_mask, self.video_cross_att, self.video_cross_layernorm)  # (N, L, D)
        else:
            encoded_sub_feat = None
        x_encoded_video_feat_ = self.video_encoder2(x_encoded_video_feat, video_mask.unsqueeze(1))
        # cross encoding video features
        if sub_feat is not None:
            x_encoded_sub_feat = self.cross_context_encoder(encoded_sub_feat, sub_mask, encoded_video_feat, video_mask, self.sub_cross_att, self.sub_cross_layernorm)  # (N, L, D)
            x_encoded_sub_feat_ = self.sub_encoder2(x_encoded_sub_feat, sub_mask.unsqueeze(1))
            # additional self encoding process
            x_encoded_video_feat = self.video_encoder3(x_encoded_video_feat_, video_mask.unsqueeze(1))
            x_encoded_sub_feat = self.sub_encoder3(x_encoded_sub_feat_, sub_mask.unsqueeze(1))
        else:
            x_encoded_sub_feat_ = None
            x_encoded_sub_feat = None
        if return_mid_output:
            return (encoded_video_feat, encoded_sub_feat, x_encoded_video_feat_, x_encoded_sub_feat_,
                    x_encoded_video_feat, x_encoded_sub_feat)
        else:
            return x_encoded_video_feat, x_encoded_sub_feat

    @staticmethod
    def cross_context_encoder(main_context_feat, main_context_mask, side_context_feat, side_context_mask,
                              cross_att_layer, norm_layer):
        """
        Args:
            main_context_feat: (N, Lq, D)
            main_context_mask: (N, Lq)
            side_context_feat: (N, Lk, D)
            side_context_mask: (N, Lk)
            cross_att_layer: cross attention layer
            norm_layer: layer norm layer
        """
        cross_mask = torch.einsum("bm,bn->bmn", main_context_mask, side_context_mask)  # (N, Lq, Lk)
        cross_out = cross_att_layer(main_context_feat, side_context_feat, side_context_feat, cross_mask)  # (N, Lq, D)
        residual_out = norm_layer(cross_out + main_context_feat)
        return residual_out

    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """
        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        return encoder_layer(feat, mask)  # (N, L, D_hidden)

    def get_modularized_queries(self, encoded_query, query_mask, return_modular_att=False):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        if return_modular_att:
            assert modular_queries.shape[1] == 2
            return modular_queries[:, 0], modular_queries[:, 1], modular_attention_scores
        else:
            assert modular_queries.shape[1] == 2
            return modular_queries[:, 0], modular_queries[:, 1]  # (N, D) * 2
    @staticmethod
    def _binarize_pm1(x: torch.Tensor) -> torch.Tensor:
        # sign(0) = 0 会带来零位；这里把 0 也当作 +1 以避免稀疏
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    # ---- helpers ----
    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        # Numerically stable version: log(cosh(x)) = |x| + log(1 + exp(-2|x|))
        abs_x = torch.abs(x)
        # Clamp to prevent overflow in exp(-2*abs_x)
        abs_x_clamped = torch.clamp(abs_x, max=20.0)  
        return abs_x_clamped + torch.log1p(torch.exp(-2.0 * abs_x_clamped))

    @staticmethod
    def hash_regularizers(B: torch.Tensor, smooth_abs: bool = True, reduction: str = "mean"):
        """
        B: (N, l)  某层的哈希码（训练时用 bin_like；评估时若要记录数值可用 sign(code)）
        return: L_q, L_b  两个标量
        """
        # L_q ：让每一位→±1
        x = B.abs() - 1.0
        lq_map = CHDL._log_cosh(x) if smooth_abs else x.abs()
        L_q = lq_map.mean() if reduction == "mean" else lq_map.sum()

        # L_b ：让各位在 batch 内均值为 0
        bit_means = B.mean(dim=0)         # (l,)
        L_b = (bit_means ** 2).mean()     # 标量
        return L_q, L_b

    # @staticmethod
    # def get_video_level_scores(modularied_query, context_feat, context_mask, hash_layer=None, epoch=None):
    #     """ Calculate video2query scores for each pair of video and query inside the batch.
    #     Args:
    #         modularied_query: (N, D)
    #         context_feat: (N, L, D), output of the first transf6ormer encoder layer
    #         context_mask: (N, L)
    #     Returns:
    #         context_query_scores: (N, N)  score of each query w.r.t. each video inside the batch,
    #             diagonal positions are positive. used to get negative samples.
    #     """

    #     N, L, Dc = context_feat.shape
    #     if hash_layer.training:
    #         Bq = F.normalize(modularied_query.bin_like, dim=-1, eps=1e-6)  # (N, Dh)
    #     else:
    #         Bq = torch.sign(modularied_query.code)               # (N, Dh)

    #     context_feat = context_feat.reshape(N * L, Dc)          # (N·L, Dc)
    #     hv_c = hash_layer(context_feat, epoch)                 # (N·L, Dh)
    #     if hash_layer.training:
    #         Bc = F.normalize(hv_c.bin_like, dim=-1, eps=1e-6).view(N, L, -1)          # (N·L, Dh)
    #         scores = torch.einsum("md,nld->mln", Bq, Bc)
    #     else:
    #         Bc = torch.sign(hv_c.code).view(N, L, -1)           # (N·L, Dh)
    #         Dh = Bc.size(-1)
    #         scores = torch.einsum("md,nld->mln", Bq, Bc) / max(Dh, 1e-6)
            
    #     cm = context_mask.transpose(0, 1).unsqueeze(0)  # (1, L, N)
    #     scores = mask_logits(scores, cm).max(dim=1).values  # (N, N)
    #     scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)


    #     # 两个正则：对 query 与 context 各算一遍后相加
    #     Lq_q, Lb_q, Lr_q = hash_layer.regularizers(modularied_query)
    #     Lq_c, Lb_c, Lr_c = hash_layer.regularizers(hv_c)
    #     reg = {"L_q": Lq_q + Lq_c, "L_b": Lb_q + Lb_c, "L_r": Lr_q + Lr_c}
    #     return scores, reg

    @staticmethod
    def get_video_level_scores(modularied_query, context_feat, context_mask, hash_layer=None, epoch=None):
        """
        支持通用形状：
        modularied_query: HashedVector（来自 self.hash2(...)），形状 (Nq, Dh)
        context_feat    : (Nc, L, D)
        context_mask    : (Nc, L)
        返回：
        scores: (Nq, Nc)  —— 每个查询对每个视频的分数（沿时间维 L 已经做了 max）
        reg   : 与原实现相同的正则项 dict
        """
        Nc, L, Dc = context_feat.shape

        # 取 Nq（eval 用 hv.code，train 用 hv.bin_like）
        if hash_layer.training:
            Nq = modularied_query.bin_like.shape[0]
            Dh = modularied_query.bin_like.shape[1]
        else:
            Nq = modularied_query.code.shape[0]
            Dh = modularied_query.code.shape[1]

        # 展平上下文后过 hash 层
        context_feat_flat = context_feat.reshape(Nc * L, Dc)    # (Nc*L, D)
        hv_c = hash_layer(context_feat_flat, epoch)              # HashedVector
        reg = {"L_q": 0.0, "L_b": 0.0, "L_r": 0.0}
        # 正则项（保持你原来的做法不变）
        if hash_layer.training:
            Lq_q, Lb_q, Lr_q = hash_layer.regularizers(modularied_query)
            Lq_c, Lb_c, Lr_c = hash_layer.regularizers(hv_c)
            reg = {"L_q": Lq_q + Lq_c, "L_b": Lb_q + Lb_c, "L_r": Lr_q + Lr_c}

        if hash_layer.training:
            # ====== 浮点可导路径（训练）======
            Bq = F.normalize(modularied_query.bin_like, dim=-1, eps=1e-6)         # (Nq, Dh)
            Bc = F.normalize(hv_c.bin_like, dim=-1, eps=1e-6).view(Nc, L, -1)     # (Nc, L, Dh)
            # 先算 (Nq, Nc, L)，再转成 (Nq, L, Nc) 以便随后按 L 维做 max
            sims_qnl = torch.einsum("qd,nld->qnl", Bq, Bc)                        # (Nq, Nc, L)
            sims_mln = sims_qnl.permute(0, 2, 1).contiguous()                      # (Nq, L, Nc)
        else:
            # ====== 位运算（推理）：XNOR + popcount + 线性缩放回余弦 ======
            # 生成 0/1 位并打包
            Bq01 = (modularied_query.bin_like > 0).to(torch.uint8)                 # (Nq, Dh)
            Bc01 = (hv_c.bin_like > 0).to(torch.uint8)                             # (Nc*L, Dh)
            Bq_packed = hash_layer._pack_bits(Bq01)                                # (Nq,   nbytes)
            Bc_packed = hash_layer._pack_bits(Bc01)                                # (Nc*L, nbytes)

            matches = hash_layer.xnor_popcount(Bq_packed, Bc_packed)               # (Nq, Nc*L) int32

            sims = (2.0 * matches.to(torch.float32) - float(Dh)) / float(Dh)       # (Nq, Nc*L)
            sims_qnl = sims.view(Nq, Nc, L)                                        # (Nq, Nc, L)
            sims_mln = sims_qnl.permute(0, 2, 1).contiguous()                       # (Nq, L, Nc)

        # 掩码与归约（与原逻辑一致）
        cm = context_mask.transpose(0, 1).unsqueeze(0)                              # (1, L, Nc)
        sims_mln = mask_logits(sims_mln, cm)                                       # (Nq, L, Nc)
        scores = sims_mln.max(dim=1).values                                        # (Nq, Nc)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        return scores, reg
    

    @staticmethod
    def get_unnormalized_video_level_scores(modularied_query, context_feat, context_mask):
        """ Calculate video2query scores for each pair of video and query inside the batch.
        Args:
            modularied_query: (N, D)
            context_feat: (N, L, D), output of the first transformer encoder layer
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)  score of each query w.r.t. each video inside the batch,
                diagonal positions are positive. used to get negative samples.
        """
        query_context_scores = torch.einsum("md,nld->mln", modularied_query, context_feat)  # (N, L, N)
        context_mask = context_mask.transpose(0, 1).unsqueeze(0)  # (1, L, N)
        query_context_scores = mask_logits(query_context_scores, context_mask)  # (N, L, N)
        return query_context_scores

    def get_merged_score(self, video_query, video_feat, sub_query, sub_feat, cross=False):
        if sub_query is not None:
            video_query = self.video_query_linear(video_query)
            sub_query = self.sub_query_linear(sub_query)
            if cross:
                video_similarity = torch.einsum("md,nld->mnl", video_query, video_feat)
                sub_similarity = torch.einsum("md,nld->mnl", sub_query, sub_feat)
                similarity = (video_similarity + sub_similarity) / 2  # (Nq, Nv, L)  from query to all videos.
            else:
                video_similarity = torch.einsum("bd,bld->bl", video_query, video_feat)  # (N, L)
                sub_similarity = torch.einsum("bd,bld->bl", sub_query, sub_feat)  # (N, L)
                similarity = (video_similarity + sub_similarity) / 2
            return similarity
        else:
            video_query = self.video_query_linear(video_query)
            if cross:
                video_similarity = torch.einsum("md,nld->mnl", video_query, video_feat)
                similarity = video_similarity
            else:
                video_similarity = torch.einsum("bd,bld->bl", video_query, video_feat)  # (N, L)
                similarity = video_similarity
            return similarity

    # def get_merged_st_ed_prob(self, similarity, context_mask, cross=False):
    #     if cross:
    #         n_q, n_c, length = similarity.shape
    #         similarity = similarity.view(n_q * n_c, 1, length)
    #         st_prob = self.merged_st_predictor(similarity).view(n_q, n_c, length)  # (Nq, Nv, L)
    #         ed_prob = self.merged_ed_predictor(similarity).view(n_q, n_c, length)  # (Nq, Nv, L)
    #     else:
    #         st_prob = self.merged_st_predictor(similarity.unsqueeze(1)).squeeze()  # (N, L)
    #         ed_prob = self.merged_ed_predictor(similarity.unsqueeze(1)).squeeze()  # (N, L)
    #     st_prob = mask_logits(st_prob, context_mask)  # (N, L)
    #     ed_prob = mask_logits(ed_prob, context_mask)
    #     return st_prob, ed_prob

    def get_merged_st_ed_prob(self, similarity, context_mask, cross=False):
        if cross:
            n_q, n_c, length = similarity.shape           # (Nq, Nv, L)
            flat = similarity.reshape(n_q * n_c, length)  # (Nq*Nv, L)
            st_prob = self.merged_st_predictor(flat)      # (Nq*Nv, L)
            ed_prob = self.merged_ed_predictor(flat)      # (Nq*Nv, L)
            st_prob = st_prob.view(n_q, n_c, length)
            ed_prob = ed_prob.view(n_q, n_c, length)
            # context_mask 这里应为 (Nq,Nv,L)；若当前只有 (Nv,L) 或 (N,L)，需在上游对齐
            st_prob = mask_logits(st_prob, context_mask)
            ed_prob = mask_logits(ed_prob, context_mask)
            return st_prob, ed_prob
        else:
            # similarity: (N, L) → (N, L)
            st_prob = self.merged_st_predictor(similarity)
            ed_prob = self.merged_ed_predictor(similarity)
            st_prob = mask_logits(st_prob, context_mask)  # (N, L)
            ed_prob = mask_logits(ed_prob, context_mask)
            return st_prob, ed_prob

    def get_pred_from_raw_query(self, query_feat, query_mask,
                            video_feat, video_mask,
                            sub_feat, sub_mask,
                            cross=False, return_query_feats=False, timing_dict=None):
        """
        sub_feat/sub_mask 可以为 None。无字幕时仅用 video 分支。
        """
        do_time = (timing_dict is not None)

        if do_time:
            _sync_if_cuda(query_feat)
            t0_q_enc = perf_counter()

        # 1) 编码 query
        video_query, sub_query, encoded_query = self.encode_query(query_feat, query_mask)

        if do_time:
            _sync_if_cuda(video_query)
            # 累积到外部字典
            timing_dict["query_enc_s"] = timing_dict.get("query_enc_s", 0.0) + (perf_counter() - t0_q_enc)

        if do_time:
            t0_vr = perf_counter()

        # 2) 计算视频级检索得分 & 哈希正则（video 必有）
        video_query_h = self.hash2(video_query, self.eta)
        video_q2ctx_scores, video_reg = self.get_video_level_scores(
            video_query_h, video_feat, video_mask, self.hash1, self.eta
        )

        # 3) 决定是否启用字幕分支
        use_sub_now = (getattr(self, "use_sub", True)
                    and (sub_feat is not None)
                    and (sub_mask is not None))

        if use_sub_now:
            sub_query_h = self.hash2(sub_query, self.eta)
            sub_q2ctx_scores, sub_reg = self.get_video_level_scores(
                sub_query_h, sub_feat, sub_mask, self.hash1, self.eta
            )
            q2ctx_scores = (video_q2ctx_scores + sub_q2ctx_scores) / 2
            reg = {
                "L_q": video_reg["L_q"] + sub_reg["L_q"],
                "L_b": video_reg["L_b"] + sub_reg["L_b"],
                "L_r": video_reg["L_r"] + sub_reg["L_r"],
            }
        else:
            q2ctx_scores = video_q2ctx_scores
            reg = video_reg
            # 可选：把 sub_query 置零，保证后面相似度汇合稳定
            sub_query = torch.zeros_like(video_query)

        if do_time:
            _sync_if_cuda(q2ctx_scores)
            # 累积到外部字典
            timing_dict["vr_score_calc_s"] = timing_dict.get("vr_score_calc_s", 0.0) + (perf_counter() - t0_vr)

        if do_time:
            t0_mr = perf_counter()

        # # 4) 起止概率：无字幕时仅用 video 相似度
        similarity = self.get_merged_score(video_query, video_feat, sub_query if use_sub_now else None, sub_feat if use_sub_now else None, cross=cross)
        st_prob, ed_prob = self.get_merged_st_ed_prob(similarity, video_mask, cross=cross)

        if do_time:
            _sync_if_cuda(st_prob)
            # 累积到外部字典
            timing_dict["mr_prob_calc_s"] = timing_dict.get("mr_prob_calc_s", 0.0) + (perf_counter() - t0_mr)

        # pairwise = (video_feat.size(0) != encoded_query.size(0))
        # st_prob, ed_prob = self.moment(
        #     video_feat, video_mask,
        #     sub_feat,   sub_mask,
        #     encoded_query, query_mask,
        #     pairwise,
        # )

        if return_query_feats:
            return video_query, sub_query, encoded_query, q2ctx_scores, st_prob, ed_prob, reg
        else:
            return q2ctx_scores, st_prob, ed_prob


    def get_video_level_loss(self, query_context_scores):
        """ ranking loss between (pos. query + pos. video) and (pos. query + neg. video) or (neg. query + pos. video)
        Args:
            query_context_scores: (N, N), cosine similarity [-1, 1],
                Each row contains the scores between the query to each of the videos inside the batch.
        """
        bsz = len(query_context_scores)
        diagonal_indices = torch.arange(bsz).to(query_context_scores.device)
        pos_scores = query_context_scores[diagonal_indices, diagonal_indices]  # (N, )
        query_context_scores_masked = copy.deepcopy(query_context_scores.data)
        # impossibly large for cosine similarity, the copy is created as modifying the original will cause error
        query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
        pos_query_neg_context_scores = self.get_neg_scores(query_context_scores, query_context_scores_masked)
        neg_query_pos_context_scores = self.get_neg_scores(query_context_scores.transpose(0, 1), query_context_scores_masked.transpose(0, 1))
        loss_neg_ctx = self.get_ranking_loss(pos_scores, pos_query_neg_context_scores)
        loss_neg_q = self.get_ranking_loss(pos_scores, neg_query_pos_context_scores)
        return loss_neg_ctx, loss_neg_q

    def get_neg_scores(self, scores, scores_masked):
        """
        scores: (N, N), cosine similarity [-1, 1],
            Each row are scores: query --> all videos. Transposed version: video --> all queries.
        scores_masked: (N, N) the same as scores, except that the diagonal (positive) positions
            are masked with a large value.
        """
        bsz = len(scores)
        batch_indices = torch.arange(bsz).to(scores.device)
        _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)
        sample_min_idx = 1  # skip the masked positive
        sample_max_idx = min(sample_min_idx + self.config.hard_pool_size, bsz) if self.config.use_hard_negative else bsz
        # (N, )
        sampled_neg_score_indices = sorted_scores_indices[batch_indices, torch.randint(sample_min_idx, sample_max_idx, size=(bsz,)).to(scores.device)]
        sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]  # (N, )
        return sampled_neg_scores

    def get_ranking_loss(self, pos_score, neg_score):
        """ Note here we encourage positive scores to be larger than negative scores.
        Args:
            pos_score: (N, ), torch.float32
            neg_score: (N, ), torch.float32
        """
        if self.config.ranking_loss_type == "hinge":  # max(0, m + S_neg - S_pos)
            return torch.clamp(self.config.margin + neg_score - pos_score, min=0).sum() / len(pos_score)
        elif self.config.ranking_loss_type == "lse":  # log[1 + exp(S_neg - S_pos)]
            return torch.log1p(torch.exp(neg_score - pos_score)).sum() / len(pos_score)
        else:
            raise NotImplementedError("Only support 'hinge' and 'lse'")

    @torch.no_grad()
    def export_query_hash(self, query_feat, query_mask):
        """
        [在线] 步骤: 仅编码查询，哈希，并返回打包的哈希码和用于MR的密集特征。
        [V2 - 修复版] 精确复现 get_video_level_scores 中的查询二值化逻辑 ( > 0 )
        """
        self.eval()
        # 1. 编码查询
        video_query, sub_query, encoded_query = self.encode_query(query_feat, query_mask) # (Nq, D)
        
        # 2. 哈希查询 (hash2)
        # 2.1 视频查询 (Video Query)
        # 调用 hash2.forward() 获取 HashedVector
        hv_q_vid = self.hash2(video_query, self.eta, export_packed=False) # (Nq, D)
        # 精确复现: (bin_like > 0)
        # hv_q_vid.bin_like 在 eval 模式下是 torch.sign(code)
        bits01_vid = (hv_q_vid.bin_like > 0).to(torch.uint8) 
        packed_q_vid = self.hash2._pack_bits(bits01_vid)

        # 2.2 字幕查询 (Sub Query)
        packed_q_sub = None
        if self.config.ctx_mode == "video_sub":
             hv_q_sub = self.hash2(sub_query, self.eta, export_packed=False)
             # 精确复现: (bin_like > 0)
             bits01_sub = (hv_q_sub.bin_like > 0).to(torch.uint8)
             packed_q_sub = self.hash2._pack_bits(bits01_sub)

        # 返回VR用的打包哈希码，和MR用的密集查询特征
        return packed_q_vid, packed_q_sub, video_query, sub_query

    @torch.no_grad()
    def export_context_index(self, video_feat, video_mask, sub_feat, sub_mask):
        """
        [离线] 步骤: 编码上下文，哈希所有帧，并返回索引。
        [V2 - 修复版] 精确复现 get_video_level_scores 中的上下文二值化逻辑 ( >= 0 )
        """
        self.eval()
        # 1. 获取上下文的密集特征
        # 确保你已经应用了上一轮的修复 (return_mid_output=True)
        _, _, _, _, x_video_feat, x_sub_feat = self.encode_context(
            video_feat, video_mask, sub_feat, sub_mask, return_mid_output=True
        ) # (Nc, L, D)

        Nc, L, D = x_video_feat.shape
        
        # 2. 为VR创建哈希索引 (hash1)
        # 2.1 视频哈希 (Video Hash)
        video_frames_flat = x_video_feat.reshape(Nc * L, D)
        # 调用 hash1.forward() 获取 HashedVector
        hv_c_vid = self.hash1(video_frames_flat, self.eta, export_packed=False)
        # 精确复现: (bin_like >= 0)
        # hv_c_vid.bin_like 在 eval 模式下是 torch.sign(code)
        bits01_vid = (hv_c_vid.bin_like >= 0).to(torch.uint8)
        packed_bits_vid = self.hash1._pack_bits(bits01_vid)

        # 2.2 字幕哈希 (Sub Hash)
        packed_bits_sub = None
        use_sub = (sub_feat is not None and x_sub_feat is not None)
        if use_sub:
            sub_frames_flat = x_sub_feat.reshape(Nc * L, D)
            hv_c_sub = self.hash1(sub_frames_flat, self.eta, export_packed=False)
            # 精确复现: (bin_like >= 0)
            bits01_sub = (hv_c_sub.bin_like >= 0).to(torch.uint8)
            packed_bits_sub = self.hash1._pack_bits(bits01_sub)
        
        # 3. 返回所有需要的数据
        indexed_data = {
            "vr_video_hash": packed_bits_vid, # (Nc*L, nbytes)
            "vr_sub_hash": packed_bits_sub,   # (Nc*L, nbytes) or None
            "mr_video_feat": x_video_feat,    # (Nc, L, D) - MR的密集特征
            "mr_sub_feat": x_sub_feat,      # (Nc, L, D) or None
            "video_mask": video_mask,       # (Nc, L)
        }
        return indexed_data

    @torch.no_grad()
    def get_pred_from_indexed_query(self, 
                                    packed_q_vid, packed_q_sub,       # 来自 export_query_hash
                                    video_query, sub_query,           # (Nq, D) 密集查询
                                    indexed_ctx,                      # 来自 export_context_index
                                    opt, timing_dict=None):
        """
        [在线] 步骤: 使用预先计算的上下文索引，快速计算VR和MR分数。
        """
        self.eval()
        do_time = (timing_dict is not None)
        
        # 1. 解包预加载的上下文索引 (已在GPU上)
        packed_ctx_vid = indexed_ctx["vr_video_hash"]   # (Nc*L, nbytes)
        packed_ctx_sub = indexed_ctx["vr_sub_hash"]   # (Nc*L, nbytes)
        ctx_video_feat = indexed_ctx["mr_video_feat"]   # (Nc, L, D)
        ctx_sub_feat = indexed_ctx["mr_sub_feat"]     # (Nc, L, D)
        ctx_video_mask = indexed_ctx["video_mask"]    # (Nc, L)

        Nq = packed_q_vid.shape[0]
        Nc, L, D = ctx_video_feat.shape
        Dh = self.hash1.encoder[-1].out_features # 哈希位数

        use_sub = (packed_q_sub is not None and packed_ctx_sub is not None and ctx_sub_feat is not None)

        # === 2. VR 任务 (哈希) ===
        if do_time: t0_vr = perf_counter()

        # 视频VR (极快)
        matches_vid = self.hash1.xnor_popcount(packed_q_vid, packed_ctx_vid) # (Nq, Nc*L)
        # 转换回 [-1, 1] 相似度
        sims_vid_flat = (2.0 * matches_vid.float() - float(Dh)) / float(Dh)   # (Nq, Nc*L)
        sims_vid_qnl = sims_vid_flat.view(Nq, Nc, L)
        sims_vid_mln = sims_vid_qnl.permute(0, 2, 1).contiguous()         # (Nq, L, Nc)

        if use_sub:
            # 字幕VR (极快)
            matches_sub = self.hash1.xnor_popcount(packed_q_sub, packed_ctx_sub) # (Nq, Nc*L)
            sims_sub_flat = (2.0 * matches_sub.float() - float(Dh)) / float(Dh)   # (Nq, Nc*L)
            sims_sub_qnl = sims_sub_flat.view(Nq, Nc, L)
            sims_sub_mln = sims_sub_qnl.permute(0, 2, 1).contiguous()     # (Nq, L, Nc)
            
            sims_mln = (sims_vid_mln + sims_sub_mln) / 2.0
        else:
            sims_mln = sims_vid_mln
            
        # 掩码 & Max-Pooling (极快)
        cm = ctx_video_mask.transpose(0, 1).unsqueeze(0)  # (1, L, Nc)
        sims_mln = mask_logits(sims_mln, cm)              # (Nq, L, Nc)
        q2ctx_scores = sims_mln.max(dim=1).values         # (Nq, Nc)
        q2ctx_scores = torch.nan_to_num(q2ctx_scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        q2ctx_scores = torch.exp(opt.q2c_alpha * q2ctx_scores) # (同原始脚本)
        
        if do_time: 
            _sync_if_cuda(opt)
            # 计时: VR分数计算 (xnor + pool)
            timing_dict["vr_score_calc_s"] = timing_dict.get("vr_score_calc_s", 0.0) + (perf_counter() - t0_vr)

        # === 3. MR 任务 (密集) ===
        # 注意: MR 仍然是密集的，需要 Nq vs Nc 的密集计算
        if do_time: t0_mr = perf_counter()

        # 密集相似度 (Nq, Nc, L) - 这是MR的主要开销
        similarity = self.get_merged_score(
            video_query, ctx_video_feat, 
            sub_query if use_sub else None, 
            ctx_sub_feat if use_sub else None, 
            cross=True
        ) # (Nq, Nc, L)

        # 扩展掩码以匹配 (Nq, Nc, L)
        ctx_mask_expanded = ctx_video_mask.unsqueeze(0).expand(Nq, Nc, L) # (Nq, Nc, L)
        st_prob, ed_prob = self.get_merged_st_ed_prob(similarity, ctx_mask_expanded, cross=True)
        
        # Softmax (同原始脚本)
        st_prob = F.softmax(st_prob, dim=-1) # (Nq, Nc, L)
        ed_prob = F.softmax(ed_prob, dim=-1)

        if do_time: 
            _sync_if_cuda(opt)
            # 计时: MR概率计算 (dense matmul + conv)
            timing_dict["mr_prob_calc_s"] = timing_dict.get("mr_prob_calc_s", 0.0) + (perf_counter() - t0_mr)

        return q2ctx_scores, st_prob, ed_prob


def mask_logits(target: torch.Tensor, mask: torch.Tensor, neg_fill: float = -1e4):
    """
    安全掩码：不会把 NaN 通过 0*NaN 传播下去。
    mask 允许广播；非零视为有效位。
    """
    # 统一 dtype 并转为布尔条件
    cond = (mask > 0).to(dtype=torch.bool)
    filled = torch.where(cond, target, target.new_full(target.shape, neg_fill))
    # 再次保险：清理 NaN / ±Inf
    filled = torch.nan_to_num(filled, nan=neg_fill, posinf=1e4, neginf=neg_fill)
    return filled


def _flatten_bits(B, mask=None):
    """
    B: (..., l) 连续哈希输出（tanh 之后），最后一维是哈希位
    mask: 与 B 前面所有维度对齐的布尔 mask（True 表示有效），可为 None
    return: (M, l)
    """
    if mask is None:
        return B.reshape(-1, B.shape[-1])
    # 将 mask broadcast 到 B 的最后一维之前
    m = mask.reshape(-1)
    return B.reshape(-1, B.shape[-1])[m]

def gather_bits_for_losses(bits_and_masks):
    """
    bits_and_masks: 列表，元素形如 (B, mask) 或 (B, None)
      例：[(video_bits, None), (sub_bits, None), (ctx_bits, ctx_mask), (subctx_bits, subctx_mask)]
    """
    pieces = []
    for B, m in bits_and_masks:
        pieces.append(_flatten_bits(B, m))
    return torch.cat(pieces, dim=0)  # (M, l)

def quantization_loss(B, smooth=True):
    """
    论文式 (14):  L_q = sum |||B|-1||_1
    """
    if smooth:
        return torch.log(torch.cosh(B.abs() - 1)).mean()
    else:
        return (B.abs() - 1).abs().mean()

def bit_balance_loss(B):
    """
    论文式 (15):  L_b = (1/l) * sum_j ( (1/N) * sum_i b_ij )^2
    """
    return (B.mean(dim=0) ** 2).mean()

