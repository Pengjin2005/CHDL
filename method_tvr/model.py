import chunk
import copy
from dataclasses import dataclass
from locale import normalize
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from method_tvr.model_components import BertAttention, LinearLayer, BertSelfAttention, TrainablePositionalEncoding, AtomicEventMomentLocalizationModule, Conv2DMomentLocalization, SpotlightMomentLocalization
from method_tvr.model_components import MILNCELoss, HashLayer
from method_tvr.contrastive import batch_video_query_loss, batch_local_token_frame_loss


class ReLoCLNet(nn.Module):
    def __init__(self, config):
        super(ReLoCLNet, self).__init__()
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

        conv_cfg = dict(in_channels=1, out_channels=1, kernel_size=config.conv_kernel_size, stride=config.conv_stride, padding=config.conv_kernel_size // 2, bias=False)
        self.merged_st_predictor = nn.Conv1d(**conv_cfg)
        self.merged_ed_predictor = nn.Conv1d(**conv_cfg)

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
        # Debug: Check input for NaN/Inf
        if torch.isnan(query_feat).any() or torch.isinf(query_feat).any():
            print("NaN/Inf detected in query_feat input!")
            return torch.tensor(0.0, requires_grad=True), {}
        if torch.isnan(video_feat).any() or torch.isinf(video_feat).any():
            print("NaN/Inf detected in video_feat input!")
            return torch.tensor(0.0, requires_grad=True), {}
        if sub_feat is not None and (torch.isnan(sub_feat).any() or torch.isinf(sub_feat).any()):
            print("NaN/Inf detected in sub_feat input!")
            return torch.tensor(0.0, requires_grad=True), {}
            
        video_feat, sub_feat, mid_x_video_feat, mid_x_sub_feat, x_video_feat, x_sub_feat = self.encode_context(
            video_feat, video_mask, sub_feat, sub_mask, return_mid_output=True)
            
        # Debug: Check encoded features for NaN/Inf
        if torch.isnan(x_video_feat).any() or torch.isinf(x_video_feat).any():
            print("NaN/Inf detected in x_video_feat after encoding!")
            return torch.tensor(0.0, requires_grad=True), {}
        if x_sub_feat is not None and (torch.isnan(x_sub_feat).any() or torch.isinf(x_sub_feat).any()):
            print("NaN/Inf detected in x_sub_feat after encoding!")
            return torch.tensor(0.0, requires_grad=True), {}
            
        # x_video_feat_hashed = self.hash1(x_video_feat, self.eta)
        # x_sub_feat_hashed = self.hash1(x_sub_feat, self.eta)
        video_query, sub_query, encoded_query, query_context_scores, st_prob, ed_prob, reg_loss = self.get_pred_from_raw_query(
            query_feat, query_mask, x_video_feat, video_mask, x_sub_feat, sub_mask, cross=False,
            return_query_feats=True)

        # Debug: Check predictions for NaN/Inf
        if torch.isnan(query_context_scores).any() or torch.isinf(query_context_scores).any():
            print("NaN/Inf detected in query_context_scores!")
            return torch.tensor(0.0, requires_grad=True), {}
        if torch.isnan(st_prob).any() or torch.isinf(st_prob).any():
            print("NaN/Inf detected in st_prob!")
            return torch.tensor(0.0, requires_grad=True), {}
        if torch.isnan(ed_prob).any() or torch.isinf(ed_prob).any():
            print("NaN/Inf detected in ed_prob!")
            return torch.tensor(0.0, requires_grad=True), {}

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
            
        # Debug: Check FCL loss
        if torch.isnan(loss_fcl) or torch.isinf(loss_fcl):
            print(f"NaN/Inf detected in loss_fcl: {loss_fcl}")
            loss_fcl = 0.0
            
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
            
        # Debug: Check VCL loss
        if torch.isnan(loss_vcl) or torch.isinf(loss_vcl):
            print(f"NaN/Inf detected in loss_vcl: {loss_vcl}")
            loss_vcl = 0.0
            
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
            
        # Debug: Check ST-ED loss
        if torch.isnan(loss_st_ed) or torch.isinf(loss_st_ed):
            print(f"NaN/Inf detected in loss_st_ed: {loss_st_ed}")
            loss_st_ed = 0.0
            
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
        lq_map = ReLoCLNet._log_cosh(x) if smooth_abs else x.abs()
        L_q = lq_map.mean() if reduction == "mean" else lq_map.sum()

        # L_b ：让各位在 batch 内均值为 0
        bit_means = B.mean(dim=0)         # (l,)
        L_b = (bit_means ** 2).mean()     # 标量
        return L_q, L_b

    @staticmethod
    def get_video_level_scores(modularied_query, context_feat, context_mask, hash_layer=None, epoch=None):
        """ Calculate video2query scores for each pair of video and query inside the batch.
        Args:
            modularied_query: (N, D)
            context_feat: (N, L, D), output of the first transf6ormer encoder layer
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)  score of each query w.r.t. each video inside the batch,
                diagonal positions are positive. used to get negative samples.
        """

        N, L, Dc = context_feat.shape
        if hash_layer.training:
            Bq = F.normalize(modularied_query.bin_like, dim=-1, eps=1e-6)  # (N, Dh)
        else:
            Bq = torch.sign(modularied_query.code)               # (N, Dh)

        context_feat = context_feat.reshape(N * L, Dc)          # (N·L, Dc)
        hv_c = hash_layer(context_feat, epoch)                 # (N·L, Dh)
        if hash_layer.training:
            Bc = F.normalize(hv_c.bin_like, dim=-1, eps=1e-6).view(N, L, -1)          # (N·L, Dh)
            scores = torch.einsum("md,nld->mln", Bq, Bc)
        else:
            Bc = torch.sign(hv_c.code).view(N, L, -1)           # (N·L, Dh)
            Dh = Bc.size(-1)
            scores = torch.einsum("md,nld->mln", Bq, Bc) / max(Dh, 1e-6)
            
        cm = context_mask.transpose(0, 1).unsqueeze(0)  # (1, L, N)
        scores = mask_logits(scores, cm).max(dim=1).values  # (N, N)
        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)


        # 两个正则：对 query 与 context 各算一遍后相加
        Lq_q, Lb_q, Lr_q = hash_layer.regularizers(modularied_query)
        # Lq_c, Lb_c, Lr_c = hash_layer.regularizers(hv_c)
        reg = {"L_q": Lq_q + Lq_q, "L_b": Lb_q + Lb_q, "L_r": Lr_q + Lr_q}
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

    def get_merged_st_ed_prob(self, similarity, context_mask, cross=False):
        if cross:
            n_q, n_c, length = similarity.shape
            similarity = similarity.view(n_q * n_c, 1, length)
            st_prob = self.merged_st_predictor(similarity).view(n_q, n_c, length)  # (Nq, Nv, L)
            ed_prob = self.merged_ed_predictor(similarity).view(n_q, n_c, length)  # (Nq, Nv, L)
        else:
            st_prob = self.merged_st_predictor(similarity.unsqueeze(1)).squeeze()  # (N, L)
            ed_prob = self.merged_ed_predictor(similarity.unsqueeze(1)).squeeze()  # (N, L)
        st_prob = mask_logits(st_prob, context_mask)  # (N, L)
        ed_prob = mask_logits(ed_prob, context_mask)
        return st_prob, ed_prob

    def get_pred_from_raw_query(self, query_feat, query_mask,
                            video_feat, video_mask,
                            sub_feat, sub_mask,
                            cross=False, return_query_feats=False):
        """
        sub_feat/sub_mask 可以为 None。无字幕时仅用 video 分支。
        """
        # 1) 编码 query
        video_query, sub_query, encoded_query = self.encode_query(query_feat, query_mask)

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

        # # 4) 起止概率：无字幕时仅用 video 相似度
        similarity = self.get_merged_score(video_query, video_feat, sub_query if use_sub_now else None, sub_feat if use_sub_now else None, cross=cross)
        st_prob, ed_prob = self.get_merged_st_ed_prob(similarity, video_mask, cross=cross)

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

