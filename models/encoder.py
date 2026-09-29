# ============================================================
# BEV Encoder Layer & Stack
# ------------------------------------------------------------
# 每个 Encoder Layer 的结构（参考 mmdet3d 原版）：
#   ┌────────────────────────────────────────────┐
#   │  BEV Query t-1（带历史帧的记忆）            │
#   │        │                                    │
#   │        ▼                                    │
#   │  [TSA] Temporal Self-Attention + Residual   │  ← 和历史 BEV 融合
#   │        │                                    │
#   │        ▼                                    │
#   │  [SCA] Spatial Cross-Attention + Residual   │  ← 从多视角图像拿信息
#   │        │                                    │
#   │        ▼                                    │
#   │  [FFN] Feed-Forward Network + Residual      │  ← 特征变换
#   │        │                                    │
#   │        ▼                                    │
#   │  BEV Query t（输出给下一层或检测头）         │
#   └────────────────────────────────────────────┘
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from .spatial_attn import SpatialCrossAttention
from .temporal_attn import TemporalSelfAttention
from .utils import FFN


class BEVEncoderLayer(nn.Module):
    """
    单层 BEV Encoder，顺序执行 TSA -> SCA -> FFN
    """

    def __init__(
        self,
        embed_dims: int = 256,
        # Spatial Cross Attention
        sca_num_heads: int = 4,
        sca_num_points: int = 4,
        # Temporal Self Attention
        tsa_num_heads: int = 4,
        num_history_frames: int = 3,
        # FFN
        ffn_dims: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dims = embed_dims

        # ---- TSA ----
        self.tsa = TemporalSelfAttention(
            embed_dims=embed_dims,
            num_heads=tsa_num_heads,
            num_history_frames=num_history_frames,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(embed_dims)

        # ---- SCA ----
        self.sca = SpatialCrossAttention(
            embed_dims=embed_dims,
            num_heads=sca_num_heads,
            num_points=sca_num_points,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dims)

        # ---- FFN ----
        self.ffn = FFN(embed_dims, ffn_dims, dropout)
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(
        self,
        bev_queries: torch.Tensor,
        bev_pos_enc: torch.Tensor,
        image_feats: List[torch.Tensor],
        lidar2cam: torch.Tensor,
        cam_intrinsic: torch.Tensor,
        history_bevs: List[torch.Tensor],
        bev_h: int,
        bev_w: int,
        bev_z: int = 1,
        x_bound: List[float] = [-50, 50],
        y_bound: List[float] = [-50, 50],
        z_bound: List[float] = [-10, 10],
        img_h: int = 256,
        img_w: int = 256,
    ) -> torch.Tensor:
        # 1. TSA: 先融合时序信息（Q = 当前帧, K/V = 历史；无历史帧时内部恒等返回）
        bev_queries = self.tsa(bev_queries, bev_pos_enc, history_bevs)

        # 2. SCA: Pre-LN 结构 —— LN 只作用于子层输入，残差裸路径保持原始尺度
        #    （SCA 注入的物体信号是加性小扰动，若对残差和做 Post-LN 会把它逐 token
        #     非线性归一化掉，导致检测头无法线性分离物体格与背景格）
        sca_out = self.sca(
            self.norm2(bev_queries), bev_pos_enc,
            image_feats, lidar2cam, cam_intrinsic,
            bev_h, bev_w, bev_z, x_bound, y_bound, z_bound,
            img_h, img_w,
        )
        bev_queries = bev_queries + sca_out  # 残差相加后不做 LN

        # 3. FFN: 内部已是 Pre-LN（identity + MLP(LN(x))），直接用
        bev_queries = self.ffn(bev_queries)

        return bev_queries


class BEVEncoder(nn.Module):
    """
    BEV Encoder：多层 BEVEncoderLayer 的堆叠
    """

    def __init__(self, num_layers: int = 3, layer_cfg: Optional[dict] = None):
        super().__init__()
        layer_cfg = layer_cfg or {}
        self.layers = nn.ModuleList([
            BEVEncoderLayer(**layer_cfg) for _ in range(num_layers)
        ])

    def forward(
        self,
        bev_queries: torch.Tensor,
        bev_pos_enc: torch.Tensor,
        image_feats: List[torch.Tensor],
        lidar2cam: torch.Tensor,
        cam_intrinsic: torch.Tensor,
        history_bevs: List[torch.Tensor],
        bev_h: int,
        bev_w: int,
        bev_z: int = 1,
        x_bound: List[float] = [-50, 50],
        y_bound: List[float] = [-50, 50],
        z_bound: List[float] = [-10, 10],
        img_h: int = 256,
        img_w: int = 256,
        return_intermediate: bool = False,
    ):
        """
        Args:
            return_intermediate: 是否返回每一层的输出（用于辅助 loss 或可视化）
        Returns:
            outputs: 最后一层输出 [B, N, C]，或每层输出的 List
        """
        intermediate = []
        for layer in self.layers:
            bev_queries = layer(
                bev_queries, bev_pos_enc,
                image_feats, lidar2cam, cam_intrinsic,
                history_bevs,
                bev_h, bev_w, bev_z,
                x_bound, y_bound, z_bound,
                img_h, img_w,
            )
            if return_intermediate:
                intermediate.append(bev_queries)
        return intermediate if return_intermediate else bev_queries
