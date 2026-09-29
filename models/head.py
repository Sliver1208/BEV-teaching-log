# ============================================================
# BEV 检测头（简化版）
# ------------------------------------------------------------
# 本工程是学习版，检测头不做复杂的 DETR-style decoder，
# 而是直接在 BEV 网格上做"中心热图 + 偏移 + 分类"预测，
# 类似 CenterPoint 的简化 2D-BEV 检测，输出足够直观，
# 便于验证 BEV 特征是否学到了有用的信息。
#
# 预测:
#   cls_logits:  [B, H, W, num_classes] 每个 BEV 网格是否存在物体
#   center_xy:   [B, H, W, 2]            物体中心相对网格的偏移（归一化 [-1,1]）
#   size_whl:    [B, H, W, 3]            物体大小（宽/长/高，米）
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class BEVDetectionHead(nn.Module):

    def __init__(
        self,
        embed_dims: int = 256,
        bev_h: int = 50,
        bev_w: int = 50,
        num_classes: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.num_classes = num_classes

        # 3 层共享 MLP 做特征提取
        shared = []
        in_dim = embed_dims
        for _ in range(2):
            shared.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = hidden_dim
        self.shared_mlp = nn.Sequential(*shared)

        # 各分支输出头
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.center_head = nn.Linear(hidden_dim, 2)   # offset within grid
        self.size_head = nn.Linear(hidden_dim, 3)     # w, l, h in meters

    def reshape_bev(self, bev_flat: torch.Tensor) -> torch.Tensor:
        """把 [B, H*W*Z, C] 的 flatten query 变形成 [B, H, W, C]，简化版 Z=1 即可"""
        B = bev_flat.shape[0]
        return bev_flat.view(B, self.bev_h, self.bev_w, self.embed_dims)

    def forward(self, bev_queries: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            bev_queries: [B, H*W*Z, C]
        Returns:
            dict {
                "cls_logits": [B, H, W, num_classes]  (sigmoid 之前)
                "center_xy":  [B, H, W, 2]
                "size_whl":   [B, H, W, 3]
            }
        """
        bev_spatial = self.reshape_bev(bev_queries)  # [B, H, W, C]
        feat = self.shared_mlp(bev_spatial)          # [B, H, W, hidden_dim]

        cls_logits = self.cls_head(feat)             # [B, H, W, num_cls]
        center_xy = self.center_head(feat)           # [B, H, W, 2]
        size_whl = F.relu(self.size_head(feat)) + 0.1  # 尺寸为正，加一个偏置防止 0

        return {
            "cls_logits": cls_logits,
            "center_xy": center_xy,
            "size_whl": size_whl,
        }
