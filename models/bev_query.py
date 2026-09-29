# ============================================================
# 核心模块 1/3: BEV Query 生成
# ------------------------------------------------------------
# 【原理简述】
# BEV Query 是 BEVFormer 的灵魂。每个 BEV Query 对应 BEV 平面
# 中的一个"锚点"（比如 50x50 网格中的某一格），在后续空间交叉
# 注意力中，每个 Query 会从图像上"看"对应于这个锚点周围的
# 像素区域，汇总得到该 BEV 位置的特征。
#
# 【与 DETR Query 的区别】
# - DETR Query 是稀疏的、对应物体数量（如 100、300 个）
# - BEV Query 是密集的、对应 BEV 网格（如 50x50=2500 个）
# - BEV Query 天然带有"空间位置"语义
#
# 【两种实现方式】
# (1) learnable: 完全可学习参数，训练后学到空间先验
# (2) geometry:  由 BEV 网格物理坐标经位置编码得到，更具可解释性
# ============================================================
import torch
import torch.nn as nn
from typing import Tuple
from .utils import position_embedding


class BEVQueryGenerator(nn.Module):
    """
    BEV Query 生成器

    输入: BEV 网格形状配置 (H, W, Z, embed_dims)
    输出: (bev_queries, bev_positional_encoding)
          bev_queries:        [1, H*W*Z, C] 可学习或几何初始化的特征
          bev_positional_enc: [1, H*W*Z, C] 正弦余弦位置编码，用于注意力中的偏置
    """

    def __init__(
        self,
        bev_h: int = 50,
        bev_w: int = 50,
        bev_z: int = 1,
        embed_dims: int = 256,
        query_type: str = "learnable",   # "learnable" 或 "geometry"
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.num_queries = bev_h * bev_w * bev_z
        self.embed_dims = embed_dims
        self.query_type = query_type

        # ------------------------------------------------------------
        # 方式 1：可学习 BEV Query（mmdet3d 原版默认方式）
        #   - 使用 nn.Parameter，shape=[1, H*W, C]，batch 维通过 expand 复用
        #   - 初始化通常用 Xavier
        # ------------------------------------------------------------
        if query_type == "learnable":
            self.bev_embedding = nn.Parameter(
                torch.Tensor(1, self.num_queries, embed_dims)
            )
            nn.init.xavier_uniform_(self.bev_embedding)

        # ------------------------------------------------------------
        # 方式 2：几何 Query（可选，更直观）
        #   - 将每个 BEV 网格的物理坐标 (x, y, z) 映射到特征空间
        #   - 训练中不直接更新 query 本身（只更新后续层）
        # ------------------------------------------------------------
        elif query_type == "geometry":
            # 把 (x, y, z) 3D 坐标映射到 embed_dims 维度
            num_feats = embed_dims // 6  # 3 个坐标 * 2(sin/cos) = 6 倍数
            self.geom_proj = nn.Sequential(
                nn.Linear(num_feats * 6, embed_dims),
                nn.GELU(),
                nn.Linear(embed_dims, embed_dims),
            )
        else:
            raise ValueError(f"不支持的 query_type: {query_type}")

    def _get_bev_grid_coords(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        生成归一化的 BEV 网格坐标（用于位置编码 / 几何 Query）
        Returns: [B, H*W*Z, 3]，值域 [0, 1]
        """
        # 先产生 [0, 1] 范围的坐标网格
        zs = torch.linspace(0, 1, self.bev_z, device=device)
        ys = torch.linspace(0, 1, self.bev_w, device=device)
        xs = torch.linspace(0, 1, self.bev_h, device=device)
        # 顺序约定：flatten 后 n = x_idx * (W*Z) + y_idx * Z + z_idx，与 SCA/检测头一致
        grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
        # stack -> [H, W, Z, 3]  -> reshape -> [H*W*Z, 3]
        coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        coords = coords.reshape(-1, 3).unsqueeze(0).expand(batch_size, -1, -1)
        return coords

    def forward(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            bev_queries: [B, H*W*Z, C]  作为注意力中的 Q
            bev_pos_enc: [B, H*W*Z, C]  加到 Q 和 K 上提供位置先验
        """
        # 1. 获取 BEV 网格归一化坐标
        grid_coords = self._get_bev_grid_coords(batch_size, device)  # [B, N, 3]

        # 2. 生成位置编码（不管是哪种 query 类型都用得到）
        #    3 个坐标各 num_feats_per_axis 的 sin/cos，总维度 = 6 * num_feats_per_axis
        #    如果 embed_dims 不能被 6 整除，则用 0 padding 补齐到 embed_dims
        num_feats_per_axis = self.embed_dims // 6
        bev_pos_enc = position_embedding(grid_coords, num_feats_per_axis)  # [B, N, 6*num_feats_per_axis]
        # 补齐到 embed_dims（例如 256: 252 -> 256，补 4 个 0）
        if bev_pos_enc.shape[-1] < self.embed_dims:
            pad = torch.zeros(
                *bev_pos_enc.shape[:-1], self.embed_dims - bev_pos_enc.shape[-1],
                device=bev_pos_enc.device, dtype=bev_pos_enc.dtype,
            )
            bev_pos_enc = torch.cat([bev_pos_enc, pad], dim=-1)

        # 3. 生成 BEV Query
        if self.query_type == "learnable":
            # expand 复制 batch 维（参数共享，不额外占显存）
            bev_queries = self.bev_embedding.expand(batch_size, -1, -1).to(device)
        else:  # geometry
            # 先用正弦编码把坐标编码，再用 MLP 映射
            init_feat = position_embedding(grid_coords, num_feats_per_axis)
            # 同样补齐到 embed_dims
            if init_feat.shape[-1] < self.embed_dims:
                pad = torch.zeros(
                    *init_feat.shape[:-1], self.embed_dims - init_feat.shape[-1],
                    device=init_feat.device, dtype=init_feat.dtype,
                )
                init_feat = torch.cat([init_feat, pad], dim=-1)
            bev_queries = self.geom_proj(init_feat)

        return bev_queries, bev_pos_enc

    def grid_index_to_xy(self, index: int) -> Tuple[int, int, int]:
        """
        辅助函数：把 flatten 后的 query index 还原成 BEV 网格坐标 (h_idx, w_idx, z_idx)
        用于 debug 或可视化定位某一个 query 对应的物理位置
        """
        z = index // (self.bev_h * self.bev_w)
        rem = index % (self.bev_h * self.bev_w)
        h = rem // self.bev_w
        w = rem % self.bev_w
        return h, w, z
