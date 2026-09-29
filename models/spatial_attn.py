# ============================================================
# 核心模块 2/3: 空间交叉注意力 (Spatial Cross-Attention, SCA)
# ------------------------------------------------------------
# 【这就是 BEVFormer 成名的技术点】
# 传统方法做视角转换（Image -> BEV）通常是：
#   - IPMPool: 基于深度估计，把每个像素"反投影"到 3D 再投到 BEV
#   - Lift-Splat: 显式预测深度分布，代价昂贵
#
# BEVFormer 的 SCA 思路完全反过来：
#   1. 每个 BEV Query 对应一个 3D reference point (x,y,z)
#   2. 用相机内外参，把这个 3D 点投影到 6 个相机图像上，得到 2D 坐标
#   3. Query 在每幅图像上，只"看"这个 2D 坐标周围 K 个采样点（默认 4 个）
#   4. 把采样到的图像特征做注意力加权，汇聚成 BEV Query 的更新值
#
# 核心好处：
#   - 不需要显式预测深度（深度信息通过相机投影和注意力隐式学习）
#   - 计算量小：每个 query 只看 6 相机 × 4 点 = 24 个图像特征
#   - 端到端可微
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


def get_reference_points_in_bev(
    bev_h: int,
    bev_w: int,
    bev_z: int = 1,
    device: torch.device = None,
) -> torch.Tensor:
    """
    生成 BEV 空间中每个 Query 对应的 reference point（归一化坐标 [0,1]）
    形状: [1, H*W*Z, 3]，(x,y,z) 顺序
    """
    zs = torch.linspace(0.5 / bev_z, 1 - 0.5 / bev_z, bev_z, device=device) if bev_z > 1 else torch.tensor([0.5], device=device)
    ys = torch.linspace(0.5 / bev_w, 1 - 0.5 / bev_w, bev_w, device=device)
    xs = torch.linspace(0.5 / bev_h, 1 - 0.5 / bev_h, bev_h, device=device)
    # 顺序约定：flatten 后 n = x_idx * W + y_idx，必须与检测头 view(B, H, W, C) 一致
    # （此前 meshgrid(zs, ys, xs) 产生 y-major 顺序，导致 SCA 采样位置与检测头格子转置错位）
    grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
    coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(1, -1, 3)
    return coords


def denormalize_bev_to_world(
    ref_points_norm: torch.Tensor,
    x_bound: List[float],
    y_bound: List[float],
    z_bound: List[float],
) -> torch.Tensor:
    """
    把 BEV 归一化坐标 [0,1] 反解到物理世界坐标（米，自车坐标系）
    ref_points_norm[...,0] 对应 x（前），映射到 [x_min, x_max]
    """
    x_min, x_max = x_bound
    y_min, y_max = y_bound
    z_min, z_max = z_bound
    world = torch.zeros_like(ref_points_norm)
    world[..., 0] = ref_points_norm[..., 0] * (x_max - x_min) + x_min
    world[..., 1] = ref_points_norm[..., 1] * (y_max - y_min) + y_min
    world[..., 2] = ref_points_norm[..., 2] * (z_max - z_min) + z_min
    return world


def project_world_to_image(
    points_world: torch.Tensor,
    lidar2cam: torch.Tensor,
    cam_intrinsic: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将自车(lidar)坐标系下的 3D 点投影到像素坐标系
    Args:
        points_world:   [B, N, 4]          齐次坐标 (x,y,z,1)
        lidar2cam:      [B, num_cam, 4, 4] 外参（自车 -> 每个相机）
        cam_intrinsic:  [B, num_cam, 3, 3] 内参 K
    Returns:
        pix_coord:  [B, num_cam, N, 2]  像素坐标 (u, v)（原始数值，未 clamp）
        in_front:   [B, num_cam, N]     该点是否在相机前方（z_cam > 0）
    """
    B, num_cam = lidar2cam.shape[:2]
    N = points_world.shape[1]

    # 1. 自车 -> 相机: P_cam = lidar2cam @ P_world
    # points_world [B,1,N,4] ⊙ lidar2cam [B,cam,4,4] -> [B,cam,N,4]
    points_world_exp = points_world.unsqueeze(1).expand(-1, num_cam, -1, -1)
    p_cam = torch.matmul(points_world_exp, lidar2cam.transpose(-2, -1))  # [B,cam,N,4]

    z_cam = p_cam[..., 2]  # [B,cam,N] 相机坐标系下的深度
    in_front = z_cam > 1e-2  # 剔除在相机后面的点

    # 2. 透视投影: [Xc, Yc, Zc] -> K @ [Xc/Zc, Yc/Zc, 1] = [u, v, 1]
    xy_cam = p_cam[..., :2] / (z_cam.unsqueeze(-1) + 1e-6)  # [B,cam,N,2]
    # 补 1 -> [B,cam,N,3] 再乘内参
    xy1_cam = torch.cat([xy_cam, torch.ones_like(xy_cam[..., :1])], dim=-1)
    pix_xyz = torch.matmul(xy1_cam, cam_intrinsic.transpose(-2, -1))  # [B,cam,N,3]
    pix_coord = pix_xyz[..., :2]  # [B,cam,N,2]，取 (u,v)
    return pix_coord, in_front


class SpatialCrossAttention(nn.Module):
    """
    空间交叉注意力：每个 BEV Query 通过相机投影
    从 6 视角图像上的对应位置采样特征，再做注意力加权。

    【简化实现说明】
    mmdet3d 原版实现了：
      - 多尺度图像特征（num_levels > 1）
      - 可变形注意力（Deformable Attention）
      - 更复杂的采样偏移学习
    这里简化为：
      - 单尺度特征
      - 手工在 reference point 周围 K 个网格点双线性采样（同样有效且易于理解）
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 4,
        num_points: int = 4,      # 每个 query 每个相机采样多少点
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.num_points = num_points
        self.scale = self.head_dims ** -0.5

        # Query / Key / Value 投影
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Conv2d(embed_dims, embed_dims, 1)  # 用 1x1 卷版本更方便图像特征
        self.v_proj = nn.Conv2d(embed_dims, embed_dims, 1)
        self.out_proj = nn.Linear(embed_dims, embed_dims)

        # 采样点的偏移量学习（每个 head × num_points × 2 偏移）
        # weight 初始为 0；bias 让 P 个点初始围绕参考点呈 3x3 网格散开
        # （若初始全部重合在中心，小目标在格子内有偏移时所有点都落在目标外，冷启动死锁）
        self.sampling_offsets = nn.Linear(
            embed_dims,
            num_heads * num_points * 2,
        )
        nn.init.constant_(self.sampling_offsets.weight, 0)
        nn.init.constant_(self.sampling_offsets.bias, 0)
        with torch.no_grad():
            # raw 偏移 ×10 = 原图像素偏移；±2.2 即 ±22px（16x16 特征图上约 ±1.4 格）
            grid_offsets = torch.tensor([
                [0.0, 0.0], [-2.2, -2.2], [-2.2, 2.2], [2.2, -2.2],
                [2.2, 2.2], [0.0, -2.2], [0.0, 2.2], [-2.2, 0.0],
                [2.2, 0.0],
            ], dtype=torch.float32)[:num_points]                # [P, 2]
            bias_init = grid_offsets.repeat(num_heads, 1).reshape(-1)  # [H*P*2]
            self.sampling_offsets.bias.copy_(bias_init)

        # 每个采样点的注意力权重（学习）
        self.attn_weights = nn.Linear(embed_dims, num_heads * num_points)

        self.dropout = nn.Dropout(dropout)

    def bilinear_sampling(
        self,
        feat_map: torch.Tensor,
        pix_coord: torch.Tensor,
        img_w: int,
        img_h: int,
    ) -> torch.Tensor:
        """
        双线性采样：从图像特征图中根据像素坐标取特征
        Args:
            feat_map:  [B, C, Hf, Wf]  单相机单尺度特征图
            pix_coord: [B, H, N, P, 2] 采样点【原图】像素坐标（P 点 × H 头 × N query）
            img_w/img_h: 原图宽高（归一化必须用原图尺寸，而非特征图尺寸，否则越界 32 倍）
        Returns:
            sampled:   [B, C, H, N, P]  采样后的特征
        """
        B, C, Hf, Wf = feat_map.shape
        # 把原图像素坐标归一化到 [-1, 1] 用于 F.grid_sample
        norm_u = 2 * pix_coord[..., 0] / img_w - 1
        norm_v = 2 * pix_coord[..., 1] / img_h - 1
        grid = torch.stack([norm_u, norm_v], dim=-1)  # [B, H, N, P, 2]

        # grid_sample 要求 [B, H*N, P, 2]，把 head 和 query 合并进 spatial 维
        grid_flat = grid.flatten(1, 2)  # [B, H*N, P, 2]
        sampled_flat = F.grid_sample(
            feat_map, grid_flat,
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )  # [B, C, H*N, P]
        sampled = sampled_flat.view(B, C, pix_coord.shape[1], pix_coord.shape[2], pix_coord.shape[3])
        return sampled

    def forward(
        self,
        bev_queries: torch.Tensor,
        bev_pos_enc: torch.Tensor,
        image_feats: List[torch.Tensor],
        lidar2cam: torch.Tensor,
        cam_intrinsic: torch.Tensor,
        bev_h: int,
        bev_w: int,
        bev_z: int = 1,
        x_bound: List[float] = [-50, 50],
        y_bound: List[float] = [-50, 50],
        z_bound: List[float] = [-10, 10],
        img_h: int = 256,
        img_w: int = 256,
    ) -> torch.Tensor:
        """
        Args:
            bev_queries:    [B, N, C]      N = H*W*Z，当前 BEV query 特征
            bev_pos_enc:    [B, N, C]      BEV query 位置编码
            image_feats:    [num_cam, B, C, Hf, Wf]  多相机图像特征列表
            lidar2cam:      [B, num_cam, 4, 4]  外参
            cam_intrinsic:  [B, num_cam, 3, 3]  内参
        Returns:
            output:         [B, N, C]  SCA 更新后的 BEV query
        """
        B, N, C = bev_queries.shape
        num_cam = lidar2cam.shape[1]
        H = self.num_heads
        P = self.num_points
        device = bev_queries.device

        # ============================================================
        # 第一步：计算 reference point（BEV 网格中心点）
        # ============================================================
        ref_norm = get_reference_points_in_bev(bev_h, bev_w, bev_z, device)  # [1, N, 3]
        ref_norm = ref_norm.expand(B, -1, -1)                                  # [B, N, 3]
        ref_world = denormalize_bev_to_world(ref_norm, x_bound, y_bound, z_bound)  # [B, N, 3]
        ref_world_h = torch.cat([ref_world, torch.ones(B, N, 1, device=device)], dim=-1)  # [B, N, 4] 齐次

        # ============================================================
        # 第二步：把 3D reference point 投影到每个相机，得到 2D 像素坐标
        # ============================================================
        pix_ref, in_front = project_world_to_image(ref_world_h, lidar2cam, cam_intrinsic)
        # pix_ref:    [B, num_cam, N, 2]  像素 (u, v)
        # in_front:   [B, num_cam, N]     是否在相机前方

        # ============================================================
        # 第三步：学习采样偏移和注意力权重
        # ============================================================
        q_with_pos = bev_queries + bev_pos_enc  # 位置编码加到 query 上
        q = self.q_proj(q_with_pos).view(B, N, H, self.head_dims)  # [B, N, H, d]

        # offsets: [B, N, H*P*2] -> reshape [B, H, N, P, 2]
        offsets = self.sampling_offsets(q_with_pos).view(B, N, H, P, 2).permute(0, 2, 1, 3, 4)
        # 控制偏移幅度：初始训练时偏移不要太大，乘以一个可学习或固定的缩放
        # 这里用固定 10 像素，简化版本
        offsets = offsets * 10.0

        # attn_weights: [B, N, H*P] -> softmax 后 [B, H, N, P]
        weights = self.attn_weights(q_with_pos).view(B, N, H, P).permute(0, 2, 1, 3)
        weights = F.softmax(weights, dim=-1)  # 沿 P 维度归一化
        weights = self.dropout(weights)

        # ============================================================
        # 第四步：多相机 × 多头 向量化采样（原双层 Python 循环版见 bilinear_sampling）
        # ============================================================
        Hf, Wf = image_feats[0].shape[-2:]

        # 4.1 6 个相机的 V 投影一次批处理：stack 成 [B,cam,C,Hf,Wf] 再 flatten
        v_all = self.v_proj(torch.stack(image_feats, dim=1).flatten(0, 1))  # [B*cam, C, Hf, Wf]
        # 注：self.k_proj 为教学保留（注意力中图像侧仅用 V），当前简化实现不参与计算

        # 4.2 组装所有相机 × 所有头的采样网格
        #   [B,cam,1,N,1,2] + [B,1,H,N,P,2] -> [B,cam,H,N,P,2]
        pix_sample = pix_ref.unsqueeze(2).unsqueeze(4) + offsets.unsqueeze(1)
        # 相机后方的点屏蔽到图像外（grid_sample 越界采到 0）
        m = in_front.view(B, num_cam, 1, N, 1, 1).float()
        pix_sample = pix_sample * m + (1.0 - m) * (-1e4)
        # 原图像素坐标 -> grid_sample 归一化坐标 [-1, 1]
        # 注意：pix 是原图尺度(0~img_w)，必须除以原图宽高，不能除以特征图宽高
        grid_u = 2 * pix_sample[..., 0] / img_w - 1
        grid_v = 2 * pix_sample[..., 1] / img_h - 1
        grid = torch.stack([grid_u, grid_v], dim=-1)  # [B,cam,H,N,P,2]

        # 4.3 每个头只在「自己的位置」采「自己那 head_dims 个通道」
        #     旧实现把 256 通道在 4 组位置上各采一遍（4 倍冗余），这里消除冗余
        heads_out = []
        for h_idx in range(H):
            v_h = v_all[:, h_idx*self.head_dims:(h_idx+1)*self.head_dims]   # [B*cam, d, Hf, Wf]
            grid_h = grid[:, :, h_idx].reshape(B * num_cam, N, P, 2)        # [B*cam, N, P, 2]
            sampled = F.grid_sample(
                v_h, grid_h,
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )  # [B*cam, d, N, P]
            # 还原相机维并对 6 个相机累加（互补视角，注意力统一加权）
            sampled = sampled.view(B, num_cam, self.head_dims, N, P).sum(dim=1)  # [B, d, N, P]
            heads_out.append(sampled)
        # [B,H,d,N,P] -> [B,H,N,P,d]
        sampled_sum = torch.stack(heads_out, dim=1).permute(0, 1, 3, 4, 2)

        # ============================================================
        # 第五步：注意力加权（沿 P 维度）+ 输出投影
        # ============================================================
        # weights: [B, H, N, P, 1] * sampled_sum: [B, H, N, P, d]
        output = (weights.unsqueeze(-1) * sampled_sum).sum(dim=3)  # [B, H, N, d]
        output = output.permute(0, 2, 1, 3).contiguous().view(B, N, C)

        return self.out_proj(output)
