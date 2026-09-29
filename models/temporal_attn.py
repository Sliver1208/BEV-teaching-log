# ============================================================
# 核心模块 3/3: 时间自注意力 (Temporal Self-Attention, TSA)
# ------------------------------------------------------------
# 【为什么需要时间融合？】
# 1. 单帧图像看不到被遮挡的物体，但历史帧可能见过
# 2. 单帧深度估计有歧义，但多帧运动后可通过视差反推深度
# 3. 平滑时序上的检测结果，避免跳变
#
# 【TSA 核心思想】
#   当前 BEV Query（帧 t）把自己当 Q，
#   把前几帧的 BEV 特征（帧 t-1, t-2, ...）当 K, V，
#   做标准 Self-Attention，在时间维度上"找自己过去的状态"。
#
# 【关键步骤：坐标对齐（Alignment）】
#   帧之间自车是移动的！同样的 BEV 网格（比如 (0,0) 自车原点）
#   在真实世界中对应的位置不同。所以做注意力之前，
#   必须把历史 BEV 特征用自车运动矩阵 warp 回当前帧坐标系。
#
# 【简化版与原版区别】
#   mmdet3d 原版用的是 Deformable DETR 风格的 deformable attention，
#   这里用标准 self-attention + 双线性 warp 实现，更易理解。
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from .utils import MultiheadAttentionWrapper, position_embedding


def get_bev_grid_world_coords(
    bev_h: int,
    bev_w: int,
    x_bound: List[float],
    y_bound: List[float],
    device: torch.device = None,
) -> torch.Tensor:
    """
    生成 BEV 网格中心在自车坐标系下的物理坐标（米）
    Returns: [H, W, 2] -> (x,y)，前为 x 正方向，左为 y 正方向
    """
    x_min, x_max = x_bound
    y_min, y_max = y_bound
    xs = torch.linspace(x_min + (x_max - x_min) / bev_h / 2,
                        x_max - (x_max - x_min) / bev_h / 2,
                        bev_h, device=device)
    ys = torch.linspace(y_min + (y_max - y_min) / bev_w / 2,
                        y_max - (y_max - y_min) / bev_w / 2,
                        bev_w, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [W, H]，注意顺序
    coords = torch.stack([grid_x, grid_y], dim=-1)  # [W, H, 2]
    coords = coords.permute(1, 0, 2)  # [H, W, 2]，让 shape 和 meshgrid 顺序对应
    return coords


def warp_history_bev(
    history_bev: torch.Tensor,
    ego_pose_prev: torch.Tensor,
    ego_pose_curr: torch.Tensor,
    x_bound: List[float],
    y_bound: List[float],
) -> torch.Tensor:
    """
    利用自车运动，把上一帧的 BEV 特征图 warp 到当前帧坐标系下
    （物理世界同一个点在两帧图像中被对齐到相同的 BEV 网格）

    Args:
        history_bev:     [B, H, W, C]    上一帧 BEV 特征（已经重排好的空间张量形式）
        ego_pose_prev:   [B, 4, 4]       上一帧自车到世界的变换 (SE(3))
        ego_pose_curr:   [B, 4, 4]       当前帧自车到世界的变换
        x_bound, y_bound: BEV 物理范围
    Returns:
        warped:          [B, H, W, C]    对齐到当前帧坐标系的历史 BEV 特征
    """
    B, H, W, C = history_bev.shape
    device = history_bev.device
    x_min, x_max = x_bound
    y_min, y_max = y_bound

    # ------------------------------------------------------------
    # 1. 计算运动变换：把当前帧 BEV 网格坐标 -> 上一帧自车坐标
    #    curr_ego -> world -> prev_ego
    #    T(prev_ego <- curr_ego) = ego_pose_prev^{-1} @ ego_pose_curr
    # ------------------------------------------------------------
    T_rel = torch.matmul(torch.inverse(ego_pose_prev), ego_pose_curr)  # [B,4,4]
    rot = T_rel[:, :2, :2]  # [B,2,2] 只取 BEV 平面的旋转平移
    trans = T_rel[:, :2, 3]  # [B,2]

    # ------------------------------------------------------------
    # 2. 获取当前帧每个 BEV 网格在 curr_ego 坐标系下的坐标
    # ------------------------------------------------------------
    grid_curr = get_bev_grid_world_coords(H, W, x_bound, y_bound, device)  # [H,W,2]
    grid_curr = grid_curr.unsqueeze(0).expand(B, -1, -1, -1)              # [B,H,W,2]
    # 把 (H, W, 2) reshape 成 (B, H*W, 2) 做矩阵乘法
    gc_flat = grid_curr.reshape(B, -1, 2)  # [B, HW, 2]

    # ------------------------------------------------------------
    # 3. 变换：P_prev = R @ P_curr + t
    # ------------------------------------------------------------
    gp_flat = torch.matmul(gc_flat, rot.transpose(-2, -1)) + trans.unsqueeze(1)  # [B,HW,2]
    grid_prev = gp_flat.view(B, H, W, 2)  # [B,H,W,2]（上一帧自车坐标系）

    # ------------------------------------------------------------
    # 4. 归一化到 [-1,1] 用于 grid_sample
    # ------------------------------------------------------------
    u = 2 * (grid_prev[..., 0] - x_min) / (x_max - x_min) - 1  # x 对应 H 维
    v = 2 * (grid_prev[..., 1] - y_min) / (y_max - y_min) - 1  # y 对应 W 维
    # grid_sample 的 grid 是 (u, v) 顺序，对应 (x_feat, y_feat)，
    # 其中 x_feat 是 W（横轴），y_feat 是 H（纵轴）
    sample_grid = torch.stack([v, u], dim=-1)  # [B,H,W,2]

    # ------------------------------------------------------------
    # 5. 双线性 warp
    #    history_bev 需要转成 [B, C, H, W] 再 grid_sample
    # ------------------------------------------------------------
    hbev = history_bev.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
    warped = F.grid_sample(
        hbev, sample_grid,
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )  # [B, C, H, W]
    return warped.permute(0, 2, 3, 1).contiguous()  # 换回 [B, H, W, C]


class TemporalSelfAttention(nn.Module):
    """
    时间自注意力层
    输入:
        curr_bev:       [B, N, C]  当前帧 BEV Query
        history_bevs:   List[[B, N, C]]  历史帧 BEV（已经过坐标对齐后的序列，可空）
        curr_pos_enc:   [B, N, C]  当前帧位置编码
    输出:
        fused_bev:      [B, N, C]  融合了时间信息的新 BEV Query
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 4,
        num_history_frames: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_history_frames = num_history_frames

        # 标准多头注意力
        self.attn = MultiheadAttentionWrapper(embed_dims, num_heads, dropout)

        # 帧间时间步编码：让模型知道"距离当前多少帧"
        # 类似 Transformer 的 sinusoidal 编码，但作用于时间维
        self.temporal_embed = nn.Parameter(torch.Tensor(num_history_frames + 1, embed_dims))
        nn.init.xavier_uniform_(self.temporal_embed)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dims)

    def forward(
        self,
        curr_bev: torch.Tensor,
        curr_pos_enc: torch.Tensor,
        history_bevs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            curr_bev:      [B, N, C]
            curr_pos_enc:  [B, N, C]
            history_bevs:  长度为 K 的列表，每一项为 [B, N, C]
                           （这里的历史 BEV 必须在调用本函数前先用 warp_history_bev 对齐坐标！）
        """
        B, N, C = curr_bev.shape

        # ------------------------------------------------------------
        # 0. 无历史帧时直接恒等通过
        #    （原版 BEVFormer 的 TSA 是 deformable attention，每个 query 只看
        #     对应空间位置；若用全连接 self-attention 代替，无历史时会对全部
        #     2500 个 query 做近似均匀平均，把 SCA 建立的局部定位信号抹平）
        # ------------------------------------------------------------
        history_bevs = history_bevs[-self.num_history_frames:]
        if len(history_bevs) == 0:
            return curr_bev

        # ------------------------------------------------------------
        # 1. K/V 只取历史帧（不再拼接当前帧自身，避免全局自平均）
        #    整体 shape: [B, K*N, C]
        # ------------------------------------------------------------
        kv_list = list(history_bevs)
        kv_seq = torch.cat(kv_list, dim=1)  # [B, K*N, C]

        # ------------------------------------------------------------
        # 2. 加上时间步编码：同一时间步的所有 query 共享同一个时间 embedding
        #    时间步: 前1帧=1, 前2帧=2, ...
        # ------------------------------------------------------------
        num_tokens_per_step = N
        L_kv = kv_seq.shape[1]
        num_steps = L_kv // num_tokens_per_step
        t_ids = torch.arange(1, num_steps + 1, device=curr_bev.device)  # 历史帧从 1 开始
        t_emb = self.temporal_embed[t_ids]  # [num_steps, C]
        t_emb_exp = t_emb.unsqueeze(1).expand(-1, num_tokens_per_step, -1).reshape(1, -1, C)
        kv_seq = kv_seq + t_emb_exp

        # 历史 token 的位置补零（历史帧几何位置与当前帧不完全对齐，简化处理）
        kv_seq = kv_seq + torch.zeros_like(curr_pos_enc).repeat(1, len(history_bevs), 1)[:, :L_kv]

        # ------------------------------------------------------------
        # 3. 多头注意力: Q = curr_bev + curr_pos_enc，K/V = kv_seq
        # ------------------------------------------------------------
        q = curr_bev + curr_pos_enc  # [B, N, C]
        out = self.attn(query=q, key=kv_seq, value=kv_seq)  # [B, N, C]

        # ------------------------------------------------------------
        # 4. Pre-LN + Residual
        # ------------------------------------------------------------
        out = self.dropout(out)
        return self.norm(curr_bev + out)
