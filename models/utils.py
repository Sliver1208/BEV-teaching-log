# ============================================================
# 模型通用工具函数
# 包含注意力机制常用函数、几何变换等
# ============================================================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def position_embedding(x: torch.Tensor, num_feats: int, temperature: int = 10000) -> torch.Tensor:
    """
    生成正弦余弦位置编码（Sinusoidal Positional Encoding）
    参考 Transformer 原文，为每个 BEV 网格提供空间位置信息。

    Args:
        x: 坐标张量 [..., 2 or 3]，值域通常归一化到 [0, 1]
        num_feats: 每个坐标维度的特征维度（总维度会 *2 因为 sin+cos）
        temperature: 频率衰减温度系数

    Returns:
        位置编码 [..., num_feats * coord_dim]
    """
    # 计算每个频率维度的缩放因子: 1 / temperature^(2i/d)
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=x.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)  # [num_feats]

    # 对每个坐标维度分别生成 sin/cos 编码
    pos_list = []
    for i in range(x.shape[-1]):
        x_i = x[..., i:i + 1] * 2 * math.pi  # [..., 1] 扩到 0~2π
        pos_sin = torch.sin(x_i / dim_t)      # [..., num_feats]
        pos_cos = torch.cos(x_i / dim_t)      # [..., num_feats]
        # sin 和 cos 交错排列，与原 Transformer 实现一致
        pos_stack = torch.stack([pos_sin, pos_cos], dim=-1).flatten(-2)
        pos_list.append(pos_stack)
    return torch.cat(pos_list, dim=-1)


class MultiheadAttentionWrapper(nn.Module):
    """
    多头注意力的通用包装，用于 SCA 和 TSA 中。
    内部实现标准 Scaled Dot-Product Attention:
        Attention(Q, K, V) = softmax(QK^T/√d) * V

    这里不用 nn.MultiheadAttention 是为了方便阅读和扩展（比如自定义采样位置）。
    """

    def __init__(self, embed_dims: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        assert self.head_dims * num_heads == embed_dims, "embed_dims 必须能被 num_heads 整除"
        self.scale = self.head_dims ** -0.5

        # 线性投影：输入特征 -> Q/K/V
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

    def reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, C] -> [B, H, N, C/H]，为多头注意力做维度准备"""
        B, N, C = x.shape
        return x.view(B, N, self.num_heads, self.head_dims).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, Lq, C]  Lq = query 数量（BEV 网格数）
            key:   [B, Lk, C]  Lk = key 数量（图像像素数或历史 BEV 数）
            value: [B, Lk, C]
            attn_mask: [B, H, Lq, Lk] 或 None，0 表示保留，-inf 表示屏蔽
        """
        # 1. 投影得到 Q, K, V
        q = self.q_proj(query)  # [B, Lq, C]
        k = self.k_proj(key)    # [B, Lk, C]
        v = self.v_proj(value)  # [B, Lk, C]

        # 2. 多头 reshape: [B, L, C] -> [B, H, L, C/H]
        q = self.reshape_heads(q)
        k = self.reshape_heads(k)
        v = self.reshape_heads(v)

        # 3-6. Scaled Dot-Product Attention
        # 用 PyTorch 融合算子（底层为 FlashAttention / memory-efficient kernel）：
        # 不显式实例化 [B,H,Lq,Lk] 注意力矩阵，显存从 O(Lq*Lk) 降到 O(Lq)
        # scale 默认 1/sqrt(head_dims)，与原手动实现一致
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        # 7. 多头合并 + 输出投影
        out = out.transpose(1, 2).contiguous().view(query.shape[0], -1, self.embed_dims)
        return self.out_proj(out)


class FFN(nn.Module):
    """
    Feed-Forward Network，Transformer 编码器标准组件
    结构: Linear -> GELU -> Dropout -> Linear -> Dropout
    注意在原始 BEVFormer 中用的是 GELU 而非 ReLU。
    """

    def __init__(self, embed_dims: int, feedforward_dims: int = 512, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dims, feedforward_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dims, embed_dims),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN 结构: LN -> FFN -> Residual
        identity = x
        x = self.norm(x)
        x = self.layers(x)
        return x + identity
