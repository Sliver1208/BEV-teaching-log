# ============================================================
# 检测头损失函数（简化版）
# ------------------------------------------------------------
# 对应 BEVDetectionHead 的 3 个输出分支:
#   1. cls_logits:  分类热图（使用 Focal Loss，正负样本极不平衡）
#   2. center_xy:   子网格偏移（L1 Loss，仅在正样本位置计算）
#   3. size_whl:    物体尺寸（L1 Loss，仅在正样本位置计算）
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class FocalLoss(nn.Module):
    """Focal Loss for classification heatmap"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: [B, H, W, cls] 未 sigmoid
            target:      [B, H, W, cls] 0/1 one-hot
            mask:        [B, H, W]      正样本所在格
        Returns:
            标量 loss
        """
        pred = torch.sigmoid(pred_logits)
        pt = torch.where(target == 1, pred, 1 - pred)
        log_pt = torch.log(torch.clamp(pt, 1e-6, 1 - 1e-6))
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        loss = -alpha_t * focal_weight * log_pt

        # 只在正样本位置做加权归一化（其他位置本来就是 0）
        num_pos = mask.sum().clamp(min=1)
        return loss.sum() / num_pos


class BEVDetectionLoss(nn.Module):

    def __init__(self, weight_cls: float = 1.0, weight_reg: float = 0.5, weight_size: float = 0.5):
        super().__init__()
        self.cls_loss = FocalLoss()
        self.weight_cls = weight_cls
        self.weight_reg = weight_reg
        self.weight_size = weight_size

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        gts: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            preds: {
                "cls_logits": [B, H, W, C],
                "center_xy":  [B, H, W, 2],
                "size_whl":   [B, H, W, 3],
            }
            gts: {
                "cls_heatmap": [B, H, W, C],
                "center_xy":   [B, H, W, 2],
                "size_whl":    [B, H, W, 3],
                "cls_mask":    [B, H, W],    1 = 该格有物体
            }
        Returns:
            {"total": ..., "cls": ..., "reg": ..., "size": ...}
        """
        mask = gts["cls_mask"]  # [B, H, W]
        num_pos = mask.sum().clamp(min=1)

        # --- classification ---
        loss_cls = self.cls_loss(preds["cls_logits"], gts["cls_heatmap"], mask)

        # --- center offset L1 (只在正样本位置) ---
        mask_exp = mask.unsqueeze(-1)  # [B, H, W, 1]
        diff_reg = (preds["center_xy"] - gts["center_xy"]).abs() * mask_exp
        loss_reg = diff_reg.sum() / num_pos

        # --- size L1 ---
        diff_size = (preds["size_whl"] - gts["size_whl"]).abs() * mask_exp
        loss_size = diff_size.sum() / num_pos

        total = (
            self.weight_cls * loss_cls
            + self.weight_reg * loss_reg
            + self.weight_size * loss_size
        )

        return {
            "total": total,
            "cls": loss_cls.detach(),
            "reg": loss_reg.detach(),
            "size": loss_size.detach(),
        }
