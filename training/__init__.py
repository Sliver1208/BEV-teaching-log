# ============================================================
# 训练相关包初始化
# ============================================================
from .loss import BEVDetectionLoss
from .train import run_training

__all__ = ['BEVDetectionLoss', 'run_training']
