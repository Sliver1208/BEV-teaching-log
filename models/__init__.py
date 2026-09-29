# ============================================================
# BEVFormer 模型包初始化
# 统一导出所有模块，方便外部调用
# ============================================================
from .backbone import SimpleResNet, ImageFPN
from .bev_query import BEVQueryGenerator
from .spatial_attn import SpatialCrossAttention
from .temporal_attn import TemporalSelfAttention
from .encoder import BEVEncoderLayer, BEVEncoder
from .head import BEVDetectionHead
from .bevformer import BEVFormer

__all__ = [
    'SimpleResNet',
    'ImageFPN',
    'BEVQueryGenerator',
    'SpatialCrossAttention',
    'TemporalSelfAttention',
    'BEVEncoderLayer',
    'BEVEncoder',
    'BEVDetectionHead',
    'BEVFormer',
]
