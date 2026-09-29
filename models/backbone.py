# ============================================================
# 图像 Backbone + Neck
# ------------------------------------------------------------
# 真实项目中一般用 ResNet-50/101 + FPN，
# 学习版实现一个轻量的 ConvNet Backbone + FPN，便于快速跑通。
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class BasicBlock(nn.Module):
    """简化的 ResNet BasicBlock（无 BottleNeck，适合小模型）"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class SimpleResNet(nn.Module):
    """
    简化版 ResNet，输出 4 阶段的特征图
    strides: [2, 2, 2, 2]  -> 总 stride = 16，和标准 ResNet stage4 对齐
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        stages: List[int] = [2, 2, 2, 2],   # 每 stage block 数
        out_indices: List[int] = [3],       # 返回哪些 stage 的输出 (0-based)
    ):
        super().__init__()
        self.out_indices = out_indices
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, 2, 3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        # stem 之后 stride = 4
        channels_list = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        strides_list = [1, 2, 2, 2]  # stage1 不 spatial 下采样，其余下采样

        self.stages = nn.ModuleList()
        in_ch = base_channels
        for i in range(4):
            blocks = []
            # first block maybe stride
            blocks.append(BasicBlock(in_ch, channels_list[i], stride=strides_list[i]))
            for _ in range(stages[i] - 1):
                blocks.append(BasicBlock(channels_list[i], channels_list[i]))
            self.stages.append(nn.Sequential(*blocks))
            in_ch = channels_list[i]

        self.channels_list = channels_list

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


class ImageFPN(nn.Module):
    """
    简化的 FPN Neck，把多层 stage 特征融合成统一 out_channels
    输出单尺度或多尺度特征图列表。
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        num_outs: int = 1,
    ):
        super().__init__()
        self.num_outs = num_outs
        # 1x1 conv 把每层统一到 out_channels
        self.laterals = nn.ModuleList([
            nn.Conv2d(in_c, out_channels, 1) for in_c in in_channels_list
        ])
        # 融合后做 3x3 smooth
        self.smooth = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        # feats 按 channel 从小到大排列（对应 stride 从小到大）
        # 简化版：只做顶层上采样融合，然后返回最后一层（或多层）
        lateral_feats = [lat(f) for lat, f in zip(self.laterals, feats)]

        # 从后向前（高层 -> 低层）做上采样融合
        for i in range(len(lateral_feats) - 1, 0, -1):
            prev = lateral_feats[i - 1]
            up = F.interpolate(lateral_feats[i], size=prev.shape[-2:], mode="nearest")
            lateral_feats[i - 1] = prev + up

        # smooth 所有层
        out = [self.smooth(f) for f in lateral_feats]

        # 如果只要求输出一层，用最后一层（最高语义）
        if self.num_outs == 1:
            return [out[-1]]
        return out[: self.num_outs]
