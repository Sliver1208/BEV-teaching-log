# ============================================================
# BEVFormer 端到端模型（学习版实现）
# ------------------------------------------------------------
# 【整体数据流】
#
#  Input:
#    images:        [B, num_cam, 3, Himg, Wimg]   多视角图像
#    lidar2cam:     [B, num_cam, 4, 4]             自车->相机外参
#    cam_intrinsic: [B, num_cam, 3, 3]             相机内参
#    ego_pose_list: List[[B,4,4]]                  各帧自车->世界 pose
#    prev_bevs:     List[[B, N, C]]                历史 BEV 特征缓存（首次为空）
#
#  Backbone + Neck:
#    [每个相机单独过 Backbone]  ->  image_feats[cam] = [B, C, Hf, Wf]
#
#  BEV Query 生成:
#    (bev_queries, bev_pos_enc) 形状都是 [B, H*W*Z, C]
#
#  BEV Encoder (×N layers):
#    for each layer:
#       TSA (融合 prev_bevs)  ->  SCA (采样图像特征)  ->  FFN
#
#  Detection Head:
#    cls_logits, center_xy, size_whl
#
#  Output:
#    preds:        Dict of detection predictions
#    curr_bev:     [B, N, C]  当前帧 BEV（可作为下一帧的 prev_bev 缓存）
# ============================================================
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import yaml

from .backbone import SimpleResNet, ImageFPN
from .bev_query import BEVQueryGenerator
from .encoder import BEVEncoder
from .head import BEVDetectionHead
from .temporal_attn import warp_history_bev


class BEVFormer(nn.Module):
    """
    BEVFormer 端到端模型

    使用方法：
        model = BEVFormer.from_config("configs/bevformer_base.yaml")
        preds, bev_feat, _ = model(images, lidar2cam, cam_intrinsic, ...)
    """

    def __init__(
        self,
        # BEV
        bev_h: int = 50,
        bev_w: int = 50,
        bev_z: int = 1,
        embed_dims: int = 256,
        x_bound: List[float] = [-50, 50],
        y_bound: List[float] = [-50, 50],
        z_bound: List[float] = [-10, 10],
        query_type: str = "learnable",
        # Image
        num_cams: int = 6,
        backbone_channels: List[int] = None,
        backbone_out_idx: int = 2,
        # Encoder
        num_layers: int = 3,
        sca_num_heads: int = 4,
        sca_num_points: int = 4,
        tsa_num_heads: int = 4,
        num_history_frames: int = 3,
        ffn_dims: int = 512,
        dropout: float = 0.1,
        # Head
        num_classes: int = 3,
        head_hidden_dim: int = 256,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound
        self.num_history_frames = num_history_frames

        # ---------- 1. 图像 Backbone + Neck（6 相机共享权重）----------
        backbone_channels = backbone_channels or [32, 64, 128, 256]
        self.backbone_out_idx = backbone_out_idx
        self.backbone = SimpleResNet(
            in_channels=3, base_channels=32,
            stages=[2, 2, 2, 2], out_indices=[backbone_out_idx],
        )
        self.neck = ImageFPN(
            in_channels_list=[backbone_channels[backbone_out_idx]],  # 与所选 stage 通道对齐
            out_channels=embed_dims,
            num_outs=1,
        )

        # ---------- 2. BEV Query 生成器 ----------
        self.bev_query_gen = BEVQueryGenerator(
            bev_h=bev_h, bev_w=bev_w, bev_z=bev_z,
            embed_dims=embed_dims, query_type=query_type,
        )

        # ---------- 3. BEV Encoder（多层）----------
        layer_cfg = dict(
            embed_dims=embed_dims,
            sca_num_heads=sca_num_heads,
            sca_num_points=sca_num_points,
            tsa_num_heads=tsa_num_heads,
            num_history_frames=num_history_frames,
            ffn_dims=ffn_dims,
            dropout=dropout,
        )
        self.encoder = BEVEncoder(num_layers=num_layers, layer_cfg=layer_cfg)

        # ---------- 4. 检测头 ----------
        self.head = BEVDetectionHead(
            embed_dims=embed_dims, bev_h=bev_h, bev_w=bev_w,
            num_classes=num_classes, hidden_dim=head_hidden_dim,
        )

    # ------------------------------------------------------------
    # 便捷接口：从 yaml 配置创建模型
    # ------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg_path: str) -> "BEVFormer":
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(
            bev_h=cfg["bev"]["H"],
            bev_w=cfg["bev"]["W"],
            bev_z=cfg["bev"]["Z"],
            embed_dims=cfg["bev"]["embed_dims"],
            x_bound=cfg["bev"]["x_bound"],
            y_bound=cfg["bev"]["y_bound"],
            z_bound=cfg["bev"]["z_bound"],
            query_type=cfg["model"]["bev_query"]["type"],
            num_cams=cfg["image"]["num_cams"],
            backbone_out_idx=cfg["model"]["image_backbone"]["out_indices"][0],
            num_layers=cfg["model"]["num_layers"],
            sca_num_heads=cfg["model"]["spatial_cross_attn"]["num_heads"],
            sca_num_points=cfg["model"]["spatial_cross_attn"]["num_points"],
            tsa_num_heads=cfg["model"]["temporal_self_attn"]["num_heads"],
            num_history_frames=cfg["model"]["temporal_self_attn"]["num_history_frames"],
            ffn_dims=cfg["model"]["ffn"]["feedforward_dims"],
            dropout=cfg["model"]["ffn"]["dropout"],
            num_classes=cfg["model"]["head"]["num_classes"],
            head_hidden_dim=cfg["model"]["head"]["hidden_dim"],
        )

    # ------------------------------------------------------------
    # 图像 Backbone 推理：把 6 个相机过共享 Backbone
    # ------------------------------------------------------------
    def extract_image_feats(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            images: [B, num_cams, 3, Himg, Wimg]
        Returns:
            image_feats: 长度 num_cams 的列表，每项 [B, C, Hf, Wf]
        """
        B, num_cams = images.shape[:2]
        # 把 (B, cam, 3, H, W) 展平成 (B*cam, 3, H, W)，一次过 backbone 提高效率
        flat_imgs = images.flatten(0, 1)
        outs = self.backbone(flat_imgs)       # List[ [B*cam, c, h, w] ]
        outs = self.neck(outs)                 # List[ [B*cam, C, Hf, Wf] ]
        feat = outs[-1]  # 只取最后一层
        feat = feat.view(B, num_cams, *feat.shape[1:])  # [B, cam, C, Hf, Wf]
        # 拆成 per-cam 的 list，方便 SCA 逐个相机处理
        return [feat[:, cam] for cam in range(num_cams)]

    # ------------------------------------------------------------
    # 前向推理
    # ------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        lidar2cam: torch.Tensor,
        cam_intrinsic: torch.Tensor,
        ego_pose: Optional[torch.Tensor] = None,
        prev_bevs: Optional[List[torch.Tensor]] = None,
        prev_ego_poses: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            images:         [B, num_cams, 3, Himg, Wimg]
            lidar2cam:      [B, num_cam, 4, 4] 当前帧自车->相机
            cam_intrinsic:  [B, num_cam, 3, 3] 当前帧内参
            ego_pose:       [B, 4, 4]          当前帧自车->世界（可选，用于时序对齐）
            prev_bevs:      List[[B, N, C]]    历史帧 BEV 特征缓存
            prev_ego_poses: List[[B, 4, 4]]    历史帧对应的自车 pose
        Returns:
            preds:          检测头输出 dict
            curr_bev:       [B, N, C]          当前帧 BEV 特征（供下一帧缓存）
            new_prev_bevs:  List[[B, N, C]]    更新后的 BEV 缓存（最新在末尾）
        """
        B = images.shape[0]
        device = images.device

        # ---------- Step 1: 图像特征 ----------
        image_feats = self.extract_image_feats(images)  # list of [B, C, Hf, Wf]

        # ---------- Step 2: 生成 BEV Query ----------
        bev_queries, bev_pos_enc = self.bev_query_gen(B, device)  # [B, N, C]

        # ---------- Step 3: 时序对齐（把历史 BEV warp 到当前坐标系）----------
        aligned_history = []
        if prev_bevs is not None and len(prev_bevs) > 0 and ego_pose is not None and prev_ego_poses is not None:
            for hist_bev, hist_pose in zip(prev_bevs, prev_ego_poses):
                # 把 [B, N, C] 变形为 [B, H, W, C] 做 warp
                hist_bev_spatial = hist_bev.view(B, self.bev_h, self.bev_w, self.embed_dims)
                warped = warp_history_bev(
                    hist_bev_spatial, hist_pose, ego_pose,
                    self.x_bound, self.y_bound,
                )
                aligned_history.append(warped.view(B, -1, self.embed_dims))

        # ---------- Step 4: BEV Encoder ----------
        img_h, img_w = images.shape[-2:]
        bev_queries = self.encoder(
            bev_queries, bev_pos_enc,
            image_feats, lidar2cam, cam_intrinsic,
            aligned_history,
            self.bev_h, self.bev_w, self.bev_z,
            self.x_bound, self.y_bound, self.z_bound,
            img_h, img_w,
        )
        curr_bev = bev_queries  # 就是编码器最后一层的输出 [B, N, C]

        # ---------- Step 5: 检测头 ----------
        preds = self.head(bev_queries)

        # ---------- Step 6: 更新 BEV 缓存（供下一帧时序用）----------
        if prev_bevs is None:
            cache_list = [curr_bev.detach()]
        else:
            cache_list = list(prev_bevs) + [curr_bev.detach()]
        if len(cache_list) > self.num_history_frames:
            cache_list = cache_list[-self.num_history_frames:]

        new_prev_bevs = cache_list

        return preds, curr_bev, new_prev_bevs
