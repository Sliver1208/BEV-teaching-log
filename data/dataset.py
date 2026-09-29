# ============================================================
# 数据集：合成数据集 + nuScenes 包装
# ------------------------------------------------------------
# 学习版默认使用 SyntheticDataset，不依赖真实数据即可跑通：
#   - 生成"假"的 6 视角图像（包含随机几何物体）
#   - 生成对应的相机内外参、自车运动 pose
#   - 生成 BEV 空间的 GT 目标（车辆、行人等 bbox）
#
# 真实训练时，把 data.type 改成 NuScenesDatasetWrapper，
# 并配置 data_root，即可接入官方 nuScenes。
# ============================================================
import os
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

from data.utils import project_bev_to_image


class SyntheticDataset(Dataset):
    """
    合成数据集，用于快速验证 BEVFormer 数据流是否正确。
    每个 sample 返回一帧（多视角图像 + 内外参 + pose + gt bbox）
    dataset 长度足够做 mini-train，梯度能正常反传即表示模型逻辑无误。
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_cams: int = 6,
        img_h: int = 256,
        img_w: int = 256,
        bev_h: int = 50,
        bev_w: int = 50,
        sequence_length: int = 4,  # 每个样本返回 sequence_length 帧（用于时序）
        num_classes: int = 3,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_cams = num_cams
        self.img_h = img_h
        self.img_w = img_w
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.sequence_length = sequence_length
        self.num_classes = num_classes 
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.num_samples

    # ------------------------------------------------------------
    # 工具：生成一组"可信"的 6 相机外参（环绕自车）
    # ------------------------------------------------------------
    def _generate_camera_params(self) -> Tuple[np.ndarray, np.ndarray]:
        num_cams = self.num_cams
        # 6 个相机均匀分布在 360 度
        yaw_angles = np.linspace(-np.pi, np.pi, num_cams, endpoint=False)  # [-180, ..., 120]
        lidar2cam = np.zeros((num_cams, 4, 4), dtype=np.float32)
        intrinsic = np.zeros((num_cams, 3, 3), dtype=np.float32)

        for i in range(num_cams):
            yaw = yaw_angles[i]
            # 小幅度随机
            yaw += self.rng.uniform(-0.1, 0.1)
            pitch = self.rng.uniform(-0.1, 0.1)
            roll = 0.0
            # 相机位置：自车四周 1.5 米，高度 1.5m
            tx = 1.5 * np.cos(yaw + np.pi / 2)
            ty = 1.5 * np.sin(yaw + np.pi / 2)
            tz = 1.5 + self.rng.uniform(-0.1, 0.1)

            # 标准针孔几何：光轴沿"远离自车"方向，Zc = 物体在朝向上的深度
            cy, sy = math.cos(yaw), math.sin(yaw)
            R = np.array([
                [cy, sy, 0],
                [0, 0, 1],
                [-sy, cy, 0],
            ], dtype=np.float32)
            cam_pos = np.array([tx, ty, tz], dtype=np.float32)
            t = -R @ cam_pos

            # T_cam_from_lidar: P_cam = R * P_lidar + t
            lidar2cam[i, :3, :3] = R
            lidar2cam[i, :3, 3] = t
            lidar2cam[i, 3, 3] = 1.0

            # 内参: 焦距约等于 img_w（典型值）
            fx = self.img_w + self.rng.uniform(-20, 20)
            fy = self.img_w + self.rng.uniform(-20, 20)
            cx = self.img_w / 2 + self.rng.uniform(-10, 10)
            cy = self.img_h / 2 + self.rng.uniform(-10, 10)
            intrinsic[i] = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ], dtype=np.float32)
        return lidar2cam, intrinsic

    # ------------------------------------------------------------
    # 工具：生成自车运动 pose（沿 x 方向缓慢前进，有轻微转向）
    # ------------------------------------------------------------
    def _generate_ego_poses(self) -> List[np.ndarray]:
        poses = []
        x, y, yaw = 0.0, 0.0, 0.0
        for t in range(self.sequence_length):
            # 每帧向前 0.5m，小角度左转
            dx = 0.5
            dy = self.rng.uniform(-0.05, 0.05)
            dyaw = self.rng.uniform(-0.03, 0.03)
            x += dx * math.cos(yaw) - dy * math.sin(yaw)
            y += dx * math.sin(yaw) + dy * math.cos(yaw)
            yaw += dyaw
            T = np.eye(4, dtype=np.float32)
            T[0, 0] = math.cos(yaw); T[0, 1] = -math.sin(yaw)
            T[1, 0] = math.sin(yaw); T[1, 1] = math.cos(yaw)
            T[0, 3] = x; T[1, 3] = y
            poses.append(T)
        return poses

    # ------------------------------------------------------------
    # 工具：生成一帧的物体列表（自车坐标系，图像与 GT 共用同一份）
    # ------------------------------------------------------------
    def _generate_scene_objects(self) -> List[dict]:
        """每帧随机 3~8 个目标，返回 cls/位置/尺寸，保证图像<->GT 一致"""
        objects = []
        for _ in range(self.rng.randint(3, 9)):
            cls = self.rng.randint(0, self.num_classes)
            default_sizes = [(4.0, 2.0, 1.5), (0.6, 0.6, 1.7), (1.8, 0.8, 1.2)]
            wl, ww, wh = default_sizes[cls]
            objects.append({
                "cls": cls,
                "x": self.rng.uniform(-20, 20),
                "y": self.rng.uniform(-20, 20),
                "w": ww * self.rng.uniform(0.8, 1.2),
                "l": wl * self.rng.uniform(0.8, 1.2),
                "h": wh * self.rng.uniform(0.8, 1.2),
            })
        return objects

    # ------------------------------------------------------------
    # 工具：把物体按真实投影画到相机图像上（与 SCA 用同一套投影数学）
    # ------------------------------------------------------------
    def _generate_images(self, objects, lidar2cam, intrinsic) -> np.ndarray:
        """[num_cams, 3, Himg, Wimg]，RGB 图像"""
        cls_colors = np.array([
            [0.9, 0.2, 0.2],   # car -> 红色
            [0.2, 0.9, 0.2],   # pedestrian -> 绿色
            [0.2, 0.2, 0.9],   # cyclist -> 蓝色
        ], dtype=np.float32)
        imgs = np.zeros((self.num_cams, 3, self.img_h, self.img_w), dtype=np.float32)
        for c in range(self.num_cams):
            # 平缓背景：相机级恒定灰调 + 轻微噪声（强逐像素噪声会在下采样后淹没色块）
            base = self.rng.uniform(0.42, 0.58)
            bg = base + self.rng.uniform(-0.03, 0.03, (3, self.img_h, self.img_w)).astype(np.float32)
            imgs[c] = np.clip(bg, 0, 1)

        # 物体中心投影到各相机（复用 SCA 的投影函数，保证几何一致）
        pts = np.array([[o["x"], o["y"], 0.0] for o in objects], dtype=np.float32)
        pts_t = torch.from_numpy(pts)[None]                    # [1, N, 3]
        l2c_t = torch.from_numpy(lidar2cam)[None]              # [1, cam, 4, 4]
        K_t = torch.from_numpy(intrinsic)[None]                # [1, cam, 3, 3]
        pix, in_front = project_bev_to_image(pts_t, l2c_t, K_t)
        pix = pix[0].numpy()          # [cam, N, 2]
        in_front = in_front[0].numpy()  # [cam, N]

        for c in range(self.num_cams):
            for n, obj in enumerate(objects):
                if not in_front[c, n]:
                    continue
                u, v = pix[c, n]
                if not (0 <= u < self.img_w and 0 <= v < self.img_h):
                    continue
                # 近大远小：像素尺寸 = 物理尺寸 * 焦距 / 深度
                P = np.array([obj["x"], obj["y"], 0.0], dtype=np.float32)
                depth = (lidar2cam[c, :3, :3] @ P + lidar2cam[c, :3, 3])[2]
                size_px = int(np.clip(max(obj["l"], obj["w"]) * intrinsic[c, 0, 0] / depth, 32, 90))
                h0 = int(np.clip(v - size_px / 2, 0, self.img_h - 1))
                w0 = int(np.clip(u - size_px / 2, 0, self.img_w - 1))
                h1 = int(min(self.img_h, h0 + size_px))
                w1 = int(min(self.img_w, w0 + size_px))
                color = cls_colors[obj["cls"]] * self.rng.uniform(0.85, 1.0, 3).astype(np.float32)
                imgs[c, :, h0:h1, w0:w1] = color[:, None, None]
                # 深色描边（模拟物体轮廓，供 CNN 边缘滤波器强响应）
                edge = max(2, size_px // 10)
                imgs[c, :, h0:min(h1, h0+edge), w0:w1] = 0.05
                imgs[c, :, max(h0, h1-edge):h1, w0:w1] = 0.05
                imgs[c, :, h0:h1, w0:min(w1, w0+edge)] = 0.05
                imgs[c, :, h0:h1, max(w0, w1-edge):w1] = 0.05
        # 归一化到 [-1,1]
        imgs = imgs * 2 - 1
        return imgs

    # ------------------------------------------------------------
    # 工具：在 BEV 空间生成 GT bbox（简化为热图 + 偏移 + 尺寸）
    # ------------------------------------------------------------
    def _generate_bev_gt(self, objects: List[dict]) -> Dict[str, np.ndarray]:
        """
        返回:
          cls_heatmap: [bev_h, bev_w, num_classes]  one-hot 热图
          center_xy:   [bev_h, bev_w, 2]            网格内偏移
          size_whl:    [bev_h, bev_w, 3]            w,l,h
          cls_mask:    [bev_h, bev_w]               有物体的格点为 1
        """
        hm = np.zeros((self.bev_h, self.bev_w, self.num_classes), dtype=np.float32)
        off = np.zeros((self.bev_h, self.bev_w, 2), dtype=np.float32)
        size = np.zeros((self.bev_h, self.bev_w, 3), dtype=np.float32)
        mask = np.zeros((self.bev_h, self.bev_w), dtype=np.float32)

        x_bound = np.array([-50, 50])
        y_bound = np.array([-50, 50])

        for obj in objects:
            cls = obj["cls"]
            # 世界坐标（gt 在当前帧自车坐标系下，与图像色块来自同一份 objects）
            px, py = obj["x"], obj["y"]
            ww, wl, wh = obj["w"], obj["l"], obj["h"]

            # 映射到 BEV 网格 index
            gx = int((px - x_bound[0]) / (x_bound[1] - x_bound[0]) * self.bev_h)
            gy = int((py - y_bound[0]) / (y_bound[1] - y_bound[0]) * self.bev_w)
            gx = np.clip(gx, 0, self.bev_h - 1)
            gy = np.clip(gy, 0, self.bev_w - 1)

            # 子像素偏移 (归一化到 [-0.5, 0.5])
            cell_size_x = (x_bound[1] - x_bound[0]) / self.bev_h
            cell_size_y = (y_bound[1] - y_bound[0]) / self.bev_w
            exact_gx = (px - x_bound[0]) / cell_size_x
            exact_gy = (py - y_bound[0]) / cell_size_y
            ox = (exact_gx - gx - 0.5) * 2  # 归一化到 [-1,1] 左右
            oy = (exact_gy - gy - 0.5) * 2

            hm[gx, gy, cls] = 1.0
            off[gx, gy, 0] = ox
            off[gx, gy, 1] = oy
            size[gx, gy, 0] = ww
            size[gx, gy, 1] = wl
            size[gx, gy, 2] = wh
            mask[gx, gy] = 1.0

        return {
            "cls_heatmap": hm,
            "center_xy": off,
            "size_whl": size,
            "cls_mask": mask,
        }

    # ------------------------------------------------------------
    # 主接口：__getitem__ 返回一个时间序列
    # ------------------------------------------------------------
    def __getitem__(self, idx):
        # 1. 相机内外参（整段序列共享，相机相对自车不动）
        lidar2cam, intrinsic = self._generate_camera_params()

        # 2. 生成 sequence_length 帧数据
        ego_poses = self._generate_ego_poses()
        frame_list = []
        for t in range(self.sequence_length):
            objects = self._generate_scene_objects()
            imgs = self._generate_images(objects, lidar2cam, intrinsic)
            gt = self._generate_bev_gt(objects)
            frame_list.append({
                "images": torch.from_numpy(imgs),  # [num_cams, 3, H, W]
                "lidar2cam": torch.from_numpy(lidar2cam),  # [num_cams, 4, 4]
                "cam_intrinsic": torch.from_numpy(intrinsic),  # [num_cams, 3, 3]
                "ego_pose": torch.from_numpy(ego_poses[t]),  # [4, 4]
                "cls_heatmap": torch.from_numpy(gt["cls_heatmap"]),  # [Hbev, Wbev, C]
                "center_xy": torch.from_numpy(gt["center_xy"]),      # [Hbev, Wbev, 2]
                "size_whl": torch.from_numpy(gt["size_whl"]),        # [Hbev, Wbev, 3]
                "cls_mask": torch.from_numpy(gt["cls_mask"]),        # [Hbev, Wbev]
            })

        # 堆叠为 sequence 维度
        seq = {}
        for k in frame_list[0].keys():
            seq[k] = torch.stack([f[k] for f in frame_list], dim=0)  # [T, ...]
        return seq


class NuScenesDatasetWrapper(Dataset):
    """
    真实 nuScenes 数据集包装（占位实现，需要 nuscenes-devkit）
    实际使用时实现此接口以接入真实数据。
    """

    def __init__(self, data_root: str, version: str = "v1.0-trainval", **kwargs):
        super().__init__()
        self.data_root = data_root
        self.version = version
        # try:
        #     from nuscenes.nuscenes import NuScenes
        #     self.nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
        # except ImportError:
        #     raise RuntimeError("请先 pip install nuscenes-devkit")
        raise NotImplementedError(
            "真实 nuScenes 数据读取需要根据官方 devkit 实现，"
            "推荐先用 SyntheticDataset 验证模型，"
            "再接入 nuScenes 官方代码。"
        )

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError


def build_dataloader(cfg: dict, split: str = "train") -> DataLoader:
    """根据配置构造 DataLoader"""
    data_cfg = cfg.get("data", {})
    data_type = data_cfg.get("type", "SyntheticDataset")

    if data_type == "SyntheticDataset":
        ds = SyntheticDataset(
            num_samples=data_cfg.get("num_samples", 100),
            num_cams=cfg["image"]["num_cams"],
            img_h=cfg["image"]["img_h"],
            img_w=cfg["image"]["img_w"],
            bev_h=cfg["bev"]["H"],
            bev_w=cfg["bev"]["W"],
            sequence_length=data_cfg.get("sequence_length", 4),
            num_classes=cfg["model"]["head"]["num_classes"],
        )
    elif data_type == "NuScenesDatasetWrapper":
        ds = NuScenesDatasetWrapper(data_root=data_cfg["data_root"])
    else:
        raise ValueError(f"未知数据集 type: {data_type}")

    train_cfg = cfg.get("train", {})
    return DataLoader(
        ds,
        batch_size=train_cfg.get("batch_size", 2),
        shuffle=(split == "train"),
        num_workers=train_cfg.get("num_workers", 0),
        drop_last=True,
    )
