# ============================================================
# 数据处理工具：3D 几何变换 / 相机投影
# 这些函数是理解 SCA 中"参考点投影"的前置知识
# ============================================================
import torch
import numpy as np
from typing import List, Tuple


def points_img_to_cam(
    pixels: np.ndarray,
    depths: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """
    像素坐标 + 深度 -> 相机坐标系 3D 点（透视投影逆运算）
    [u, v, 1] * z = K @ [Xc, Yc, Zc]
    所以: [Xc, Yc, Zc] = z * K^{-1} @ [u, v, 1]

    Args:
        pixels:     [N, 2] 像素 (u, v)
        depths:     [N]    每个像素对应的深度 Zc（米）
        intrinsic:  [3, 3] 相机内参 K = [[fx, 0, cx], [0, fy, cy], [0,0,1]]
    Returns:
        cam_pts:    [N, 3] 相机坐标系下 (Xc, Yc, Zc)
    """
    K_inv = np.linalg.inv(intrinsic)
    uv1 = np.hstack([pixels, np.ones((len(pixels), 1))])  # [N, 3]
    cam_dir = uv1 @ K_inv.T                                # [N, 3]
    cam_pts = cam_dir * depths[:, None]                    # 乘深度得到 3D 点
    return cam_pts


def points_cam_to_lidar(cam_pts: np.ndarray, lidar2cam: np.ndarray) -> np.ndarray:
    """
    相机坐标系 -> 自车(lidar)坐标系
    P_lidar = lidar2cam^{-1} @ P_cam

    Args:
        cam_pts:     [N, 3]
        lidar2cam:   [4, 4] 齐次外参（lidar -> cam 的变换）
    Returns:
        lidar_pts:   [N, 3]
    """
    cam2lidar = np.linalg.inv(lidar2cam)
    cam_h = np.hstack([cam_pts, np.ones((len(cam_pts), 1))])
    lidar_h = cam_h @ cam2lidar.T
    return lidar_h[:, :3]


def points_lidar_to_bev_grid(
    lidar_pts: np.ndarray,
    x_bound: List[float], y_bound: List[float], z_bound: List[float],
    bev_h: int, bev_w: int, bev_z: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    自车 3D 点 -> BEV 网格 index
    返回 (grid_index, valid_mask)
    """
    x, y, z = lidar_pts[:, 0], lidar_pts[:, 1], lidar_pts[:, 2]
    x_min, x_max = x_bound
    y_min, y_max = y_bound
    z_min, z_max = z_bound

    mask = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max) & (z >= z_min) & (z < z_max)

    x_idx = np.clip(((x - x_min) / (x_max - x_min) * bev_h).astype(int), 0, bev_h - 1)
    y_idx = np.clip(((y - y_min) / (y_max - y_min) * bev_w).astype(int), 0, bev_w - 1)
    z_idx = np.clip(((z - z_min) / (z_max - z_min) * bev_z).astype(int), 0, bev_z - 1)

    flat_idx = z_idx * bev_h * bev_w + x_idx * bev_w + y_idx
    return flat_idx, mask


def get_reference_points_in_bev(
    bev_h: int, bev_w: int, bev_z: int = 1,
    batch_size: int = 1, device: torch.device = None,
) -> torch.Tensor:
    """归一化 reference point，用于 SCA"""
    zs = torch.linspace(0, 1, bev_z, device=device) if bev_z > 1 else torch.tensor([0.5], device=device)
    ys = torch.linspace(0, 1, bev_w, device=device)
    xs = torch.linspace(0, 1, bev_h, device=device)
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
    coords = torch.stack([gx, gy, gz], dim=-1).reshape(1, -1, 3)
    return coords.expand(batch_size, -1, -1).contiguous()


def project_bev_to_image(
    ref_points_world: torch.Tensor,  # [B, N, 3]
    lidar2cam: torch.Tensor,         # [B, cam, 4, 4]
    cam_intrinsic: torch.Tensor,     # [B, cam, 3, 3]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """把 BEV 空间的 3D reference points 投影到每个相机的像素平面"""
    B, N = ref_points_world.shape[:2]
    num_cam = lidar2cam.shape[1]
    pts_h = torch.cat([ref_points_world, torch.ones(B, N, 1, device=ref_points_world.device)], dim=-1)
    pts_h = pts_h.unsqueeze(1).expand(-1, num_cam, -1, -1)  # [B,cam,N,4]

    p_cam = torch.matmul(pts_h, lidar2cam.transpose(-2, -1))  # [B,cam,N,4]
    z_cam = p_cam[..., 2]
    in_front = z_cam > 1e-2
    xy_cam = p_cam[..., :2] / (z_cam.unsqueeze(-1) + 1e-6)
    xy1 = torch.cat([xy_cam, torch.ones_like(xy_cam[..., :1])], dim=-1)
    pix = torch.matmul(xy1, cam_intrinsic.transpose(-2, -1))
    return pix[..., :2], in_front
