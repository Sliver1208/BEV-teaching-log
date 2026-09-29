# ============================================================
# 可视化工具
# ------------------------------------------------------------
# 主要做三件事：
#   1. 把 BEV 空间的分类热图画成热力图
#   2. 把检测 bbox（从 preds 解码出来）画在 BEV 平面上
#   3. 把 BEV reference point 投影到图像上的位置画出来（用于理解 SCA）
# ============================================================
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 无头环境
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Optional


def visualize_bev_heatmap(
    cls_prob: np.ndarray,
    save_path: str,
    x_bound: List[float] = [-50, 50],
    y_bound: List[float] = [-50, 50],
    class_names: List[str] = None,
):
    """
    画分类概率热图，每行一个子图（每类一个）
    Args:
        cls_prob: [Hbev, Wbev, C]  sigmoid 后的概率 [0,1]
    """
    class_names = class_names or [f"cls{i}" for i in range(cls_prob.shape[-1])]
    C = cls_prob.shape[-1]
    fig, axes = plt.subplots(1, C, figsize=(6 * C, 5))
    if C == 1:
        axes = [axes]
    for c, ax in enumerate(axes):
        im = ax.imshow(
            cls_prob[:, :, c].T, origin="lower",
            extent=[x_bound[0], x_bound[1], y_bound[0], y_bound[1]],
            vmin=0, vmax=1, cmap="viridis", aspect="auto",
        )
        ax.set_title(f"BEV Heatmap - {class_names[c]}")
        ax.set_xlabel("X (front, m)")
        ax.set_ylabel("Y (left, m)")
        ax.axvline(0, color="w", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="w", linestyle="--", linewidth=0.8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def _decode_boxes_from_preds(
    cls_logits: np.ndarray,
    center_xy: np.ndarray,
    size_whl: np.ndarray,
    x_bound: List[float],
    y_bound: List[float],
    score_thr: float = 0.3,
) -> List[Dict]:
    """从检测头输出解码 bbox（含类别无关 NMS 去除重复框）"""
    prob = 1 / (1 + np.exp(-cls_logits))
    bev_h, bev_w = prob.shape[:2]
    x_cell = (x_bound[1] - x_bound[0]) / bev_h
    y_cell = (y_bound[1] - y_bound[0]) / bev_w
    boxes = []
    for c in range(prob.shape[-1]):
        idxs = np.where(prob[:, :, c] >= score_thr)
        for gx, gy in zip(*idxs):
            score = prob[gx, gy, c]
            cx_m = x_bound[0] + (gx + 0.5 + center_xy[gx, gy, 0] * 0.5) * x_cell
            cy_m = y_bound[0] + (gy + 0.5 + center_xy[gx, gy, 1] * 0.5) * y_cell
            ww, wl, _ = size_whl[gx, gy]
            boxes.append({
                "cx": cx_m, "cy": cy_m,
                "w": ww, "l": wl,
                "score": float(score),
                "class": c,
            })
    # 类别无关 NMS：同一物体上多个网格同时放电时只保留得分最高的框
    boxes = sorted(boxes, key=lambda x: -x["score"])
    keep = []
    for b in boxes:
        suppressed = False
        for k in keep:
            # 轴对齐 IoU（l 沿 x 方向，w 沿 y 方向）
            ix = max(0.0, min(b["cx"] + b["l"] / 2, k["cx"] + k["l"] / 2)
                     - max(b["cx"] - b["l"] / 2, k["cx"] - k["l"] / 2))
            iy = max(0.0, min(b["cy"] + b["w"] / 2, k["cy"] + k["w"] / 2)
                     - max(b["cy"] - b["w"] / 2, k["cy"] - k["w"] / 2))
            inter = ix * iy
            union = b["l"] * b["w"] + k["l"] * k["w"] - inter
            if union > 0 and inter / union > 0.25:
                suppressed = True
                break
        if not suppressed:
            keep.append(b)
    return keep[:30]


def visualize_detection_results(
    preds: Dict[str, np.ndarray],
    gts: Optional[Dict[str, np.ndarray]] = None,
    save_path: str = "vis_results/detection.png",
    x_bound: List[float] = [-50, 50],
    y_bound: List[float] = [-50, 50],
    class_names: List[str] = None,
    score_thr: float = 0.3,
):
    """BEV 平面上画 bbox（preds vs 可选 GT）"""
    class_names = class_names or ["car", "ped", "cyc"]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim(x_bound[0], x_bound[1])
    ax.set_ylim(y_bound[0], y_bound[1])
    ax.set_aspect("equal")
    ax.set_xlabel("X (front, m)")
    ax.set_ylabel("Y (left, m)")
    ax.set_title("BEV Detection Results")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    # 画自车
    ax.add_patch(patches.Rectangle((-2.0, -1.0), 4.0, 2.0,
                                   fill=False, color="black", linewidth=2, label="Ego"))

    # GT
    if gts is not None and "cls_heatmap" in gts:
        gt_boxes = _decode_boxes_from_preds(
            np.log(np.clip(gts["cls_heatmap"], 1e-6, 1 - 1e-6) + 1e-6),
            gts["center_xy"], gts["size_whl"],
            x_bound, y_bound, score_thr=0.5,
        )
        for b in gt_boxes:
            rect = patches.Rectangle(
                (b["cx"] - b["l"] / 2, b["cy"] - b["w"] / 2), b["l"], b["w"],
                fill=False, linestyle="--", edgecolor="black", linewidth=1.2, alpha=0.7,
            )
            ax.add_patch(rect)

    # Preds
    pred_boxes = _decode_boxes_from_preds(
        preds["cls_logits"], preds["center_xy"], preds["size_whl"],
        x_bound, y_bound, score_thr=score_thr,
    )
    for b in pred_boxes:
        c = colors[b["class"] % len(colors)]
        rect = patches.Rectangle(
            (b["cx"] - b["l"] / 2, b["cy"] - b["w"] / 2), b["l"], b["w"],
            fill=False, edgecolor=c, linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(b["cx"], b["cy"] + b["w"] / 2 + 0.5,
                f"{class_names[b['class']]}\n{b['score']:.2f}",
                ha="center", va="bottom", fontsize=8, color=c)

    # Legend
    legend_handles = [
        patches.Patch(facecolor="none", edgecolor="black", linestyle="--", label="GT"),
    ] + [patches.Patch(facecolor="none", edgecolor=colors[i], label=f"Pred {class_names[i]}") for i in range(len(class_names))]
    ax.legend(handles=legend_handles, loc="upper right")
    ax.grid(True, alpha=0.2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path, pred_boxes


def visualize_camera_projection(
    images: np.ndarray,
    pix_coords: np.ndarray,
    save_path: str = "vis_results/projection.png",
    num_cams_to_show: int = 3,
):
    """
    把一组 BEV reference point 投影到图像上的位置画出来（用来验证/理解 SCA 的投影）
    Args:
        images:     [num_cams, H, W, 3]  (0~1 float)
        pix_coords: [num_cams, N, 2]     像素坐标 (u, v)
    """
    num_cams = min(len(images), num_cams_to_show)
    fig, axes = plt.subplots(1, num_cams, figsize=(6 * num_cams, 5))
    if num_cams == 1:
        axes = [axes]
    for i in range(num_cams):
        ax = axes[i]
        img = np.clip(images[i] * 0.5 + 0.5, 0, 1)  # 从 [-1,1] 反归一化
        ax.imshow(img)
        # 投影点
        uvs = pix_coords[i]
        valid = (uvs[:, 0] >= 0) & (uvs[:, 0] < images[i].shape[1]) & \
                (uvs[:, 1] >= 0) & (uvs[:, 1] < images[i].shape[0])
        ax.scatter(uvs[valid, 0], uvs[valid, 1], s=5, c="red", alpha=0.6, label="projected refs")
        ax.set_title(f"Cam {i}: {valid.sum()} points in image")
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path
