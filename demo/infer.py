# ============================================================
# 推理 Demo：单文件跑通 end-to-end
# 用法: python -m demo.infer --config configs/bevformer_base.yaml --seq-len 4
# ============================================================
import os
import sys
import argparse
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bevformer import BEVFormer
from data.dataset import SyntheticDataset
from demo.visualize import (
    visualize_bev_heatmap,
    visualize_detection_results,
    visualize_camera_projection,
)
from data.utils import project_bev_to_image, get_reference_points_in_bev


def run_inference_demo(config_path: str, seq_len: int = 4, save_dir: str = "./vis_results"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] 使用设备: {device}")

    # 1. 模型（随机初始化权重，仅演示推理逻辑）
    model = BEVFormer.from_config(config_path).eval().to(device)
    # 加载训练好的权重（存在才加载，避免 checkpoint 缺失时崩溃）
    ckpt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "work_dirs", "bevformer_base", "epoch_30.pth",
    )
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[INFO] 已加载 epoch_{ckpt['epoch']} 权重")
    else:
        print(f"[WARN] 未找到权重 {ckpt_path}，使用随机初始化（此时 heatmap 无意义属正常）")
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] 模型参数量: {total_params:.2f} M")

    # 2. 取一组合成数据（Batch=1）
    ds = SyntheticDataset(
        num_samples=10,
        num_cams=cfg["image"]["num_cams"],
        img_h=cfg["image"]["img_h"],
        img_w=cfg["image"]["img_w"],
        bev_h=cfg["bev"]["H"],
        bev_w=cfg["bev"]["W"],
        sequence_length=seq_len,
        num_classes=cfg["model"]["head"]["num_classes"],
    )
    seq = ds[0]
    # 加 batch 维度
    for k in seq:
        seq[k] = seq[k].unsqueeze(0).to(device)

    B, T = seq["images"].shape[:2]
    num_cam = cfg["image"]["num_cams"]
    class_names = ["car", "pedestrian", "cyclist"]
    x_bound = cfg["bev"]["x_bound"]
    y_bound = cfg["bev"]["y_bound"]
    z_bound = cfg["bev"]["z_bound"]

    # 3. 时序推理
    prev_bevs = None
    prev_poses = None
    with torch.no_grad():
        for t in range(T):
            images = seq["images"][:, t]
            lidar2cam = seq["lidar2cam"][:, t]
            cam_in = seq["cam_intrinsic"][:, t]
            ego_pose = seq["ego_pose"][:, t]
            gt = {
                "cls_heatmap": seq["cls_heatmap"][:, t].cpu().numpy()[0],
                "center_xy":   seq["center_xy"][:, t].cpu().numpy()[0],
                "size_whl":    seq["size_whl"][:, t].cpu().numpy()[0],
                "cls_mask":    seq["cls_mask"][:, t].cpu().numpy()[0],
            }

            preds, curr_bev, new_cache = model(
                images, lidar2cam, cam_in,
                ego_pose=ego_pose,
                prev_bevs=prev_bevs,
                prev_ego_poses=prev_poses,
            )

            # 更新缓存
            prev_bevs = new_cache
            if prev_poses is None:
                prev_poses = [ego_pose]
            else:
                prev_poses.append(ego_pose)
            if len(prev_poses) > model.num_history_frames:
                prev_poses = prev_poses[-model.num_history_frames:]

            # 4. 可视化当前帧结果
            print(f"  [Frame {t}] preds cls_logits shape: {preds['cls_logits'].shape}")
            preds_np = {k: v.cpu().numpy()[0] for k, v in preds.items()}

            # 4a. 热图（sigmoid 后）
            prob = 1 / (1 + np.exp(-preds_np["cls_logits"]))
            hm_path = visualize_bev_heatmap(
                prob, f"{save_dir}/frame{t}_heatmap.png",
                x_bound, y_bound, class_names,
            )
            print(f"       保存热图: {hm_path}")

            # 4b. 检测结果（bbox）
            det_path, det_boxes = visualize_detection_results(
                preds_np, gt,
                f"{save_dir}/frame{t}_detection.png",
                x_bound, y_bound, class_names, score_thr=0.3,
            )
            print(f"       保存检测图: {det_path}  (detected {len(det_boxes)} boxes)")

            # 4c. 仅第 0 帧可视化一次投影（理解 SCA 的参考点投影）
            if t == 0:
                ref_world_norm = get_reference_points_in_bev(
                    cfg["bev"]["H"], cfg["bev"]["W"], cfg["bev"]["Z"],
                    batch_size=1, device=device,
                )
                # 归一化 -> 物理
                x_min, x_max = x_bound
                y_min, y_max = y_bound
                z_min, z_max = z_bound
                ref_world = ref_world_norm.clone()
                ref_world[..., 0] = ref_world_norm[..., 0] * (x_max - x_min) + x_min
                ref_world[..., 1] = ref_world_norm[..., 1] * (y_max - y_min) + y_min
                ref_world[..., 2] = ref_world_norm[..., 2] * (z_max - z_min) + z_min
                pix, _ = project_bev_to_image(ref_world, lidar2cam, cam_in)
                # [B, cam, N, 2] -> [cam, N, 2]
                pix_np = pix.cpu().numpy()[0]
                # images: [B, cam, 3, H, W] -> [cam, H, W, 3]
                imgs_np = images.cpu().numpy()[0].transpose(0, 2, 3, 1)
                proj_path = visualize_camera_projection(
                    imgs_np, pix_np, f"{save_dir}/frame0_projection.png",
                )
                print(f"       保存投影可视化: {proj_path}")

    print(f"\n[DONE] 推理完成，所有可视化结果保存在: {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bevformer_base.yaml")
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="./vis_results")
    args = parser.parse_args()
    run_inference_demo(args.config, args.seq_len, args.save_dir)
