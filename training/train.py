# ============================================================
# 训练入口脚本
# 用法: python -m training.train --config configs/bevformer_base.yaml
# ============================================================
import os
import sys
import time
import argparse
import yaml
import torch
import random
import numpy as np
from tqdm import tqdm
from typing import Dict

# 保证能找到项目根目录下的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bevformer import BEVFormer
from data.dataset import build_dataloader
from training.loss import BEVDetectionLoss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)


def train_one_epoch(
    model: BEVFormer,
    loader,
    criterion: BEVDetectionLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    print_freq: int,
) -> Dict[str, float]:
    model.train()
    meters = {"total": 0.0, "cls": 0.0, "reg": 0.0, "size": 0.0, "n": 0}
    start = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch}", ncols=100)
    for it, seq in enumerate(pbar):
        # seq[k] shape = [B, T, ...]，T 是时序长度
        B, T = seq["images"].shape[:2]
        # 把每帧送入模型，维护 prev_bev 缓存做时序
        prev_bevs = None
        prev_poses = None
        total_loss = torch.tensor(0.0, device=device)

        for t in range(T):
            images = seq["images"][:, t].to(device)           # [B, cam, 3, H, W]
            lidar2cam = seq["lidar2cam"][:, t].to(device)      # [B, cam, 4, 4]
            cam_in = seq["cam_intrinsic"][:, t].to(device)     # [B, cam, 3, 3]
            ego_pose = seq["ego_pose"][:, t].to(device)        # [B, 4, 4]
            gt = {
                "cls_heatmap": seq["cls_heatmap"][:, t].to(device),
                "center_xy":   seq["center_xy"][:, t].to(device),
                "size_whl":    seq["size_whl"][:, t].to(device),
                "cls_mask":    seq["cls_mask"][:, t].to(device),
            }

            preds, curr_bev, new_cache = model(
                images, lidar2cam, cam_in,
                ego_pose=ego_pose,
                prev_bevs=prev_bevs,
                prev_ego_poses=prev_poses,
            )
            loss_dict = criterion(preds, gt)
            total_loss = total_loss + loss_dict["total"]

            # 更新缓存
            prev_bevs = new_cache
            if prev_poses is None:
                prev_poses = [ego_pose]
            else:
                prev_poses = prev_poses + [ego_pose]
            if len(prev_poses) > model.num_history_frames:
                prev_poses = prev_poses[-model.num_history_frames:]

        # 反向传播（累积了 T 帧 loss）
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # 统计
        meters["total"] += total_loss.item() / T
        meters["cls"] += loss_dict["cls"].item()
        meters["reg"] += loss_dict["reg"].item()
        meters["size"] += loss_dict["size"].item()
        meters["n"] += 1

        if (it + 1) % print_freq == 0:
            n = meters["n"]
            pbar.set_postfix({
                "total": f"{meters['total'] / n:.3f}",
                "cls": f"{meters['cls'] / n:.3f}",
                "reg": f"{meters['reg'] / n:.3f}",
                "sz": f"{meters['size'] / n:.3f}",
            })

    n = meters.pop("n")
    for k in meters:
        meters[k] /= n
    meters["time"] = time.time() - start
    return meters


def run_training(config_path: str):
    # 1. 读取配置
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    work_dir = cfg.get("work_dir", "./work_dirs/bevformer_base")
    os.makedirs(work_dir, exist_ok=True)

    # 2. 构建模型
    model = BEVFormer.from_config(config_path).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] 模型参数量: {total_params:.2f} M")

    # 3. 数据加载
    loader = build_dataloader(cfg, split="train")
    print(f"[INFO] 训练集 samples: {len(loader.dataset)}, iters per epoch: {len(loader)}")

    # 4. 优化器和学习率调度
    train_cfg = cfg.get("train", {})
    criterion = BEVDetectionLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    sched_cfg = train_cfg.get("lr_scheduler", {"type": "CosineAnnealingLR", "T_max": 20})
    if sched_cfg["type"] == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sched_cfg.get("T_max", train_cfg.get("epochs", 20)),
        )
    else:
        scheduler = None

    # 5. 训练循环
    epochs = train_cfg.get("epochs", 20)
    print_freq = cfg.get("print_freq", 10)
    save_freq = cfg.get("save_freq", 5)
    log_file = os.path.join(work_dir, "train_log.txt")
    best_total = float("inf")

    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(model, loader, criterion, optimizer, device, epoch, print_freq)
        if scheduler is not None:
            scheduler.step()

        msg = (f"[Epoch {epoch}/{epochs}] total={metrics['total']:.4f} "
               f"cls={metrics['cls']:.4f} reg={metrics['reg']:.4f} "
               f"size={metrics['size']:.4f} time={metrics['time']:.1f}s")
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        if epoch % save_freq == 0 or metrics["total"] < best_total:
            save_checkpoint({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": metrics,
                "cfg": cfg,
            }, os.path.join(work_dir, f"epoch_{epoch}.pth" if epoch % save_freq == 0 else os.path.join(work_dir, "best.pth")))
            best_total = min(best_total, metrics["total"])

    print("[DONE] 训练完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEVFormer Training")
    parser.add_argument("--config", type=str, default="configs/bevformer_base.yaml")
    args = parser.parse_args()
    run_training(args.config)
