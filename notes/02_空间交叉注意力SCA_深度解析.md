# 核心概念 2/3：空间交叉注意力 (SCA) 深度解析

> 对应源码文件：`models/spatial_attn.py` 中的 `SpatialCrossAttention` 类
> 对应 mmdet3d 源码：
>   - `mmdet3d/models/bevformer/attention.py` 中的 `SpatialCrossAttention`
>   - 关键函数 `point_sampling()`、`get_reference_points()`、`attention()`

---

## 一、先理解"视角转换"问题（为什么需要 SCA？）

自动驾驶中我们有 6 个环绕相机的图像（前/前左/前右/后/后左/后右），
每个像素是 2D 的，我们要把它变成 3D BEV（鸟瞰）表示。

传统做法的两条路：
- **A. 深度估计 + 反投影**：先预测每个像素的深度 d，然后
  `P_cam = d · K^{-1} [u,v,1]^T`，再投到 BEV 网格。代表：Lift-Splat, BEVDet
  - 缺点：显式深度不准，误差直接累计；算力大（H×W 个像素都投）
- **B. Transformer 全注意力**：让 BEV Query 和所有 H×W 像素做全 attention
  - 缺点：计算量 = N_query × H×W × num_cam，太大不可行

**BEVFormer SCA 的天才思路：用相机几何"指导"注意力，只看必要的像素。**

> 每个 BEV Query 只看：
> 1. 它对应的 3D reference point 投影到每个相机图像上的位置
> 2. 该位置周围 K=4 个可学习偏移点
> 
> 总计算量：N_query × num_cam × K = 2500 × 6 × 4 = 60,000
> 对比全注意力：2500 × 6 × (256×256) ≈ 1 亿（差 3 个数量级！）

---

## 二、SCA 五步详解（对照代码）

### Step 1：生成每个 BEV Query 的 3D reference point

```python
# get_reference_points_in_bev()
# 在 BEV 空间生成 H×W×Z 个 3D 锚点（归一化坐标 [0,1]）
ref_norm = [
  [(0.5/50, 0.5/50, 0.5),    # (h=0, w=0, z=0)
   (0.5/50, 1.5/50, 0.5),    # (h=0, w=1, z=0)
   ...],
  ...
]
```

然后反归一化到真实物理坐标（米）：
```python
# 物理 x = (norm_x) * (x_max-x_min) + x_min
ref_world = denormalize_bev_to_world(ref_norm, [-50,50], [-50,50], [-10,10])
```

### Step 2：把 3D 点投影到 6 个相机图像上

这是 SCA 的几何基础：
```
      自车坐标系 (x, y, z)              相机坐标系 (Xc, Yc, Zc)          像素 (u, v)
    ┌───────────────────┐    lidar2cam   ┌───────────────────┐      K    ┌─────────────┐
    │  ref_world (x,y,z)│ ─────────────▶ │ P_cam = R·x + t   │ ────────▶ │ u, v ∈ 图像 │
    └───────────────────┘                 └───────────────────┘           └─────────────┘
```

```python
# 1) 齐次坐标 P_world -> P_cam
p_cam = lidar2cam @ ref_world_h
# 2) 透视投影 P_cam -> pixel
z_cam = p_cam[..., 2]
xy1 = [Xc/Zc, Yc/Zc, 1]
pix_coord = K @ xy1
```

注意：投影后有些点在相机后方（z_cam < 0）或图像外（u<0 或 u>W），这些会被 mask 掉。
这就是为什么 **SCA 天然会"丢弃"看不到的相机**——只有当前相机能看到的 reference point 才真正参与注意力。

### Step 3：学习采样偏移 offsets 和注意力权重 weights

```python
offsets = self.sampling_offsets(q_with_pos)  # [B, H, N, P, 2]
weights = F.softmax(self.attn_weights(q_with_pos))  # [B, H, N, P]
```

- **offsets**：每个 head 每个 Query 每个 sampling point 学一个偏移（像素级），
  初始是 0，训练后会学会看 reference point 周围最合适的 4 个位置（比如车的四个角）。
- **weights**：这 K=4 个采样点的相对重要性，softmax 后 sum=1。

### Step 4：多相机多采样点双线性采样（Bilinear Sampling）

```python
# F.grid_sample 是 PyTorch 标准的可微采样
sampled_feat = F.grid_sample(feat_map, pix_sample_grid)
```

对于每一个 query，6 个相机 × 4 个点 = 24 个采样点。
每个采样点的特征是它周围 4 个像素的双线性插值——全流程可微。

### Step 5：注意力加权 + 输出投影

```python
# weights: [B, H, N, P, 1]
# sampled: [B, H, N, P, d]
output = (weights * sampled).sum(dim=3)  # P 维度加权求和
output = out_proj(output.flatten_head()) # 合并 head + 线性变换
```

---

## 三、和 mmdet3d 原版的差异

| 项目 | 本学习版 | mmdet3d 原版 |
|------|---------|-------------|
| 图像特征 | 单尺度（stage4） | 多尺度（FPN P3~P5，共 4 层） |
| 采样实现 | 标准 F.grid_sample + offsets | **MS Deformable Attention** 自定义 CUDA 算子 |
| 偏移学习 | 只在 2D 像素平面学偏移 | 在 (level, u, v) 三维空间学偏移（可跨尺度） |
| Key 计算 | 直接对整张图做 1×1 Conv | 在采样点处做 projection（显存更省） |

核心数学原理 **完全一致**。原版用 Deformable Attention 主要是工程优化（速度/显存），
不影响你理解"基于相机几何做稀疏采样注意力"的核心思想。

---

## 四、可视化验证（推荐跑一下 Demo 看）

运行推理脚本时，第一张图会生成 `frame0_projection.png`：
```
每个相机上的红点 = BEV reference point 投影到该相机上的位置
```

你会观察到：
1. 前方两个相机（cam0, cam1）红点集中在图像中央（前向视野）
2. 后方相机红点集中在不同区域
3. 约 1/6 的红点会落在某个相机图像里（其余被 mask）

这就是 SCA 几何指导效果的直观体现。

---

## 五、SCA 让模型学到了什么？（训练后分析）

训练完毕后，可以分析：
1. **offsets 的分布**：车类别对应的 Query，offsets 会扩散到车宽/车长
   （相当于 Query 看的是车的四个角，而不只是中心点）
2. **哪个相机被"看"得最多**：前方 Query 看前方相机权重最大
3. **多尺度的作用**：原版用多尺度后，小物体看高分辨率层，大物体看低分辨率层

这就是 SCA 的强大之处：**把相机几何先验和可学习注意力完美结合，既高效又有几何可解释性。**
