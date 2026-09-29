# 核心概念 3/3：时间融合 & 时间自注意力 (TSA) 深度解析

> 对应源码文件：`models/temporal_attn.py` 中的 `TemporalSelfAttention` 和 `warp_history_bev`
> 对应 mmdet3d 源码：
>   - `mmdet3d/models/bevformer/bevformer_head.py` 中的 `TemporalSelfAttention`
>   - 关键函数 `get_reference_points()`、`get_history_bev()`、`align_memory()`

---

## 一、为什么 BEV 需要时间融合？

单帧图像有三个根本局限：
1. **遮挡问题**：前车挡住了后车，单帧看不到（但前几帧可能看到过后车）
2. **深度歧义**：单目图像无法唯一确定深度（近大远小 / 远大近小都可能）
3. **检测抖动**：单帧检测 bbox 每一帧轻微跳动，导致下游规划不可靠

**时间融合就是用"前几帧看到过什么"来帮助当前帧做出更好的判断。**

在自动驾驶场景里，BEVFormer 用了 2~4 帧历史（大约 0.5~1 秒），效果显著：
- NuScenes NDS 指标提升 3~5 个点
- 远处小目标召回率提升明显
- 遮挡目标重识别率大幅提升

---

## 二、时间融合的关键：先对齐，再融合

这是 TSA 最容易被忽略但最重要的一步：**自车在移动，BEV 网格在"世界坐标系"中的实际位置变了！**

```
t-1 时刻自车位置              t 时刻自车位置
    ↑(+x 前)                      ↑(+x 前)
    │                             │
  ──┼── 自车(旧)                ──┼── 自车(新，向前开了 Δx)
    │                             │
旧 BEV 网格原点              新 BEV 网格原点

→ 旧 BEV 网格中的 (x=10m, y=0)，在"世界"里其实是新 BEV 的 (x=10-Δx, y=0)
→ 如果直接相加，相当于把"同一物体"错位相加，特征就会糊掉！
```

### 2.1 坐标对齐（Warp）的数学推导

我们有：
- `ego_pose_curr`：当前帧自车 → 世界 的变换矩阵（4×4 SE(3)）
- `ego_pose_prev`：历史帧自车 → 世界 的变换矩阵

对于同一个真实世界点 `P_world`：
```
P_world = ego_pose_curr @ P_curr_ego
P_world = ego_pose_prev @ P_prev_ego
```

所以 `P_curr_ego`（当前自车坐标系下的点）和 `P_prev_ego`（历史自车坐标系下的点）关系：
```
ego_pose_curr @ P_curr_ego = ego_pose_prev @ P_prev_ego
        P_prev_ego = ego_pose_prev^{-1} @ ego_pose_curr @ P_curr_ego
        P_prev_ego = T_rel @ P_curr_ego
```

**也就是：想知道"当前 BEV 网格 (x, y) 在历史帧自车坐标系下对应的 (x', y')"，
只需乘上 `T_rel = inv(ego_pose_prev) @ ego_pose_curr`**。

实现这个逻辑的就是 `warp_history_bev()`（本项目）或 mmdet3d 中的
`align_memory()` + `bev_align()`。

### 2.2 Warp 用的技术：双线性重采样

```python
# 对于新 BEV 的每个 (gx, gy)，算出它在"历史 BEV 特征图"上应该取哪个位置
sample_grid = (归一化后的 (x', y')) 
# 然后用 F.grid_sample 对历史特征图做双线性采样
warped_feat = F.grid_sample(history_bev, sample_grid)
```

注意：对齐后会出现"边缘区域"历史帧没有覆盖过（自车新进入的区域），
这些地方的特征是 0，TSA 自然就不会注意它们——这符合物理直觉。

---

## 三、时间自注意力（TSA）本体

对齐之后，注意力就非常直观了：

```
K/V 序列 = [ curr_bev_flatten | warped_bev_-1_flatten | warped_bev_-2_flatten | ... ]
  长度 = (1 + num_history) × (H×W)

Q 序列 = curr_bev_flatten（当前帧的每个 Query）
  长度 = H×W
```

每个 Query（BEV 位置）会问：
- "我当前自己是什么特征"（self-attention 部分）
- "我在 0.5s 前、1.0s 前的对应位置是什么特征"（temporal 部分）

这种设计比"简单平均"或"卷积 LSTM"更灵活：
- 如果当前帧有遮挡，Q 会把权重更多地放历史帧上
- 如果当前帧清晰（比如第一次看到新物体），Q 就更多取自己
- 模型可以学到"时间上的平滑先验"（车不应该一帧瞬移 10 米）

### 3.1 时间步编码（Temporal Embedding）

```python
# 对应我们代码中的 temporal_embed
temporal_embed = nn.Parameter(shape=[num_history_frames+1, C])
kv_seq = kv_seq + temporal_embed[time_step_id]
```

为什么要加这个？因为：
- 没有时间步编码的话，模型不知道哪个 token 是当前帧，哪个是 0.5s 前
- 加了之后，模型能学到"越旧的帧权重应该越低"等时间衰减规律
- 类似 Transformer 的位置编码，但作用在"时间维度"而不是"空间维度"

---

## 四、TSA 完整流程图

```
  t-2 帧 BEV ─────┐
                   │
  t-1 帧 BEV ─────┤
                   ▼
              ┌─────────┐
              │  Warp   │ ←─ 根据 ego_pose_{t-i} 和 ego_pose_t 做坐标对齐
              └─────────┘
                   │
  当前 t BEV ──┐   │
              ▼   ▼
        ┌────────────────────────┐
        │ Concat along N 维度    │ → K/V 序列：[t | t-1 | t-2] BEV 特征
        └────────────────────────┘
                   │
                   ▼
        ┌────────────────────────┐
        │ + Temporal Embedding   │ → 给不同时间步加可区分标记
        └────────────────────────┘
                   │
                   ▼
        ┌────────────────────────┐
        │ Multi-Head Self Attn   │ ← Q = 当前 BEV (加 pos_enc)
        └────────────────────────┘
                   │
                   ▼
             融合后的 BEV 特征 (送入 SCA)
```

---

## 五、对照 mmdet3d 源码

mmdet3d 中 TSA 的实际位置：
```python
# mmdet3d/models/bevformer/bevformer_head.py → BEVFormerHead.point_attention()
# 时序对齐在 forward() 函数里做：
def forward(self, ..., bev_queue, metas):
    ...
    for i, past_bev in enumerate(bev_queue):
        # 计算相对变换 T_rel = T_curr_from_past
        prev_bev = bev_align(past_bev, T_rel, ...)
        ...
    # 然后把 prev_bev 和 prev_bev_mask 传给 TSA
    ...

# 注意力实现：mmdet3d/models/bevformer/attention.py → TemporalSelfAttention
# 原版是用 MSDeformAttn 实现的（多采样点），我们用标准 MHA 更容易理解
```

**核心流程：Warp 对齐 → 拼 K/V 序列 → 做多头注意力 → 输出**
本项目和 mmdet3d 原版完全一致，只是把 Deformable 换成了标准 MHA，更干净。

---

## 六、如何验证 TSA 是否真的工作？

训练时建议做 2 组 ablation（消融实验）：
| 实验 | 配置 | 预期效果 |
|-----|------|---------|
| Baseline | 启用 TSA | 分类 loss 下降最快 |
| No TSA | `num_history_frames=0` | 收敛慢 1~2 epoch，最终 loss 高 5~10% |
| No Warp | 注释掉 `warp_history_bev`（直接对齐但不做 warp）| 时序融合反而会有负效果，loss 上升 |

关键结论：**Warp 对齐是时序融合的前提，没有对齐还不如不用时序！**

---

## 七、TSA 的内存优化工程细节（进阶）

mmdet3d 原版中 BEV 缓存有两个重要技巧（本项目已实现简化版）：
1. **缓存 detach**：历史 BEV 特征是 `.detach()` 的（不算梯度），防止显存随着时间爆炸
   ```python
   cache_list = list(prev_bevs) + [curr_bev.detach()]
   ```
2. **缓存上限**：只保留最近 `num_history_frames` 帧（通常 3 帧），多余的 pop

真正跑长视频流时，你还会用到：
- **Queues**：每个样本维护独立 queue（分布式训练每个 rank 独立）
- **Metric 重置**：每个 epoch 开始清空 queue（避免跨 sample 污染）

到这里，BEVFormer 的三大核心（BEV Query / SCA / TSA）就全部打通了。
把这三个机制组合起来，就构成了完整的 BEVFormer Encoder！
