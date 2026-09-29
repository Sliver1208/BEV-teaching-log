# 核心概念 1/3：BEV Query 深度解析

> 对应源码文件：`models/bev_query.py` 中的 `BEVQueryGenerator` 类
> 对应 mmdet3d 源码：`mmdet3d/models/bevformer/bevformer_head.py` 中的
> `BEVFormerHead._init_bev_embedding()` 和 `bev_embedding` / `bev_pos` 参数

---

## 一、BEV Query 是什么？一句话理解

> BEV Query = **鸟瞰平面上每个"格子"的"问题"**，每个 Query 都会通过注意力去问图像：
> "我的这个格子对应的 3D 空间里，有什么物体？"

BEVFormer 把 BEV 空间（比如前后 50m × 左右 50m）离散化成 50 × 50 = 2500 个网格，
每个网格对应一个 Query（向量维度 256）。

这是一种 **Dense Query**（密集查询），和 DETR 里 100/300 个 **Sparse Query**（稀疏查询，直接对应最终 bbox）有本质区别：

| 类型 | Query 数量 | 含义 | 输出方式 |
|------|-----------|------|---------|
| DETR Query | 100~900 | 每个 Query 直接预测一个 bbox | Query 本身就是预测 |
| BEV Query | H×W（如 2500）| 每个 Query 表示 BEV 平面一个位置的特征 | 再送检测头解码 bbox |

---

## 二、为什么要用 Query 的形式？不用 "CNN 卷积 BEV 特征图"？

CNN 也可以生成 BEV 特征图（比如 Lift-Splat、VPN、VAD 早期版本），但它们的问题：
1. **只能处理单尺度/固定视角**：卷积核感受野受限，难以"看"远处的物体
2. **多相机融合不灵活**：6 个相机的信息只能在早期拼接
3. **时序融合麻烦**：历史帧难以和当前帧做像素级对齐

而 Query + Transformer 的形式天然支持：
- Query 可以去任何位置"看"（通过 SCA 投影到任意相机图像上采样）
- Query 本身有"位置语义"，可以把参考点投到图像对应位置
- 时序上 Query 可以互相问："我前一帧在做什么"（TSA）

---

## 三、两种实现方式：Learnable vs Geometry

### 方式 1：Learnable BEV Query（mmdet3d 原版默认）

```python
# 对应我们代码中的写法
self.bev_embedding = nn.Parameter(torch.Tensor(1, H*W, C))
nn.init.xavier_uniform_(self.bev_embedding)
```

- 把 2500 个向量当作 `nn.Parameter` 随机初始化
- 训练中通过反向传播学到"前面的 Query 偏向关注车辆，侧面的 Query 偏向关注行人"等先验
- **优点**：端到端训练，最终性能通常略好
- **缺点**：可解释性差，完全是黑盒

### 方式 2：Geometry-based BEV Query（教学实现）

```python
# 对应我们代码中的 geometry 分支
init_feat = position_embedding(grid_coords, num_feats_per_axis)
bev_queries = self.geom_proj(init_feat)
```

- 每个 Query 的初始值 = 其 BEV 网格的 (x, y, z) 坐标经过正弦编码后，再过一个 MLP
- 初始化时就有"我在空间哪个位置"的明确语义
- **优点**：可解释性强，适合理解和 debug
- **缺点**：性能上略低于可学习版本（训练后也能收敛）

---

## 四、BEV Positional Encoding 的作用

Query 本身还会加上一个额外的位置编码 `bev_pos_enc`：
```python
q_with_pos = bev_queries + bev_pos_enc  # 做注意力前会加
```

为什么 Query 本身已经有位置信息了还要再加一层位置编码？

- `bev_embedding`：学习到的"语义先验"（可训练）
- `bev_pos_enc`：提供给注意力的"坐标先验"（固定或半固定）
  - 作用于 Q 和 K，让注意力机制知道两个 Query 的相对位置关系
  - 类似 Transformer 中 `src + pos_embed` 的标准用法

---

## 五、对照 mmdet3d 源码的阅读指引

在 mmdet3d 中，BEV Query 相关的关键位置：

```python
# 1. 参数初始化：mmdet3d/models/bevformer/bevformer_head.py
class BEVFormerHead(nn.Module):
    def __init__(self, ...):
        # 这两个就是我们的 bev_queries（可学习参数）+ 位置编码
        self.bev_embedding = nn.Embedding(bev_h*bev_w, embed_dims)
        self.bev_pos = nn.Embedding(bev_h*bev_w, embed_dims)

    # 2. 实际取出时的逻辑
    def forward(...):
        bev_queries = self.bev_embedding.weight.unsqueeze(0).expand(B, -1, -1)
        bev_pos = self.bev_pos.weight.unsqueeze(0).expand(B, -1, -1)
        ...
```

注意：mmdet3d 原版把 H×W 的 index 直接当 `nn.Embedding` 输入，而我们用
`nn.Parameter(shape=[1, H*W, C])` 再 expand 到 batch 维度——
**本质完全等价**，只是写法不同。

---

## 六、调试小技巧（理解 BEV Query）

运行推理 demo 时，你可以：
1. 打印 `bev_queries.shape`，确认是 `[B, 2500, 256]`
2. 用 `grid_index_to_xy(1234)` 查第 1234 号 Query 对应哪个 BEV 网格
3. 计算两个相邻 Query 的 L2 距离：同一行相邻的距离应小于跨行列的距离（证明位置编码起作用）
