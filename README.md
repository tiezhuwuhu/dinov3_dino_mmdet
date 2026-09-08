# Point-DINO Stage 3：像素空间欧氏距离定位损失与匈牙利匹配几何权重消融

> 状态：Stage 3 已完成。  
> 数据集：ShanghaiTech Part B。  
> 输入协议：统一使用原始分辨率 1024×768。  
> 正式超参数消融：**8 × 8 = 64 组实验**。  
> 所有正式结果统一使用 **`point/f1@8px` 最优 checkpoint** 进行测试与汇总。

---

## 1. Stage 3 的目标

Stage 2 已经完成了 Point-DINO 从原始 DINO 检测框回归到纯二维点回归的结构改造，包括：

- decoder regression branch 从 4D `(cx, cy, w, h)` 改为 2D `(x, y)`；
- reference point 改为二维点；
- encoder proposal 改为二维点；
- Hungarian matching 改为点匹配；
- 保留分类分支；
- 加入 Point DN；
- 去掉 bbox width / height、IoU、GIoU 等不再适用于纯点定位任务的监督。

Stage 3 不再继续改变 Point-DINO 的基本预测范式，而是集中解决一个问题：

> **如何进一步提高点的精确定位能力，尤其是 4px 严格阈值下的定位性能。**

Stage 3 主要从两个不同环节进行改进：

1. **提高 Hungarian matching 中 PointL1Cost 的权重**，让 query-GT 分配阶段更加重视空间位置；
2. **增加像素空间 Euclidean localization loss**，让已经匹配成功的 query 在训练时直接优化真实像素距离。

这两个参数作用位置不同：

```text
PointL1Cost
    ↓
Hungarian assignment
    ↓
决定“哪个 query 匹配哪个 GT”

Euclidean λ
    ↓
matched-query regression
    ↓
决定“匹配之后这个 query 应该多强地向 GT 靠近”
```

因此二者不是同一个东西，也不是简单重复的 loss，而是分别控制：

```text
匹配质量
+
匹配后的定位精度
```

---

## 2. Stage 3 的统一实验协议

### 2.1 数据集

ShanghaiTech Crowd Counting Dataset Part B。

```text
训练图像：400
测试图像：316
```

任务为单类别点定位。

---

### 2.2 输入尺寸

Stage 3 所有正式实验统一采用：

```text
1024 × 768
```

即 ShanghaiTech Part B 原始图像分辨率。

此前部分实验沿用了原始 DINO 测试协议：

```python
Resize(
    scale=(1333, 800),
    keep_ratio=True
)
```

对于 1024×768 的 Part B 图像，该操作会将模型实际输入变为约：

```text
1067 × 800
```

虽然最终预测会通过 `scale_factor` 映射回原图坐标，但推理阶段模型实际看到的并不是原始分辨率。

因此 Stage 3 开始统一使用 native-resolution 协议：

```text
训练输入：1024×768
验证输入：1024×768
测试输入：1024×768
评测坐标：原始 1024×768 坐标系
```

正式 Stage 3 结果不再与旧的约 1067×800 输入实验混用。

---

## 3. Stage 3 固定设置

除 λ 和 PointL1Cost 外，64 组正式实验全部保持一致：

```text
num_queries              = 900
Point DN                 = True
point_noise_scale        = 0.01
FocalLossCost            = 2.0
score_threshold          = 0.5
distance_thresholds      = [4px, 8px]
训练轮数                 = 12 epochs
best checkpoint 保存指标 = point/f1@8px
输入                     = 1024×768
```

Stage 3 的两个消融变量：

```text
Euclidean λ:
0.00
0.05
0.10
0.20
0.30
0.40
0.50
0.60
```

以及：

```text
PointL1Cost:
5
10
15
20
25
30
35
40
```

总共：

```text
8 × 8 = 64 组正式实验
```

---

# 4. Stage 3 的代码改动

## 4.1 Hungarian matching 保持为点匹配

Point-DINO 的 Hungarian matcher 仍然由两部分组成：

```python
match_costs = [
    dict(
        type='FocalLossCost',
        weight=2.0),
    dict(
        type='PointL1Cost',
        weight=COST)
]
```

其中：

```text
FocalLossCost
```

负责分类匹配，

```text
PointL1Cost
```

负责空间位置匹配。

需要特别强调：

> `PointL1Cost` 是 Hungarian assignment 阶段的 **matching cost 权重**，不是训练时 `loss_point` 的 loss weight。

Stage 3 只改变这个匹配代价的权重：

```text
5 → 10 → 15 → 20 → 25 → 30 → 35 → 40
```

---

## 4.2 不把 Euclidean distance 加入 matcher

Stage 3 中没有把新的 Euclidean loss 加入 Hungarian matcher。

即 matcher 始终保持：

```text
FocalLossCost
+
PointL1Cost
```

Euclidean loss 只作用于：

```text
Hungarian 已经匹配成功的 positive queries
```

因此实验能够清楚区分：

```text
matching geometry
```

与：

```text
regression geometry
```

的作用。

---

## 4.3 新增像素空间 Euclidean localization loss

新增函数：

```python
_loss_point_euclidean(...)
```

预测点和 GT target 在网络内部均为 normalized 坐标：

```text
pred = (x_pred, y_pred)
gt   = (x_gt, y_gt)
```

范围：

```text
[0, 1]
```

首先根据当前图像大小将 normalized error 转换为真实像素误差。

对于：

```text
W = image width
H = image height
```

定义：

```text
dx_pixel = (x_pred - x_gt) × W
dy_pixel = (y_pred - y_gt) × H
```

然后计算二维欧氏距离：

```text
distance =
sqrt(
    dx_pixel²
    +
    dy_pixel²
    +
    eps
)
```

其中：

```text
eps = 1e-6
```

用于避免数值不稳定。

---

## 4.4 只对 positive matched queries 计算 Euclidean loss

Point-DINO Hungarian matching 后：

```text
positive query
```

表示已经和 GT 匹配的 query。

因此 Euclidean loss 只计算：

```python
positive_weights = point_weights[..., 0]
```

即：

```text
matched query → 参与 Euclidean loss
negative query → 不参与 Euclidean regression
```

---

## 4.5 采用 8px 作为 Euclidean loss 的归一化尺度

单个点的像素欧氏距离为：

```text
d_pixel
```

Stage 3 将其除以：

```text
8
```

因此：

```text
L_euc_raw = d_pixel / 8
```

原因是当前 localization benchmark 的主要阈值之一就是：

```text
8 px
```

这样 Euclidean loss 的数值尺度更加稳定。

---

## 4.6 Euclidean λ 的作用

Stage 3 使用：

```text
λ
```

控制 Euclidean loss 的强度。

因此总体定位监督可以理解为：

```text
原有 Point L1 loss
+
λ × Pixel Euclidean loss
```

注意：

> Euclidean loss 是附加项，并没有替代原有 Point L1 loss。

---

## 4.7 Euclidean loss 接入所有相关层

Euclidean localization supervision 不只作用于最终 decoder layer。

它同时接入：

```text
最终 decoder layer
所有 auxiliary decoder layers
encoder proposal branch
Point DN 最终层
Point DN auxiliary decoder layers
```

因此训练日志中可以看到：

```text
loss_point_euclidean

d0.loss_point_euclidean
d1.loss_point_euclidean
d2.loss_point_euclidean
d3.loss_point_euclidean
d4.loss_point_euclidean

enc_loss_point_euclidean

dn_loss_point_euclidean

d0.dn_loss_point_euclidean
d1.dn_loss_point_euclidean
d2.dn_loss_point_euclidean
d3.dn_loss_point_euclidean
d4.dn_loss_point_euclidean
```

这保证 Point-DINO 的：

```text
matching branch
encoder branch
decoder branch
DN branch
```

都受到一致的二维像素空间定位监督。

---

# 5. Stage 3 修改后的冒烟测试

在正式训练前完成了以下检查：

```text
1. forward 正常
2. loss_by_feat 正常
3. DN loss 正常
4. Euclidean loss keys 正常出现
5. regression gradients finite = True
6. regression gradients nonzero = True
7. DN embedding gradients nonzero = True
8. native-resolution validation 正常
```

曾出现过的两个实现问题也已经修复：

### 问题 1

```text
ValueError:
not enough values to unpack
(expected 3, got 2)
```

原因是 Euclidean loss 初版只支持：

```text
[B, Q, 2]
```

但实际部分调用会传入：

```text
[B*Q, 2]
```

之后增加了对两种 shape 的兼容处理。

### 问题 2

```text
NameError:
name 'point_preds' is not defined
```

原因是在 DN loss 中误用了变量名 `point_preds`，而实际变量为：

```text
dn_point_preds
```

修正后 Point DN backward test 正常通过。

---

# 6. Stage 2 原图协议下的基线结果

Stage 3 开始前，先重新在 native 1024×768 输入协议下复现 Stage 2。

| 设置 | P@4 | R@4 | F1@4 | MLE@4 | P@8 | R@8 | F1@8 | MLE@8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pure 2D + No DN | 56.21 | 53.00 | 54.56 | 2.2831 | 83.25 | 78.50 | 80.80 | 3.3504 |
| DN=0.05 | 57.56 | 54.97 | 56.23 | 2.2678 | 83.25 | 79.51 | 81.34 | 3.2917 |
| DN=0.005 | 61.65 | 57.89 | 59.71 | 2.2020 | 85.04 | 79.86 | 82.36 | 3.1207 |
| DN=0.01 | 60.76 | 57.38 | 59.02 | 2.2300 | 85.05 | 80.32 | 82.61 | 3.1842 |

Stage 3 正式固定：

```text
Point DN noise = 0.01
```

---

# 7. 正式 8×8 消融之前的探索实验

正式 64 组 grid 之前，先进行了少量单独实验，用于判断两个参数是否值得大规模搜索。

## 7.1 λ=0 时 PointL1Cost 探索

| PointL1Cost | F1@4 | F1@8 | MLE@4 | MLE@8 |
|---:|---:|---:|---:|---:|
| 5 | 59.02 | 82.61 | 2.2300 | 3.1842 |
| 10 | 60.05 | 83.23 | 2.2081 | 3.1386 |
| 15 | 60.40 | 83.40 | 2.2192 | 3.1430 |
| 20 | 60.49 | 83.55 | 2.2037 | 3.1183 |

说明：

```text
提高 PointL1Cost 是有效的
```

但单独实验还不能确定完整的最佳区域。

---

## 7.2 Euclidean λ=0.1 的探索实验

在：

```text
λ = 0.10
PointL1Cost = 5
```

条件下，结果为：

```text
F1@4  = 60.06
F1@8  = 82.75
MLE@4 = 2.1941
MLE@8 = 3.1069
```

与当时 native baseline：

```text
59.02 / 82.61
```

相比，Euclidean loss 对 4px strict localization 有明显改善。

因此决定进一步进行完整二维联合消融。

---

# 8. 正式 64 组 8×8 消融结果

以下结果均来自 **统一重新训练的正式 grid**。

前面的单独探索实验不与本表混用。

每个单元格表示：

```text
F1@4 / F1@8 (%)
```

| λ ↓ / PointL1Cost → | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.00** | 59.09 / 82.72 | 59.69 / 82.99 | 60.17 / 83.37 | 60.60 / 83.03 | 60.86 / 83.31 | 60.87 / 83.18 | 60.90 / 83.44 | 61.12 / 83.44 |
| **0.05** | 59.72 / 82.65 | 60.06 / 82.98 | 60.86 / 83.47 | 61.19 / 83.36 | 61.44 / 83.51 | 61.14 / 83.27 | 61.37 / 83.56 | 61.48 / 83.31 |
| **0.10** | 60.51 / 82.92 | 60.79 / 83.21 | 61.56 / 83.57 | 61.40 / 83.63 | 61.86 / 83.70 | 61.69 / 83.58 | 61.78 / 83.78 | 61.98 / 83.70 |
| **0.20** | 60.93 / 83.16 | 61.34 / 83.48 | 61.75 / 83.51 | 62.28 / 83.99 | 62.28 / 83.56 | 62.00 / 83.29 | 61.97 / 83.39 | 62.50 / 83.50 |
| **0.30** | 61.21 / 83.05 | 61.57 / 83.34 | 62.24 / 83.68 | 61.88 / 83.45 | 62.33 / 83.78 | 62.14 / 83.35 | 62.42 / 83.26 | 62.49 / 83.31 |
| **0.40** | 61.44 / 83.19 | 62.04 / 83.46 | 62.42 / 83.64 | 62.17 / 83.57 | 62.42 / 83.47 | 62.57 / 83.45 | 62.57 / 83.58 | 62.60 / 83.23 |
| **0.50** | 61.68 / 83.37 | 61.71 / 83.46 | 61.33 / 83.29 | 62.35 / 83.36 | 62.62 / 83.53 | 62.63 / 83.57 | 62.65 / 83.37 | 62.28 / 83.05 |
| **0.60** | 61.52 / 83.04 | 62.13 / 83.27 | 62.59 / 83.44 | 62.42 / 83.43 | 62.67 / 83.66 | 62.59 / 83.42 | 62.64 / 83.02 | 62.48 / 83.19 |

---

# 9. 完整 F1@4 消融表（%）

| λ ↓ / PointL1Cost → | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.00** | 59.09 | 59.69 | 60.17 | 60.60 | 60.86 | 60.87 | 60.90 | 61.12 |
| **0.05** | 59.72 | 60.06 | 60.86 | 61.19 | 61.44 | 61.14 | 61.37 | 61.48 |
| **0.10** | 60.51 | 60.79 | 61.56 | 61.40 | 61.86 | 61.69 | 61.78 | 61.98 |
| **0.20** | 60.93 | 61.34 | 61.75 | 62.28 | 62.28 | 62.00 | 61.97 | 62.50 |
| **0.30** | 61.21 | 61.57 | 62.24 | 61.88 | 62.33 | 62.14 | 62.42 | 62.49 |
| **0.40** | 61.44 | 62.04 | 62.42 | 62.17 | 62.42 | 62.57 | 62.57 | 62.60 |
| **0.50** | 61.68 | 61.71 | 61.33 | 62.35 | 62.62 | 62.63 | 62.65 | 62.28 |
| **0.60** | 61.52 | 62.13 | 62.59 | 62.42 | 62.67 | 62.59 | 62.64 | 62.48 |

---

# 10. 完整 F1@8 消融表（%）

| λ ↓ / PointL1Cost → | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.00** | 82.72 | 82.99 | 83.37 | 83.03 | 83.31 | 83.18 | 83.44 | 83.44 |
| **0.05** | 82.65 | 82.98 | 83.47 | 83.36 | 83.51 | 83.27 | 83.56 | 83.31 |
| **0.10** | 82.92 | 83.21 | 83.57 | 83.63 | 83.70 | 83.58 | 83.78 | 83.70 |
| **0.20** | 83.16 | 83.48 | 83.51 | 83.99 | 83.56 | 83.29 | 83.39 | 83.50 |
| **0.30** | 83.05 | 83.34 | 83.68 | 83.45 | 83.78 | 83.35 | 83.26 | 83.31 |
| **0.40** | 83.19 | 83.46 | 83.64 | 83.57 | 83.47 | 83.45 | 83.58 | 83.23 |
| **0.50** | 83.37 | 83.46 | 83.29 | 83.36 | 83.53 | 83.57 | 83.37 | 83.05 |
| **0.60** | 83.04 | 83.27 | 83.44 | 83.43 | 83.66 | 83.42 | 83.02 | 83.19 |

---

# 11. 完整 MLE@4 消融表（px）

| λ ↓ / PointL1Cost → | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.00** | 2.2204 | 2.2086 | 2.2139 | 2.1962 | 2.1986 | 2.1997 | 2.1987 | 2.1979 |
| **0.05** | 2.2076 | 2.2139 | 2.1798 | 2.1882 | 2.1829 | 2.1869 | 2.1899 | 2.1748 |
| **0.10** | 2.2054 | 2.1899 | 2.1752 | 2.1873 | 2.1772 | 2.1789 | 2.1700 | 2.1748 |
| **0.20** | 2.1765 | 2.1809 | 2.1823 | 2.1648 | 2.1636 | 2.1713 | 2.1598 | 2.1584 |
| **0.30** | 2.1710 | 2.1688 | 2.1653 | 2.1578 | 2.1634 | 2.1609 | 2.1534 | 2.1501 |
| **0.40** | 2.1703 | 2.1624 | 2.1570 | 2.1560 | 2.1470 | 2.1516 | 2.1568 | 2.1467 |
| **0.50** | 2.1604 | 2.1759 | 2.1801 | 2.1490 | 2.1522 | 2.1481 | 2.1599 | 2.1533 |
| **0.60** | 2.1621 | 2.1567 | 2.1456 | 2.1466 | 2.1443 | 2.1521 | 2.1427 | 2.1394 |

---

# 12. 完整 MLE@8 消融表（px）

| λ ↓ / PointL1Cost → | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.00** | 3.1835 | 3.1439 | 3.1437 | 3.0919 | 3.0890 | 3.0879 | 3.0916 | 3.0827 |
| **0.05** | 3.1364 | 3.1380 | 3.0844 | 3.0841 | 3.0654 | 3.0744 | 3.0730 | 3.0503 |
| **0.10** | 3.1091 | 3.0942 | 3.0640 | 3.0794 | 3.0508 | 3.0540 | 3.0570 | 3.0351 |
| **0.20** | 3.0803 | 3.0786 | 3.0616 | 3.0345 | 3.0253 | 3.0241 | 3.0163 | 3.0000 |
| **0.30** | 3.0637 | 3.0519 | 3.0288 | 3.0273 | 3.0254 | 3.0167 | 2.9851 | 2.9866 |
| **0.40** | 3.0576 | 3.0313 | 3.0183 | 3.0168 | 2.9934 | 2.9905 | 3.0045 | 2.9808 |
| **0.50** | 3.0359 | 3.0561 | 3.0727 | 3.0010 | 2.9932 | 2.9947 | 2.9922 | 2.9937 |
| **0.60** | 3.0387 | 3.0153 | 2.9922 | 2.9921 | 3.0013 | 2.9948 | 2.9747 | 2.9781 |

---

# 13. 全局最佳结果

## 13.1 F1@4 最高配置

正式 64 组实验中最高 F1@4：

```text
λ = 0.60
PointL1Cost = 25

F1@4  = 62.67%
F1@8  = 83.66%
MLE@4 = 2.1443 px
MLE@8 = 3.0013 px
```

相对于正式 grid 的基础配置：

```text
λ=0
Cost=5

F1@4 = 59.09
F1@8 = 82.72
```

提升：

```text
F1@4:
59.09 → 62.67
+3.58 percentage points

F1@8:
82.72 → 83.66
+0.94 percentage points
```

---

## 13.2 F1@8 最高配置

正式 grid 中最高 F1@8：

```text
λ = 0.20
PointL1Cost = 20

F1@4  = 62.28%
F1@8  = 83.99%
MLE@4 = 2.1648 px
MLE@8 = 3.0345 px
```

相对于正式基础配置：

```text
F1@4:
59.09 → 62.28
+3.19 percentage points

F1@8:
82.72 → 83.99
+1.27 percentage points
```

---

## 13.3 最低 MLE@4

完整 grid 中最低 MLE@4：

```text
λ = 0.60
PointL1Cost = 40

MLE@4 = 2.1394 px
```

---

## 13.4 最低 MLE@8

完整 grid 中最低 MLE@8：

```text
λ = 0.60
PointL1Cost = 35

MLE@8 = 2.9747 px
```

最低 MLE 配置并不等于最高 F1 配置。

说明：

```text
matched-point localization error
```

与：

```text
set-level Precision / Recall / F1
```

虽然相关，但不是完全相同的优化目标。

---

# 14. Pareto 最优区域

如果同时观察 F1@4 和 F1@8，具有代表性的三个 Pareto 点为：

| λ | PointL1Cost | F1@4 | F1@8 | 说明 |
|---:|---:|---:|---:|---|
| **0.20** | **20** | 62.28 | **83.99** | F1@8 全局最佳 |
| **0.30** | **25** | 62.33 | 83.78 | 中间折中 |
| **0.60** | **25** | **62.67** | 83.66 | F1@4 全局最佳 |

因此 Stage 3 并不存在一个同时最大化：

```text
F1@4
和
F1@8
```

的参数组合。

---

# 15. Euclidean λ 消融分析

为了观察 λ 的总体趋势，对每个 λ 下 8 个 Cost 的结果求平均。

## 15.1 F1@4 行平均

```text
λ=0.00 → 60.41
λ=0.05 → 60.91
λ=0.10 → 61.45
λ=0.20 → 61.88
λ=0.30 → 62.04
λ=0.40 → 62.28
λ=0.50 → 62.16
λ=0.60 → 62.38
```

整体趋势非常明显：

> **提高 Euclidean λ 对严格 4px 定位非常有效。**

从：

```text
λ = 0
```

到：

```text
λ = 0.4~0.6
```

F1@4 整体明显提高。

同时可以看出：

```text
0.4 ~ 0.6
```

已经进入平台区。

---

## 15.2 F1@8 行平均

```text
λ=0.00 → 83.19
λ=0.05 → 83.26
λ=0.10 → 83.51
λ=0.20 → 83.49
λ=0.30 → 83.40
λ=0.40 → 83.45
λ=0.50 → 83.38
λ=0.60 → 83.31
```

F1@8 的规律与 F1@4 不同。

最佳总体区域约为：

```text
λ = 0.10 ~ 0.20
```

继续增大 λ 后：

```text
F1@4 仍然可以继续受益
但 F1@8 不再继续提高
```

说明 Euclidean loss 太强之后，模型越来越强调：

```text
matched point 的精确定位
```

而不一定继续改善：

```text
整个预测点集的 Precision / Recall balance
```

---

# 16. PointL1Cost 消融分析

对每个 Cost 下全部 8 个 λ 求平均：

| PointL1Cost | Avg F1@4 | Avg F1@8 |
|---:|---:|---:|
| 5 | 60.76 | 83.01 |
| 10 | 61.17 | 83.27 |
| 15 | 61.62 | 83.50 |
| 20 | 61.79 | 83.48 |
| 25 | **62.06** | **83.57** |
| 30 | 61.95 | 83.39 |
| 35 | 62.04 | 83.43 |
| 40 | 62.12 | 83.34 |

从：

```text
Cost=5
```

增加到：

```text
Cost=20~25
```

F1 明显提升。

但继续：

```text
25 → 30 → 35 → 40
```

没有稳定收益。

因此 Cost 的主要结论不是：

```text
越大越好
```

而是：

```text
5 → 20/25:
明显改善 Hungarian geometry matching

20/25 之后:
进入饱和区
```

较合理区域：

```text
PointL1Cost = 15 ~ 25
```

其中：

```text
20 ~ 25
```

是本实验中整体表现最好的区域。

---

# 17. λ 与 Cost 的交互作用

完整 64 组 grid 说明：

> Euclidean λ 和 PointL1Cost 不是简单独立参数，它们存在明显互补关系。

原因在于：

```text
PointL1Cost
    ↓
改变 Hungarian assignment
    ↓
改善“谁和谁匹配”

Euclidean λ
    ↓
改变 matched-query regression
    ↓
改善“匹配之后定位得多精确”
```

因此：

```text
更合理的 matching
+
更强的 pixel-space localization
```

可以叠加提升。

但二者都不能无限增大。

最终出现：

```text
F1@8 最优：
λ=0.20, Cost=20

F1@4 最优：
λ=0.60, Cost=25

最低 MLE：
出现在更高 Cost 区域
```

说明继续增强 geometry pressure 后：

```text
定位误差仍可能下降
```

但：

```text
Precision / Recall / F1
```

已经开始饱和甚至下降。

---

# 18. Stage 3 最重要的实验结论

Stage 3 可以总结为以下几点。

### 结论 1：Pixel-space Euclidean loss 有效

相对于只使用 Point L1 的情况：

```text
Euclidean loss
```

能够稳定改善严格定位性能，并降低 MLE。

其主要收益体现在：

```text
F1@4
+
mean localization error
```

而不是单纯增加预测点数量。

---

### 结论 2：提高 PointL1Cost 有效

将 Hungarian matcher 中的：

```text
PointL1Cost = 5
```

提高到：

```text
20~25
```

能够明显改善 query-GT matching。

但：

```text
30~40
```

没有稳定增益。

---

### 结论 3：Matching 和 Regression 是互补的

Stage 3 的核心方法不是单纯：

```text
“加一个 loss”
```

而是同时强化：

```text
assignment geometry
+
regression geometry
```

即：

```text
先匹配得更合理
+
再定位得更准确
```

---

### 结论 4：4px 与 8px 最优参数不同

严格定位：

```text
F1@4
```

更偏好：

```text
较大的 Euclidean λ
```

而整体 8px performance 更偏好：

```text
中等 λ
```

因此出现：

```text
F1@4 最佳：
λ=0.60, Cost=25

F1@8 最佳：
λ=0.20, Cost=20
```

---

# 19. Stage 3 推荐配置

如果目标是 ShanghaiTech benchmark，并且仍然按照：

```text
F1@8
```

选择最佳 checkpoint，则推荐：

```text
Euclidean λ   = 0.20
PointL1Cost   = 20
FocalLossCost = 2
Point DN      = 0.01
Input         = 1024×768
```

对应：

```text
F1@4 = 62.28%
F1@8 = 83.99%
```

如果更强调严格定位能力，例如未来工业打磨点任务，则可以考虑：

```text
Euclidean λ   = 0.60
PointL1Cost   = 25
```

对应：

```text
F1@4 = 62.67%
F1@8 = 83.66%
MLE@4 = 2.1443 px
MLE@8 = 3.0013 px
```

---

# 20. 关于 checkpoint 选择的注意事项

所有正式 64 组实验统一采用：

```text
best_point_f1@8px_epoch_*.pth
```

作为 best checkpoint。

因此表中的：

```text
F1@4
```

实际上是：

> **在 F1@8 最优 checkpoint 上对应测得的 F1@4。**

所以严格写论文时：

```text
62.67% F1@4
```

更准确的表述应为：

> 在统一使用 F1@8 进行 model selection 的协议下，最高观测到 F1@4=62.67%。

如果以后要专门宣称：

```text
best F1@4 checkpoint
```

则需要单独以：

```text
point/f1@4px
```

作为 checkpoint 保存指标重新比较。

---

# 21. 正式 64 组实验目录结构

完整 8×8 grid 分四块完成：

```text
stage3_grid/
    λ = 0.00 ~ 0.20
    Cost = 5 ~ 20

stage3_grid_topright/
    λ = 0.00 ~ 0.20
    Cost = 25 ~ 40

stage3_grid_bottomleft/
    λ = 0.30 ~ 0.60
    Cost = 5 ~ 20

stage3_grid_ext/
    λ = 0.30 ~ 0.60
    Cost = 25 ~ 40
```

四块合并后：

```text
8 λ
×
8 Cost
=
64 experiments
```

每组实验目录中应保留：

```text
merged_config.py
train.log
best_test.log
best_point_f1@8px_epoch_*.pth
```

同时保存各 block 自动生成的：

```text
*.csv
*.md
```

用于最终论文复现。

---

# 22. Stage 3 与后续阶段的边界

Stage 3 到这里应当视为完成。

Stage 3 已经解决：

```text
Hungarian point geometry weight
+
pixel-space Euclidean localization
```

以及二者的完整二维参数敏感性问题。

后续如果继续提升模型，不建议继续扩大：

```text
λ
Cost
```

做无休止参数搜索。

下一阶段应该引入新的模型思想，例如：

```text
local structure supervision
edge-aware supervision
geometry-aware point prediction
3D / depth / normal / curvature information
```

并单独定义为新的 Stage。

---

# 23. Stage 3 最终汇总

```text
模型基础：
Pure 2D Point-DINO + Point DN

输入：
ShanghaiTech Part B native 1024×768

Stage 3 新增：
Pixel-space Euclidean localization loss

Stage 3 同时研究：
Hungarian PointL1Cost

正式消融：
8 × 8 = 64 组

λ：
0 / 0.05 / 0.10 / 0.20 /
0.30 / 0.40 / 0.50 / 0.60

PointL1Cost：
5 / 10 / 15 / 20 /
25 / 30 / 35 / 40

F1@8 最优：
λ=0.20
Cost=20
F1@4=62.28
F1@8=83.99

F1@4 最优：
λ=0.60
Cost=25
F1@4=62.67
F1@8=83.66

最低 MLE@4：
λ=0.60
Cost=40
MLE@4=2.1394 px

最低 MLE@8：
λ=0.60
Cost=35
MLE@8=2.9747 px
```

Stage 3 的核心结论可以概括为：

> **通过同时增强 Hungarian assignment 阶段的点几何约束，以及 matched-query 回归阶段的像素空间 Euclidean 定位约束，Point-DINO 在原始 1024×768 输入协议下获得了明显的严格点定位提升。两种几何约束具有互补性，但均存在饱和区；中等 Euclidean 权重更有利于 F1@8，而较强 Euclidean 权重更有利于 F1@4 和定位误差。**
