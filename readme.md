# Point-DINO Stage 3: Native-Resolution Matching and Euclidean Localization

## 1. Stage 3 目标

Stage 3 的目标是在统一、严格的 ShanghaiTech Part B 原始分辨率协议下，重新验证 Point-DINO 的匹配策略与点定位损失。

此前部分实验沿用了原始 DINO 的测试预处理：

```text
Resize(scale=(1333, 800), keep_ratio=True)
```

对于 ShanghaiTech Part B 的 `1024×768` 图像，模型实际输入约为：

```text
1067×800
```

虽然最终预测点会通过 `scale_factor` 映射回原始图像坐标，并在原始坐标系计算 `4px / 8px` 指标，但模型推理时看到的并非原始分辨率。

为了与 ShanghaiTech Part B 常见 localization benchmark 的原图测试协议对齐，从 Stage 3 开始统一采用：

```text
Input resolution: 1024×768
Evaluation coordinates: original-image coordinates
Evaluation thresholds: 4 px / 8 px
Score threshold: 0.5
```

此前基于约 `1067×800` 输入得到的实验结果仅作为历史内部实验记录，不再作为正式 Stage 3 结果。

---

# 2. 统一实验设置

## Dataset

ShanghaiTech Crowd Counting Dataset Part B

```text
Train images: 400
Test images: 316
Total images: 716
```

点标注格式：

```python
{
    "point": [x, y],
    "point_label": 0,
    "ignore_flag": 0
}
```

当前任务为单类别点预测：

```text
num_classes = 1
```

---

## Input protocol

训练与测试统一使用：

```python
Resize(
    scale=(1024, 768),
    keep_ratio=False
)
```

对于原始 `1024×768` ShanghaiTech Part B 图像，该操作等效于保持原始尺寸，并保留 `scale_factor=(1,1)` 供预测阶段使用。

因此 Stage 3 所有正式实验均满足：

```text
Training input : 1024×768
Testing input  : 1024×768
Evaluation     : original 1024×768 coordinate system
```

---

# 3. Point-DINO 基础结构

Stage 3 基于 Stage 2 完成后的 Pure Point-DINO。

主要结构包括：

```text
Decoder reference points: 2D (x, y)
Encoder proposals:        2D (x, y)
Regression branches:      2D (x, y)
Matching target:          point only
Denoising query:          Point DN
```

不再使用：

```text
bbox width / height
IoU loss
GIoU loss
bbox matching cost
pseudo boxes
```

每个 query 输出：

```text
classification score + point (x, y)
```

---

# 4. Hungarian Matching

基础 Hungarian matching：

```python
match_costs = [
    dict(
        type='FocalLossCost',
        weight=2.0),
    dict(
        type='PointL1Cost',
        weight=5.0)
]
```

其中：

```text
FocalLossCost
```

负责类别匹配，

```text
PointL1Cost
```

负责预测点与 GT 点之间的位置匹配。

Stage 3 对 PointL1Cost 权重进行了额外消融。

---

# 5. Point DN 设置

Stage 2 已验证 Point DN 对 Point-DINO 有明显作用。

Stage 3 中主要固定：

```text
Point DN noise scale = 0.01
```

除专门的 DN ablation 外，其余实验均使用该设置。

---

# 6. Native-resolution Stage 2 Ablation

重新在原始 `1024×768` 输入下运行 Stage 2 的主要实验。

## Results

| Setting | P@4 | R@4 | F1@4 | MLE@4 | P@8 | R@8 | F1@8 | MLE@8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2D + No DN | 56.21 | 53.00 | 54.56 | 2.2831 | 83.25 | 78.50 | 80.80 | 3.3504 |
| DN = 0.05 | 57.56 | 54.97 | 56.23 | 2.2678 | 83.25 | 79.51 | 81.34 | 3.2917 |
| DN = 0.005 | 61.65 | 57.89 | 59.71 | 2.2020 | 85.04 | 79.86 | 82.36 | 3.1207 |
| DN = 0.01 | 60.76 | 57.38 | 59.02 | 2.2300 | 85.05 | 80.32 | 82.61 | 3.1842 |

上述实验的 Hungarian matcher 均采用：

```text
FocalLossCost = 2
PointL1Cost   = 5
```

---

## DN observations

No DN：

```text
F1@4 = 54.56
F1@8 = 80.80
```

加入 DN=0.05：

```text
F1@4 = 56.23
F1@8 = 81.34
```

说明 Point DN 对 Point-DINO 有稳定提升。

进一步减小 DN noise：

```text
DN=0.005:
F1@4 = 59.71
F1@8 = 82.36

DN=0.01:
F1@4 = 59.02
F1@8 = 82.61
```

表现出一定的阈值差异：

```text
DN=0.005 更有利于 4px 精细定位
DN=0.01  更有利于 8px 整体匹配
```

后续实验暂时采用：

```text
Point DN noise = 0.01
```

---

# 7. Hungarian PointL1Cost Ablation

在以下条件固定的情况下：

```text
Input            = 1024×768
Point DN         = 0.01
FocalLossCost    = 2
Euclidean Loss   = OFF
```

只改变：

```text
PointL1Cost weight
```

## Results

| PointL1Cost | P@4 | R@4 | F1@4 | MLE@4 | P@8 | R@8 | F1@8 | MLE@8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 60.76 | 57.38 | 59.02 | 2.2300 | 85.05 | 80.32 | 82.61 | 3.1842 |
| 10 | 62.11 | 58.12 | 60.05 | 2.2081 | 86.09 | 80.55 | 83.23 | 3.1386 |
| 15 | 62.68 | 58.28 | 60.40 | 2.2192 | 86.55 | 80.48 | 83.40 | 3.1430 |
| 20 | 64.07 | 57.28 | **60.49** | **2.2037** | 88.50 | 79.12 | **83.55** | **3.1183** |

---

## Matching-cost observations

从 `PointL1Cost=5` 提高到 `10`：

```text
F1@4: 59.02 → 60.05   (+1.03)
F1@8: 82.61 → 83.23   (+0.62)
```

继续提高：

```text
Cost 10 → 15:
F1@4: +0.35
F1@8: +0.17

Cost 15 → 20:
F1@4: +0.09
F1@8: +0.15
```

说明提高 point matching cost 对 Point-DINO 有效，但随着权重增加收益逐渐减小。

`Cost=20` 当前获得最高 F1：

```text
F1@4 = 60.49
F1@8 = 83.55
```

同时可以观察到较明显的 precision / recall trade-off：

```text
Cost=15:
P@8 = 86.55
R@8 = 80.48

Cost=20:
P@8 = 88.50
R@8 = 79.12
```

即更高的位置匹配权重进一步提高 Precision，但开始牺牲 Recall。

因此 PointL1Cost 的最佳区域目前约位于：

```text
15 ~ 20
```

不过 Stage 3 的 Euclidean Loss 实验暂时仍以：

```text
PointL1Cost = 5
```

作为控制变量，以便单独分析新 loss 本身的效果。

---

# 8. Pixel-space Euclidean Localization Loss

## Motivation

Point-DINO 原始 point regression loss 使用 normalized coordinate L1：

```text
|x_pred - x_gt| + |y_pred - y_gt|
```

但最终 localization metric 使用的是原始图像像素空间中的欧氏距离：

```text
sqrt(
    (x_pred - x_gt)^2 +
    (y_pred - y_gt)^2
)
```

因此 Stage 3 增加 pixel-space Euclidean localization loss，使训练目标与最终 localization metric 更直接对齐。

---

## Definition

首先将 normalized point error 转换到当前输入图像像素尺度：

```text
dx_pixel = (x_pred - x_gt) × W
dy_pixel = (y_pred - y_gt) × H
```

然后：

```text
d = sqrt(
    dx_pixel² +
    dy_pixel² +
    eps
)
```

当前实现进一步除以 8 px 进行尺度归一化：

```text
L_euclidean =
    λ × d / 8
```

其中：

```text
eps = 1e-6
```

用于数值稳定。

Euclidean Loss 为额外损失，不替代原有 Point L1 Loss。

即：

```text
Localization Loss
=
Point L1 Loss
+
Pixel Euclidean Loss
```

---

# 9. Euclidean Loss Experiment

为了单独验证 Euclidean loss 的作用，本轮固定：

```text
Input             = native 1024×768
Point DN          = 0.01
FocalLossCost     = 2
PointL1Cost       = 5
Point L1 Loss     = ON
Euclidean Loss    = ON
Euclidean weight  = 0.1
```

注意：

```text
Euclidean loss 只参与训练 loss，
不加入 Hungarian matching cost。
```

Hungarian matcher 仍为：

```text
FocalLossCost = 2
PointL1Cost   = 5
```

---

## Result

### Baseline

```text
Point L1 only
Euclidean λ = 0
```

结果：

```text
P@4   = 60.76
R@4   = 57.38
F1@4  = 59.02
MLE@4 = 2.2300

P@8   = 85.05
R@8   = 80.32
F1@8  = 82.61
MLE@8 = 3.1842
```

### + Pixel Euclidean Loss

```text
Euclidean λ = 0.1
```

结果：

```text
P@4   = 62.07
R@4   = 58.18
F1@4  = 60.06
MLE@4 = 2.1941

P@8   = 85.52
R@8   = 80.15
F1@8  = 82.75
MLE@8 = 3.1069
```

---

## Comparison

| Setting | P@4 | R@4 | F1@4 | MLE@4 | P@8 | R@8 | F1@8 | MLE@8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Point L1 | 60.76 | 57.38 | 59.02 | 2.2300 | 85.05 | 80.32 | 82.61 | 3.1842 |
| **Point L1 + Euclidean λ=0.1** | **62.07** | **58.18** | **60.06** | **2.1941** | **85.52** | 80.15 | **82.75** | **3.1069** |
| Δ | +1.31 | +0.80 | **+1.04** | **-0.0359** | +0.47 | -0.17 | **+0.14** | **-0.0773** |

---

# 10. Euclidean Loss Observation

Pixel-space Euclidean loss 对 `4px` 精细定位的改善更加明显：

```text
F1@4:
59.02 → 60.06
+1.04 percentage points
```

同时：

```text
Precision@4:
60.76 → 62.07

Recall@4:
57.38 → 58.18
```

Precision 和 Recall 同时提高。

对于更宽松的 `8px`：

```text
F1@8:
82.61 → 82.75
+0.14
```

提升较小。

但是 localization error 在两个阈值下均下降：

```text
MLE@4:
2.2300 → 2.1941

MLE@8:
3.1842 → 3.1069
```

因此当前结果支持以下判断：

> Pixel-space Euclidean loss 主要改善 Point-DINO 的精细空间定位能力，而不是单纯增加检测数量。

这与该损失的设计目标一致。

---

# 11. Current Stage 3 Status

目前在统一 native-resolution 协议下得到：

## Best matcher-only result

```text
Point DN        = 0.01
FocalLossCost   = 2
PointL1Cost     = 20
Euclidean       = OFF

F1@4  = 60.49
F1@8  = 83.55
MLE@4 = 2.2037
MLE@8 = 3.1183
```

## Euclidean-loss result

```text
Point DN         = 0.01
FocalLossCost    = 2
PointL1Cost      = 5
Euclidean weight = 0.1

F1@4  = 60.06
F1@8  = 82.75
MLE@4 = 2.1941
MLE@8 = 3.1069
```

这两组目前不能直接解释为谁替代谁，因为它们研究的是两个不同变量：

```text
Matcher experiment:
改变 Hungarian assignment 中 PointL1Cost

Euclidean experiment:
保持 matcher 不变，
改变训练阶段 localization objective
```

下一阶段如果继续实验，应研究二者能否叠加：

```text
stronger PointL1 matching
+
pixel-space Euclidean localization
```

但截至当前 Stage 3，本 README 只记录已经完成并在统一 `1024×768` 输入协议下得到的正式结果。

---

# 12. Experiment Directory Reference

当前 native-resolution matcher experiments：

```text
work_dirs/
├── point_dino_r50_shanghaitech_stage2_nodn_12e
├── point_dino_r50_shanghaitech_stage2_12e
├── point_dino_r50_shanghaitech_stage2_dn0005_12e
├── point_dino_r50_shanghaitech_stage2_dn001_12e
├── point_dino_r50_shanghaitech_stage2_dn001_match10_12e
├── point_dino_r50_shanghaitech_stage2_dn001_match15_12e
└── point_dino_r50_shanghaitech_stage2_dn001_match20_12e
```

Euclidean experiment：

```text
work_dirs/
└── point_dino_r50_shanghaitech_stage2_dn001_euc01_native_12e
```

---

# 13. Important Protocol Note

从本阶段开始，正式实验必须保持：

```text
ShanghaiTech Part B
Native 1024×768 input
Original-coordinate evaluation
Score threshold = 0.5
Distance thresholds = 4 px / 8 px
```

旧的：

```text
1024×768
→ Resize keep_ratio
→ approximately 1067×800
```

实验不再与 Stage 3 正式结果混合使用。

Stage 3 所有后续消融应基于相同 native-resolution protocol 进行。