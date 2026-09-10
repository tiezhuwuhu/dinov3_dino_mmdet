# Point-DINO Stage 4 README

## 1. Stage 4 目标

Stage 4 的目标是在 **不改变 Point-DINO 最终任务形式** 的前提下，引入 FIDT（Focal Inverse Distance Transform）式空间监督，验证 dense spatial supervision 是否能够提升最终的 query-based point localization。

Point-DINO 的最终预测仍保持为：

\[
\{(\hat{x}_i,\hat{y}_i,\hat{s}_i)\}_{i=1}^{N_q}
\]

即每个 query 直接预测二维点坐标 `(x, y)` 和分类置信度。

Stage 4 **没有**恢复 bbox、没有使用伪框、没有使用固定宽高，也没有把任务改成 density-map prediction。FIDT map 仅作为训练辅助监督，推理阶段最终输出仍然是 query point set。

```text
Backbone / Neck
      │
      ├──────────────→ DINO Transformer → Queries → (x, y)   [最终输出]
      │
      └──────────────→ FIDT Auxiliary Head → FIDT Map        [仅训练辅助]
```

---

## 2. Stage 4 基线设置

Stage 4 延续 Stage 3 已确定的 Point-DINO 主干配置：

```text
num_classes                 = 1
num_queries                 = 900
Point L1 loss weight        = 5
Euclidean point loss weight = 0.20
Hungarian FocalLossCost     = 2
Hungarian PointL1Cost       = 20
Point DN noise scale        = 0.01
```

Stage 4 的核心新增项只有 FIDT auxiliary supervision。

---

## 3. Stage 4 主要代码改动

### 3.1 新增 FIDT Auxiliary Head

主要文件：

```text
mmdet/models/dense_heads/point_fidt_aux_head.py
```

FIDT auxiliary head 从 DINO 使用的最高分辨率多尺度特征 `mlvl_feats[0]` 生成 dense FIDT prediction map。

在 ShanghaiTech Part B 原生 `1024×768` 输入下，实测 feature shapes：

```text
level 0: [B, 256, 96, 128]   stride 8
level 1: [B, 256, 48, 64]    stride 16
level 2: [B, 256, 24, 32]    stride 32
level 3: [B, 256, 12, 16]    stride 64
```

FIDT head：

```text
[B, 256, H/8, W/8]
        ↓
Conv 3×3 + GN + ReLU
        ↓
Bilinear Upsample ×2
        ↓
Conv 3×3 + GN + ReLU
        ↓
Conv 1×1
        ↓
Sigmoid
        ↓
[B, 1, H/4, W/4]
```

因此在 `1024×768` 输入下：

```text
FIDT output = [B, 1, 192, 256]
```

对应 stride 4。

### 3.2 FIDT Target

Stage 4 使用 FIDT 形式：

\[
T(x,y)=\frac{1}{D(x,y)^{\gamma D(x,y)+\phi}+\xi}
\]

其中：

\[
\gamma=0.02,\qquad \phi=0.75,\qquad \xi=1
\]

`D(x,y)` 表示当前位置到最近 GT point 的欧氏距离。

Target 根据经过当前数据增强后的 float GT points 在线生成。

FIDT feature cell `(u,v)` 使用 cell center 映射到当前 padded image：

\[
x=(u+0.5)\frac{W_{pad}}{W_f}
\]

\[
y=(v+0.5)\frac{H_{pad}}{H_f}
\]

随后计算到所有 GT point 的最近距离。为控制显存，最近距离计算使用 chunked `cdist`。

### 3.3 Full-map FIDT

所有 valid FIDT cells 参与 MSE：

\[
L_{\mathrm{FIDT}}=
\frac{\sum_{q\in valid}(\hat T(q)-T(q))^2}{N_{valid}}
\]

padding 区域不参与 loss。空 GT 图片 target 为全 0。

总训练目标：

\[
L=L_{\mathrm{Point-DINO}}+\lambda_{\mathrm{FIDT}}L_{\mathrm{FIDT}}
\]

### 3.4 FIDT 梯度路径检查

对 FIDT-only loss 做独立 backward 后：

```text
FIDT head               gradient > 0
Backbone                gradient > 0
Neck                    gradient > 0
Transformer encoder     gradient = 0
Transformer decoder     gradient = 0
Point prediction head   gradient = 0
```

因此 FIDT branch 只能通过共享 backbone / neck 间接影响最终 query-based point prediction。FIDT head 在 inference 阶段不参与最终输出。

### 3.5 Local FIDT

Local mask：

\[
M(q)=M_{valid}(q)\land[D(q)\le R]
\]

Local FIDT loss：

\[
L_{\mathrm{local}}=
\frac{\sum_q M(q)(\hat T(q)-T(q))^2}{\max(\sum_q M(q),1)}
\]

实现特性：

- 仅 `D(q) <= R` 的 valid cells 参与 loss；
- radius 外直接 ignore，而不是设为 0 后继续平均；
- 多 GT 使用最近距离形成局部区域 union；
- overlap 区域只计算一次；
- empty-GT image 在 Local 模式下不产生 selected cells；
- 整个 batch 没有 selected cells 时返回 graph-connected zero；
- `local_radius_px=None` 时严格保留 Full-map 行为；
- Full / Local 共用最近距离计算，不重复 `cdist`。

Stage 4 Local-FIDT 正式实验使用：

```text
R = 16 px
```

---

# 4. ShanghaiTech Part B 实验

## 4.1 固定协议

```text
Dataset                    ShanghaiTech Part B
Input                      native 1024×768
Epochs                     12
num_classes                1
num_queries                900
Euclidean point weight     0.20
Point L1 loss weight       5
FocalLossCost              2
PointL1Cost                20
Point DN noise             0.01
score threshold            0.5
Metric                     F1@4 / F1@8
Checkpoint selection       best F1@8
Initialization             common Stage 2 Point-DINO init
```

所有 Stage 4 正式 sweep 都从同一个 Stage 2 init 开始：

```text
/root/autodl-tmp/dinov3_dino_mmdet/checkpoints/mmdet/point_dino_stage2_step2_init.pth
```

没有从已训练 Stage 3 checkpoint 再续训。

## 4.2 Full-map FIDT λ Sweep

主要配置：

```text
configs/dino/point_dino_r50_shanghaitech_stage4_fidt_native_12e.py
```

扫描：

\[
\lambda_{\mathrm{FIDT}}\in\{0,0.25,0.5,1,2\}
\]

| λFIDT | F1@4 | F1@8 | MLE@4 | MLE@8 |
|---:|---:|---:|---:|---:|
| 0 | 62.04 | 83.60 | 2.1764 | 3.0422 |
| 0.25 | 61.75 | 83.50 | — | — |
| 0.5 | **62.09** | **83.67** | — | — |
| 1 | 61.82 | 83.60 | — | — |
| 2 | 61.99 | 83.58 | — | — |

相对 λ=0：

```text
best F1@4: 62.04 → 62.09  (+0.05 pp)
best F1@8: 83.60 → 83.67  (+0.07 pp)
```

**结论：Full-map FIDT 没有带来有意义的提升。**

### 4.2.1 Full-map FIDT loss 训练现象

观察到 raw FIDT MSE 大致：

```text
训练开始        ≈ 0.06
epoch 1 后      ≈ 0.004
训练后期        ≈ 0.0007 ~ 0.0008
```

说明 FIDT map supervision 很快被优化到较小值，后期其加权 contribution 很小；这只是实验现象，不证明背景占比是唯一原因。

## 4.3 Local-FIDT Sweep

主要配置：

```text
configs/dino/point_dino_r50_shanghaitech_stage4_local_fidt_native_12e.py
```

固定：

```text
R = 16 px
```

扫描：

\[
\lambda_{\mathrm{FIDT}}\in\{0,0.5,1,2,4\}
\]

| λFIDT | F1@4 | F1@8 |
|---:|---:|---:|
| 0 | 61.96 | 83.60 |
| 0.5 | 61.74 | 83.50 |
| 1 | **62.01** | 83.31 |
| 2 | 61.94 | **83.62** |
| 4 | 61.90 | 83.60 |

相对 λ=0：

```text
best F1@4: 61.96 → 62.01  (+0.05 pp)
best F1@8: 83.60 → 83.62  (+0.02 pp)
```

**结论：Local FIDT `R=16` 同样没有带来有意义的提升。**

需要注意：Local-FIDT sweep 中 λ=0 retrain 为 `61.96 / 83.60`，历史 Stage 3 某次结果为 `62.28 / 83.99`。两者不是同一次训练，且 sweep 使用 `deterministic=False`，因此不能用它们直接估计 FIDT 增益；FIDT 消融应比较同一 sweep 内的 λ=0 control。

---

# 5. BCData 实验

## 5.1 数据审计

数据路径：

```text
/root/autodl-tmp/dinov3_dino_mmdet/data/BCData/
```

| Split | Images | Positive | Negative | Total points | Max GT/image |
|---|---:|---:|---:|---:|---:|
| Train | 803 | 33,058 | 60,780 | 93,838 | 374 |
| Validation | 133 | 7,701 | 14,103 | 21,804 | 374 |
| Test | 402 | 21,864 | 43,568 | 65,432 | **421** |
| **Total** | **1,338** | **62,623** | **118,451** | **181,074** | **421** |

全部图片均为：

```text
640×640
```

数据检查：

```text
missing positive h5 = 0
missing negative h5 = 0
non-640x640 images = 0
invalid/out-of-bound coordinates = 0
```

最大单图 GT = 421，小于 `num_queries=900`，因此当前 query capacity 足够。

## 5.2 Task Definition

第一阶段只评价 localization：

```text
positive + negative
        ↓
single class: cell
```

即：

```text
num_classes = 1
```

暂不评价 Ki-67 positive / negative 分类。

## 5.3 输入尺度和评价协议

BCData 原图：

```text
640×640
```

为了与已有 localization baseline 对齐，模型输入统一：

```text
512×512
```

训练 / validation / test pipeline：

```text
640×640
    ↓ Resize
512×512
```

BCData inference 使用：

```text
rescale_points = False
```

从而保证：

```text
GT coordinates   = 512-space
Pred coordinates = 512-space
```

最终 metric：

```text
F1@5px
F1@10px
```

## 5.4 PointMetric key 修复

原实现：

```python
key = str(int(threshold))
```

会把：

```text
6.25 → 6
12.5 → 12
```

虽然实际 matching 仍然使用 float threshold，但 metric key 会错误截断。

修改为：

```python
key = f'{threshold:g}'
```

该修改只修复日志 / checkpoint key，不改变距离计算。

## 5.5 Point-DINO inference 坐标系改动

文件：

```text
mmdet/models/dense_heads/dino_head.py
```

`_predict_by_feat_single()` 先把 normalized `(x,y)` 乘当前 `img_shape` 得到 resize-space pixel coordinates。为了 BCData 直接在 512-space 评价，引入：

```text
rescale_points=False
```

使预测点不再除以 `scale_factor` 映射回 640-space。

最终：

```text
model input       = 512×512
GT coordinates    = 512×512
Pred coordinates  = 512×512
Metric            = 5 / 10 px
```

## 5.6 BCData 固定训练设置

```text
Input                      = 512×512
Epochs                     = 12
num_classes                = 1
num_queries                = 900
max_per_img                = 900
Euclidean point weight     = 0.20
Point L1 loss weight       = 5
FocalLossCost              = 2
PointL1Cost                = 20
Point DN noise             = 0.01
Local FIDT radius          = 16 px
Metric                     = F1@5 / F1@10
Checkpoint selection       = validation F1@10
Initialization             = common Stage 2 Point-DINO init
```

---

# 6. BCData Confidence Threshold Sweep

初始沿用：

```text
score_threshold = 0.5
```

但模型呈现明显 high precision / low recall，因此固定 checkpoint 后，只在 validation set 上扫 confidence threshold：

\[
score=0.05,0.075,0.10,\ldots,0.90
\]

步长：

```text
0.025
```

正式选择规则：

> 使用 validation F1@10 最大的 confidence threshold；test set 不参与 confidence tuning。

## 6.1 λFIDT = 2：Validation Sweep

`score=0.5` sanity check：

```text
Predictions = 17,518
F1@5  = 75.88
F1@10 = 84.12
```

最佳 confidence：

```text
0.325
```

| Metric | Result |
|---|---:|
| Predictions | 21,679 |
| P@5 | 79.39 |
| R@5 | 78.94 |
| **F1@5** | **79.16** |
| P@10 | 88.07 |
| R@10 | 87.57 |
| **F1@10** | **87.82** |

F1@5 和 F1@10 的最佳 confidence 恰好都是 `0.325`。

## 6.2 λFIDT = 2：正式 Test

固定：

```text
checkpoint = validation-best
score = 0.325
```

Test set：

```text
402 images
65,432 GT points
```

| Metric | 5 px | 10 px |
|---|---:|---:|
| Precision | 82.31 | 89.93 |
| Recall | 79.46 | 86.82 |
| **F1** | **80.86** | **88.35** |
| MLE | 2.1271 px | 2.4978 px |
| TP | 51,995 | 56,808 |
| FP | 11,173 | 6,360 |
| FN | 13,437 | 8,624 |

GT consistency：

```text
51,995 + 13,437 = 65,432
56,808 + 8,624  = 65,432
```

与完整 test GT 数一致。

---

# 7. BCData FIDT Ablation：λFIDT = 0

为判断 BCData 上的性能是否来自 FIDT branch，进行了严格 control：

```text
Stage 4 code path 保持不变
FIDT head 保持存在
R = 16 不变
仅设置 fidt_loss_weight = 0
```

因此该实验准确名称应为：

```text
Point-DINO w/o FIDT
```

而不是代码意义上完全删除 Stage 4。

## 7.1 λFIDT = 0：Validation Sweep

`score=0.5`：

```text
Predictions = 17,309
F1@5  = 75.37
F1@10 = 83.65
```

最佳 confidence：

```text
0.325
```

| Metric | Result |
|---|---:|
| Predictions | 21,479 |
| P@5 | 79.57 |
| R@5 | 78.38 |
| **F1@5** | **78.97** |
| P@10 | 88.39 |
| R@10 | 87.08 |
| **F1@10** | **87.73** |

## 7.2 λFIDT = 0：正式 Test

固定：

```text
score = 0.325
```

| Metric | 5 px | 10 px |
|---|---:|---:|
| Precision | 82.55 | 90.18 |
| Recall | 79.23 | 86.56 |
| **F1** | **80.85** | **88.33** |
| MLE | 2.1348 px | 2.5056 px |
| TP | 51,840 | 56,635 |
| FP | 10,960 | 6,165 |
| FN | 13,592 | 8,797 |

---

# 8. BCData FIDT Ablation 汇总

| Variant | λFIDT | Val-selected score | F1@5 | F1@10 | MLE@5 | MLE@10 |
|---|---:|---:|---:|---:|---:|---:|
| Point-DINO w/o FIDT | 0 | 0.325 | 80.85 | 88.33 | 2.1348 | 2.5056 |
| Point-DINO + Local FIDT | 2 | 0.325 | 80.86 | 88.35 | 2.1271 | 2.4978 |
| **Δ (λ2 - λ0)** | — | — | **+0.01 pp** | **+0.02 pp** | **-0.0077 px** | **-0.0078 px** |

Precision / Recall 变化：

```text
5 px:
P 82.55 → 82.31  (-0.24 pp)
R 79.23 → 79.46  (+0.23 pp)

10 px:
P 90.18 → 89.93  (-0.25 pp)
R 86.56 → 86.82  (+0.26 pp)
```

说明 FIDT 只产生极小的 precision / recall balance 变化，最终 F1 几乎不变。

---

# 9. BCData 与已有 Localization Baselines 对比

在对齐的：

```text
Input  = 512×512
Metric = F1@5 / F1@10
```

协议下，本项目记录的参考结果如下：

| Method | P@5 | R@5 | F1@5 | P@10 | R@10 | F1@10 |
|---|---:|---:|---:|---:|---:|---:|
| U-CSRNet | 76.9 | 77.5 | 77.2 | 86.6 | 87.2 | 86.9 |
| VGG+FPN | 76.9 | 78.9 | 77.9 | 86.1 | 88.3 | 87.1 |
| U-Net | 81.7 | 72.1 | 76.7 | 80.7 | 91.4 | 85.7 |
| HRNet | 80.1 | 78.3 | 79.2 | 88.4 | 86.3 | 87.3 |
| M-HRNet | 80.2 | 78.3 | 79.3 | 88.4 | 86.2 | 87.2 |
| **Point-DINO w/o FIDT** | **82.55** | **79.23** | **80.85** | **90.18** | **86.56** | **88.33** |
| **Point-DINO + Local FIDT** | 82.31 | **79.46** | **80.86** | 89.93 | **86.82** | **88.35** |

相对表中已有最好 F1：

```text
F1@5:
79.3 → 80.85 / 80.86
≈ +1.55 ~ +1.56 pp

F1@10:
87.3 → 88.33 / 88.35
≈ +1.03 ~ +1.05 pp
```

最重要的是：

```text
Point-DINO w/o FIDT:
F1@5  = 80.85
F1@10 = 88.33
```

因此 BCData 上的强 localization 表现并不依赖 FIDT auxiliary supervision。

---

# 10. Stage 4 所有实验汇总

## 10.1 ShanghaiTech Part B — Full FIDT

| λFIDT | F1@4 | F1@8 |
|---:|---:|---:|
| 0 | 62.04 | 83.60 |
| 0.25 | 61.75 | 83.50 |
| 0.5 | **62.09** | **83.67** |
| 1 | 61.82 | 83.60 |
| 2 | 61.99 | 83.58 |

## 10.2 ShanghaiTech Part B — Local FIDT, R=16

| λFIDT | F1@4 | F1@8 |
|---:|---:|---:|
| 0 | 61.96 | 83.60 |
| 0.5 | 61.74 | 83.50 |
| 1 | **62.01** | 83.31 |
| 2 | 61.94 | **83.62** |
| 4 | 61.90 | 83.60 |

## 10.3 BCData — Validation confidence sweep

| Variant | Score@0.5 F1@5 | Score@0.5 F1@10 | Best score | Best Val F1@5 | Best Val F1@10 |
|---|---:|---:|---:|---:|---:|
| λFIDT=0 | 75.37 | 83.65 | **0.325** | **78.97** | **87.73** |
| λFIDT=2 | 75.88 | 84.12 | **0.325** | **79.16** | **87.82** |

## 10.4 BCData — Test

| Variant | P@5 | R@5 | F1@5 | P@10 | R@10 | F1@10 |
|---|---:|---:|---:|---:|---:|---:|
| λFIDT=0 | 82.55 | 79.23 | **80.85** | 90.18 | 86.56 | **88.33** |
| λFIDT=2 | 82.31 | 79.46 | **80.86** | 89.93 | 86.82 | **88.35** |

---

# 11. Stage 4 最终结论

Stage 4 的核心研究问题是：

> 额外增加一个 FIDT dense spatial auxiliary branch，能否提升 Point-DINO 最终 query-based point localization？

现有实验给出高度一致的结果。

ShanghaiTech Part B：

```text
Full FIDT:
最大提升仅约 +0.05 / +0.07 pp

Local FIDT:
最大提升仅约 +0.05 / +0.02 pp
```

BCData：

```text
λFIDT=0:
F1@5  = 80.85
F1@10 = 88.33

λFIDT=2:
F1@5  = 80.86
F1@10 = 88.35
```

FIDT 增益只有：

```text
+0.01 pp @5
+0.02 pp @10
```

因此当前证据支持：

> **Stage 4 的 branch-style FIDT auxiliary supervision 对 Point-DINO 最终 query point prediction 没有带来有意义的性能提升。**

该结论已经在 ShanghaiTech Part B 和 BCData 两个性质差异明显的数据集上得到一致观察。

---

# 12. Stage 4 的重要正向结果

虽然 FIDT auxiliary branch 无效，但 BCData 实验验证了：

> **Point-DINO 的直接 query → `(x,y)` formulation 本身具有较强的 pixel-level point localization 能力。**

无 FIDT 的 Point-DINO：

```text
F1@5   = 80.85
F1@10  = 88.33
MLE@5  = 2.1348 px
MLE@10 = 2.5056 px
```

因此 ShanghaiTech Part B 上较低的 strict F1 不能简单归因于 direct query point regression 本身缺乏精确定位能力。

---

# 13. Stage 4 后续建议

Stage 4 到此应停止继续进行：

```text
FIDT λ sweep
FIDT radius sweep
Full / Local FIDT variants
```

因为在两个数据集上的收益都已接近 0。

如果后续继续提高 Point-DINO localization，更值得研究的是让 spatial localization supervision **直接进入 query 的最终定位机制**，而不是继续增加一个 inference 时被丢弃的 backbone auxiliary map head。

例如后续可以研究：

```text
query
  ↓
query-conditioned spatial localization distribution
  ↓
soft-argmax / integral coordinate
  ↓
(x, y)
```

同时继续保留：

```text
DINO query set prediction
Hungarian matching
direct point-set output
```

这属于 Stage 5 / 后续阶段，不属于 Stage 4。

---

# 14. Stage 4 关键文件

核心实现：

```text
mmdet/models/dense_heads/point_fidt_aux_head.py
mmdet/models/dense_heads/dino_head.py
mmdet/evaluation/metrics/point_metric.py
```

ShanghaiTech Stage 4 配置：

```text
configs/dino/point_dino_r50_shanghaitech_stage4_fidt_native_12e.py
configs/dino/point_dino_r50_shanghaitech_stage4_local_fidt_native_12e.py
```

BCData Stage 4 配置：

```text
configs/dino/point_dino_r50_bcdata_stage4_local_fidt_512_12e.py
configs/dino/point_dino_r50_bcdata_stage4_local_fidt_lam0_512_12e.py
```

BCData confidence sweep：

```text
sweep_bcdata_confidence.py
```

BCData visualization：

```text
visualize_bcdata_lam0_test.py
```

BCData：

```text
/root/autodl-tmp/dinov3_dino_mmdet/data/BCData/
```

Stage 2 common initialization：

```text
/root/autodl-tmp/dinov3_dino_mmdet/checkpoints/mmdet/point_dino_stage2_step2_init.pth
```

---

# 15. 一句话总结

> **Stage 4 证明了 FIDT-based dense auxiliary supervision 对 Point-DINO 基本无效，但同时验证了不依赖 FIDT 的直接 query-to-point Point-DINO 在 BCData 上具有很强的严格点定位能力。**
