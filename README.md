# Point-DINO 阶段二改造记录与消融实验

> 项目：基于 MMDetection DINO-R50 4-scale 的 Pure Point-DINO  
> 基础版本：阶段一 Minimal Point-DINO  
> 数据集：ShanghaiTech Part B  
> MMDetection 源码基线：`v3.3.0-1-g30b574b`  
> 阶段二目标：**将阶段一仍保留的 DINO 内部 4D box geometry 全部改造成纯 2D point geometry，并重新设计 Point DN。**

---

# 1. 阶段二总体目标

阶段一已经完成：

```text
输入图像
  ↓
DINO Backbone / Encoder / Decoder
  ↓
输出 score + (x, y)
  ↓
Focal Loss + Point L1 Loss
```

但是阶段一内部仍然保留：

```text
Encoder proposal      (x, y, w, h)
Decoder reference     (x, y, w, h)
Regression branch     (dx, dy, dw, dh)
DN                    关闭
```

因此阶段一只是“外部输出 point”，DINO 内部仍然存在 box geometry。

阶段二的目标是将整个主路径统一为：

```text
Encoder proposal      (x, y)
Encoder regression    (dx, dy)
Top-k proposal        (x, y)
Decoder reference     (x, y)
Decoder regression    (dx, dy)
Matching output       score + (x, y)
DN query              noisy (x, y)
DN regression         (dx, dy)
```

最终形成 **Pure Point-DINO**。

---

# 2. 阶段二整体改动概览

阶段二实际完成了以下核心改动：

1. Decoder initial reference：`4D → 2D`
2. Decoder iterative reference refinement：`4D → 2D`
3. Decoder reference positional encoding：`512D → 256D`
4. Two-stage encoder proposal：`(x,y,w,h) → (x,y)`
5. Top-k proposal gather：`4D → 2D`
6. Regression branches：`Linear(256,4) → Linear(256,2)`
7. 原始 Box DN：`关闭 → Point DN`
8. DN position query：`4D bbox → 2D point`
9. DN target：`bbox target → point target`
10. DN loss：`Focal + BBox L1 + GIoU → Focal + Point L1`
11. 正式验证指标：`P/R/F1 @4px、@8px`
12. 正式 checkpoint：仍使用 `point/f1@8px` 选择 best

---

# 3. Decoder 初始 Reference 由 4D 改成 2D

## 3.1 修改文件

```text
mmdet/models/detectors/dino.py
```

阶段一中：

```python
reference_points = topk_coords_unact
```

其中：

```text
topk_coords_unact.shape = [B, 900, 4]
```

语义：

```text
(x, y, w, h)
```

阶段二第一步改为：

```python
reference_points = topk_coords_unact[..., :2]
```

因此 Decoder 的 initial reference 变为：

```text
[B, 900, 2]

(x, y)
```

这一阶段暂时保持 encoder proposal 和 regression branch 为 4D，用于隔离验证 Decoder 2D reference 是否能够正常工作。

---

# 4. Decoder Iterative Refinement 支持 2D Reference

## 4.1 修改文件

```text
mmdet/models/layers/transformer/dino_layers.py
```

原始 DINO refinement 强制：

```python
assert reference_points.shape[-1] == 4
```

并执行：

```python
new_reference_points = (
    tmp +
    inverse_sigmoid(reference_points)
).sigmoid()
```

阶段二改成同时兼容 4D / 2D：

```python
if reference_points.shape[-1] == 4:
    new_reference_points = (
        tmp +
        inverse_sigmoid(
            reference_points,
            eps=1e-3)
    )

elif reference_points.shape[-1] == 2:
    new_reference_points = (
        tmp[..., :2] +
        inverse_sigmoid(
            reference_points,
            eps=1e-3)
    )

else:
    raise ValueError(...)
```

随后：

```python
new_reference_points = \
    new_reference_points.sigmoid()

reference_points = \
    new_reference_points.detach()
```

因此 2D 路径变成：

```text
reference_l = (x_l, y_l)
        ↓
reg branch
(dx, dy, dw, dh)      # 过渡阶段
        ↓
只使用 (dx, dy)
        ↓
reference_l+1 = (x_l+1, y_l+1)
```

---

# 5. Decoder Reference Positional Head：512D → 256D

## 5.1 问题

当 reference 从 4D 改成 2D 后：

```text
coordinate_to_encoding(4D)
→ 512D

coordinate_to_encoding(2D)
→ 256D
```

但原始 DINO 中：

```python
self.ref_point_head = MLP(
    self.embed_dims * 2,
    self.embed_dims,
    self.embed_dims,
    2)
```

在：

```text
embed_dims = 256
```

时输入固定为：

```text
512
```

第一次测试时实际报错：

```text
RuntimeError:
mat1 and mat2 shapes cannot be multiplied
(900x256 and 512x256)
```

## 5.2 修改文件

```text
mmdet/models/layers/transformer/dino_layers.py
```

将：

```python
self.ref_point_head = MLP(
    self.embed_dims * 2,
    self.embed_dims,
    self.embed_dims,
    2)
```

改为：

```python
self.ref_point_head = MLP(
    self.embed_dims,
    self.embed_dims,
    self.embed_dims,
    2)
```

因此：

```text
2D reference
    ↓
coordinate_to_encoding
    ↓
256D sine embedding
    ↓
ref_point_head
256 → 256 → 256
```

---

# 6. Stage 1 → Stage 2 Step 1 权重转换

阶段一 best checkpoint：

```text
/root/autodl-tmp/dinov3_dino_mmdet/
work_dirs/point_dino_r50_shanghaitech_12e/
best_point_f1_epoch_12.pth
```

原权重：

```text
decoder.ref_point_head.layers.0.weight
(256, 512)
```

4D positional encoding 拼接顺序为：

```text
[pos_y, pos_x, pos_w, pos_h]
```

2D Point-DINO 只需要：

```text
[pos_y, pos_x]
```

因此保留前 256 个输入维度：

```python
weight[:, :256]
```

转换：

```text
(256,512)
    ↓
(256,256)
```

生成：

```text
/root/autodl-tmp/dinov3_dino_mmdet/
checkpoints/mmdet/
point_dino_stage2_step1_init.pth
```

这样保留了 Stage 1 已学习的 `(x,y)` positional mapping，而不是重新随机初始化。

---

# 7. Two-stage Encoder Proposal：4D → 2D

## 7.1 修改文件

```text
mmdet/models/detectors/dino.py
```

原始 Deformable DETR / DINO 的 encoder proposal 为：

```text
grid(x,y)
+
人工构造的 (w,h)
```

最终：

```text
(x,y,w,h)
```

阶段二在 DINO 类中覆盖：

```python
gen_encoder_output_proposals()
```

只保留 normalized feature-grid center：

```python
proposal = grid.view(
    bs, -1, 2)
```

不再构造：

```python
wh = ...
proposal = torch.cat(
    (grid, wh),
    -1)
```

因此：

```text
Encoder feature grid
        ↓
normalized grid center
        ↓
2D proposal (x,y)
```

---

# 8. Encoder Proposal Refinement 改为 2D

## 8.1 修改文件

```text
mmdet/models/detectors/dino.py
```

过渡阶段 regression branch 仍为 4D，因此 encoder proposal refinement 改为：

```python
enc_outputs_coord_unact = \
    self.bbox_head.reg_branches[
        self.decoder.num_layers
    ](output_memory)[..., :2] \
    + output_proposals
```

即：

```text
reg branch:
(dx,dy,dw,dh)
        ↓
只使用 dx,dy
        +
2D proposal(x,y)
        ↓
refined encoder point
```

---

# 9. Encoder Top-k Gather：4D → 2D

原代码：

```python
topk_coords_unact = torch.gather(
    enc_outputs_coord_unact,
    1,
    topk_indices.unsqueeze(-1).repeat(
        1, 1, 4))
```

阶段二改为：

```python
topk_coords_unact = torch.gather(
    enc_outputs_coord_unact,
    1,
    topk_indices.unsqueeze(-1).repeat(
        1, 1, 2))
```

因此：

```text
enc_outputs_coord_unact
[B,N,2]
       ↓
classification Top-k
       ↓
topk_coords_unact
[B,900,2]
```

Top-k 仍然由 encoder classification score 决定，分类 proposal selection 机制不变。

---

# 10. Decoder Initial Reference 不再需要截断

当 encoder proposal 已经完全是 2D 后：

```text
topk_coords_unact.shape
=
[B,900,2]
```

因此之前 Step 1 的：

```python
reference_points = \
    topk_coords_unact[..., :2]
```

恢复为：

```python
reference_points = \
    topk_coords_unact
```

此时进入 Decoder 的 reference 天然就是 2D。

---

# 11. Regression Branch：4D → 2D

## 11.1 修改文件

```text
mmdet/models/dense_heads/dino_head.py
```

没有直接修改公共：

```text
DeformableDETRHead
```

而是在 `DINOHead` 中覆盖：

```python
_init_layers()
```

阶段一 / 原始 DINO 最后一层：

```python
Linear(
    self.embed_dims,
    4)
```

阶段二改为：

```python
Linear(
    self.embed_dims,
    2)
```

最终 7 个 regression branch 均为：

```text
weight = (2,256)
bias   = (2,)
```

模型实测：

```text
reg branch 0: (2,256) (2,)
reg branch 1: (2,256) (2,)
reg branch 2: (2,256) (2,)
reg branch 3: (2,256) (2,)
reg branch 4: (2,256) (2,)
reg branch 5: (2,256) (2,)
reg branch 6: (2,256) (2,)
```

---

# 12. Stage 2 Regression 权重转换

Stage 2 Step 1 checkpoint：

```text
point_dino_stage2_step1_init.pth
```

原 regression final layer：

```text
weight: (4,256)
bias:   (4,)
```

保留前两维：

```python
weight[:2, :]
bias[:2]
```

对应：

```text
(dx,dy,dw,dh)
      ↓
(dx,dy)
```

7 个 regression branches 共转换：

```text
7 weights
+
7 biases
=
14 tensors
```

生成：

```text
/root/autodl-tmp/dinov3_dino_mmdet/
checkpoints/mmdet/
point_dino_stage2_step2_init.pth
```

该 checkpoint 是后续所有 Stage 2 正式实验的初始化权重。

---

# 13. Pure 2D 主路径 Backward Smoke Test

在真实 ShanghaiTech Part B batch 上测试：

```text
batch size = 2

sample 0:
GT points = 28

sample 1:
GT points = 45
```

模型结构：

```text
ref_point_head input:
(256,256)

7 × regression branch:
(2,256)
```

Loss keys：

```text
loss_cls
loss_point

d0.loss_cls
d0.loss_point
...
d4.loss_cls
d4.loss_point

enc_loss_cls
enc_loss_point
```

没有：

```text
loss_bbox
loss_iou
loss_giou
```

7 个 regression branch 均满足：

```text
gradient exists
gradient finite = True
gradient nonzero = True
gradient shape = (2,256)
```

最终：

```text
STAGE-2 PURE-2D BACKWARD SMOKE TEST PASSED
```

说明 Pure 2D matching 主路径已经能够正常训练。

---

# 14. Point DN Query Generator

阶段一关闭了原始 DINO Box DN：

```python
use_dn=False
```

阶段二重新设计 Point DN。

## 14.1 修改文件

```text
mmdet/models/layers/transformer/dino_layers.py
```

新增：

```python
PointCdnQueryGenerator
```

继承：

```python
CdnQueryGenerator
```

保留：

```text
label embedding
DN grouping
DN attention mask
```

删除 Box DN 几何语义。

---

# 15. Point DN 输入

原始 Box DN：

```text
gt_instances.bboxes
      ↓
(cx,cy,w,h)
      ↓
bbox noise
      ↓
inverse sigmoid
      ↓
[B,Ndn,4]
```

Point DN：

```text
gt_instances.points
      ↓
pixel (x,y)
      ↓
/ [W,H]
      ↓
normalized (x,y)
      ↓
point noise
      ↓
inverse sigmoid
      ↓
[B,Ndn,2]
```

GT point 归一化：

```python
factor = gt_points.new_tensor(
    [img_w, img_h]).unsqueeze(0)

gt_points_normalized = \
    gt_points / factor
```

---

# 16. Point DN Noise

Point DN 使用二维 radial noise。

正 DN point：

```text
radius ∈ [0, scale)
```

负 DN point：

```text
radius ∈ [scale, 2×scale)
```

方向：

```text
angle ∈ [0,2π)
```

offset：

```text
dx = cos(angle) × radius
dy = sin(angle) × radius
```

随后：

```python
noisy_points = \
    (gt_points + offset).clamp(
        0.0, 1.0)
```

最后：

```python
dn_point_query = inverse_sigmoid(
    noisy_points)
```

---

# 17. Point DN Query Collation

原 Box DN position query：

```text
[B,Ndn,4]
```

阶段二改成：

```text
[B,Ndn,2]
```

测试结果：

```text
DN generator:
PointCdnQueryGenerator

GT:
sample 0 = 28
sample 1 = 45

dn_label_query:
(2,180,256)

dn_point_query:
(2,180,2)

dn_mask:
(1080,1080)

dn_meta:
{
    'num_denoising_queries': 180,
    'num_denoising_groups': 2
}
```

这里：

```text
1080
=
180 DN queries
+
900 matching queries
```

说明 900 个正常 matching query 始终保持不变。

---

# 18. Point DN 接入 Decoder

## 修改文件

```text
mmdet/models/detectors/dino.py
```

阶段二训练时：

```python
if self.training and self.use_dn:
    dn_label_query, \
    dn_point_query, \
    dn_mask, \
    dn_meta = \
        self.dn_query_generator(
            batch_data_samples)

    query = torch.cat(
        [dn_label_query, query],
        dim=1)

    reference_points = torch.cat(
        [
            dn_point_query,
            topk_coords_unact
        ],
        dim=1)
```

此时：

```text
DN reference:
[B,Ndn,2]

matching reference:
[B,900,2]
```

因此整个 Decoder 只接收 2D reference。

---

# 19. Point DN Target

## 修改文件

```text
mmdet/models/dense_heads/dino_head.py
```

重写：

```python
_get_dn_targets_single()
get_dn_targets()
```

正 DN query：

```text
classification target = GT label
point target = normalized GT (x,y)
point weight = 1
```

负 DN query：

```text
classification target = background
point weight = 0
```

因此负 DN query 只参与分类，不参与坐标回归。

---

# 20. Point DN Loss

重写：

```python
_loss_dn_single()
loss_dn()
```

原始 DINO DN：

```text
Focal classification
+
BBox L1
+
GIoU
```

Point-DINO Stage 2：

```text
Focal classification
+
Point L1
```

Loss keys：

```text
dn_loss_cls
dn_loss_point

d0.dn_loss_cls
d0.dn_loss_point
...
d4.dn_loss_cls
d4.dn_loss_point
```

不再存在：

```text
dn_loss_bbox
dn_loss_iou
dn_loss_giou
```

---

# 21. Point DN 接入 `loss_by_feat()`

Stage 1 中存在：

```python
assert all_layers_denoising_cls_scores is None
```

用于强制 DN 关闭。

阶段二删除该限制，并在普通 matching loss 和 encoder point loss 后增加：

```python
if all_layers_denoising_cls_scores is not None:
    dn_losses_cls, \
    dn_losses_point = self.loss_dn(...)
```

最后一层：

```text
dn_loss_cls
dn_loss_point
```

前 5 个 auxiliary decoder layers：

```text
d0~d4.dn_loss_cls
d0~d4.dn_loss_point
```

---

# 22. Point DN Backward Smoke Test

Point DN 完整接入后：

```text
use_dn:
True

DN generator:
PointCdnQueryGenerator
```

完整 loss keys：

```text
loss_cls
loss_point

d0~d4.loss_cls
d0~d4.loss_point

enc_loss_cls
enc_loss_point

dn_loss_cls
dn_loss_point

d0~d4.dn_loss_cls
d0~d4.dn_loss_point
```

Point DN 测试总 loss：

```text
6.255548
```

所有 2D regression branch：

```text
grad finite = True
```

DN label embedding：

```text
grad shape:
(1,256)

nonzero:
True
```

最终：

```text
POINT-DINO DN BACKWARD TEST PASSED
```

---

# 23. 阶段二正式训练配置

Stage 2 正式配置：

```text
configs/dino/
point_dino_r50_shanghaitech_stage2_12e.py
```

初始化：

```text
point_dino_stage2_step2_init.pth
```

正式训练：

```text
12 epochs
```

Best checkpoint 选择标准：

```python
save_best='point/f1@8px'
rule='greater'
```

评价距离：

```text
4 px
8 px
```

默认测试 score threshold：

```text
0.5
```

---

# 24. 阶段一固定阈值 Baseline

为了保证 Stage 1 / Stage 2 可直接比较，结构消融统一使用：

```text
score_threshold = 0.5
```

Stage 1 Minimal Point-DINO：

| Metric | @4px | @8px |
|---|---:|---:|
| Precision | 56.41% | 83.88% |
| Recall | 50.97% | 75.80% |
| F1 | 53.55% | 79.64% |
| Mean Localization Error | 2.2979 px | 3.3763 px |
| TP | 19,985 | 29,719 |
| FP | 15,444 | 5,710 |
| FN | 19,223 | 9,489 |

---

# 25. Stage 1 Score Threshold Scan

为了确认 Stage 1 的低 Recall 是否主要由：

```text
score_threshold = 0.5
```

导致，对 Stage 1 同一个 12-epoch checkpoint 扫描：

```text
0.1 / 0.2 / 0.3 / 0.4 / 0.5 / 0.6 / 0.7
```

结果：

| Score | P@4 | R@4 | F1@4 | P@8 | R@8 | F1@8 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 41.40 | 57.35 | 48.08 | 61.56 | **85.28** | 71.50 |
| 0.2 | 47.96 | 55.77 | 51.57 | 71.58 | 83.25 | 76.98 |
| 0.3 | 51.85 | 54.31 | 53.05 | 77.40 | 81.08 | 79.19 |
| **0.4** | 54.54 | **52.70** | **53.60** | 81.22 | **78.47** | **79.82** |
| 0.5 | **56.41** | 50.97 | 53.55 | **83.88** | 75.80 | 79.64 |
| 0.6 | 57.70 | 49.46 | 53.26 | 85.65 | 73.42 | 79.06 |
| 0.7 | 58.67 | 48.19 | 52.92 | 86.88 | 71.36 | 78.36 |

说明：

```text
Stage 1 best scanned F1@4:
53.60% @ score=0.4

Stage 1 best scanned F1@8:
79.82% @ score=0.4
```

相对 `score=0.5`：

```text
F1@4:
53.55 → 53.60
仅 +0.05 percentage point

F1@8:
79.64 → 79.82
仅 +0.18 percentage point
```

因此 Stage 1 的性能瓶颈并不是简单由 `score_threshold=0.5` 造成。

> 注意：该 threshold scan 是在当前 test split 上进行的，因此用于诊断，不应作为严格论文中通过 test set 选择超参数的依据。

---

# 26. Stage 2 消融 1：Pure 2D + No DN

配置：

```text
point_dino_r50_shanghaitech_stage2_nodn_12e.py
```

结构：

```text
Pure 2D geometry
+
use_dn=False
```

结果：

| Metric | @4px | @8px |
|---|---:|---:|
| Precision | 56.90% | 84.86% |
| Recall | 51.67% | 77.06% |
| F1 | 54.16% | 80.77% |
| Mean Localization Error | 2.3170 px | 3.3885 px |
| TP | 20,259 | 30,213 |
| FP | 15,345 | 5,391 |
| FN | 18,949 | 8,995 |

相对 Stage 1：

```text
F1@4:
53.55 → 54.16
+0.61 percentage point

F1@8:
79.64 → 80.77
+1.13 percentage points
```

因此即使不使用 DN：

> **将内部 4D geometry 改成纯 2D geometry 本身就是有效的。**

---

# 27. Stage 2 消融 2：Point DN Noise = 0.05

初始 Point DN 配置：

```python
point_noise_scale = 0.05
```

结果：

| Metric | @4px | @8px |
|---|---:|---:|
| Precision | 57.93% | 85.09% |
| Recall | 53.18% | 78.11% |
| F1 | 55.45% | 81.45% |
| Mean Localization Error | 2.3111 px | 3.3429 px |
| TP | 20,851 | 30,627 |
| FP | 15,144 | 5,368 |
| FN | 18,357 | 8,581 |

相对 `2D + No DN`：

```text
F1@4:
54.16 → 55.45
+1.29 percentage points

F1@8:
80.77 → 81.45
+0.68 percentage point
```

因此：

> **Point DN 机制本身有效，直接关闭 DN 会造成性能下降。**

---

# 28. Stage 2 消融 3：Point DN Noise = 0.01

配置：

```text
point_dino_r50_shanghaitech_stage2_dn001_12e.py
```

唯一主要改动：

```python
point_noise_scale:
0.05 → 0.01
```

Matcher：

```text
FocalLossCost = 2
PointL1Cost   = 5
```

结果：

| Metric | @4px | @8px |
|---|---:|---:|
| Precision | 62.35% | 86.18% |
| Recall | 57.16% | 79.00% |
| F1 | **59.64%** | 82.44% |
| Mean Localization Error | **2.2194 px** | **3.1357 px** |
| TP | 22,410 | 30,975 |
| FP | 13,531 | 4,966 |
| FN | 16,798 | 8,233 |

相对 `DN=0.05`：

```text
P@4:
57.93 → 62.35
+4.42 percentage points

R@4:
53.18 → 57.16
+3.98 percentage points

F1@4:
55.45 → 59.64
+4.19 percentage points

F1@8:
81.45 → 82.44
+0.99 percentage point

MLE@4:
2.3111 → 2.2194
-0.0917 px

MLE@8:
3.3429 → 3.1357
-0.2072 px
```

TP / FP / FN 也同时改善：

```text
@4:
TP +1559
FP -1613
FN -1559

@8:
TP +348
FP -402
FN -348
```

结论：

> `point_noise_scale=0.05` 对当前 ShanghaiTech Part B 精细点定位偏大，缩小到 `0.01` 后产生显著提升。

---

# 29. Stage 2 消融 4：Point DN Noise = 0.005

配置：

```text
point_dino_r50_shanghaitech_stage2_dn0005_12e.py
```

设置：

```python
point_noise_scale = 0.005
```

结果：

| Metric | @4px | @8px |
|---|---:|---:|
| Precision | 61.37% | 85.56% |
| Recall | 56.92% | **79.35%** |
| F1 | 59.07% | 82.34% |
| Mean Localization Error | 2.2316 px | 3.1710 px |
| TP | 22,319 | **31,113** |
| FP | 14,046 | 5,252 |
| FN | 16,889 | **8,095** |

相对 `DN=0.01`：

```text
F1@4:
59.64 → 59.07
-0.57 percentage point

F1@8:
82.44 → 82.34
-0.10 percentage point

MLE@4:
2.2194 → 2.2316
+0.0122 px

MLE@8:
3.1357 → 3.1710
+0.0353 px
```

虽然：

```text
R@8:
79.00 → 79.35
```

略有提高，但 Precision、F1 和 localization error 均略差。

因此：

> **当前实验中 `point_noise_scale=0.01` 比 `0.005` 更均衡。**

---

# 30. Stage 2 消融 5：Hungarian PointL1Cost = 10

当前主要配置：

```text
Point DN noise = 0.01
```

原 Hungarian Matcher：

```text
FocalLossCost = 2
PointL1Cost   = 5
```

实验改成：

```text
FocalLossCost = 2
PointL1Cost   = 10
```

注意：

> 这里改的是 Hungarian matching cost，不是反向传播使用的 `loss_point` 权重。

结果：

| Metric | @4px | @8px |
|---|---:|---:|
| Precision | **63.61%** | **88.25%** |
| Recall | 55.78% | 77.38% |
| F1 | 59.44% | **82.46%** |
| Mean Localization Error | 2.2221 px | 3.1441 px |
| TP | 21,870 | 30,341 |
| FP | **12,512** | **4,041** |
| FN | 17,338 | 8,867 |

相对 `PointL1Cost=5`：

```text
P@4:
62.35 → 63.61
+1.26 percentage points

R@4:
57.16 → 55.78
-1.38 percentage points

F1@4:
59.64 → 59.44
-0.20 percentage point

P@8:
86.18 → 88.25
+2.07 percentage points

R@8:
79.00 → 77.38
-1.62 percentage points

F1@8:
82.44 → 82.46
+0.02 percentage point

MLE@4:
2.2194 → 2.2221
+0.0027 px

MLE@8:
3.1357 → 3.1441
+0.0084 px
```

结论：

> 增大 Hungarian `PointL1Cost` 主要提高 Precision、降低 FP，但同时降低 Recall。  
> `F1@8` 仅增加 0.02 percentage point，而 `F1@4` 和定位误差略有退化，因此没有证据表明继续增大 matcher point cost 能改善精细定位。

---

# 31. 当前全部结构消融汇总

所有下表结果均使用：

```text
score_threshold = 0.5
distance thresholds = 4px / 8px
```

因此可直接做结构消融比较。

| 实验 | Geometry | DN | DN Noise | PointL1Cost | F1@4 | F1@8 | MLE@4 | MLE@8 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Stage 1 | 内部4D / 外部2D | Off | — | 5 | 53.55 | 79.64 | 2.2979 | 3.3763 |
| Stage 2 No-DN | Pure 2D | Off | — | 5 | 54.16 | 80.77 | 2.3170 | 3.3885 |
| Stage 2 DN | Pure 2D | On | 0.05 | 5 | 55.45 | 81.45 | 2.3111 | 3.3429 |
| **Stage 2 DN** | **Pure 2D** | **On** | **0.01** | **5** | **59.64** | 82.44 | **2.2194** | **3.1357** |
| Stage 2 DN | Pure 2D | On | 0.005 | 5 | 59.07 | 82.34 | 2.2316 | 3.1710 |
| Stage 2 DN Match-10 | Pure 2D | On | 0.01 | 10 | 59.44 | **82.46** | 2.2221 | 3.1441 |

---

# 32. 当前实验中的“最佳”需要分指标说明

不能只写一个统一的“best”，因为当前不同指标的最佳配置不同。

## 最高 F1@4

```text
Pure 2D
Point DN = On
point_noise_scale = 0.01
PointL1Cost = 5

F1@4 = 59.64%
```

## 最高 F1@8

```text
Pure 2D
Point DN = On
point_noise_scale = 0.01
PointL1Cost = 10

F1@8 = 82.46%
```

相比 `PointL1Cost=5`：

```text
82.46 - 82.44
=
0.02 percentage point
```

该提升非常小，同时 F1@4 和 MLE 略差。

## 最低 MLE@4

```text
point_noise_scale = 0.01
PointL1Cost = 5

MLE@4 = 2.2194 px
```

## 最低 MLE@8

```text
point_noise_scale = 0.01
PointL1Cost = 5

MLE@8 = 3.1357 px
```

因此如果目标是综合考虑：

```text
F1@4
F1@8
Localization Error
```

当前最均衡配置仍然是：

```text
point_noise_scale = 0.01
FocalLossCost = 2
PointL1Cost = 5
```

如果严格只按当前设置的：

```text
save_best = point/f1@8px
```

比较不同实验的最高数值，则 `PointL1Cost=10` 当前略高：

```text
82.46% vs 82.44%
```

---

# 33. 从 Stage 1 到当前 Stage 2 的总体提升

使用更均衡的 Stage 2 配置：

```text
Pure 2D
Point DN noise = 0.01
PointL1Cost = 5
```

与 Stage 1 固定 `score=0.5` 比较：

| Metric | Stage 1 | Stage 2 | 变化 |
|---|---:|---:|---:|
| P@4 | 56.41 | 62.35 | **+5.94** |
| R@4 | 50.97 | 57.16 | **+6.19** |
| F1@4 | 53.55 | 59.64 | **+6.09** |
| P@8 | 83.88 | 86.18 | **+2.30** |
| R@8 | 75.80 | 79.00 | **+3.20** |
| F1@8 | 79.64 | 82.44 | **+2.80** |
| MLE@4 | 2.2979 | 2.2194 | **-0.0785 px** |
| MLE@8 | 3.3763 | 3.1357 | **-0.2406 px** |

TP / FP / FN：

```text
@4:
TP  19985 → 22410   +2425
FP  15444 → 13531   -1913
FN  19223 → 16798   -2425

@8:
TP  29719 → 30975   +1256
FP   5710 →  4966    -744
FN   9489 →  8233   -1256
```

因此 Stage 2 的提升不是单纯由增加预测数量得到，而表现为：

```text
TP 上升
FP 下降
FN 下降
```

---

# 34. 阶段二当前得到的主要实验结论

## 结论 1：Pure 2D geometry 有效

```text
Stage 1 4D:
F1@8 = 79.64

Pure 2D + No DN:
F1@8 = 80.77
```

即使完全关闭 DN，纯 2D geometry 本身仍有提升。

---

## 结论 2：Point DN 有效

```text
2D + No DN:
F1@4 = 54.16
F1@8 = 80.77

2D + Point DN(0.05):
F1@4 = 55.45
F1@8 = 81.45
```

Point DN 带来额外提升。

---

## 结论 3：Point DN 噪声尺度非常敏感

```text
noise = 0.05:
F1@4 = 55.45

noise = 0.01:
F1@4 = 59.64

noise = 0.005:
F1@4 = 59.07
```

当前最佳区域明显接近：

```text
0.01
```

而不是原始设置：

```text
0.05
```

---

## 结论 4：0.005 已经开始过小

`0.005` 虽然：

```text
R@8 = 79.35
```

略高于 `0.01` 的：

```text
79.00
```

但：

```text
F1@4
F1@8
MLE@4
MLE@8
```

均略差，因此当前没有继续向更小 noise scale 搜索的依据。

---

## 结论 5：提高 Hungarian PointL1Cost 不改善精细定位

```text
PointL1Cost:
5 → 10
```

带来的主要现象：

```text
Precision ↑
Recall ↓
FP ↓
```

但：

```text
F1@4 ↓ 0.20
MLE@4 略差
MLE@8 略差
```

因此当前不建议继续提高：

```text
PointL1Cost = 15 / 20
```

---

# 35. 阶段二当前推荐配置

综合：

```text
F1@4
F1@8
MLE@4
MLE@8
```

当前推荐保留：

```python
num_queries = 900

use_dn = True

dn_cfg = dict(
    label_noise_scale=0.5,
    point_noise_scale=0.01,
    group_cfg=dict(
        dynamic=True,
        num_groups=None,
        num_dn_queries=100
    )
)

train_cfg = dict(
    assigner=dict(
        type='HungarianAssigner',
        match_costs=[
            dict(
                type='FocalLossCost',
                weight=2.0),
            dict(
                type='PointL1Cost',
                weight=5.0)
        ]
    )
)
```

注意：

```text
num_queries = 900
```

是正常 matching queries 数量。

而：

```text
num_dn_queries = 100
```

是 Point DN dynamic grouping 的预算参数，二者不是同一含义。

---

# 36. 当前仍然存在的问题

虽然 Stage 2 已经明显改善，但当前仍然存在：

```text
F1@4 < F1@8
```

且差距较大。

当前推荐配置：

```text
F1@4 = 59.64
F1@8 = 82.44
```

说明仍然有较多预测点：

```text
落在 GT 的 4~8 px 区间
```

因此下一阶段优化重点应该从：

```text
query capacity
confidence threshold
matcher cost
```

转向真正的：

```text
fine localization
```

候选方向包括：

1. 重新设计 point regression loss；
2. 增加与 pixel-space Euclidean distance 更直接对齐的 loss；
3. 引入更高分辨率特征，例如 stride-4 feature；
4. 研究局部 point refinement；
5. 后续再进行正式验证集划分，避免在 test split 上选择超参数。

---

# 37. 阶段二最终结构

当前 Pure Point-DINO 可以概括为：

```text
Image
  ↓
ResNet-50 + multi-scale features
  ↓
Transformer Encoder
  ↓
2D encoder point proposals
(x,y)
  ↓
encoder score Top-K
  ↓
900 × 2D point proposals
  ↓
Point DN queries + matching queries
  ↓
DINO Decoder
2D reference only
  ↓
6-layer iterative point refinement
(dx,dy)
  ↓
score + (x,y)
```

训练：

```text
Matching:
FocalLossCost
+
PointL1Cost
+
Hungarian assignment
```

监督：

```text
Matching query:
Focal Loss
+
Point L1 Loss
```

Point DN：

```text
Noisy GT point
+
Focal Loss
+
Point L1 Loss
```

彻底不再使用：

```text
w
h
BBoxL1Cost
IoUCost
GIoU Loss
BBox DN
4D reference
4D regression
```

---

# 38. 阶段二当前状态

当前已经完成：

```text
2D decoder reference                 ✓
2D reference positional encoding     ✓
2D encoder proposal                  ✓
2D encoder regression                ✓
2D top-k proposal                    ✓
2D decoder regression                ✓
Point DN generator                   ✓
Point DN target                      ✓
Point DN loss                        ✓
Point DN backward                    ✓
10-iter smoke training               ✓
12-epoch formal training             ✓
No-DN ablation                       ✓
DN noise 0.05                        ✓
DN noise 0.01                        ✓
DN noise 0.005                       ✓
PointL1Cost 5→10 ablation            ✓
4px / 8px evaluation                 ✓
```

当前最重要的实验结果：

```text
Stage 1:
F1@4 = 53.55
F1@8 = 79.64

Stage 2 recommended:
Point DN = 0.01
PointL1Cost = 5

F1@4 = 59.64
F1@8 = 82.44
```

对应总体提升：

```text
F1@4:
+6.09 percentage points

F1@8:
+2.80 percentage points
```

阶段二已经证明：

> **将 DINO 从内部 box geometry 改造成纯 point geometry，并加入针对 point 任务重新设计的 Point DN，可以明显改善 ShanghaiTech Part B 上的点定位性能。**
