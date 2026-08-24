# Point-DINO 阶段一改造记录

> 项目：基于 MMDetection DINO-R50 4-scale 的 Minimal Point-DINO  
> 基线模型：`dino-4scale_r50_8xb2-12e_coco`  
> 初始权重：`dino_r50_4scale_coco.pth`  
> MMDetection 源码版本：本地代码基于 `v3.3.0-1-g30b574b`  
> 阶段一目标：**保留 DINO 内部 4D reference / two-stage 几何结构，对外改造成 `score + (x, y)` 的点预测模型。**

---

## 1. 阶段一整体设计

阶段一没有直接把 DINO 内部所有 `(x, y, w, h)` 都改成 `(x, y)`，而是采用最小改造方式：

- DINO 内部 decoder reference 暂时仍为 4D。
- two-stage encoder proposal 暂时仍为 4D。
- `reg_branches` 暂时仍输出 4D。
- 对外输出只保留前两维 `(x, y)`。
- Hungarian Matcher 改为：
  - `FocalLossCost`
  - `PointL1Cost`
- 训练 Loss 改为：
  - `Focal Loss`
  - `Point L1 Loss`
- 删除普通 matching 路径的：
  - BBox L1 Loss
  - GIoU Loss
- 第一阶段关闭原始 box-based DN。
- 数据输入改为纯 point annotation，不构造伪 bbox。
- 推理输出改为 `scores + labels + points`。
- 新增 Point Metric，用点距离进行一对一评估。

---

# 2. 模型输出改造

## 2.1 文件

```text
mmdet/models/dense_heads/dino_head.py
```

## 2.2 新增 `forward()`

原始 `DINOHead` 继承 `DeformableDETRHead.forward()`，输出回归结果：

```text
(cx, cy, w, h)
```

阶段一中没有直接修改底层：

```python
Linear(self.embed_dims, 4)
```

因为这组 `reg_branches` 同时被 decoder iterative refinement 和 two-stage 使用。

因此在 `DINOHead` 中新增 `forward()`，内部仍执行原始 4D 回归，只在对外输出时截取前两维：

```python
def forward(self, hidden_states: Tensor,
            references: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """Forward the DINO head and keep only point coordinates.

    The internal DINO regression branches still predict 4D
    (cx, cy, w, h) for decoder refinement and two-stage proposals.
    For the external prediction task, only (cx, cy) is returned.
    """
    all_layers_cls_scores, all_layers_bbox_preds = super().forward(
        hidden_states, references)

    all_layers_point_preds = all_layers_bbox_preds[..., :2]

    return all_layers_cls_scores, all_layers_point_preds
```

### 结果

对外回归张量由：

```text
[num_decoder_layers, B, Q, 4]
```

变为：

```text
[num_decoder_layers, B, Q, 2]
```

语义变为：

```text
(x, y)
```

但内部 decoder / encoder 仍可继续使用 4D 几何。

---

# 3. 关闭原始 DINO DN

## 3.1 文件

```text
mmdet/models/detectors/dino.py
```

## 3.2 增加 `use_dn`

给 `DINO.__init__()` 增加：

```python
use_dn: bool = True
```

并保存：

```python
self.use_dn = use_dn
```

原始：

```python
self.dn_query_generator = CdnQueryGenerator(**dn_cfg)
```

没有删除。

这样做的原因是：

- 第一阶段只关闭 DN 的使用；
- 保留模块本身，避免 checkpoint 结构发生更多不必要变化；
- 第二阶段后续可以继续改造成 Point DN。

## 3.3 训练时跳过 DN query

原逻辑：

```python
if self.training:
    dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
        self.dn_query_generator(batch_data_samples)

    query = torch.cat([dn_label_query, query], dim=1)

    reference_points = torch.cat(
        [dn_bbox_query, topk_coords_unact], dim=1)
else:
    reference_points = topk_coords_unact
    dn_mask, dn_meta = None, None
```

改为：

```python
if self.training and self.use_dn:
    dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
        self.dn_query_generator(batch_data_samples)

    query = torch.cat([dn_label_query, query], dim=1)

    reference_points = torch.cat(
        [dn_bbox_query, topk_coords_unact], dim=1)
else:
    reference_points = topk_coords_unact
    dn_mask, dn_meta = None, None
```

## 3.4 DN label embedding 特殊逻辑增加开关

原逻辑：

```python
if len(query) == self.num_queries:
    inter_states[0] += \
        self.dn_query_generator.label_embedding.weight[0, 0] * 0.0
```

改为：

```python
if len(query) == self.num_queries and self.use_dn:
    inter_states[0] += \
        self.dn_query_generator.label_embedding.weight[0, 0] * 0.0
```

---

# 4. `split_outputs()` 适配 DN 关闭

## 4.1 文件

```text
mmdet/models/dense_heads/dino_head.py
```

原实现会先执行：

```python
num_denoising_queries = dn_meta['num_denoising_queries']
```

再判断：

```python
if dn_meta is not None:
```

当第一阶段：

```python
dn_meta = None
```

时会直接报错。

因此改为：

```python
if dn_meta is not None:
    num_denoising_queries = dn_meta['num_denoising_queries']

    all_layers_denoising_cls_scores = \
        all_layers_cls_scores[:, :, :num_denoising_queries, :]
    all_layers_denoising_bbox_preds = \
        all_layers_bbox_preds[:, :, :num_denoising_queries, :]
    all_layers_matching_cls_scores = \
        all_layers_cls_scores[:, :, num_denoising_queries:, :]
    all_layers_matching_bbox_preds = \
        all_layers_bbox_preds[:, :, num_denoising_queries:, :]
else:
    all_layers_denoising_cls_scores = None
    all_layers_denoising_bbox_preds = None
    all_layers_matching_cls_scores = all_layers_cls_scores
    all_layers_matching_bbox_preds = all_layers_bbox_preds
```

DN 相关：

```python
loss_dn()
_loss_dn_single()
get_dn_targets()
_get_dn_targets_single()
```

第一阶段保留原代码，不调用。

---

# 5. Hungarian Matcher 改造成 Point Matcher

## 5.1 新增 `PointL1Cost`

### 文件

```text
mmdet/models/task_modules/assigners/match_cost.py
```

新增：

```python
@TASK_UTILS.register_module()
class PointL1Cost(BaseMatchCost):
    """L1 matching cost for 2D point prediction."""

    def __init__(self, weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:

        pred_points = pred_instances.points
        gt_points = gt_instances.points

        img_h, img_w = img_meta['img_shape']

        factor = gt_points.new_tensor(
            [img_w, img_h]).unsqueeze(0)

        gt_points = gt_points / factor

        point_cost = torch.cdist(
            pred_points,
            gt_points,
            p=1)

        return point_cost * self.weight
```

### 数学含义

预测点是归一化坐标：

```text
(x_pred, y_pred) ∈ [0,1]
```

GT point 是像素坐标，先除以：

```text
(W, H)
```

再计算：

```text
|x_pred - x_gt| + |y_pred - y_gt|
```

---

## 5.2 注册 `PointL1Cost`

### 文件

```text
mmdet/models/task_modules/assigners/__init__.py
```

增加：

```python
from .match_cost import (..., PointL1Cost)
```

并加入：

```python
__all__
```

---

## 5.3 修改 DINO Matcher 配置

### 文件

```text
configs/dino/dino-4scale_r50_8xb2-12e_coco.py
```

原配置：

```python
match_costs=[
    dict(type='FocalLossCost', weight=2.0),
    dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
    dict(type='IoUCost', iou_mode='giou', weight=2.0)
]
```

改为：

```python
match_costs=[
    dict(type='FocalLossCost', weight=2.0),
    dict(type='PointL1Cost', weight=5.0)
]
```

因此阶段一 Hungarian Matcher 为：

```text
Classification Cost + Point L1 Cost
```

不再使用 bbox cost 和 IoU cost。

---

# 6. 普通 matching target 改为 point target

## 6.1 文件

```text
mmdet/models/dense_heads/dino_head.py
```

在 `DINOHead` 中重写：

```python
_get_targets_single()
```

核心变化：

原始父类创建：

```python
pred_instances = InstanceData(
    scores=cls_score,
    bboxes=bbox_pred)
```

阶段一改成：

```python
pred_instances = InstanceData(
    scores=cls_score,
    points=point_pred)
```

GT 使用：

```python
gt_instances.points
gt_instances.labels
```

而不是：

```python
gt_instances.bboxes
```

### point target

GT point 原始为像素坐标：

```python
gt_points
```

按图像大小归一化：

```python
img_h, img_w = img_meta['img_shape']

factor = gt_points.new_tensor(
    [img_w, img_h]).unsqueeze(0)

pos_gt_points_normalized = pos_gt_points / factor
```

构建：

```python
point_targets
point_weights
```

仅 Hungarian 匹配到的 positive query：

```python
point_weights[pos_inds] = 1.0
```

---

# 7. Loss 改为 `Focal Loss + Point L1 Loss`

## 7.1 文件

```text
mmdet/models/dense_heads/dino_head.py
```

## 7.2 新增 `loss_by_feat_single()`

阶段一不再调用父类的 bbox loss 逻辑，而是在 `DINOHead` 中新增 point 版本。

分类部分继续使用：

```python
self.loss_cls
```

即原 DINO 的 Focal Loss。

回归部分继续复用原配置中的：

```python
self.loss_bbox
```

其模块类型本身是：

```text
L1Loss
```

但输入由 4D bbox 改为二维 point：

```python
point_preds = point_preds.reshape(-1, 2)

loss_point = self.loss_bbox(
    point_preds,
    point_targets,
    point_weights,
    avg_factor=num_total_pos)
```

因此虽然成员变量仍叫：

```text
loss_bbox
```

实际计算内容已经变为：

```text
Point L1 Loss
```

---

## 7.3 重写 `loss_by_feat()`

普通 decoder matching loss 变为：

```text
loss_cls
loss_point
```

辅助 decoder layer：

```text
d0.loss_cls
d0.loss_point
...
d4.loss_cls
d4.loss_point
```

### Encoder two-stage loss

第一阶段 encoder proposal 内部仍是 4D：

```text
(x, y, w, h)
```

但训练时只取：

```python
enc_point_preds = enc_bbox_preds[..., :2]
```

因此 encoder loss 变为：

```text
enc_loss_cls
enc_loss_point
```

### 第一阶段普通训练路径中删除

```text
loss_bbox
loss_iou
GIoU
```

### DN loss

由于：

```python
use_dn=False
```

DN loss 不进入训练。

---

# 8. 单类别化

## 8.1 文件

```text
configs/dino/dino-4scale_r50_8xb2-12e_coco.py
```

将：

```python
num_classes=80
```

改为：

```python
num_classes=1
```

因此：

```text
cls_out_channels = 1
```

每个 query 输出一个前景 point score。

---

# 9. COCO checkpoint 兼容处理

单类别化后，原 COCO checkpoint 有 15 个 shape mismatch：

```text
bbox_head.cls_branches.0.weight
bbox_head.cls_branches.0.bias
...
bbox_head.cls_branches.6.weight
bbox_head.cls_branches.6.bias
dn_query_generator.label_embedding.weight
```

其中：

- 7 个 cls branch：
  - 7 个 weight
  - 7 个 bias
- 1 个 DN label embedding

共：

```text
15
```

这些参数由：

```text
80 类
```

变成：

```text
1 类
```

因此生成过滤后的初始化权重：

```text
dino_r50_4scale_coco_point1cls.pth
```

仅删除这 15 个 shape 不兼容参数。

其余 COCO DINO 预训练参数继续加载，包括：

- ResNet-50 backbone
- neck
- encoder
- decoder
- attention
- 4D regression branches
- two-stage 相关参数

验证结果：

```text
Missing keys: 15
Unexpected keys: 0
```

---

# 10. Point annotation 打包

## 10.1 文件

```text
mmdet/datasets/transforms/formatting.py
```

修改：

```python
class PackDetInputs(BaseTransform):
```

中的：

```python
mapping_table
```

增加：

```python
'gt_points': 'points',
'gt_points_labels': 'labels',
```

最终 point 数据会进入：

```python
data_sample.gt_instances.points
data_sample.gt_instances.labels
```

而不是 bbox。

---

# 11. Point annotation 加载

## 11.1 文件

```text
mmdet/datasets/transforms/loading.py
```

给：

```python
LoadAnnotations
```

增加：

```python
with_point: bool = False
```

并保存：

```python
self.with_point = with_point
```

## 11.2 新增 `_load_points()`

输入统一实例格式：

```python
{
    'point': [x, y],
    'point_label': 0,
    'ignore_flag': 0
}
```

转换为 pipeline 字段：

```python
results['gt_points']
results['gt_points_labels']
results['gt_ignore_flags']
```

## 11.3 `transform()` 增加调用

加入：

```python
if self.with_point:
    self._load_points(results)
```

---

# 12. RandomFlip 支持 point

## 12.1 文件

```text
mmdet/datasets/transforms/transforms.py
```

在：

```python
class RandomFlip
```

的 `_flip()` 中新增 point 变换。

### 水平翻转

```python
x' = W - 1 - x
```

### 垂直翻转

```python
y' = H - 1 - y
```

### diagonal

同时更新 x、y。

实现核心：

```python
if results.get('gt_points', None) is not None:
    h, w = img_shape
    direction = results['flip_direction']

    if direction == 'horizontal':
        results['gt_points'][:, 0] = \
            w - 1 - results['gt_points'][:, 0]

    elif direction == 'vertical':
        results['gt_points'][:, 1] = \
            h - 1 - results['gt_points'][:, 1]

    elif direction == 'diagonal':
        ...
```

---

# 13. Resize 支持 point

## 13.1 文件

```text
mmdet/datasets/transforms/transforms.py
```

在：

```python
class Resize
```

中新建：

```python
_resize_points()
```

核心：

```python
w_scale, h_scale = results['scale_factor']

results['gt_points'][:, 0] *= w_scale
results['gt_points'][:, 1] *= h_scale
```

并在：

```python
transform()
```

中增加：

```python
self._resize_points(results)
```

因此 `RandomChoiceResize` 内部调用 Resize 时，point 会自动同步变换。

---

# 14. RandomCrop 支持 point

## 14.1 文件

```text
mmdet/datasets/transforms/transforms.py
```

在：

```python
class RandomCrop
```

的 `_crop_data()` 中增加：

1. point 减去 crop 偏移：

```python
points[:, 0] -= offset_w
points[:, 1] -= offset_h
```

2. 判断 point 是否仍在 crop 内：

```python
valid_inds = (
    (points[:, 0] >= 0) &
    (points[:, 0] < crop_w) &
    (points[:, 1] >= 0) &
    (points[:, 1] < crop_h)
)
```

3. 删除 crop 外 point。

4. 同步过滤：

```python
gt_points_labels
gt_ignore_flags
```

5. 当：

```python
allow_negative_crop=False
```

且 crop 后没有 point 时返回 `None`。

当前 DINO pipeline 使用：

```python
allow_negative_crop=True
```

因此允许生成无目标点的负样本 crop。

---

# 15. Train pipeline 改为 point

## 15.1 文件

```text
configs/dino/dino-4scale_r50_8xb2-12e_coco.py
```

原：

```python
dict(type='LoadAnnotations', with_bbox=True)
```

改为：

```python
dict(
    type='LoadAnnotations',
    with_bbox=False,
    with_label=False,
    with_point=True)
```

后续：

- RandomFlip
- RandomChoice
- Resize
- RandomCrop
- PackDetInputs

均已经完成 point 适配。

---

# 16. Val / Test pipeline 改为 point

## 16.1 文件

```text
configs/dino/dino-4scale_r50_8xb2-12e_coco.py
```

新增 / 覆盖：

```python
test_pipeline
```

核心：

```python
dict(type='LoadImageFromFile', ...),
dict(type='Resize', scale=(1333, 800), keep_ratio=True),
dict(
    type='LoadAnnotations',
    with_bbox=False,
    with_label=False,
    with_point=True),
dict(type='PackDetInputs', ...)
```

同时覆盖：

```python
val_dataloader.dataset.pipeline
test_dataloader.dataset.pipeline
```

避免继续继承 COCO bbox pipeline。

---

# 17. 推理后处理改为 point

## 17.1 文件

```text
mmdet/models/dense_heads/dino_head.py
```

在 `DINOHead` 中重写：

```python
_predict_by_feat_single()
```

### 分类

保持 sigmoid：

```python
cls_score = cls_score.sigmoid()
```

### top-k

根据 score 选择 query。

### 坐标

不再进行：

```text
cxcywh → xyxy
```

而是直接选择：

```python
det_points = point_pred[point_indexes]
```

归一化坐标恢复到 resized image：

```python
det_points[:, 0] *= img_w
det_points[:, 1] *= img_h
```

若：

```python
rescale=True
```

则除以：

```python
scale_factor
```

恢复到原始图像坐标。

### 最终输出

```python
results = InstanceData()
results.points = det_points
results.scores = scores
results.labels = det_labels
```

不再返回：

```python
results.bboxes
```

---

# 18. `max_per_img` 调整

## 18.1 文件

```text
configs/dino/dino-4scale_r50_8xb2-12e_coco.py
```

原：

```python
test_cfg=dict(max_per_img=300)
```

改为：

```python
test_cfg=dict(max_per_img=900)
```

原因：

```text
num_queries = 900
```

Point-DINO 单类别时每个 query 最多对应一个点。

若仍限制 300，会人为限制每张图最多只能输出 300 个 point。

---

# 19. ShanghaiTech Part B 转统一 Point JSON

## 19.1 新文件

```text
tools/dataset_converters/shanghaitech_to_point_json.py
```

读取 ShanghaiTech 原始 `.mat`：

```python
points = mat['image_info'][0, 0][0, 0][0]
```

转换为统一格式：

```json
{
  "metainfo": {
    "classes": ["point"]
  },
  "data_list": [
    {
      "img_id": 0,
      "img_path": "IMG_1.jpg",
      "width": 1024,
      "height": 768,
      "instances": [
        {
          "point": [x, y],
          "point_label": 0,
          "ignore_flag": 0
        }
      ]
    }
  ]
}
```

生成：

```text
train_point.json
test_point.json
```

实际数据：

```text
train: 400 images
test: 316 images
```

阶段一最终没有使用 ShanghaiTech 专用 Dataset 类，而是采用通用：

```text
BaseDetDataset + 统一 Point JSON
```

因此以后工业数据只需要转换成相同 JSON 格式即可复用。

---

# 20. 数据集配置改为 `BaseDetDataset`

## 20.1 文件

```text
configs/dino/dino-4scale_r50_8xb2-12e_coco.py
```

数据根目录：

```text
/root/autodl-tmp/dinov3_dino_mmdet/data/
ShanghaiTech_Crowd_Counting_Dataset/part_B_final/
```

### Train

```python
type='BaseDetDataset'
ann_file='train_point.json'
data_prefix=dict(
    img_path='train_data/images/')
```

### Val / Test

```python
type='BaseDetDataset'
ann_file='test_point.json'
data_prefix=dict(
    img_path='test_data/images/')
```

使用 `_delete_=True` 覆盖原 COCO dataset 配置。

---

# 21. 新增 PointMetric

## 21.1 新文件

```text
mmdet/evaluation/metrics/point_metric.py
```

继承：

```python
BaseMetric
```

注册：

```python
@METRICS.register_module()
class PointMetric(BaseMetric):
```

支持两个配置参数：

```python
distance_threshold
score_threshold
```

---

## 21.2 PointMetric 匹配方式

预测点先按：

```python
score >= score_threshold
```

过滤。

随后预测点和 GT point 计算欧氏距离：

```text
sqrt((x_pred-x_gt)^2 + (y_pred-y_gt)^2)
```

使用 Hungarian assignment 做一对一匹配。

仅满足：

```text
distance <= distance_threshold
```

的 pair 计为 TP。

统计：

```text
TP
FP
FN
Precision
Recall
F1
Mean Localization Error
```

---

## 21.3 适配 Runner 的 dict 输入

初始版本按：

```python
data_sample.pred_instances
```

访问。

正式 Runner 验证发现 MMEngine `Evaluator` 实际传给 metric 的是普通 `dict`。

因此 `process()` 改为兼容：

```python
if isinstance(data_sample, dict):
    pred_instances = data_sample['pred_instances']
    gt_instances = data_sample['gt_instances']
else:
    pred_instances = data_sample.pred_instances
    gt_instances = data_sample.gt_instances
```

因此：

- 手工单元测试可用；
- MMEngine Runner 正式验证也可用。

---

# 22. 注册 PointMetric

## 22.1 文件

```text
mmdet/evaluation/metrics/__init__.py
```

增加：

```python
from .point_metric import PointMetric
```

并加入：

```python
__all__
```

---

# 23. Val / Test evaluator 改为 PointMetric

## 23.1 文件

```text
configs/dino/dino-4scale_r50_8xb2-12e_coco.py
```

覆盖 COCO evaluator：

```python
val_evaluator = dict(
    _delete_=True,
    type='PointMetric',
    distance_threshold=10.0,
    score_threshold=0.5)

test_evaluator = dict(
    _delete_=True,
    type='PointMetric',
    distance_threshold=10.0,
    score_threshold=0.5)
```

其中：

```text
10 px
0.5 score
```

阶段一首先作为 baseline / 联调参数使用。

---

# 24. Smoke Test 配置

## 24.1 新文件

```text
configs/dino/point_dino_r50_shanghaitech_smoke.py
```

继承：

```python
_base_ = './dino-4scale_r50_8xb2-12e_coco.py'
```

使用：

```text
IterBasedTrainLoop
max_iters=10
batch_size=1
InfiniteSampler
```

用于验证真实训练是否能：

```text
data
→ model
→ point matcher
→ point loss
→ backward
→ checkpoint
```

10 iter 训练成功。

训练日志只出现：

```text
loss_cls
loss_point
d0~d4.loss_cls
d0~d4.loss_point
enc_loss_cls
enc_loss_point
```

没有：

```text
loss_bbox
loss_iou
DN loss
```

---

# 25. 正式 12 epoch 配置

## 25.1 新文件

```text
configs/dino/point_dino_r50_shanghaitech_12e.py
```

作为阶段一正式 baseline。

主要设置：

```python
load_from = '.../dino_r50_4scale_coco_point1cls.pth'
```

训练：

```text
EpochBasedTrainLoop
12 epochs
val every epoch
```

Optimizer：

```text
AdamW
lr=1e-4
weight_decay=1e-4
```

Backbone：

```text
lr_mult=0.1
```

Gradient clipping：

```text
max_norm=0.1
```

Scheduler：

```text
MultiStepLR
milestone=11
gamma=0.1
```

Batch：

```text
batch_size=2
```

Checkpoint：

```text
save_best = point/f1
```

---

# 26. 环境兼容处理

这部分不属于 Point-DINO 方法本身，但属于实际工程运行环境。

## 26.1 PyTorch 2.6 checkpoint 加载

由于 PyTorch 2.6 默认：

```text
torch.load(weights_only=True)
```

MMDetection 官方旧 checkpoint 中包含 MMEngine 对象，加载时会报错。

运行前使用：

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

---

## 26.2 确保使用本地修改后的 MMDetection

曾出现：

```text
site-packages/mmdet
```

覆盖本地源码的问题。

训练前增加：

```bash
export PYTHONPATH=/root/autodl-tmp/dinov3_dino_mmdet/mmdetection:$PYTHONPATH
```

确保：

```python
import mmdet
print(mmdet.__file__)
```

指向：

```text
/root/autodl-tmp/dinov3_dino_mmdet/mmdetection/mmdet/
```

而不是：

```text
/root/miniconda3/lib/python3.12/site-packages/mmdet/
```

---

# 27. 阶段一完成的验证

阶段一实际完成了以下验证。

## 模块级

- `PointL1Cost` 数学测试通过。
- HungarianAssigner point matching 测试通过。
- `PackDetInputs` point 测试通过。
- Point annotation loader 测试通过。
- RandomFlip point 测试通过。
- Resize point 测试通过。
- RandomCrop point 测试通过。
- train pipeline 完整 point 联调通过。
- val pipeline 完整 point 联调通过。
- Point postprocess 测试通过。
- PointMetric 数学测试通过。
- PointMetric Runner dict 测试通过。

## 真实数据

- ShanghaiTech Part B `.mat → Point JSON` 转换成功。
- 真实 `BaseDetDataset → gt_instances.points` 测试通过。
- 真实模型推理输出：
  - `points`
  - `scores`
  - `labels`
- 不包含 `bboxes`。

## 真实训练

完整：

```text
image
→ backbone
→ encoder
→ two-stage
→ decoder
→ Point-DINO head
→ Hungarian point matching
→ Focal Loss + Point L1 Loss
→ backward
```

测试通过。

10 iteration smoke training 通过。

正式 12 epoch baseline 已启动并正常训练。

---

# 28. 阶段一最终模型定义

阶段一的 Minimal Point-DINO 可以概括为：

## 内部

仍保留：

```text
4D encoder proposal
4D decoder reference
4D reg_branches
```

## 对外

每个 query 输出：

```text
score + (x, y)
```

## Matcher

```text
FocalLossCost + PointL1Cost
```

## Loss

```text
Focal Loss + Point L1 Loss
```

包括：

```text
decoder final loss
decoder auxiliary loss
encoder two-stage loss
```

## DN

```text
关闭
```

## 数据

```text
纯 point annotation
不使用 bbox
不构造伪 bbox
```

## 评估

```text
Point distance + Hungarian one-to-one matching
Precision / Recall / F1 / Localization Error
```

---

# 29. 阶段一当前 baseline 结果

当前阶段一正式训练已经得到一组有效验证结果：

```text
Precision: 0.8797
Recall:    0.7949
F1:        0.8352
Mean Localization Error: 3.6646 px

TP: 31168
FP: 4261
FN: 8040
```

该结果说明第一阶段 Minimal Point-DINO 已经不只是工程可运行，而是具备有效的 point detection 能力。

---

# 30. 第二阶段待改内容

第二阶段目标是把阶段一保留的内部 4D 几何真正去掉。

计划顺序：

1. Decoder internal reference：
   ```text
   (x, y, w, h) → (x, y)
   ```

2. Two-stage encoder proposal：
   ```text
   4D proposal → 2D point proposal
   ```

3. Regression branch：
   ```text
   4D internal regression → 2D point regression
   ```

4. Point DN：
   - 重新开启 DN；
   - GT 只扰动 `(x,y)`；
   - 不再生成或扰动 `w,h`。

阶段二最终目标：

> **DINO 内部和外部几何表达均统一为二维 point。**
