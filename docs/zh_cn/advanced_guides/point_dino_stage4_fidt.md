# Stage 4：Full-map / Local FIDT Auxiliary Supervision

Stage 4 在现有 Point-DINO 的 query-level 点监督之外，将同一份点标注表示为
dense spatial distance-field target，直接监督共享空间特征。研究目标是验证这种
梯度分布是否改善 fine localization，尤其是 F1@4；没有增加新的标注信息，
也不预设 F1 会提高。最终预测始终是 query 直接输出 `(x, y)`。

旧 Full-map 模式继续保留。新增 `local_radius_px` 仅改变参与 MSE 的空间位置：
`None` 使用所有 valid cells；正数使用 `valid AND D<=R`，R 的单位是当前增强后
训练图像的像素。FIDT 公式、空间坐标映射和辅助 head 结构均保持原样。

## 当前 Stage 3 实现

本仓库根目录直接包含 `mmdet/`、`configs/`、`tools/`，没有额外的
`mmdetection/` 子目录。以下路径均相对仓库根目录。

- detector：`mmdet/models/detectors/dino.py`。
- 特征提取和原训练/推理入口：`mmdet/models/detectors/base_detr.py`。
- 多尺度展开与 encoder：`mmdet/models/detectors/deformable_detr.py`。
- 点 head、Euclidean、decoder/encoder/DN losses：
  `mmdet/models/dense_heads/dino_head.py`。
- 二维 reference refinement 与 Point DN：
  `mmdet/models/layers/transformer/dino_layers.py`。
- PointL1Cost：`mmdet/models/task_modules/assigners/match_cost.py`。
- 选定基线：`configs/dino/point_dino_stage3_lam020_cost20.py`。

真实训练调用链：`extract_feat → backbone → neck → forward_transformer →
pre_transformer → forward_encoder → pre_decoder (Point DN) → forward_decoder →
DINOHead.loss → loss_by_feat`。回归层和 reference 均为二维；部分已有 docstring
仍描述 boxes，不能据此将当前实现误判为 bbox DINO。

基线使用 ResNet50，`out_indices=(1,2,3)`，ChannelMapper 输出 256 通道。
对 `[B,3,768,1024]` 输入，形状如下（H 在前，W 在后）：

| Tensor | Shape | Stride |
| --- | --- | --- |
| backbone[0] | `[B,512,96,128]` | 8 |
| backbone[1] | `[B,1024,48,64]` | 16 |
| backbone[2] | `[B,2048,24,32]` | 32 |
| mlvl_feats[0] | `[B,256,96,128]` | 8 |
| mlvl_feats[1] | `[B,256,48,64]` | 16 |
| mlvl_feats[2] | `[B,256,24,32]` | 32 |
| mlvl_feats[3] | `[B,256,12,16]` | 64 |

分类训练权重为 1、Point L1 训练权重为 5。Hungarian 使用 FocalLossCost=2、
PointL1Cost=20，后者不是训练 loss 权重。Euclidean 保留现有正样本 pixel-space
距离、按 8 像素归一化及 lambda=0.20。Point DN noise=0.01。6 层 decoder、
encoder、6 层 DN 共 39 个原有 loss 项全部保留。

## 辅助分支与训练损失

新增 `mmdet/models/dense_heads/point_fidt_aux_head.py`，由 DINO 直接构造，
不添加 registry/import 导出。仅 `DINO.loss()` 调用它。

```text
shared mlvl_feats[0] [B,256,96,128]
  ├─ 原有 transformer → query → (x,y), scores
  └─ Conv3x3(256→64) → GN(8) → ReLU
       → bilinear x2 (align_corners=False)
       → Conv3x3(64→64) → GN(8) → ReLU
       → Conv1x1(64→1) → float32 sigmoid
       → [B,1,192,256] FIDT prediction (仅训练)
```

detector 选择空间面积最大的 transformer 输入，本配置中就是 `mlvl_feats[0]`。
原特征不 detach、不原地修改，原 transformer 输入分辨率不变。辅助分支约
18.5 万参数，无 Transformer/attention；梯度可以进入共享 neck 和未冻结的 backbone。
若实际 backbone 改变，需重新检查 feature shape 并配置 `upsample_factor`：
stride 4 用 1，stride 8 用 2。本实现不为其他 stride 猜测重采样方案。

定义 raw 辅助 loss 为：

```text
M = valid                         # local_radius_px=None，旧 Full-map
M = valid AND nearest_distance<=R # local_radius_px>0，Local
L_fidt_raw = sum(M * (sigmoid(raw_output) - target)^2) / max(sum(M), 1)
loss_fidt = lambda_fidt * L_fidt_raw
L_total = L_stage3 + lambda_fidt * L_fidt_raw
```

分母是整个 batch 的所选 cell 数，不是整张图面积，也不先对每张图求 mean。
全无所选 cell 时，masked squared error 的零和仍连接 prediction 图，反向梯度为零。
FIDT 只计算一次，不参与 Hungarian、encoder matching、decoder auxiliary 或 Point DN。
`debug=True` 额外返回 detached 标量 `fidt_mse`；MMEngine parser 只将名称包含
`loss` 的项累加，因此这个统计不会再次参与总损失。`loss_fidt` 始终是加权后值。

## 在线 FIDT target

数据链为 `LoadAnnotations(with_point=True) → Resize → RandomFlip → PackDetInputs`。
`gt_instances.points` 是增强后的 float32 pixel coordinates，直接使用，不取整。
所有 target/grid/distance 计算都在 `torch.no_grad()` 中完成，无预生成文件，
不依赖 scipy/OpenCV distance transform。

对于 output map `[H_f,W_f]`，从实际 `batch_inputs.shape[-2:]` 得到 padded H/W：

```text
x_u = (u + 0.5) * padded_width  / W_f
y_v = (v + 0.5) * padded_height / H_f
D(q) = min_i ||q - g_i||_2
T(q) = 1 / (D(q)^(0.02*D(q) + 0.75) + 1)
```

`D=0` 时 `T=1`。实现用等价 log-domain 公式避免远距离幂运算溢出；非常远的
float32 target 可以数值下溢为 0。多个 GT 取最近距离，绝不相加。

`PackDetInputs` 提供 `img_shape`；`DetDataPreprocessor` 添加
`batch_input_shape` 和每图 `pad_shape`。后者可能比 batch tensor 小，不能用于
映射整个 batch grid。代码校验已有 `batch_input_shape` 与 tensor 一致，并仅让
`x < img_width && y < img_height` 的 cell center 进入 loss。padding 不进入分子
或分母。GT 为空时 target 全 0：Full-map 模式仍监督有效背景；Local 模式的 mask
全 False，该图没有 FIDT supervision。非空图的 Local 半径外 target 保持原值，
只是从 loss 中排除，不新增 background/ring loss。

每次对至多 4096 个有效 spatial locations 和 256 个 GT 计算 `torch.cdist`，
跨 GT chunks 继续取最小值；临时距离矩阵最多约 4 MiB（float32）。使用直接
Euclidean kernel，避免矩阵乘法形式在接近 GT 时的相消误差。没有 `[B,H*W,N]`
距离张量。sigmoid/MSE/target 均采用 float32 数值。

## 配置与运行

新配置：`configs/dino/point_dino_r50_shanghaitech_stage4_fidt_native_12e.py`。
继承选定 Stage 3 配置，显式固定 Euclidean=0.20、matcher=2/20、DN noise=0.01；
native 输入、数据格式、优化器、query 数量与 4/8 px metric 均继承不变。

辅助配置在 `model.point_fidt_head`：`enabled=True`、`gamma=0.02`、`phi=0.75`、
`xi=1.0`、`upsample_factor=2`、`fidt_loss_weight=1.0`、`debug=True`。
**1.0 仅用于功能 smoke，未经过权重选择，也不是最佳实验参数。** 先记录
`fidt_mse`、`loss_fidt` 和主损失量级，再决定后续实验权重。

关闭方式：省略 `point_fidt_head`、`enabled=False` 或 `fidt_loss_weight=0`。
关闭时不创建辅助参数、不消耗辅助初始化 RNG，直接走原 `super().loss()`。
开启时，Stage 3 state_dict 用 `strict=False` 加载，只允许新的
`point_fidt_head.*` 参数 missing，原有参数形状不变。

继承的 `data_root` 和 `load_from` 是 `/root/autodl-tmp/...` 路径。
`load_from` 仍是已有 Stage 2 初始化；公平比较 Stage 3、Full-map、Local 时必须
从这个相同初始化开始，不改为 Stage 3 best checkpoint。迁移机器时按实际路径
覆盖同一初始化文件及 train/val/test dataset 的 data_root。
不要将不存在的 checkpoint 当成已经验证。

```bash
python tools/train.py configs/dino/point_dino_r50_shanghaitech_stage4_fidt_native_12e.py --cfg-options model.point_fidt_head.debug=True
```

`predict()`、`test_step()` 和 PointMetric 没有修改；这些路径不执行辅助 head，
输出仍只有 query points/scores/labels。没有 FIDT peak、NMS、密度/count map，
也没有额外 FIDT 推理 FLOPs。开启分支后的训练会更新共享权重，因此不同训练
checkpoint 的预测数值可能变化；相同 Stage 3 权重的推理结果应完全一致。

## Sanity 与回归测试

```bash
python -m pytest -o addopts= -q -s tests/test_models/test_dense_heads/test_point_fidt_aux_head.py
python -m pytest -o addopts= -q -s tests/test_models/test_detectors/test_point_dino_stage4.py
```

第一组只需 PyTorch、pytest、MMEngine：配置、单 GT、双 GT nearest 而非求和、
subpixel、empty GT、padding、chunk 内存上界、AMP 数值、辅助梯度和 native 输出。
第二组需要可导入 compiled ops 的 MMCV 及项目 runtime 依赖，使用真实 detector
调用链验证合成训练、39 个原 loss、parser、共享梯度、推理不调用分支、Stage 3
state_dict 兼容，以及 native backbone/neck shape。合成 smoke 的训练规模缩小，
不能替代真实数据/训练 checkpoint 的功能测试或 1024×768 训练性能测试。

## 初始 Full-map 本地验证记录（2026-09-09）

已有 Python 环境没有 torch/MMCV/MMEngine。本次只在临时隔离 venv 安装测试
依赖，未改全局或现有 conda 环境：

```text
C:\Users\14222\AppData\Local\Temp\point-dino-stage4-venv\Scripts\python.exe
Python 3.10.20 / torch 2.1.0+cpu / torchvision 0.16.0+cpu
mmcv 2.1.0（官方 Windows CPU wheel）/ mmengine 0.10.7
numpy 1.26.4 / pytest 9.1.1
```

- 第一组 9 项通过；第二组 5 项通过，共 14 项，无跳过或失败。
- 实际 native backbone/neck 与辅助输出形状均符合上表。
- 真实合成训练前向、完整损失反向和一次 SGD 参数更新通过。
- raw MSE=`0.1631237566`；weight=1 的 `loss_fidt=0.1631237566`；
  Stage 3 total=`33.69027710`；加入 FIDT 后 total=`33.85340118`。
  数值均 finite，仅说明此合成样本的功能/量级正常，不决定实验权重。
- 同 core weights、同 DN 随机种子下，原有 39 个 loss 与 Stage 3 逐项完全一致。
- FIDT 单独反向：辅助 head、neck、backbone.layer2 有非零有限梯度；
  encoder、decoder、query、point head 和 Point DN 无 FIDT 梯度。
- `fidt_mse` 可被真实 MMEngine parser 记录，移除它不改变总 loss。
- `predict/test_step` 没有辅助 head 调用，同 core weights 的 points/scores/labels
  与 Stage 3 逐元素完全相等，合成测试点形状为 `[30,2]`；正式配置仍为 900 queries。
- zero-weight 和 disabled 的参数键、形状、初始化值/RNG、预测都与 Stage 3 一致；
  zero-weight 另外验证了全部 39 项训练 loss 一致且不返回 FIDT 项。
- Stage 3 随机初始化 state_dict 加载仅缺 8 个辅助参数键，无 unexpected keys
  或旧参数 shape mismatch。未将此声称为真实训练 checkpoint 的加载测试。
- 本地没有 ShanghaiTech 数据或训练好的 Stage 3 checkpoint；真实数据训练、
  实际 checkpoint inference、CUDA/DDP 和 native 全模型训练尚未验证。

集成测试的规模是 256×256、30 matching queries、2 encoder layers、
6 decoder layers；所有算子、matcher、Point DN 和模型调用链均为真实实现，
没有 mock。native 768×1024 只测试真实特征提取和辅助 head 的前向。

## Local FIDT 配置与实现

新增 `configs/dino/point_dino_r50_shanghaitech_stage4_local_fidt_native_12e.py`，
继承已有 Full-map 配置，只增加 `local_radius_px=16.0` 和独立 work_dir；
`fidt_loss_weight=1.0`、`debug=True`。12 epochs、native 1024×768、Euclidean
0.20、matcher 2/20、Point L1 训练权重 5、DN noise 0.01 全部保留。
初始化仍为 `point_dino_stage2_step2_init.pth`，不加载 Stage 3 best。

R=16、weight=1 仅是初始功能测试值，没有最优声明，也没有添加 radius/lambda
sweep。旧 Full-map 的 0.25/0.5/1/2 权重结果不能直接确定 Local 的权重范围。
正式训练使用 `debug=False`。

最近距离仍由原有 chunked cdist 计算。同一个 `nearest` 同时产生原 FIDT target
和 `nearest<=local_radius_px`。重叠邻域是布尔 union，每个 cell 最多计一次。
没有重复距离计算，不截断或重新归一化 target，不增加半径外监督。

`generate_fidt_target()` 默认仍返回两个 tensor：`target, loss_mask`。
`local_radius_px=None` 时第二项就是原有 valid mask；内部统计可显式请求
`return_valid_mask=True`，额外获得第三项原 valid mask，避免重新生成 grid。

Local 的 debug 统计为 detached 浮点标量：

- `fidt_mse`：所选 local valid cells 的 raw MSE。
- `fidt_local_count`：本 batch 所选 cell 总数。
- `fidt_local_ratio`：所选 cell 数 / 本 batch 全部 valid image cell 数。

空 GT 图也计入 ratio 的 valid-image 分母，但没有选中 cell，也不进入 MSE 分母。
全部统计名称不包含 `loss`，不会被 MMEngine 加进总 loss。

## Local 本地验证与真实 CUDA smoke

本轮 target/head 测试 **22 项通过**，真实 detector 集成测试 **7 项通过**，
合计 **29 项通过**。包括 local 单点和半径边界、多点 union/重叠去重、padding、
empty/零选中cell反传、不同图选中数量的 global denominator、target 不变、
没有额外 cdist、debug/parser、39 项 Stage 3 loss 保留、梯度隔离及推理零调用。

另外将本轮修改前保存的真实 head 文件与修改后 `local_radius_px=None` 并行加载，
使用相同输入、权重和 seed，验证 raw/weighted loss、输入梯度、全部参数梯度及
初始化逐元素完全相同。该比较包含 padding 和 empty-GT 图：

```text
before/after fidt_mse  = 0.1471332311630249
before/after loss_fidt = 0.10299325734376907  (weight=0.7)
```

Local R=16、weight=1 的真实模型合成 CPU smoke 输出：

```text
loss_fidt             = 0.08782979846
fidt_mse              = 0.08782979846
fidt_local_ratio      = 0.0244140625
fidt_local_count      = 100
loss_point            = 1.290283203125
loss_point_euclidean  = 1.41990363598
parsed total loss     = 33.77810668945
```

这些数值来自 256×256 合成输入、30 matching queries、2 encoder/6 decoder
layers。FIDT-only 反向仅进入 auxiliary head、neck 和未冻结的 backbone；
encoder、decoder、query、point head、DN 没有该项梯度。Full/Local 的
predict/test_step 均零次调用 FIDT，且相同 core weights 输出逐元素一致。

本地环境仍为 CPU PyTorch，且没有本地真实训练数据或 Stage 2 初始化文件，
因此未运行真实数据 CUDA smoke。已有 CUDA 训练环境中，使用原训练入口并开启
逐步日志即可看到上述七项，MMEngine 日志的 `loss` 就是 parsed total：

```bash
python tools/train.py configs/dino/point_dino_r50_shanghaitech_stage4_local_fidt_native_12e.py --cfg-options model.point_fidt_head.debug=True default_hooks.logger.interval=1 log_processor.window_size=1
```

该命令沿用 12e 配置；先观察真实数据前几步的 loss scale，再确定正式权重实验。
正式实验关闭 debug 并恢复常规日志间隔。旧 Full-map 配置和 detector 文件在本轮
保持逐字节不变；本地未找到 `point_dino_stage4_fidt_lam*.py` 文件，因而未单独加载
这些远端实验配置，但省略半径参数会继续走原 Full-map 路径。
