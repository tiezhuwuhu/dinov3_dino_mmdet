import torch

from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner import load_checkpoint

from mmdet.datasets import OCRDinoDataset
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_overlaps
from mmdet.utils import register_all_modules


def get_box_tensor(boxes):
    if hasattr(boxes, 'tensor'):
        return boxes.tensor
    return boxes


register_all_modules()

config_path = (
    'configs/ocr_dino/'
    'ocr-dino-4scale_r50_overfit8_eval.py'
)

checkpoint_path = (
    '/root/autodl-tmp/work_dirs/'
    'ocr_dino_overfit8/iter_300.pth'
)

cfg = Config.fromfile(config_path)

model = MODELS.build(cfg.model)

load_checkpoint(
    model,
    checkpoint_path,
    map_location='cpu',
    strict=False,
)

device = (
    'cuda'
    if torch.cuda.is_available()
    else 'cpu'
)

model = model.to(device)
model.eval()

dataset = OCRDinoDataset(
    data_root='/root/autodl-tmp/dataset/OmniDocBench/',
    ann_file='ocrdino_v1_overfit8.json',
    data_prefix=dict(
        img_path='',
    ),
    pipeline=cfg.test_pipeline,
    serialize_data=False,
)

print('=' * 80)
print('OCR-DINO ERROR INSPECTION')
print('=' * 80)

total_ocr = 0
exact_ocr = 0

for sample_index in range(len(dataset)):
    sample = dataset[sample_index]

    batch = pseudo_collate(
        [sample]
    )

    with torch.no_grad():
        result = model.test_step(
            batch
        )[0]

    pred = result.pred_instances
    gt = result.gt_instances

    pred_boxes = get_box_tensor(
        pred.bboxes
    ).detach().cpu()

    gt_boxes = get_box_tensor(
        gt.bboxes
    ).detach().cpu()

    scale_factor = result.metainfo.get(
        'scale_factor',
        None,
    )

    if scale_factor is not None:
        scale = gt_boxes.new_tensor(
            scale_factor
        ).flatten()

        if scale.numel() == 2:
            scale = scale.repeat(2)

        gt_boxes = gt_boxes / scale

    pred_labels = pred.labels.detach().cpu()
    gt_labels = gt.labels.detach().cpu()

    pred_scores = pred.scores.detach().cpu()

    order = torch.argsort(
        pred_scores,
        descending=True,
    )

    matched_gt = set()

    for pred_index in order.tolist():
        candidates = []

        for gt_index in range(len(gt)):
            if gt_index in matched_gt:
                continue

            if (
                int(gt_labels[gt_index])
                != int(pred_labels[pred_index])
            ):
                continue

            candidates.append(
                gt_index
            )

        if not candidates:
            continue

        ious = bbox_overlaps(
            pred_boxes[
                pred_index:
                pred_index + 1
            ],
            gt_boxes[candidates],
        )[0]

        best_position = int(
            torch.argmax(ious)
        )

        best_iou = float(
            ious[best_position]
        )

        if best_iou < 0.5:
            continue

        gt_index = candidates[
            best_position
        ]

        matched_gt.add(
            gt_index
        )

        text_type = gt.ocr_text_types[
            gt_index
        ]

        if text_type not in (
            'text',
            'latex',
        ):
            continue

        reference = gt.ocr_texts[
            gt_index
        ]

        if not reference:
            continue

        prediction = pred.texts[
            pred_index
        ]

        total_ocr += 1

        if prediction == reference:
            exact_ocr += 1
            continue

        print()
        print('-' * 80)
        print('FAILED OCR BLOCK')
        print('-' * 80)

        print(
            'sample:',
            sample_index,
        )

        print(
            'image:',
            result.img_path,
        )

        print(
            'GT index:',
            gt_index,
        )

        print(
            'prediction index:',
            pred_index,
        )

        print(
            'IoU:',
            round(
                best_iou,
                4,
            ),
        )

        print(
            'score:',
            round(
                float(
                    pred_scores[pred_index]
                ),
                4,
            ),
        )

        print(
            'text type:',
            text_type,
        )

        print(
            'GT length:',
            len(reference),
        )

        print(
            'prediction length:',
            len(prediction),
        )

        print()
        print('GT:')
        print(repr(reference))

        print()
        print('PREDICTION:')
        print(repr(prediction))

print()
print('=' * 80)
print(
    'Exact:',
    exact_ocr,
    '/',
    total_ocr,
)
print('=' * 80)