import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings

from mmdet.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize OCRDINO predictions on Total-Text test set.'
    )

    parser.add_argument(
        '--config',
        default=(
            'configs/ocrdino/'
            'ocrdino_r50_totaltext_'
            'stage1pretrain_finetune_5e_'
            '2gpu_bs4_acc2.py'
        ),
    )

    parser.add_argument(
        '--checkpoint',
        default=(
            '/root/autodl-tmp/work_dirs/'
            'ocrdino_r50_totaltext_'
            'stage1pretrain_finetune_5e_'
            '2gpu_bs4_acc2/'
            'best_ocrdino_e2e_f1_epoch_5.pth'
        ),
    )

    parser.add_argument(
        '--output-dir',
        default=(
            '/root/autodl-tmp/work_dirs/'
            'ocrdino_totaltext_visualization_epoch5'
        ),
    )

    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='Prediction score threshold used for visualization.',
    )

    parser.add_argument(
        '--device',
        default='cuda:0',
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=2,
    )

    parser.add_argument(
        '--line-width',
        type=int,
        default=3,
    )

    parser.add_argument(
        '--font-size',
        type=int,
        default=18,
    )

    return parser.parse_args()


def get_font(font_size):
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
    ]

    for path in candidates:
        if os.path.isfile(path):
            return ImageFont.truetype(
                path,
                font_size,
            )

    return ImageFont.load_default()


def boxes_to_numpy(boxes):
    if boxes is None:
        return np.empty(
            (0, 4),
            dtype=np.float32,
        )

    if hasattr(boxes, 'tensor'):
        boxes = boxes.tensor

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()

    boxes = np.asarray(
        boxes,
        dtype=np.float32,
    )

    return boxes.reshape(-1, 4)


def scores_to_numpy(scores):
    if scores is None:
        return np.empty(
            (0,),
            dtype=np.float32,
        )

    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    return np.asarray(
        scores,
        dtype=np.float32,
    ).reshape(-1)


def normalize_text_list(values):
    if values is None:
        return []

    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()

    return [
        str(value)
        for value in values
    ]


def clean_label_text(text):
    text = str(text)

    text = text.replace(
        '\n',
        '\\n',
    )

    text = text.replace(
        '\r',
        '\\r',
    )

    text = text.replace(
        '\t',
        '\\t',
    )

    return text


def clamp_box(
    box,
    width,
    height,
):
    x1, y1, x2, y2 = [
        float(value)
        for value in box
    ]

    x1 = max(
        0.0,
        min(x1, width - 1),
    )

    x2 = max(
        0.0,
        min(x2, width - 1),
    )

    y1 = max(
        0.0,
        min(y1, height - 1),
    )

    y2 = max(
        0.0,
        min(y2, height - 1),
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


def draw_label(
    draw,
    xy,
    text,
    font,
    color,
):
    x, y = xy

    text = clean_label_text(
        text
    )

    try:
        bbox = draw.textbbox(
            (x, y),
            text,
            font=font,
        )

        text_w = (
            bbox[2]
            - bbox[0]
        )

        text_h = (
            bbox[3]
            - bbox[1]
        )

    except Exception:
        text_w = max(
            20,
            len(text) * 9,
        )

        text_h = 18

    draw.rectangle(
        [
            x,
            y,
            x + text_w + 6,
            y + text_h + 6,
        ],
        fill=color,
    )

    draw.text(
        (
            x + 3,
            y + 2,
        ),
        text,
        fill=(255, 255, 255),
        font=font,
    )


def draw_box_with_label(
    draw,
    box,
    label,
    font,
    color,
    line_width,
    image_width,
    image_height,
):
    x1, y1, x2, y2 = clamp_box(
        box=box,
        width=image_width,
        height=image_height,
    )

    draw.rectangle(
        [
            x1,
            y1,
            x2,
            y2,
        ],
        outline=color,
        width=line_width,
    )

    label_y = (
        y1
        - font.size
        - 10
    )

    if label_y < 0:
        label_y = y1 + 2

    draw_label(
        draw=draw,
        xy=(
            int(x1),
            int(label_y),
        ),
        text=label,
        font=font,
        color=color,
    )


def extract_gt(sample):
    if not hasattr(
        sample,
        'gt_instances',
    ):
        return (
            np.empty(
                (0, 4),
                dtype=np.float32,
            ),
            [],
        )

    gt = sample.gt_instances

    gt_boxes = boxes_to_numpy(
        getattr(
            gt,
            'bboxes',
            None,
        )
    )

    gt_texts = normalize_text_list(
        getattr(
            gt,
            'ocr_texts',
            None,
        )
    )

    if not gt_texts:
        gt_texts = [
            ''
            for _ in range(
                len(gt_boxes)
            )
        ]

    if (
        len(gt_texts)
        != len(gt_boxes)
    ):
        print(
            '[WARNING] GT box/text mismatch: '
            f'{len(gt_boxes)} boxes, '
            f'{len(gt_texts)} texts'
        )

        count = min(
            len(gt_boxes),
            len(gt_texts),
        )

        gt_boxes = gt_boxes[:count]
        gt_texts = gt_texts[:count]

    return (
        gt_boxes,
        gt_texts,
    )


def extract_predictions(
    sample,
    score_thr,
):
    pred = sample.pred_instances

    pred_boxes = boxes_to_numpy(
        pred.bboxes
    )

    pred_scores = scores_to_numpy(
        pred.scores
    )

    pred_texts = normalize_text_list(
        getattr(
            pred,
            'rec_texts',
            None,
        )
    )

    if not pred_texts:
        pred_texts = [
            ''
            for _ in range(
                len(pred_boxes)
            )
        ]

    if not (
        len(pred_boxes)
        == len(pred_scores)
        == len(pred_texts)
    ):
        raise RuntimeError(
            'Prediction fields are not aligned: '
            f'boxes={len(pred_boxes)}, '
            f'scores={len(pred_scores)}, '
            f'texts={len(pred_texts)}'
        )

    valid = (
        np.isfinite(
            pred_boxes
        ).all(
            axis=1
        )
        & np.isfinite(
            pred_scores
        )
    )

    valid &= (
        pred_boxes[:, 2]
        > pred_boxes[:, 0]
    )

    valid &= (
        pred_boxes[:, 3]
        > pred_boxes[:, 1]
    )

    valid &= (
        pred_scores
        >= score_thr
    )

    indexes = np.where(
        valid
    )[0]

    pred_boxes = (
        pred_boxes[indexes]
    )

    pred_scores = (
        pred_scores[indexes]
    )

    pred_texts = [
        pred_texts[index]
        for index in indexes
    ]

    order = np.argsort(
        -pred_scores
    )

    pred_boxes = (
        pred_boxes[order]
    )

    pred_scores = (
        pred_scores[order]
    )

    pred_texts = [
        pred_texts[index]
        for index in order
    ]

    return (
        pred_boxes,
        pred_scores,
        pred_texts,
    )


def validate_test_pipeline(cfg):
    pipeline = (
        cfg
        .test_dataloader
        .dataset
        .pipeline
    )

    transform_types = [
        item['type']
        for item in pipeline
    ]

    print(
        'Test pipeline:'
    )

    for index, name in enumerate(
        transform_types
    ):
        print(
            f'  {index}: {name}'
        )

    resize_indexes = [
        index
        for index, name
        in enumerate(
            transform_types
        )
        if name == 'Resize'
    ]

    ann_indexes = [
        index
        for index, name
        in enumerate(
            transform_types
        )
        if name
        == 'LoadOCRAnnotations'
    ]

    if (
        not resize_indexes
        or not ann_indexes
    ):
        raise RuntimeError(
            'Test pipeline must contain both '
            'Resize and LoadOCRAnnotations.'
        )

    if (
        resize_indexes[0]
        > ann_indexes[0]
    ):
        raise RuntimeError(
            '\n'
            'Unsafe test pipeline detected.\n'
            '\n'
            'Current order is:\n'
            'LoadOCRAnnotations -> Resize\n'
            '\n'
            'This causes GT boxes to be in resize-space '
            'while predictions are rescaled to original-space.\n'
            '\n'
            'Expected test order:\n'
            'LoadImageFromFile -> Resize -> '
            'LoadOCRAnnotations -> ...\n'
        )


def main():
    args = parse_args()

    config_path = os.path.abspath(
        args.config
    )

    checkpoint_path = os.path.abspath(
        args.checkpoint
    )

    output_dir = Path(
        args.output_dir
    )

    pred_only_dir = (
        output_dir
        / 'pred_only'
    )

    gt_pred_dir = (
        output_dir
        / 'gt_pred'
    )

    pred_only_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gt_pred_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not os.path.isfile(
        config_path
    ):
        raise FileNotFoundError(
            config_path
        )

    if not os.path.isfile(
        checkpoint_path
    ):
        raise FileNotFoundError(
            checkpoint_path
        )

    register_all_modules(
        init_default_scope=True
    )

    cfg = Config.fromfile(
        config_path
    )

    custom_imports = cfg.get(
        'custom_imports',
        None,
    )

    if custom_imports:
        import_modules_from_strings(
            **custom_imports
        )

    cfg.launcher = 'none'

    cfg.load_from = (
        checkpoint_path
    )

    cfg.resume = False

    cfg.work_dir = str(
        output_dir
        / '_runner'
    )

    # Visualization is easier and deterministic
    # with one image per batch.
    cfg.test_dataloader.batch_size = 1

    cfg.test_dataloader.num_workers = (
        args.workers
    )

    cfg.test_dataloader.persistent_workers = (
        args.workers > 0
    )

    # Make absolutely sure the validation coordinate
    # convention is the corrected one.
    validate_test_pipeline(
        cfg
    )

    print()
    print(
        'Config:',
        config_path,
    )

    print(
        'Checkpoint:',
        checkpoint_path,
    )

    print(
        'Score threshold:',
        args.score_thr,
    )

    print(
        'Output:',
        output_dir,
    )

    print()

    runner = Runner.from_cfg(
        cfg
    )

    runner.load_or_resume()

    model = runner.model

    model.to(
        args.device
    )

    model.eval()

    data_loader = (
        runner.build_dataloader(
            cfg.test_dataloader
        )
    )

    dataset_size = len(
        data_loader.dataset
    )

    print(
        'Total test images:',
        dataset_size,
    )

    font = get_font(
        args.font_size
    )

    # GT color.
    gt_color = (
        0,
        190,
        0,
    )

    # Prediction color.
    pred_color = (
        230,
        35,
        35,
    )

    all_records = []

    total_predictions = 0
    total_gt = 0

    with torch.no_grad():

        for batch_index, data_batch in enumerate(
            data_loader
        ):

            outputs = model.test_step(
                data_batch
            )

            if not isinstance(
                outputs,
                (list, tuple),
            ):
                outputs = [
                    outputs
                ]

            for local_index, sample in enumerate(
                outputs
            ):
                image_index = (
                    batch_index
                    * cfg.test_dataloader.batch_size
                    + local_index
                )

                meta = sample.metainfo

                img_path = meta.get(
                    'img_path',
                    None,
                )

                if not img_path:
                    raise RuntimeError(
                        'img_path is missing from '
                        'DetDataSample metainfo.'
                    )

                if not os.path.isfile(
                    img_path
                ):
                    raise FileNotFoundError(
                        img_path
                    )

                image = Image.open(
                    img_path
                ).convert(
                    'RGB'
                )

                image_width, image_height = (
                    image.size
                )

                (
                    pred_boxes,
                    pred_scores,
                    pred_texts,
                ) = extract_predictions(
                    sample=sample,
                    score_thr=args.score_thr,
                )

                (
                    gt_boxes,
                    gt_texts,
                ) = extract_gt(
                    sample
                )

                total_predictions += len(
                    pred_boxes
                )

                total_gt += len(
                    gt_boxes
                )

                # ---------------------------------------------
                # 1. Prediction-only visualization
                # ---------------------------------------------

                pred_image = (
                    image.copy()
                )

                pred_draw = ImageDraw.Draw(
                    pred_image
                )

                for (
                    box,
                    score,
                    text,
                ) in zip(
                    pred_boxes,
                    pred_scores,
                    pred_texts,
                ):
                    label = (
                        f'P {score:.3f}: '
                        f'{text}'
                    )

                    draw_box_with_label(
                        draw=pred_draw,
                        box=box,
                        label=label,
                        font=font,
                        color=pred_color,
                        line_width=(
                            args.line_width
                        ),
                        image_width=(
                            image_width
                        ),
                        image_height=(
                            image_height
                        ),
                    )

                # ---------------------------------------------
                # 2. GT + prediction visualization
                # ---------------------------------------------

                compare_image = (
                    image.copy()
                )

                compare_draw = (
                    ImageDraw.Draw(
                        compare_image
                    )
                )

                for (
                    box,
                    text,
                ) in zip(
                    gt_boxes,
                    gt_texts,
                ):
                    label = (
                        f'GT: {text}'
                    )

                    draw_box_with_label(
                        draw=compare_draw,
                        box=box,
                        label=label,
                        font=font,
                        color=gt_color,
                        line_width=(
                            args.line_width
                        ),
                        image_width=(
                            image_width
                        ),
                        image_height=(
                            image_height
                        ),
                    )

                for (
                    box,
                    score,
                    text,
                ) in zip(
                    pred_boxes,
                    pred_scores,
                    pred_texts,
                ):
                    label = (
                        f'P {score:.3f}: '
                        f'{text}'
                    )

                    draw_box_with_label(
                        draw=compare_draw,
                        box=box,
                        label=label,
                        font=font,
                        color=pred_color,
                        line_width=(
                            args.line_width
                        ),
                        image_width=(
                            image_width
                        ),
                        image_height=(
                            image_height
                        ),
                    )

                stem = Path(
                    img_path
                ).stem

                output_name = (
                    f'{image_index:04d}_'
                    f'{stem}.jpg'
                )

                pred_output_path = (
                    pred_only_dir
                    / output_name
                )

                compare_output_path = (
                    gt_pred_dir
                    / output_name
                )

                pred_image.save(
                    pred_output_path,
                    quality=95,
                )

                compare_image.save(
                    compare_output_path,
                    quality=95,
                )

                record = {
                    'index': int(
                        image_index
                    ),
                    'img_path': str(
                        img_path
                    ),
                    'width': int(
                        image_width
                    ),
                    'height': int(
                        image_height
                    ),
                    'score_thr': float(
                        args.score_thr
                    ),
                    'gt': [],
                    'predictions': [],
                }

                for (
                    box,
                    text,
                ) in zip(
                    gt_boxes,
                    gt_texts,
                ):
                    record[
                        'gt'
                    ].append(
                        {
                            'bbox': [
                                float(value)
                                for value
                                in box
                            ],
                            'text': str(
                                text
                            ),
                        }
                    )

                for (
                    box,
                    score,
                    text,
                ) in zip(
                    pred_boxes,
                    pred_scores,
                    pred_texts,
                ):
                    record[
                        'predictions'
                    ].append(
                        {
                            'bbox': [
                                float(value)
                                for value
                                in box
                            ],
                            'score': float(
                                score
                            ),
                            'text': str(
                                text
                            ),
                        }
                    )

                all_records.append(
                    record
                )

                print(
                    f'[{image_index + 1:03d}/'
                    f'{dataset_size:03d}] '
                    f'{Path(img_path).name} | '
                    f'GT={len(gt_boxes)} | '
                    f'Pred>={args.score_thr:.2f}'
                    f'={len(pred_boxes)}'
                )

    json_path = (
        output_dir
        / 'predictions.json'
    )

    with open(
        json_path,
        'w',
        encoding='utf-8',
    ) as file:
        json.dump(
            all_records,
            file,
            ensure_ascii=False,
            indent=2,
        )

    summary = {
        'config': config_path,
        'checkpoint': checkpoint_path,
        'score_threshold': float(
            args.score_thr
        ),
        'num_images': int(
            len(all_records)
        ),
        'num_gt': int(
            total_gt
        ),
        'num_predictions': int(
            total_predictions
        ),
    }

    summary_path = (
        output_dir
        / 'summary.json'
    )

    with open(
        summary_path,
        'w',
        encoding='utf-8',
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        '=' * 80
    )

    print(
        'VISUALIZATION FINISHED'
    )

    print(
        '=' * 80
    )

    print(
        'Images:',
        len(all_records),
    )

    print(
        'GT boxes:',
        total_gt,
    )

    print(
        f'Predictions >= {args.score_thr}:',
        total_predictions,
    )

    print()
    print(
        'Prediction-only images:'
    )

    print(
        pred_only_dir
    )

    print()
    print(
        'GT + prediction images:'
    )

    print(
        gt_pred_dir
    )

    print()
    print(
        'Prediction JSON:'
    )

    print(
        json_path
    )


if __name__ == '__main__':
    main()