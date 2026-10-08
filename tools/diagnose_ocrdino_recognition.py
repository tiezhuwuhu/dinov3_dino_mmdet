import copy
from collections import Counter

import numpy as np
import torch

from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.utils import import_modules_from_strings

from mmdet.registry import DATASETS, MODELS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.utils import register_all_modules


CONFIG = (
    'configs/ocrdino/'
    'ocrdino_r50_totaltext_original_24e.py'
)

CHECKPOINT = (
    '/root/autodl-tmp/work_dirs/'
    'ocrdino_r50_totaltext_original_24e_val/'
    'epoch_9.pth'
)

DEVICE = 'cuda:0'

SCORE_THR = 0.3
IOU_THR = 0.5

MAX_PRINT_MATCHES = 30

EOS_ID = 96
UNK_ID = 95
NUM_CLASSES = 97
REC_LENGTH = 25


def decode_tokens(tokens):
    chars = []

    for token in tokens:
        token = int(token)

        if token == EOS_ID:
            break

        if 0 <= token <= 94:
            chars.append(
                chr(token + 32)
            )
        elif token == UNK_ID:
            chars.append('?')
        else:
            chars.append('?')

    return ''.join(chars)


def token_to_string(token):
    token = int(token)

    if token == EOS_ID:
        return 'EOS'

    if token == UNK_ID:
        return 'UNK'

    if 0 <= token <= 94:
        ch = chr(token + 32)

        if ch == ' ':
            return 'SPACE'

        return ch

    return f'INVALID_{token}'


def bbox_iou_matrix(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros(
            (
                boxes1.shape[0],
                boxes2.shape[0],
            )
        )

    lt = torch.maximum(
        boxes1[:, None, :2],
        boxes2[None, :, :2],
    )

    rb = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:],
    )

    wh = (
        rb - lt
    ).clamp(
        min=0
    )

    intersection = (
        wh[..., 0]
        * wh[..., 1]
    )

    area1 = (
        (
            boxes1[:, 2]
            - boxes1[:, 0]
        ).clamp(
            min=0
        )
        * (
            boxes1[:, 3]
            - boxes1[:, 1]
        ).clamp(
            min=0
        )
    )

    area2 = (
        (
            boxes2[:, 2]
            - boxes2[:, 0]
        ).clamp(
            min=0
        )
        * (
            boxes2[:, 3]
            - boxes2[:, 1]
        ).clamp(
            min=0
        )
    )

    union = (
        area1[:, None]
        + area2[None, :]
        - intersection
    )

    return torch.where(
        union > 0,
        intersection / union,
        torch.zeros_like(
            intersection
        ),
    )


def greedy_match(
    pred_boxes,
    pred_scores,
    gt_boxes,
):
    keep = torch.where(
        pred_scores >= SCORE_THR
    )[0]

    if (
        keep.numel() == 0
        or gt_boxes.numel() == 0
    ):
        return []

    keep_scores = (
        pred_scores[
            keep
        ]
    )

    order = torch.argsort(
        keep_scores,
        descending=True,
    )

    keep = keep[
        order
    ]

    boxes = pred_boxes[
        keep
    ]

    ious = bbox_iou_matrix(
        boxes,
        gt_boxes,
    )

    gt_used = torch.zeros(
        gt_boxes.shape[0],
        dtype=torch.bool,
        device=gt_boxes.device,
    )

    matches = []

    for local_pred_index in range(
        boxes.shape[0]
    ):
        available = torch.where(
            ~gt_used
        )[0]

        if available.numel() == 0:
            break

        current_ious = (
            ious[
                local_pred_index,
                available,
            ]
        )

        best_local = int(
            torch.argmax(
                current_ious
            ).item()
        )

        gt_index = int(
            available[
                best_local
            ].item()
        )

        best_iou = float(
            ious[
                local_pred_index,
                gt_index,
            ].item()
        )

        if best_iou < IOU_THR:
            continue

        gt_used[
            gt_index
        ] = True

        pred_index = int(
            keep[
                local_pred_index
            ].item()
        )

        matches.append(
            (
                pred_index,
                gt_index,
                best_iou,
            )
        )

    return matches


def first_eos_position(tokens):
    for index, token in enumerate(
        tokens
    ):
        if int(token) == EOS_ID:
            return index

    return REC_LENGTH


def main():
    register_all_modules()

    cfg = Config.fromfile(
        CONFIG
    )

    if cfg.get(
        'custom_imports',
        None,
    ) is not None:
        import_modules_from_strings(
            **cfg.custom_imports
        )

    print()
    print(
        '============================================'
    )
    print(
        'OCRDINO RECOGNITION DIAGNOSTIC'
    )
    print(
        '============================================'
    )
    print(
        'config:',
        CONFIG,
    )
    print(
        'checkpoint:',
        CHECKPOINT,
    )
    print(
        'device:',
        DEVICE,
    )
    print(
        'score threshold:',
        SCORE_THR,
    )
    print(
        'iou threshold:',
        IOU_THR,
    )

    model = MODELS.build(
        cfg.model
    )

    load_checkpoint(
        model,
        CHECKPOINT,
        map_location='cpu',
        strict=False,
    )

    model.to(
        DEVICE
    )

    model.eval()

    dataset_cfg = copy.deepcopy(
        cfg.test_dataloader.dataset
    )

    dataset_cfg.ann_file = (
        'ocrdino_train.json'
    )

    dataset_cfg.test_mode = True
    dataset_cfg.serialize_data = False

    dataset = DATASETS.build(
        dataset_cfg
    )

    print()
    print(
        'dataset length:',
        len(dataset),
    )

    gt_token_counter = Counter()
    pred_token_counter = Counter()

    gt_first_token_counter = Counter()
    pred_first_token_counter = Counter()

    predicted_first_eos_counter = Counter()
    gt_first_eos_counter = Counter()

    total_gt_words = 0
    total_gt_tokens = 0
    total_gt_eos_tokens = 0
    total_gt_char_tokens = 0

    total_matches = 0

    total_matched_tokens = 0
    total_token_correct = 0

    total_matched_char_tokens = 0
    total_char_token_correct = 0

    total_matched_eos_tokens = 0
    total_eos_token_correct = 0

    total_pred_tokens = 0
    total_pred_eos_tokens = 0

    exact_word_correct = 0

    total_detection_score = 0.0
    total_match_iou = 0.0

    printed = 0

    for image_index in range(
        len(dataset)
    ):
        raw_sample = dataset[
            image_index
        ]

        batch = pseudo_collate(
            [
                raw_sample
            ]
        )

        processed = (
            model.data_preprocessor(
                batch,
                training=False,
            )
        )

        batch_inputs = processed[
            'inputs'
        ]

        batch_data_samples = processed[
            'data_samples'
        ]

        data_sample = (
            batch_data_samples[0]
        )

        gt_instances = (
            data_sample.gt_instances
        )

        gt_boxes = (
            gt_instances.bboxes
        )

        gt_rec = (
            gt_instances.rec
        )

        gt_texts = (
            gt_instances.ocr_texts
        )

        if (
            gt_rec.ndim != 2
            or gt_rec.shape[1]
            != REC_LENGTH
        ):
            raise RuntimeError(
                'Unexpected GT rec shape: '
                f'{tuple(gt_rec.shape)}'
            )

        if len(gt_texts) != len(
            gt_boxes
        ):
            raise RuntimeError(
                'GT text and bbox count mismatch.'
            )

        gt_rec_cpu = (
            gt_rec
            .detach()
            .cpu()
        )

        total_gt_words += (
            gt_rec_cpu.shape[0]
        )

        total_gt_tokens += (
            gt_rec_cpu.numel()
        )

        total_gt_eos_tokens += int(
            (
                gt_rec_cpu
                == EOS_ID
            ).sum().item()
        )

        total_gt_char_tokens += int(
            (
                gt_rec_cpu
                != EOS_ID
            ).sum().item()
        )

        for row in gt_rec_cpu:
            row_list = row.tolist()

            for token in row_list:
                gt_token_counter[
                    int(token)
                ] += 1

            if row_list:
                gt_first_token_counter[
                    int(
                        row_list[0]
                    )
                ] += 1

            gt_first_eos_counter[
                first_eos_position(
                    row_list
                )
            ] += 1

        with torch.no_grad():
            img_feats = (
                model.extract_feat(
                    batch_inputs
                )
            )

            (
                head_inputs_dict,
                text_outputs_dict,
            ) = (
                model._forward_transformer_impl(
                    img_feats=img_feats,
                    batch_data_samples=(
                        batch_data_samples
                    ),
                )
            )

            hidden_states = (
                head_inputs_dict[
                    'hidden_states'
                ]
            )

            references = (
                head_inputs_dict[
                    'references'
                ]
            )

            recognition_logits = (
                text_outputs_dict[
                    'recognition_logits'
                ]
            )

            (
                all_cls_scores,
                all_bbox_preds,
            ) = model.bbox_head(
                hidden_states,
                references,
            )

            cls_score = (
                all_cls_scores[
                    -1,
                    0,
                ]
            )

            bbox_pred = (
                all_bbox_preds[
                    -1,
                    0,
                ]
            )

            rec_logits = (
                recognition_logits[
                    -1,
                    0,
                ]
            )

            if (
                rec_logits.shape
                != (
                    model.num_queries,
                    REC_LENGTH,
                    NUM_CLASSES,
                )
            ):
                raise RuntimeError(
                    'Unexpected rec logits shape: '
                    f'{tuple(rec_logits.shape)}'
                )

            if model.bbox_head.loss_cls.use_sigmoid:
                class_scores = (
                    cls_score.sigmoid()
                )

                flat_scores = (
                    class_scores.reshape(-1)
                )

                num_classes = (
                    class_scores.shape[-1]
                )

                max_per_img = int(
                    model.test_cfg.get(
                        'max_per_img',
                        model.num_queries,
                    )
                )

                num_topk = min(
                    max_per_img,
                    flat_scores.numel(),
                )

                (
                    det_scores,
                    topk_indices,
                ) = flat_scores.topk(
                    num_topk
                )

                query_indices = (
                    topk_indices
                    // num_classes
                )

            else:
                class_scores = (
                    cls_score.softmax(
                        dim=-1
                    )[
                        ...,
                        :-1
                    ]
                )

                flat_scores = (
                    class_scores.reshape(-1)
                )

                num_classes = (
                    class_scores.shape[-1]
                )

                max_per_img = int(
                    model.test_cfg.get(
                        'max_per_img',
                        model.num_queries,
                    )
                )

                num_topk = min(
                    max_per_img,
                    flat_scores.numel(),
                )

                (
                    det_scores,
                    topk_indices,
                ) = flat_scores.topk(
                    num_topk
                )

                query_indices = (
                    topk_indices
                    // num_classes
                )

            selected_bbox = (
                bbox_pred[
                    query_indices
                ]
            )

            selected_rec_logits = (
                rec_logits[
                    query_indices
                ]
            )

            pred_boxes = (
                bbox_cxcywh_to_xyxy(
                    selected_bbox
                )
            )

            img_h = float(
                data_sample.img_shape[0]
            )

            img_w = float(
                data_sample.img_shape[1]
            )

            pred_boxes[
                :,
                0::2
            ] *= img_w

            pred_boxes[
                :,
                1::2
            ] *= img_h

            pred_boxes[
                :,
                0::2
            ].clamp_(
                min=0.0,
                max=img_w,
            )

            pred_boxes[
                :,
                1::2
            ].clamp_(
                min=0.0,
                max=img_h,
            )

            rec_probs = (
                selected_rec_logits.softmax(
                    dim=-1
                )
            )

            (
                pred_token_probs,
                pred_tokens,
            ) = rec_probs.max(
                dim=-1
            )

        matches = greedy_match(
            pred_boxes=pred_boxes,
            pred_scores=det_scores,
            gt_boxes=gt_boxes,
        )

        for (
            pred_index,
            gt_index,
            match_iou,
        ) in matches:
            total_matches += 1

            pred_token_row = (
                pred_tokens[
                    pred_index
                ]
                .detach()
                .cpu()
            )

            pred_prob_row = (
                pred_token_probs[
                    pred_index
                ]
                .detach()
                .cpu()
            )

            gt_token_row = (
                gt_rec_cpu[
                    gt_index
                ]
            )

            pred_list = (
                pred_token_row.tolist()
            )

            gt_list = (
                gt_token_row.tolist()
            )

            pred_text = decode_tokens(
                pred_list
            )

            gt_text = str(
                gt_texts[
                    gt_index
                ]
            )

            score = float(
                det_scores[
                    pred_index
                ].item()
            )

            total_detection_score += (
                score
            )

            total_match_iou += (
                match_iou
            )

            correct = (
                pred_token_row
                == gt_token_row
            )

            char_mask = (
                gt_token_row
                != EOS_ID
            )

            eos_mask = (
                gt_token_row
                == EOS_ID
            )

            total_matched_tokens += (
                REC_LENGTH
            )

            total_token_correct += int(
                correct.sum().item()
            )

            total_matched_char_tokens += int(
                char_mask.sum().item()
            )

            total_char_token_correct += int(
                (
                    correct
                    & char_mask
                ).sum().item()
            )

            total_matched_eos_tokens += int(
                eos_mask.sum().item()
            )

            total_eos_token_correct += int(
                (
                    correct
                    & eos_mask
                ).sum().item()
            )

            total_pred_tokens += (
                REC_LENGTH
            )

            pred_eos_count = int(
                (
                    pred_token_row
                    == EOS_ID
                ).sum().item()
            )

            total_pred_eos_tokens += (
                pred_eos_count
            )

            if pred_text == gt_text:
                exact_word_correct += 1

            for token in pred_list:
                pred_token_counter[
                    int(token)
                ] += 1

            pred_first_token_counter[
                int(
                    pred_list[0]
                )
            ] += 1

            predicted_first_eos_counter[
                first_eos_position(
                    pred_list
                )
            ] += 1

            if printed < MAX_PRINT_MATCHES:
                printed += 1

                print()
                print(
                    '--------------------------------------------'
                )

                print(
                    'MATCH',
                    printed,
                )

                print(
                    'image index:',
                    image_index,
                )

                print(
                    'img path:',
                    data_sample.img_path,
                )

                print(
                    'det score:',
                    f'{score:.6f}',
                )

                print(
                    'iou:',
                    f'{match_iou:.6f}',
                )

                print(
                    'GT text:',
                    repr(
                        gt_text
                    ),
                )

                print(
                    'PRED text:',
                    repr(
                        pred_text
                    ),
                )

                print(
                    'GT tokens:'
                )

                print(
                    ' '.join(
                        str(
                            int(token)
                        )
                        for token
                        in gt_list
                    )
                )

                print(
                    'PRED tokens:'
                )

                print(
                    ' '.join(
                        str(
                            int(token)
                        )
                        for token
                        in pred_list
                    )
                )

                print(
                    'PRED symbols:'
                )

                print(
                    ' '.join(
                        token_to_string(
                            token
                        )
                        for token
                        in pred_list
                    )
                )

                print(
                    'TOP1 probs:'
                )

                print(
                    ' '.join(
                        f'{float(prob):.3f}'
                        for prob
                        in pred_prob_row
                    )
                )

                print(
                    'pred EOS count:',
                    pred_eos_count,
                    '/',
                    REC_LENGTH,
                )

                if char_mask.any():
                    char_correct = int(
                        (
                            correct
                            & char_mask
                        ).sum().item()
                    )

                    char_total = int(
                        char_mask.sum().item()
                    )

                    print(
                        'real char token correct:',
                        char_correct,
                        '/',
                        char_total,
                    )

                if eos_mask.any():
                    eos_correct = int(
                        (
                            correct
                            & eos_mask
                        ).sum().item()
                    )

                    eos_total = int(
                        eos_mask.sum().item()
                    )

                    print(
                        'EOS token correct:',
                        eos_correct,
                        '/',
                        eos_total,
                    )

        if (
            image_index + 1
        ) % 100 == 0:
            print()
            print(
                'processed images:',
                image_index + 1,
                '/',
                len(dataset),
                'matches:',
                total_matches,
            )

    print()
    print()
    print(
        '============================================'
    )

    print(
        'GLOBAL DIAGNOSTIC SUMMARY'
    )

    print(
        '============================================'
    )

    print(
        'images:',
        len(dataset),
    )

    print(
        'GT words:',
        total_gt_words,
    )

    print(
        'matched predictions:',
        total_matches,
    )

    if total_matches > 0:
        print(
            'mean matched det score:',
            f'{total_detection_score / total_matches:.6f}',
        )

        print(
            'mean matched IoU:',
            f'{total_match_iou / total_matches:.6f}',
        )

        print(
            'exact word accuracy:',
            f'{exact_word_correct / total_matches:.6f}',
        )

    print()
    print(
        'GT TARGET DISTRIBUTION'
    )

    if total_gt_tokens > 0:
        print(
            'GT total tokens:',
            total_gt_tokens,
        )

        print(
            'GT character tokens:',
            total_gt_char_tokens,
        )

        print(
            'GT EOS tokens:',
            total_gt_eos_tokens,
        )

        print(
            'GT EOS ratio:',
            f'{total_gt_eos_tokens / total_gt_tokens:.6f}',
        )

    print()
    print(
        'MATCHED TOKEN ACCURACY'
    )

    if total_matched_tokens > 0:
        print(
            'all 25-position token accuracy:',
            f'{total_token_correct / total_matched_tokens:.6f}',
        )

    if total_matched_char_tokens > 0:
        print(
            'real character token accuracy:',
            f'{total_char_token_correct / total_matched_char_tokens:.6f}',
        )

    if total_matched_eos_tokens > 0:
        print(
            'EOS-position accuracy:',
            f'{total_eos_token_correct / total_matched_eos_tokens:.6f}',
        )

    if total_pred_tokens > 0:
        print(
            'predicted EOS ratio:',
            f'{total_pred_eos_tokens / total_pred_tokens:.6f}',
        )

    print()
    print(
        'TOP GT TOKENS'
    )

    for token, count in (
        gt_token_counter.most_common(
            15
        )
    ):
        print(
            token,
            token_to_string(
                token
            ),
            count,
        )

    print()
    print(
        'TOP PREDICTED TOKENS ON MATCHED BOXES'
    )

    for token, count in (
        pred_token_counter.most_common(
            15
        )
    ):
        print(
            token,
            token_to_string(
                token
            ),
            count,
        )

    print()
    print(
        'FIRST TOKEN DISTRIBUTION'
    )

    print(
        'GT top first tokens:'
    )

    for token, count in (
        gt_first_token_counter.most_common(
            10
        )
    ):
        print(
            token,
            token_to_string(
                token
            ),
            count,
        )

    print(
        'PRED top first tokens:'
    )

    for token, count in (
        pred_first_token_counter.most_common(
            10
        )
    ):
        print(
            token,
            token_to_string(
                token
            ),
            count,
        )

    print()
    print(
        'FIRST EOS POSITION'
    )

    print(
        'position 0 means prediction starts with EOS'
    )

    print(
        'position 25 means no EOS'
    )

    print()

    print(
        'GT first EOS positions:'
    )

    for position in sorted(
        gt_first_eos_counter
    ):
        print(
            position,
            gt_first_eos_counter[
                position
            ],
        )

    print()

    print(
        'PRED first EOS positions:'
    )

    for position in sorted(
        predicted_first_eos_counter
    ):
        print(
            position,
            predicted_first_eos_counter[
                position
            ],
        )

    print()
    print(
        '============================================'
    )

    if total_pred_tokens > 0:
        pred_eos_ratio = (
            total_pred_eos_tokens
            / total_pred_tokens
        )
    else:
        pred_eos_ratio = 0.0

    if total_gt_tokens > 0:
        gt_eos_ratio = (
            total_gt_eos_tokens
            / total_gt_tokens
        )
    else:
        gt_eos_ratio = 0.0

    print(
        'GT EOS ratio:',
        f'{gt_eos_ratio:.6f}',
    )

    print(
        'PRED EOS ratio:',
        f'{pred_eos_ratio:.6f}',
    )

    if total_matched_char_tokens > 0:
        real_char_acc = (
            total_char_token_correct
            / total_matched_char_tokens
        )
    else:
        real_char_acc = 0.0

    if total_matched_eos_tokens > 0:
        eos_acc = (
            total_eos_token_correct
            / total_matched_eos_tokens
        )
    else:
        eos_acc = 0.0

    print(
        'real character token accuracy:',
        f'{real_char_acc:.6f}',
    )

    print(
        'EOS-position accuracy:',
        f'{eos_acc:.6f}',
    )

    print(
        '============================================'
    )


if __name__ == '__main__':
    main()