import torch
import torch.nn.functional as F

from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner import load_checkpoint

from mmdet.datasets import OCRDinoDataset
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules
from mmdet.utils.ocr_tokenizer import OCRDinoCharTokenizer


register_all_modules()

config_path = (
    'configs/ocr_dino/'
    'ocr-dino-4scale_r50_overfit8.py'
)

checkpoint_path = (
    '/root/autodl-tmp/work_dirs/'
    'ocr_dino_overfit8/iter_300.pth'
)

vocab_file = (
    '/root/autodl-tmp/dataset/OmniDocBench/'
    'ocrdino_debug_vocab.json'
)

sample_index = 1
target_gt_index = 1


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

tokenizer = OCRDinoCharTokenizer(
    vocab_file
)


pipeline = [
    dict(
        type='LoadImageFromFile',
    ),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_label=True,
    ),
    dict(
        type='Resize',
        scale=(800, 480),
        keep_ratio=True,
    ),
    dict(
        type='TokenizeOCRText',
        vocab_file=vocab_file,
        text_types=('text', 'latex'),
    ),
    dict(
        type='PackOCRDinoInputs',
    ),
]


dataset = OCRDinoDataset(
    data_root='/root/autodl-tmp/dataset/OmniDocBench/',
    ann_file='ocrdino_v1_overfit8.json',
    data_prefix=dict(
        img_path='',
    ),
    pipeline=pipeline,
    serialize_data=False,
)

sample = dataset[sample_index]

batch = pseudo_collate(
    [sample]
)

batch = model.data_preprocessor(
    batch,
    training=False,
)

inputs = batch['inputs']
data_samples = batch['data_samples']

data_sample = data_samples[0]
gt_instances = data_sample.gt_instances


with torch.no_grad():
    feats = model.extract_feat(
        inputs
    )

    outputs = model.forward_transformer(
        feats,
        data_samples,
    )

    (
        matching_hidden_states,
        matching_cls_scores,
        matching_bbox_preds,
    ) = model.bbox_head._get_matching_outputs(
        hidden_states=outputs['hidden_states'],
        references=outputs['references'],
        dn_meta=None,
    )

    (
        query_inds,
        gt_inds,
    ) = model.bbox_head._assign_single_image(
        cls_score=matching_cls_scores[0],
        bbox_pred=matching_bbox_preds[0],
        gt_instances=gt_instances,
        img_meta=data_sample.metainfo,
    )


matched_query_index = None

for query_idx, gt_idx in zip(
    query_inds.tolist(),
    gt_inds.tolist(),
):
    if gt_idx == target_gt_index:
        matched_query_index = query_idx
        break


if matched_query_index is None:
    raise RuntimeError(
        'Target GT was not matched.'
    )


token_ids = gt_instances.ocr_token_ids[
    target_gt_index
]

token_tensor = torch.tensor(
    token_ids,
    dtype=torch.long,
    device=device,
)

input_ids = token_tensor[
    :-1
].view(
    1,
    1,
    -1,
)

target_ids = token_tensor[
    1:
].view(
    1,
    1,
    -1,
)


selected_queries = (
    matching_hidden_states[
        0:1,
        matched_query_index:
        matched_query_index + 1,
        :,
    ]
)

selected_boxes = (
    matching_bbox_preds[
        0:1,
        matched_query_index:
        matched_query_index + 1,
        :,
    ]
)


with torch.no_grad():
    block_visual_features = (
        model.bbox_head.block_visual_extractor(
            object_queries=selected_queries,
            object_boxes=selected_boxes,
            memory=outputs['ocr_memory'],
            memory_mask=outputs[
                'ocr_memory_mask'
            ],
            spatial_shapes=outputs[
                'ocr_spatial_shapes'
            ],
            level_start_index=outputs[
                'ocr_level_start_index'
            ],
            valid_ratios=outputs[
                'ocr_valid_ratios'
            ],
        )
    )

    teacher_logits = (
        model.bbox_head.block_text_decoder(
            block_visual_features=(
                block_visual_features
            ),
            input_ids=input_ids,
            block_queries=selected_queries,
        )
    )

    teacher_loss = F.cross_entropy(
        teacher_logits.reshape(
            -1,
            model.bbox_head.ocr_vocab_size,
        ),
        target_ids.reshape(-1),
        reduction='mean',
    )

    teacher_pred = teacher_logits.argmax(
        dim=-1
    )

    teacher_accuracy = (
        teacher_pred.eq(
            target_ids
        )
        .float()
        .mean()
    )

    mismatch_mask = teacher_pred.ne(
        target_ids
    )
    
    mismatch_positions = torch.nonzero(
        mismatch_mask[0, 0],
        as_tuple=False,
    ).squeeze(-1)
    
    print()
    print('Teacher forcing mismatches:')
    print('count:', mismatch_positions.numel())
    
    for position in mismatch_positions.tolist():
        logits_at_position = teacher_logits[
            0,
            0,
            position,
        ]
    
        probabilities = torch.softmax(
            logits_at_position,
            dim=-1,
        )
    
        top_probs, top_ids = torch.topk(
            probabilities,
            k=10,
        )
    
        target_id = int(
            target_ids[
                0,
                0,
                position,
            ]
        )
    
        predicted_id = int(
            teacher_pred[
                0,
                0,
                position,
            ]
        )
    
        target_token = tokenizer.id_to_token.get(
            target_id,
            '<UNKNOWN>',
        )
    
        predicted_token = tokenizer.id_to_token.get(
            predicted_id,
            '<UNKNOWN>',
        )
    
        target_probability = float(
            probabilities[target_id]
        )
    
        target_rank = int(
            (
                probabilities
                > probabilities[target_id]
            ).sum().item()
            + 1
        )
    
        print()
        print('position:', position)
        print(
            'target:',
            target_id,
            repr(target_token),
        )
        print(
            'predicted:',
            predicted_id,
            repr(predicted_token),
        )
        print(
            'target probability:',
            target_probability,
        )
        print(
            'target rank:',
            target_rank,
        )
    
        print('top candidates:')
    
        for rank, (
            token_id,
            probability,
        ) in enumerate(
            zip(
                top_ids.tolist(),
                top_probs.tolist(),
            ),
            start=1,
        ):
            token = tokenizer.id_to_token.get(
                token_id,
                '<UNKNOWN>',
            )
    
            print(
                rank,
                token_id,
                repr(token),
                probability,
            )

    greedy_tokens = (
        model.bbox_head._greedy_decode_ocr(
            block_visual_features=(
                block_visual_features
            ),
            block_queries=(
                selected_queries
            ),
        )[0]
    )


reference = gt_instances.ocr_texts[
    target_gt_index
]

teacher_text = tokenizer.decode(
    [
        tokenizer.bos_token_id,
        *teacher_pred[
            0,
            0,
        ].tolist(),
    ]
)

greedy_text = tokenizer.decode(
    greedy_tokens
)


print('=' * 80)
print('FAILED BLOCK DIAGNOSTIC')
print('=' * 80)

print('sample:', sample_index)
print('GT index:', target_gt_index)
print('query index:', matched_query_index)

print()
print(
    'teacher forcing CE:',
    float(teacher_loss),
)

print(
    'teacher forcing token accuracy:',
    float(teacher_accuracy),
)

print()
print('GT length:', len(reference))
print('greedy length:', len(greedy_text))

print()
print('GROUND TRUTH:')
print(repr(reference))

print()
print('TEACHER FORCING ARGMAX:')
print(repr(teacher_text))

print()
print('GREEDY:')
print(repr(greedy_text))

print()
print(
    'teacher exact:',
    teacher_text == reference,
)

print(
    'greedy exact:',
    greedy_text == reference,
)

print('=' * 80)