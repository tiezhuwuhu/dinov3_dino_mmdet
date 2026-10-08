from __future__ import annotations

import copy
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from mmengine.config import Config

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


OFFICIAL_CHECKPOINT = Path(
    '/root/autodl-tmp/checkpoint/ESTextSpotter/pretrain.pth'
)

BASE_CHECKPOINT = Path(
    '/root/autodl-tmp/checkpoint/ocrdino/'
    'ocrdino_r50_totaltext_init.pth'
)

OUTPUT_CHECKPOINT = Path(
    '/root/autodl-tmp/checkpoint/ocrdino/'
    'ocrdino_r50_estextspotter_pretrain_init.pth'
)

REPORT_PATH = Path(
    '/root/autodl-tmp/checkpoint/ocrdino/'
    'ocrdino_r50_estextspotter_pretrain_merge_report.txt'
)

CONFIG_PATH = Path(
    '/root/autodl-tmp/mmdetection/configs/ocrdino/'
    'ocrdino_r50_totaltext_rec1e4_dn04_100e.py'
)

NUM_HEADS = 8
NUM_POINTS = 4


def load_checkpoint(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)

    return torch.load(
        str(path),
        map_location='cpu',
        weights_only=False,
    )


def extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(
            'Checkpoint must be a dictionary.'
        )

    for key in (
        'state_dict',
        'model',
    ):
        value = checkpoint.get(key)

        if isinstance(value, dict):
            tensor_count = sum(
                torch.is_tensor(v)
                for v in value.values()
            )

            if tensor_count > 0:
                return value

    tensor_count = sum(
        torch.is_tensor(v)
        for v in checkpoint.values()
    )

    if tensor_count > 0:
        return checkpoint

    raise RuntimeError(
        'Cannot find a state dict in checkpoint.'
    )


def strip_wrapper_prefix(key: str) -> str:
    changed = True

    while changed:
        changed = False

        for prefix in (
            'module.',
            '_orig_mod.',
            'model.',
        ):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True

    return key


def normalize_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    result = {}

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue

        normalized_key = strip_wrapper_prefix(key)

        if normalized_key in result:
            if not torch.equal(
                result[normalized_key],
                value,
            ):
                raise RuntimeError(
                    'Duplicate normalized source key '
                    f'with different values: '
                    f'{normalized_key}'
                )

            continue

        result[normalized_key] = value

    return result


def adapt_feature_level_parameter(
    source: torch.Tensor,
    target: torch.Tensor,
    target_key: str,
) -> Optional[torch.Tensor]:
    source_shape = tuple(source.shape)
    target_shape = tuple(target.shape)

    if source_shape == target_shape:
        return source

    if (
        target_key == 'level_embed'
        and source.ndim == 2
        and target.ndim == 2
        and source.shape[1] == target.shape[1]
        and source.shape[0] >= target.shape[0]
    ):
        return source[:target.shape[0]].clone()

    if (
        target_key == 'query_embedding.weight'
        and source.ndim == 2
        and target.ndim == 2
        and source.shape[1] == target.shape[1]
        and source.shape[0] >= target.shape[0]
    ):
        return source[:target.shape[0]].clone()

    if target_key.endswith(
        'sampling_offsets.weight'
    ):
        if (
            source.ndim == 2
            and target.ndim == 2
            and source.shape[1] == target.shape[1]
        ):
            source_levels = (
                source.shape[0]
                // (
                    NUM_HEADS
                    * NUM_POINTS
                    * 2
                )
            )

            target_levels = (
                target.shape[0]
                // (
                    NUM_HEADS
                    * NUM_POINTS
                    * 2
                )
            )

            if (
                source_levels >= target_levels
                and source_levels > 0
                and target_levels > 0
            ):
                reshaped = source.reshape(
                    NUM_HEADS,
                    source_levels,
                    NUM_POINTS,
                    2,
                    source.shape[1],
                )

                adapted = reshaped[
                    :,
                    :target_levels,
                ].reshape(target.shape)

                return adapted.clone()

    if target_key.endswith(
        'sampling_offsets.bias'
    ):
        if (
            source.ndim == 1
            and target.ndim == 1
        ):
            source_levels = (
                source.shape[0]
                // (
                    NUM_HEADS
                    * NUM_POINTS
                    * 2
                )
            )

            target_levels = (
                target.shape[0]
                // (
                    NUM_HEADS
                    * NUM_POINTS
                    * 2
                )
            )

            if (
                source_levels >= target_levels
                and source_levels > 0
                and target_levels > 0
            ):
                reshaped = source.reshape(
                    NUM_HEADS,
                    source_levels,
                    NUM_POINTS,
                    2,
                )

                adapted = reshaped[
                    :,
                    :target_levels,
                ].reshape(target.shape)

                return adapted.clone()

    if target_key.endswith(
        'attention_weights.weight'
    ):
        if (
            source.ndim == 2
            and target.ndim == 2
            and source.shape[1] == target.shape[1]
        ):
            source_levels = (
                source.shape[0]
                // (
                    NUM_HEADS
                    * NUM_POINTS
                )
            )

            target_levels = (
                target.shape[0]
                // (
                    NUM_HEADS
                    * NUM_POINTS
                )
            )

            if (
                source_levels >= target_levels
                and source_levels > 0
                and target_levels > 0
            ):
                reshaped = source.reshape(
                    NUM_HEADS,
                    source_levels,
                    NUM_POINTS,
                    source.shape[1],
                )

                adapted = reshaped[
                    :,
                    :target_levels,
                ].reshape(target.shape)

                return adapted.clone()

    if target_key.endswith(
        'attention_weights.bias'
    ):
        if (
            source.ndim == 1
            and target.ndim == 1
        ):
            source_levels = (
                source.shape[0]
                // (
                    NUM_HEADS
                    * NUM_POINTS
                )
            )

            target_levels = (
                target.shape[0]
                // (
                    NUM_HEADS
                    * NUM_POINTS
                )
            )

            if (
                source_levels >= target_levels
                and source_levels > 0
                and target_levels > 0
            ):
                reshaped = source.reshape(
                    NUM_HEADS,
                    source_levels,
                    NUM_POINTS,
                )

                adapted = reshaped[
                    :,
                    :target_levels,
                ].reshape(target.shape)

                return adapted.clone()

    return None


def map_backbone(
    key: str,
) -> List[Tuple[str, str]]:
    results = []

    prefix = 'backbone.0.body.'

    if key.startswith(prefix):
        suffix = key[len(prefix):]

        results.append(
            (
                'backbone.' + suffix,
                'backbone',
            )
        )

    return results


def map_input_proj(
    key: str,
) -> List[Tuple[str, str]]:
    results = []

    match = re.match(
        r'^input_proj\.(\d+)\.(\d+)\.(.+)$',
        key,
    )

    if match is None:
        return results

    level = int(match.group(1))
    submodule = int(match.group(2))
    suffix = match.group(3)

    if submodule == 0:
        module_name = 'conv'
    elif submodule == 1:
        module_name = 'gn'
    else:
        return results

    if level <= 2:
        target = (
            f'neck.convs.{level}.'
            f'{module_name}.{suffix}'
        )

        results.append(
            (
                target,
                'neck',
            )
        )

    elif level == 3:
        target = (
            'neck.extra_convs.0.'
            f'{module_name}.{suffix}'
        )

        results.append(
            (
                target,
                'neck',
            )
        )

    return results


def map_encoder_layer(
    key: str,
) -> List[Tuple[str, str]]:
    results = []

    match = re.match(
        r'^transformer\.encoder\.layers\.'
        r'(\d+)\.(.+)$',
        key,
    )

    if match is None:
        return results

    layer = int(match.group(1))
    suffix = match.group(2)

    source_prefix = (
        f'transformer.encoder.layers.{layer}.'
    )

    target_prefix = (
        f'encoder.layers.{layer}.'
    )

    if suffix.startswith('self_attn.'):
        target = (
            target_prefix
            + suffix
        )

        results.append(
            (
                target,
                'encoder',
            )
        )

        return results

    simple_map = {
        'linear1.weight':
            'ffn.layers.0.0.weight',
        'linear1.bias':
            'ffn.layers.0.0.bias',
        'linear2.weight':
            'ffn.layers.1.weight',
        'linear2.bias':
            'ffn.layers.1.bias',
        'norm1.weight':
            'norms.0.weight',
        'norm1.bias':
            'norms.0.bias',
        'norm2.weight':
            'norms.1.weight',
        'norm2.bias':
            'norms.1.bias',
    }

    mapped_suffix = simple_map.get(suffix)

    if mapped_suffix is not None:
        results.append(
            (
                target_prefix
                + mapped_suffix,
                'encoder',
            )
        )

    return results


def map_decoder_layer(
    key: str,
) -> List[Tuple[str, str]]:
    results = []

    match = re.match(
        r'^transformer\.decoder\.layers\.'
        r'(\d+)\.(.+)$',
        key,
    )

    if match is None:
        return results

    layer = int(match.group(1))
    suffix = match.group(2)

    target_prefix = (
        f'decoder.layers.{layer}.'
    )

    mappings = (
        (
            'attn_m.',
            'vlc_attention.',
        ),
        (
            'norm_m.',
            'vlc_attention_norm.',
        ),
        (
            'linear1_m.',
            'vlc_ffn_linear1.',
        ),
        (
            'linear2_m.',
            'vlc_ffn_linear2.',
        ),
        (
            'norm3_m.',
            'vlc_ffn_norm.',
        ),
        (
            'attn_intra.',
            'self_attention.attn_intra.',
        ),
        (
            'norm_intra.',
            'self_attention.norm_intra.',
        ),
        (
            'attn_inter.',
            'self_attention.attn_inter.',
        ),
        (
            'norm_inter.',
            'self_attention.norm_inter.',
        ),
        (
            'cross_attn.',
            'cross_attention.cross_attn.',
        ),
        (
            'norm1.',
            'cross_attention.norm.',
        ),
        (
            'linear1.',
            'ffn_linear1.',
        ),
        (
            'linear2.',
            'ffn_linear2.',
        ),
        (
            'norm3.',
            'ffn_norm.',
        ),
    )

    for source_prefix, target_suffix in mappings:
        if suffix.startswith(source_prefix):
            remainder = suffix[len(source_prefix):]

            target = (
                target_prefix
                + target_suffix
                + remainder
            )

            results.append(
                (
                    target,
                    'decoder',
                )
            )

            break

    return results


def map_decoder_global(
    key: str,
) -> List[Tuple[str, str]]:
    results = []

    prefix = 'transformer.decoder.ref_point_head.'

    if key.startswith(prefix):
        results.append(
            (
                'decoder.ref_point_head.'
                + key[len(prefix):],
                'decoder',
            )
        )

    prefix = 'transformer.decoder.norm.'

    if key.startswith(prefix):
        results.append(
            (
                'decoder.norm.'
                + key[len(prefix):],
                'decoder',
            )
        )

    return results


def map_recognition(
    key: str,
) -> List[Tuple[str, str]]:
    results = []

    if re.search(
        r'(^|\.)rec_cls\.weight$',
        key,
    ):
        results.append(
            (
                'decoder.recognition_semantics.'
                'rec_classifier.weight',
                'recognition',
            )
        )

    if re.search(
        r'(^|\.)rec_cls\.bias$',
        key,
    ):
        results.append(
            (
                'decoder.recognition_semantics.'
                'rec_classifier.bias',
                'recognition',
            )
        )

    if re.search(
        r'(^|\.)rec_proj\.weight$',
        key,
    ):
        results.append(
            (
                'decoder.recognition_semantics.'
                'rec_projection.weight',
                'recognition',
            )
        )

    if re.search(
        r'(^|\.)rec_proj\.bias$',
        key,
    ):
        results.append(
            (
                'decoder.recognition_semantics.'
                'rec_projection.bias',
                'recognition',
            )
        )

    return results


def map_bbox_head(
    key: str,
) -> List[Tuple[str, str]]:
    results = []

    normalized = key

    prefix = 'transformer.decoder.'

    if normalized.startswith(prefix):
        normalized = normalized[len(prefix):]

    bbox_match = re.match(
        r'^bbox_embed\.(\d+)\.'
        r'layers\.(\d+)\.(weight|bias)$',
        normalized,
    )

    if bbox_match is not None:
        branch = int(
            bbox_match.group(1)
        )

        official_layer = int(
            bbox_match.group(2)
        )

        parameter = (
            bbox_match.group(3)
        )

        layer_map = {
            0: 0,
            1: 2,
            2: 4,
        }

        if official_layer in layer_map:
            target_layer = (
                layer_map[official_layer]
            )

            target = (
                f'bbox_head.reg_branches.'
                f'{branch}.{target_layer}.'
                f'{parameter}'
            )

            results.append(
                (
                    target,
                    'bbox_head',
                )
            )

    cls_match = re.match(
        r'^class_embed\.(\d+)\.'
        r'(weight|bias)$',
        normalized,
    )

    if cls_match is not None:
        branch = int(
            cls_match.group(1)
        )

        parameter = (
            cls_match.group(2)
        )

        target = (
            f'bbox_head.cls_branches.'
            f'{branch}.{parameter}'
        )

        results.append(
            (
                target,
                'bbox_head',
            )
        )

    return results


def map_global_dino(
    key: str,
) -> List[Tuple[str, str]]:
    results = []

    if key == 'transformer.level_embed':
        results.append(
            (
                'level_embed',
                'transformer',
            )
        )

    if key == 'transformer.tgt_embed.weight':
        results.append(
            (
                'query_embedding.weight',
                'transformer',
            )
        )

    prefix = 'transformer.enc_output.'

    if key.startswith(prefix):
        results.append(
            (
                'memory_trans_fc.'
                + key[len(prefix):],
                'transformer',
            )
        )

    prefix = 'transformer.enc_output_norm.'

    if key.startswith(prefix):
        results.append(
            (
                'memory_trans_norm.'
                + key[len(prefix):],
                'transformer',
            )
        )

    if key == 'label_enc.weight':
        results.append(
            (
                'dn_query_generator.'
                'label_embedding.weight',
                'dn',
            )
        )

    return results


def candidate_targets(
    key: str,
) -> List[Tuple[str, str]]:
    candidates = []

    candidates.append(
        (
            key,
            'exact',
        )
    )

    candidates.extend(
        map_backbone(key)
    )

    candidates.extend(
        map_input_proj(key)
    )

    candidates.extend(
        map_encoder_layer(key)
    )

    candidates.extend(
        map_decoder_layer(key)
    )

    candidates.extend(
        map_decoder_global(key)
    )

    candidates.extend(
        map_recognition(key)
    )

    candidates.extend(
        map_bbox_head(key)
    )

    candidates.extend(
        map_global_dino(key)
    )

    unique = []
    seen = set()

    for candidate in candidates:
        if candidate[0] in seen:
            continue

        seen.add(candidate[0])
        unique.append(candidate)

    return unique


def verify_required_targets(
    loaded_targets,
):
    required = [
        'decoder.recognition_semantics.'
        'rec_classifier.weight',

        'decoder.recognition_semantics.'
        'rec_classifier.bias',

        'decoder.recognition_semantics.'
        'rec_projection.weight',

        'decoder.recognition_semantics.'
        'rec_projection.bias',
    ]

    for layer in range(6):
        required.extend(
            [
                (
                    f'decoder.layers.{layer}.'
                    'vlc_attention.in_proj_weight'
                ),
                (
                    f'decoder.layers.{layer}.'
                    'self_attention.attn_intra.'
                    'in_proj_weight'
                ),
                (
                    f'decoder.layers.{layer}.'
                    'self_attention.attn_inter.'
                    'in_proj_weight'
                ),
            ]
        )

    missing = [
        key
        for key in required
        if key not in loaded_targets
    ]

    if missing:
        print()
        print('Required ESTextSpotter weights missing:')

        for key in missing:
            print('  ', key)

        raise RuntimeError(
            'Official recognition/task-aware '
            'decoder weights were not fully mapped. '
            'Do not train this checkpoint.'
        )


def main():
    print('=' * 80)
    print('ESTextSpotter -> OCRDINO checkpoint merge')
    print('=' * 80)

    print('Official:')
    print(OFFICIAL_CHECKPOINT)

    print('Base:')
    print(BASE_CHECKPOINT)

    print('Output:')
    print(OUTPUT_CHECKPOINT)

    if (
        OUTPUT_CHECKPOINT.resolve()
        == BASE_CHECKPOINT.resolve()
    ):
        raise RuntimeError(
            'Output checkpoint must not overwrite '
            'the base checkpoint.'
        )

    if OUTPUT_CHECKPOINT.exists():
        raise FileExistsError(
            f'Output already exists: '
            f'{OUTPUT_CHECKPOINT}'
        )

    official_checkpoint = load_checkpoint(
        OFFICIAL_CHECKPOINT
    )

    base_checkpoint = load_checkpoint(
        BASE_CHECKPOINT
    )

    official_state = normalize_state_dict(
        extract_state_dict(
            official_checkpoint
        )
    )

    base_state_raw = extract_state_dict(
        base_checkpoint
    )

    base_state = {
        strip_wrapper_prefix(key):
            value.clone()
            if torch.is_tensor(value)
            else value

        for key, value
        in base_state_raw.items()
        if torch.is_tensor(value)
    }

    print()
    print(
        'Official tensor count:',
        len(official_state),
    )

    print(
        'OCRDINO base tensor count:',
        len(base_state),
    )

    merged_state = {
        key: value.clone()
        for key, value
        in base_state.items()
    }

    loaded_targets = {}
    skipped_source = []
    shape_mismatches = []
    duplicate_same = []
    group_counter = Counter()

    for source_key, source_tensor in (
        official_state.items()
    ):
        matched = False

        for target_key, group in (
            candidate_targets(source_key)
        ):
            if target_key not in merged_state:
                continue

            target_tensor = (
                merged_state[target_key]
            )

            adapted = (
                adapt_feature_level_parameter(
                    source=source_tensor,
                    target=target_tensor,
                    target_key=target_key,
                )
            )

            if adapted is None:
                shape_mismatches.append(
                    (
                        source_key,
                        tuple(source_tensor.shape),
                        target_key,
                        tuple(target_tensor.shape),
                    )
                )

                continue

            if (
                tuple(adapted.shape)
                != tuple(target_tensor.shape)
            ):
                raise RuntimeError(
                    'Adapter returned wrong shape: '
                    f'{source_key} -> {target_key}'
                )

            if target_key in loaded_targets:
                previous_source = (
                    loaded_targets[
                        target_key
                    ]
                )

                if torch.equal(
                    merged_state[target_key],
                    adapted,
                ):
                    duplicate_same.append(
                        (
                            source_key,
                            target_key,
                        )
                    )

                    matched = True
                    break

                raise RuntimeError(
                    'Conflicting official parameters '
                    'map to the same OCRDINO key:\n'
                    f'  previous: {previous_source}\n'
                    f'  current:  {source_key}\n'
                    f'  target:   {target_key}'
                )

            merged_state[
                target_key
            ] = adapted.clone()

            loaded_targets[
                target_key
            ] = source_key

            group_counter[group] += 1

            matched = True
            break

        if not matched:
            skipped_source.append(
                source_key
            )

    verify_required_targets(
        loaded_targets
    )

    print()
    print('=' * 80)
    print('MERGE SUMMARY')
    print('=' * 80)

    print(
        'Official tensors:',
        len(official_state),
    )

    print(
        'OCRDINO tensors:',
        len(base_state),
    )

    print(
        'Updated OCRDINO tensors:',
        len(loaded_targets),
    )

    print()

    for group, count in sorted(
        group_counter.items()
    ):
        print(
            f'{group:20s}: {count}'
        )

    print()
    print(
        'Skipped official tensors:',
        len(skipped_source),
    )

    print(
        'Shape mismatch candidates:',
        len(shape_mismatches),
    )

    print(
        'Duplicate equivalent mappings:',
        len(duplicate_same),
    )

    print()
    print(
        'Recognition classifier source:',
        loaded_targets.get(
            'decoder.recognition_semantics.'
            'rec_classifier.weight'
        ),
    )

    print(
        'Recognition projection source:',
        loaded_targets.get(
            'decoder.recognition_semantics.'
            'rec_projection.weight'
        ),
    )

    print()
    print(
        'Verifying checkpoint against '
        'current OCRDINO model...'
    )

    register_all_modules(
        init_default_scope=True
    )

    cfg = Config.fromfile(
        str(CONFIG_PATH)
    )

    model = MODELS.build(
        cfg.model
    )

    load_result = model.load_state_dict(
        merged_state,
        strict=False,
    )

    missing_keys = list(
        load_result.missing_keys
    )

    unexpected_keys = list(
        load_result.unexpected_keys
    )

    print(
        'Model missing keys:',
        len(missing_keys),
    )

    print(
        'Model unexpected keys:',
        len(unexpected_keys),
    )

    if missing_keys:
        print()

        print('Missing keys:')

        for key in missing_keys:
            print('  ', key)

    if unexpected_keys:
        print()

        print('Unexpected keys:')

        for key in unexpected_keys:
            print('  ', key)

    if missing_keys or unexpected_keys:
        raise RuntimeError(
            'Merged state dict does not exactly '
            'match the current OCRDINO model.'
        )

    output_meta = copy.deepcopy(
        base_checkpoint.get(
            'meta',
            {},
        )
        if isinstance(
            base_checkpoint,
            dict,
        )
        else {}
    )

    output_meta[
        'ocrdino_init_source'
    ] = str(
        BASE_CHECKPOINT
    )

    output_meta[
        'estextspotter_pretrain_source'
    ] = str(
        OFFICIAL_CHECKPOINT
    )

    output_meta[
        'estextspotter_updated_tensors'
    ] = len(
        loaded_targets
    )

    output_checkpoint = dict(
        meta=output_meta,
        state_dict=merged_state,
    )

    OUTPUT_CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        output_checkpoint,
        str(OUTPUT_CHECKPOINT),
    )

    report_lines = []

    report_lines.append(
        'ESTextSpotter -> OCRDINO merge report'
    )

    report_lines.append(
        '=' * 80
    )

    report_lines.append(
        f'Official: {OFFICIAL_CHECKPOINT}'
    )

    report_lines.append(
        f'Base: {BASE_CHECKPOINT}'
    )

    report_lines.append(
        f'Output: {OUTPUT_CHECKPOINT}'
    )

    report_lines.append('')

    report_lines.append(
        f'Official tensors: '
        f'{len(official_state)}'
    )

    report_lines.append(
        f'OCRDINO tensors: '
        f'{len(base_state)}'
    )

    report_lines.append(
        f'Updated tensors: '
        f'{len(loaded_targets)}'
    )

    report_lines.append('')

    report_lines.append(
        'Updated groups:'
    )

    for group, count in sorted(
        group_counter.items()
    ):
        report_lines.append(
            f'  {group}: {count}'
        )

    report_lines.append('')

    report_lines.append(
        'Mapped tensors:'
    )

    for target_key in sorted(
        loaded_targets
    ):
        source_key = (
            loaded_targets[target_key]
        )

        report_lines.append(
            f'  {source_key}'
        )

        report_lines.append(
            f'    -> {target_key}'
        )

    report_lines.append('')

    report_lines.append(
        'Shape mismatches:'
    )

    for (
        source_key,
        source_shape,
        target_key,
        target_shape,
    ) in shape_mismatches:
        report_lines.append(
            f'  {source_key} '
            f'{source_shape}'
        )

        report_lines.append(
            f'    -> {target_key} '
            f'{target_shape}'
        )

    report_lines.append('')

    report_lines.append(
        'Skipped official keys:'
    )

    for key in skipped_source:
        report_lines.append(
            f'  {key}'
        )

    REPORT_PATH.write_text(
        '\n'.join(report_lines)
        + '\n',
        encoding='utf-8',
    )

    print()
    print('=' * 80)
    print('MERGE PASSED')
    print('=' * 80)

    print(
        'New checkpoint:',
        OUTPUT_CHECKPOINT,
    )

    print(
        'Report:',
        REPORT_PATH,
    )

    print()
    print(
        'The original OCRDINO init checkpoint '
        'was not modified.'
    )


if __name__ == '__main__':
    main()