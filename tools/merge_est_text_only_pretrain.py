from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Tuple

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
    'ocrdino_r50_dino_det_est_text_pretrain_init.pth'
)

REPORT_PATH = Path(
    '/root/autodl-tmp/checkpoint/ocrdino/'
    'ocrdino_r50_dino_det_est_text_pretrain_report.txt'
)

CONFIG_PATH = Path(
    '/root/autodl-tmp/mmdetection/configs/ocrdino/'
    'ocrdino_r50_totaltext_rec1e4_dn04_100e.py'
)

NUM_DECODER_LAYERS = 6
EXPECTED_TRANSFER_COUNT = 148


def load_checkpoint(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f'Checkpoint does not exist: {path}'
        )

    return torch.load(
        str(path),
        map_location='cpu',
        weights_only=False,
    )


def strip_prefix(key: str) -> str:
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


def extract_state_dict(
    checkpoint,
) -> Dict[str, torch.Tensor]:

    if not isinstance(checkpoint, dict):
        raise TypeError(
            'Checkpoint must be a dictionary.'
        )

    for container_key in (
        'state_dict',
        'model',
    ):
        candidate = checkpoint.get(
            container_key,
            None,
        )

        if isinstance(candidate, dict):
            tensor_count = sum(
                torch.is_tensor(value)
                for value in candidate.values()
            )

            if tensor_count > 0:
                return candidate

    tensor_count = sum(
        torch.is_tensor(value)
        for value in checkpoint.values()
    )

    if tensor_count > 0:
        return checkpoint

    raise RuntimeError(
        'No tensor state dict was found.'
    )


def normalize_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:

    output = {}

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue

        new_key = strip_prefix(key)

        if new_key in output:
            if not torch.equal(
                output[new_key],
                value,
            ):
                raise RuntimeError(
                    'Conflicting duplicate key: '
                    f'{new_key}'
                )

            continue

        output[new_key] = value.detach().cpu()

    return output


def resolve_source(
    official_state: Dict[str, torch.Tensor],
    suffix: str,
    target_tensor: torch.Tensor,
) -> Tuple[str, torch.Tensor]:

    matches: List[
        Tuple[str, torch.Tensor]
    ] = []

    expected_shape = tuple(
        target_tensor.shape
    )

    for key, tensor in official_state.items():

        key_matches = (
            key == suffix
            or key.endswith(
                '.' + suffix
            )
        )

        if not key_matches:
            continue

        if tuple(tensor.shape) != expected_shape:
            continue

        matches.append(
            (
                key,
                tensor,
            )
        )

    if not matches:
        raise RuntimeError(
            'Cannot find official source tensor:\n'
            f'  suffix: {suffix}\n'
            f'  target shape: {expected_shape}'
        )

    if len(matches) == 1:
        return matches[0]

    first_tensor = matches[0][1]

    all_equal = all(
        torch.equal(
            first_tensor,
            tensor,
        )
        for _, tensor in matches[1:]
    )

    if not all_equal:
        print()
        print(
            'Ambiguous official source keys:'
        )

        for key, tensor in matches:
            print(
                '  ',
                key,
                tuple(tensor.shape),
            )

        raise RuntimeError(
            'Multiple non-identical official '
            'tensors match one target.'
        )

    exact_matches = [
        item
        for item in matches
        if item[0] == suffix
    ]

    if exact_matches:
        return exact_matches[0]

    decoder_matches = [
        item
        for item in matches
        if item[0].startswith(
            'transformer.decoder.'
        )
    ]

    if decoder_matches:
        decoder_matches.sort(
            key=lambda item: len(item[0])
        )

        return decoder_matches[0]

    matches.sort(
        key=lambda item: len(item[0])
    )

    return matches[0]


def build_transfer_plan(
    official_state: Dict[str, torch.Tensor],
    base_state: Dict[str, torch.Tensor],
):
    plan = []

    def add(
        source_suffix: str,
        target_key: str,
        group: str,
    ):
        if target_key not in base_state:
            raise RuntimeError(
                'Target key does not exist:\n'
                f'  {target_key}'
            )

        source_key, source_tensor = (
            resolve_source(
                official_state=official_state,
                suffix=source_suffix,
                target_tensor=base_state[
                    target_key
                ],
            )
        )

        plan.append(
            dict(
                source_key=source_key,
                target_key=target_key,
                source_tensor=source_tensor,
                group=group,
            )
        )

    mha_suffixes = (
        'in_proj_weight',
        'in_proj_bias',
        'out_proj.weight',
        'out_proj.bias',
    )

    norm_suffixes = (
        'weight',
        'bias',
    )

    linear_suffixes = (
        'weight',
        'bias',
    )

    for layer in range(
        NUM_DECODER_LAYERS
    ):
        source_prefix = (
            f'transformer.decoder.'
            f'layers.{layer}.'
        )

        target_prefix = (
            f'decoder.layers.{layer}.'
        )

        for suffix in mha_suffixes:
            add(
                source_prefix
                + 'attn_m.'
                + suffix,
                target_prefix
                + 'vlc_attention.'
                + suffix,
                'vlc_attention',
            )

        for suffix in norm_suffixes:
            add(
                source_prefix
                + 'norm_m.'
                + suffix,
                target_prefix
                + 'vlc_attention_norm.'
                + suffix,
                'vlc_attention_norm',
            )

        for suffix in linear_suffixes:
            add(
                source_prefix
                + 'linear1_m.'
                + suffix,
                target_prefix
                + 'vlc_ffn_linear1.'
                + suffix,
                'vlc_ffn',
            )

            add(
                source_prefix
                + 'linear2_m.'
                + suffix,
                target_prefix
                + 'vlc_ffn_linear2.'
                + suffix,
                'vlc_ffn',
            )

        for suffix in norm_suffixes:
            add(
                source_prefix
                + 'norm3_m.'
                + suffix,
                target_prefix
                + 'vlc_ffn_norm.'
                + suffix,
                'vlc_ffn',
            )

        for suffix in mha_suffixes:
            add(
                source_prefix
                + 'attn_intra.'
                + suffix,
                target_prefix
                + 'self_attention.'
                + 'attn_intra.'
                + suffix,
                'intra_attention',
            )

        for suffix in norm_suffixes:
            add(
                source_prefix
                + 'norm_intra.'
                + suffix,
                target_prefix
                + 'self_attention.'
                + 'norm_intra.'
                + suffix,
                'intra_attention',
            )

        for suffix in mha_suffixes:
            add(
                source_prefix
                + 'attn_inter.'
                + suffix,
                target_prefix
                + 'self_attention.'
                + 'attn_inter.'
                + suffix,
                'inter_attention',
            )

        for suffix in norm_suffixes:
            add(
                source_prefix
                + 'norm_inter.'
                + suffix,
                target_prefix
                + 'self_attention.'
                + 'norm_inter.'
                + suffix,
                'inter_attention',
            )

    add(
        'rec_cls.weight',
        (
            'decoder.recognition_semantics.'
            'rec_classifier.weight'
        ),
        'recognition_semantics',
    )

    add(
        'rec_cls.bias',
        (
            'decoder.recognition_semantics.'
            'rec_classifier.bias'
        ),
        'recognition_semantics',
    )

    add(
        'rec_proj.weight',
        (
            'decoder.recognition_semantics.'
            'rec_projection.weight'
        ),
        'recognition_semantics',
    )

    add(
        'rec_proj.bias',
        (
            'decoder.recognition_semantics.'
            'rec_projection.bias'
        ),
        'recognition_semantics',
    )

    return plan


def verify_plan(
    plan,
):
    target_keys = [
        item['target_key']
        for item in plan
    ]

    if len(target_keys) != len(
        set(target_keys)
    ):
        raise RuntimeError(
            'Duplicate target key in '
            'transfer plan.'
        )

    if (
        len(target_keys)
        != EXPECTED_TRANSFER_COUNT
    ):
        raise RuntimeError(
            'Unexpected number of '
            'transfer tensors:\n'
            f'  got: {len(target_keys)}\n'
            f'  expected: '
            f'{EXPECTED_TRANSFER_COUNT}'
        )


def verify_base_against_model(
    base_state,
):
    register_all_modules(
        init_default_scope=True
    )

    cfg = Config.fromfile(
        str(CONFIG_PATH)
    )

    model = MODELS.build(
        cfg.model
    )

    model_state = model.state_dict()

    model_keys = set(
        model_state.keys()
    )

    base_keys = set(
        base_state.keys()
    )

    missing = sorted(
        model_keys - base_keys
    )

    unexpected = sorted(
        base_keys - model_keys
    )

    if missing:
        print()
        print(
            'Base checkpoint is missing keys:'
        )

        for key in missing:
            print(
                '  ',
                key,
            )

    if unexpected:
        print()
        print(
            'Base checkpoint has '
            'unexpected keys:'
        )

        for key in unexpected:
            print(
                '  ',
                key,
            )

    if missing or unexpected:
        raise RuntimeError(
            'Base OCRDINO init checkpoint '
            'does not exactly match the '
            'current model.'
        )

    load_result = model.load_state_dict(
        base_state,
        strict=False,
    )

    if (
        load_result.missing_keys
        or load_result.unexpected_keys
    ):
        raise RuntimeError(
            'Base checkpoint failed exact '
            'model loading.'
        )

    if (
        len(model.decoder.layers)
        != NUM_DECODER_LAYERS
    ):
        raise RuntimeError(
            'Unexpected decoder layer count.'
        )

    return model


def verify_protected_parameters(
    base_state,
    merged_state,
    transferred_targets,
):
    transferred_targets = set(
        transferred_targets
    )

    changed_protected = []

    for key in base_state:

        if key in transferred_targets:
            continue

        if not torch.equal(
            base_state[key],
            merged_state[key],
        ):
            changed_protected.append(
                key
            )

    if changed_protected:
        print()
        print(
            'ERROR: protected parameters '
            'were modified:'
        )

        for key in changed_protected:
            print(
                '  ',
                key,
            )

        raise RuntimeError(
            'Non-text parameters changed.'
        )


def verify_detection_parameters(
    base_state,
    merged_state,
):
    protected_prefixes = (
        'backbone.',
        'neck.',
        'encoder.',
        'bbox_head.',
        'dn_query_generator.',
        'query_embedding.',
        'memory_trans_fc.',
        'memory_trans_norm.',
    )

    protected_exact = (
        'level_embed',
    )

    protected_decoder_fragments = (
        '.cross_attention.',
        '.ffn_linear1.',
        '.ffn_linear2.',
        '.ffn_norm.',
        'decoder.ref_point_head.',
    )

    checked = 0

    for key in base_state:

        is_protected = (
            key in protected_exact
            or key.startswith(
                protected_prefixes
            )
            or any(
                fragment in key
                for fragment
                in protected_decoder_fragments
            )
        )

        if not is_protected:
            continue

        checked += 1

        if not torch.equal(
            base_state[key],
            merged_state[key],
        ):
            raise RuntimeError(
                'Detection/shared parameter '
                'was changed:\n'
                f'  {key}'
            )

    print(
        'Detection/shared tensors checked:',
        checked,
    )


def verify_merged_against_model(
    model,
    merged_state,
):
    result = model.load_state_dict(
        merged_state,
        strict=False,
    )

    if result.missing_keys:
        print()
        print('Missing keys:')

        for key in result.missing_keys:
            print(
                '  ',
                key,
            )

    if result.unexpected_keys:
        print()
        print('Unexpected keys:')

        for key in result.unexpected_keys:
            print(
                '  ',
                key,
            )

    if (
        result.missing_keys
        or result.unexpected_keys
    ):
        raise RuntimeError(
            'Merged checkpoint does not '
            'exactly match current model.'
        )


def main():
    print('=' * 80)
    print(
        'Strict EST text-only transfer'
    )
    print('=' * 80)

    print('Official:')
    print(
        OFFICIAL_CHECKPOINT
    )

    print('Base:')
    print(
        BASE_CHECKPOINT
    )

    print('Output:')
    print(
        OUTPUT_CHECKPOINT
    )

    if OUTPUT_CHECKPOINT.exists():
        raise FileExistsError(
            'Output already exists. '
            'Delete it manually if you '
            'really want to regenerate it:\n'
            f'{OUTPUT_CHECKPOINT}'
        )

    if (
        OUTPUT_CHECKPOINT.resolve()
        == BASE_CHECKPOINT.resolve()
    ):
        raise RuntimeError(
            'Output must not overwrite '
            'the base checkpoint.'
        )

    official_checkpoint = (
        load_checkpoint(
            OFFICIAL_CHECKPOINT
        )
    )

    base_checkpoint = (
        load_checkpoint(
            BASE_CHECKPOINT
        )
    )

    official_state = (
        normalize_state_dict(
            extract_state_dict(
                official_checkpoint
            )
        )
    )

    base_state = (
        normalize_state_dict(
            extract_state_dict(
                base_checkpoint
            )
        )
    )

    print()
    print(
        'Official tensors:',
        len(official_state),
    )

    print(
        'Base OCRDINO tensors:',
        len(base_state),
    )

    print()
    print(
        'Verifying base checkpoint...'
    )

    model = (
        verify_base_against_model(
            base_state
        )
    )

    print(
        'Base checkpoint: PASS'
    )

    print()
    print(
        'Building strict transfer plan...'
    )

    plan = build_transfer_plan(
        official_state=official_state,
        base_state=base_state,
    )

    verify_plan(
        plan
    )

    print(
        'Transfer tensor count:',
        len(plan),
    )

    merged_state = {
        key: tensor.clone()
        for key, tensor
        in base_state.items()
    }

    report_lines = []

    report_lines.append(
        'Strict EST text-only '
        'transfer report'
    )

    report_lines.append(
        '=' * 80
    )

    report_lines.append(
        f'Official: '
        f'{OFFICIAL_CHECKPOINT}'
    )

    report_lines.append(
        f'Base: '
        f'{BASE_CHECKPOINT}'
    )

    report_lines.append(
        f'Output: '
        f'{OUTPUT_CHECKPOINT}'
    )

    report_lines.append('')

    group_counts = {}

    transferred_targets = []

    equal_to_base = 0

    for item in plan:

        source_key = (
            item['source_key']
        )

        target_key = (
            item['target_key']
        )

        source_tensor = (
            item['source_tensor']
        )

        group = (
            item['group']
        )

        target_tensor = (
            base_state[target_key]
        )

        if (
            tuple(source_tensor.shape)
            != tuple(target_tensor.shape)
        ):
            raise RuntimeError(
                'Shape mismatch:\n'
                f'  source: {source_key} '
                f'{tuple(source_tensor.shape)}\n'
                f'  target: {target_key} '
                f'{tuple(target_tensor.shape)}'
            )

        converted = source_tensor.to(
            dtype=target_tensor.dtype
        )

        if torch.equal(
            converted,
            target_tensor,
        ):
            equal_to_base += 1

        merged_state[
            target_key
        ] = converted.clone()

        transferred_targets.append(
            target_key
        )

        group_counts[group] = (
            group_counts.get(
                group,
                0,
            )
            + 1
        )

        report_lines.append(
            source_key
        )

        report_lines.append(
            '  -> '
            + target_key
        )

    print()
    print('Transferred groups:')

    for group in sorted(
        group_counts
    ):
        print(
            f'  {group:24s}'
            f'{group_counts[group]}'
        )

    print()
    print(
        'Official tensors equal to '
        'base before transfer:',
        equal_to_base,
    )

    print()
    print(
        'Verifying all protected '
        'parameters...'
    )

    verify_protected_parameters(
        base_state=base_state,
        merged_state=merged_state,
        transferred_targets=(
            transferred_targets
        ),
    )

    print(
        'All non-transfer tensors '
        'unchanged: PASS'
    )

    verify_detection_parameters(
        base_state=base_state,
        merged_state=merged_state,
    )

    print(
        'Detection/shared parameter '
        'protection: PASS'
    )

    print()
    print(
        'Verifying merged checkpoint '
        'against current model...'
    )

    verify_merged_against_model(
        model=model,
        merged_state=merged_state,
    )

    print(
        'Model state validation: PASS'
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
        'base_checkpoint'
    ] = str(
        BASE_CHECKPOINT
    )

    output_meta[
        'estextspotter_checkpoint'
    ] = str(
        OFFICIAL_CHECKPOINT
    )

    output_meta[
        'transfer_policy'
    ] = (
        'EST text-specific modules only; '
        'all standard DINO detection and '
        'shared decoder modules preserved '
        'from OCRDINO base init'
    )

    output_meta[
        'transferred_tensor_count'
    ] = len(
        transferred_targets
    )

    output_checkpoint = dict(
        meta=output_meta,
        state_dict=merged_state,
    )

    OUTPUT_CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        '\n'.join(
            report_lines
        )
        + '\n',
        encoding='utf-8',
    )

    torch.save(
        output_checkpoint,
        str(OUTPUT_CHECKPOINT),
    )

    reloaded = load_checkpoint(
        OUTPUT_CHECKPOINT
    )

    reloaded_state = (
        normalize_state_dict(
            extract_state_dict(
                reloaded
            )
        )
    )

    if set(
        reloaded_state.keys()
    ) != set(
        merged_state.keys()
    ):
        raise RuntimeError(
            'Saved checkpoint key set '
            'changed after reload.'
        )

    for key in merged_state:
        if not torch.equal(
            merged_state[key],
            reloaded_state[key],
        ):
            raise RuntimeError(
                'Saved tensor changed '
                'after reload:\n'
                f'  {key}'
            )

    print()
    print('=' * 80)
    print('MERGE PASSED')
    print('=' * 80)

    print(
        'Transferred tensors:',
        len(transferred_targets),
    )

    print(
        'Expected tensors:',
        EXPECTED_TRANSFER_COUNT,
    )

    print(
        'Protected tensor changes: 0'
    )

    print()
    print(
        'New checkpoint:'
    )

    print(
        OUTPUT_CHECKPOINT
    )

    print()
    print(
        'Report:'
    )

    print(
        REPORT_PATH
    )

    print()
    print(
        'Original base checkpoint '
        'was not modified.'
    )


if __name__ == '__main__':
    main()