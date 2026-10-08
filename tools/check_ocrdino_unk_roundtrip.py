import inspect

from mmengine.config import Config

from mmdet.registry import TRANSFORMS
from mmdet.utils import register_all_modules


CONFIG = (
    'configs/ocrdino/'
    'ocrdino_r50_totaltext_rec1e4_dn04_100e.py'
)


def get_tokenize_cfg(cfg):
    candidates = []

    if 'train_pipeline' in cfg:
        candidates.extend(
            cfg.train_pipeline
        )

    try:
        candidates.extend(
            cfg.train_dataloader.dataset.pipeline
        )
    except Exception:
        pass

    for item in candidates:
        if (
            isinstance(item, dict)
            and item.get('type')
            == 'TokenizeOCRText'
        ):
            return dict(item)

    raise RuntimeError(
        'TokenizeOCRText config not found.'
    )


def run_case(
    transform,
    text,
):
    results = {
        'gt_texts': [
            text
        ],
        'gt_text_types': [
            'text'
        ],
    }

    output = transform(
        results
    )

    if output is None:
        raise RuntimeError(
            'TokenizeOCRText returned None.'
        )

    token_ids = output.get(
        'gt_ocr_token_ids'
    )

    trainable = output.get(
        'gt_ocr_trainable'
    )

    print()
    print(
        'text:',
        repr(text),
    )

    print(
        'gt_ocr_token_ids:',
        token_ids,
    )

    print(
        'gt_ocr_trainable:',
        trainable,
    )

    for key in (
        'ocr_vocab_size',
        'ocr_pad_token_id',
        'ocr_bos_token_id',
        'ocr_eos_token_id',
        'ocr_unk_token_id',
    ):
        if key in output:
            print(
                key,
                '=',
                output[key],
            )

    if not isinstance(
        token_ids,
        list,
    ):
        raise RuntimeError(
            'gt_ocr_token_ids is not a list.'
        )

    if len(token_ids) != 1:
        raise RuntimeError(
            'Expected exactly one OCR target.'
        )

    return token_ids[0]


def strip_eos_tail(
    ids,
):
    ids = list(ids)

    if not ids:
        return ids

    # For this OCR vocabulary, source data has
    # normal chars 0..94, UNK 95, EOS 96.
    #
    # We only compare the active character part.
    if 96 in ids:
        first_eos = ids.index(96)
        return ids[:first_eos]

    return ids


def main():
    register_all_modules(
        init_default_scope=True
    )

    cfg = Config.fromfile(
        CONFIG
    )

    token_cfg = get_tokenize_cfg(
        cfg
    )

    print(
        'TokenizeOCRText config:'
    )

    print(
        token_cfg
    )

    transform = TRANSFORMS.build(
        token_cfg
    )

    print()
    print(
        'transform class:',
        transform.__class__
    )

    print(
        'transform module:',
        transform.__class__.__module__
    )

    print(
        'transform source file:',
        inspect.getsourcefile(
            transform.__class__
        )
    )

    print(
        'transform attributes:',
        sorted(
            vars(transform).keys()
        )
    )

    print()
    print(
        'Running actual transform tests...'
    )

    ids_the = run_case(
        transform,
        'the',
    )

    ids_unk1 = run_case(
        transform,
        '\ufffd',
    )

    ids_unk3 = run_case(
        transform,
        '\ufffd\ufffd\ufffd',
    )

    ids_mix = run_case(
        transform,
        'A\ufffdB',
    )

    active_the = strip_eos_tail(
        ids_the
    )

    active_unk1 = strip_eos_tail(
        ids_unk1
    )

    active_unk3 = strip_eos_tail(
        ids_unk3
    )

    active_mix = strip_eos_tail(
        ids_mix
    )

    print()
    print(
        '=' * 80
    )

    print(
        'ACTIVE TOKEN SUMMARY'
    )

    print(
        '=' * 80
    )

    print(
        "'the'     ->",
        active_the,
    )

    print(
        "'\\uFFFD'  ->",
        active_unk1,
    )

    print(
        "'\\uFFFDx3'->",
        active_unk3,
    )

    print(
        "'A\\uFFFDB'->",
        active_mix,
    )

    if active_the != [
        84,
        72,
        69,
    ]:
        raise RuntimeError(
            'ASCII mapping failed: '
            f"'the' -> {active_the}"
        )

    if active_unk1 != [
        95
    ]:
        raise RuntimeError(
            'UNK mapping failed: '
            f"'\\uFFFD' -> {active_unk1}"
        )

    if active_unk3 != [
        95,
        95,
        95,
    ]:
        raise RuntimeError(
            'Repeated UNK mapping failed: '
            f'{active_unk3}'
        )

    if active_mix != [
        33,
        95,
        34,
    ]:
        raise RuntimeError(
            'Mixed mapping failed: '
            f'{active_mix}'
        )

    print()
    print(
        'UNK ROUND-TRIP PASS'
    )


if __name__ == '__main__':
    main()