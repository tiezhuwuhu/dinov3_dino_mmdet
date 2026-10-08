import json
from pathlib import Path


ROOT = Path(
    '/root/autodl-tmp/dataset/ocrdino_pretrain/syntext2'
)

A_PATH = ROOT / 'train.json'

B_PATH = (
    ROOT
    / 'annotations'
    / 'ecms_v1_maxlen25.json'
)


def load_json(path):
    with path.open(
        'r',
        encoding='utf-8',
    ) as f:
        return json.load(f)


def remove_id(annotation):
    return {
        key: value
        for key, value in annotation.items()
        if key != 'id'
    }


a = load_json(A_PATH)
b = load_json(B_PATH)

a_annotations = a['annotations']
b_annotations = b['annotations']

a_by_id = {
    ann['id']: ann
    for ann in a_annotations
}

b_by_id = {
    ann['id']: ann
    for ann in b_annotations
}


print('A annotations:', len(a_annotations))
print('B annotations:', len(b_annotations))

assert len(b_annotations) == len(a_annotations) + 1

assert a['images'] == b['images']
assert a['categories'] == b['categories']

print('images equal: PASS')
print('categories equal: PASS')


first_difference = None

for ann_id in range(
    1,
    len(a_annotations) + 1,
):
    if (
        ann_id in b_by_id
        and a_by_id[ann_id]
        == b_by_id[ann_id]
    ):
        continue

    first_difference = ann_id
    break


print(
    'first different annotation id:',
    first_difference,
)


if first_difference is None:
    raise RuntimeError(
        'No difference found.'
    )


before_ok = True

for ann_id in range(
    1,
    first_difference,
):
    if (
        a_by_id[ann_id]
        != b_by_id[ann_id]
    ):
        before_ok = False

        print(
            'Mismatch before insertion:',
            ann_id,
        )

        break


print(
    'all annotations before insertion identical:',
    before_ok,
)


shift_ok = True
shift_mismatch = None

for a_id in range(
    first_difference,
    len(a_annotations) + 1,
):
    b_id = a_id + 1

    a_ann = remove_id(
        a_by_id[a_id]
    )

    b_ann = remove_id(
        b_by_id[b_id]
    )

    if a_ann != b_ann:
        shift_ok = False
        shift_mismatch = (
            a_id,
            b_id,
        )

        break


print(
    'all remaining annotations match with +1 id shift:',
    shift_ok,
)

print(
    'first shift mismatch:',
    shift_mismatch,
)


extra = b_by_id[
    first_difference
]

print()
print(
    'inserted annotation:'
)

print(
    json.dumps(
        extra,
        indent=2,
        ensure_ascii=False,
    )
)


extra_without_id = remove_id(
    extra
)

duplicate_in_a = []

for ann in a_annotations:
    if (
        remove_id(ann)
        == extra_without_id
    ):
        duplicate_in_a.append(
            ann['id']
        )


print()
print(
    'inserted annotation duplicate IDs in A:',
    duplicate_in_a,
)


image_id = extra['image_id']

a_same_image = [
    ann
    for ann in a_annotations
    if ann['image_id'] == image_id
]

b_same_image = [
    ann
    for ann in b_annotations
    if ann['image_id'] == image_id
]


print()
print(
    'inserted annotation image_id:',
    image_id,
)

print(
    'A annotation count for image:',
    len(a_same_image),
)

print(
    'B annotation count for image:',
    len(b_same_image),
)


if (
    before_ok
    and shift_ok
    and not duplicate_in_a
):
    print()
    print('=' * 80)

    print(
        'RELATION VERIFIED:'
    )

    print(
        'B is exactly A plus one unique '
        'annotation inserted at id '
        f'{first_difference}, with all '
        'subsequent annotation ids shifted by +1.'
    )

    print('=' * 80)

else:
    print()
    print('=' * 80)

    print(
        'RELATION NOT FULLY VERIFIED'
    )

    print('=' * 80)