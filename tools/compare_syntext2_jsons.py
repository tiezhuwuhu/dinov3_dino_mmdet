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


def load(path):
    with path.open(
        'r',
        encoding='utf-8',
    ) as f:
        return json.load(f)


a = load(A_PATH)
b = load(B_PATH)


print(
    'images equal:',
    a['images'] == b['images'],
)

print(
    'categories equal:',
    a['categories'] == b['categories'],
)

print(
    'A annotations:',
    len(a['annotations']),
)

print(
    'B annotations:',
    len(b['annotations']),
)


a_by_id = {
    ann['id']: ann
    for ann in a['annotations']
}

b_by_id = {
    ann['id']: ann
    for ann in b['annotations']
}


a_ids = set(a_by_id)
b_ids = set(b_by_id)


only_a = sorted(
    a_ids - b_ids
)

only_b = sorted(
    b_ids - a_ids
)

common = (
    a_ids & b_ids
)


different = [
    ann_id
    for ann_id in common
    if a_by_id[ann_id]
    != b_by_id[ann_id]
]


print(
    'only in train.json:',
    only_a,
)

print(
    'only in ecms json:',
    only_b,
)

print(
    'same id but different record count:',
    len(different),
)

print(
    'different ids:',
    different[:50],
)


for ann_id in only_a[:10]:
    print()
    print(
        'ONLY A:',
        a_by_id[ann_id],
    )


for ann_id in only_b[:10]:
    print()
    print(
        'ONLY B:',
        b_by_id[ann_id],
    )


for ann_id in different[:10]:
    print()
    print(
        'DIFF ID:',
        ann_id,
    )

    print(
        'A:',
        a_by_id[ann_id],
    )

    print(
        'B:',
        b_by_id[ann_id],
    )