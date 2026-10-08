#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
from collections import Counter
from typing import Dict, List, Tuple, Optional

from PIL import Image


def is_image_file(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def get_image_size(path: str) -> Tuple[int, int]:
    with Image.open(path) as img:
        w, h = img.size
    return w, h


def safe_str(x) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return x.strip()


def is_finite_number(x) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return math.isfinite(v)


def flatten_lsvt_points(points) -> Optional[List[float]]:
    """
    LSVT format:
    "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    if not isinstance(points, list) or len(points) == 0:
        return None

    flat = []
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            return None
        x, y = p
        if not is_finite_number(x) or not is_finite_number(y):
            return None
        flat.append(float(x))
        flat.append(float(y))
    return flat


def flatten_rects_points(points) -> Optional[List[float]]:
    """
    ReCTS format:
    "points": [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    if not isinstance(points, list) or len(points) == 0:
        return None

    flat = []
    for v in points:
        if not is_finite_number(v):
            return None
        flat.append(float(v))

    if len(flat) % 2 != 0:
        return None

    return flat


def all_minus_one(points: List[float]) -> bool:
    if points is None or len(points) == 0:
        return False
    return all(v == -1.0 for v in points)


def clip_bbox(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[float, float, float, float]:
    x1 = max(0.0, min(float(w), x1))
    y1 = max(0.0, min(float(h), y1))
    x2 = max(0.0, min(float(w), x2))
    y2 = max(0.0, min(float(h), y2))
    return x1, y1, x2, y2


def points_to_bbox(points: List[float]) -> Tuple[float, float, float, float]:
    xs = points[0::2]
    ys = points[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def build_instance(
    text: str,
    ignore: int,
    bbox: List[float],
    bbox_label: int = 0,
) -> Dict:
    return {
        "bbox": bbox,
        "bbox_label": int(bbox_label),
        "ignore": int(ignore),
        "text": text,
        "text_type": "text",
    }


def validate_and_convert_instance(
    text: str,
    ignore: int,
    raw_points: List[float],
    image_w: int,
    image_h: int,
    stats: Counter,
    source_name: str,
) -> Optional[Dict]:
    """
    Rules:
    1. If all points are -1, drop directly.
    2. If points invalid, drop.
    3. Convert polygon/quad to bbox.
    4. Clip bbox to image boundary.
    5. If clipped bbox becomes degenerate, drop.
    """
    if raw_points is None:
        stats["skipped_invalid_points"] += 1
        return None

    if len(raw_points) < 8:
        stats["skipped_too_few_points"] += 1
        return None

    if all_minus_one(raw_points):
        stats["skipped_all_minus_one"] += 1
        return None

    x1, y1, x2, y2 = points_to_bbox(raw_points)

    raw_bbox = [x1, y1, x2, y2]
    clipped = clip_bbox(x1, y1, x2, y2, image_w, image_h)
    cx1, cy1, cx2, cy2 = clipped

    if [cx1, cy1, cx2, cy2] != raw_bbox:
        stats["clipped_bboxes"] += 1

    if not (cx2 > cx1 and cy2 > cy1):
        stats["skipped_degenerate_or_outside"] += 1
        return None

    bbox = [float(cx1), float(cy1), float(cx2), float(cy2)]

    instance = build_instance(
        text=text,
        ignore=ignore,
        bbox=bbox,
        bbox_label=0,
    )
    stats["valid_instances"] += 1
    if ignore:
        stats["ignored_instances"] += 1
    else:
        stats["non_ignored_instances"] += 1

    return instance


def write_records(out_path: str, records: List[Dict], overwrite: bool):
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(f"Output exists: {out_path}. Use --overwrite to replace it.")
    save_json(out_path, records)


def build_lsvt_image_index(lsvt_root: str) -> Dict[str, str]:
    """
    Map basename -> relative path
    Example:
    gt_1234.jpg -> LSVT/train_full_images_0/gt_1234.jpg
    """
    search_dirs = [
        os.path.join(lsvt_root, "train_full_images_0"),
        os.path.join(lsvt_root, "train_full_images_1"),
        os.path.join(lsvt_root, "test_part1_images"),
    ]

    index = {}
    for abs_dir in search_dirs:
        if not os.path.isdir(abs_dir):
            continue
        for name in os.listdir(abs_dir):
            if not is_image_file(name):
                continue
            abs_path = os.path.join(abs_dir, name)
            rel_path = os.path.relpath(abs_path, os.path.dirname(lsvt_root))
            index[name] = rel_path
    return index


def convert_lsvt(root: str, overwrite: bool) -> Dict:
    lsvt_root = os.path.join(root, "LSVT")
    ann_path = os.path.join(lsvt_root, "train_full_labels.json")
    out_path = os.path.join(lsvt_root, "ocrdino_train.json")

    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"LSVT label file not found: {ann_path}")

    print("=" * 10, "LSVT TRAIN", "=" * 10)
    data = load_json(ann_path)
    if not isinstance(data, dict):
        raise TypeError("LSVT train_full_labels.json must be a dict")

    image_index = build_lsvt_image_index(lsvt_root)
    stats = Counter()
    records = []

    total = len(data)
    for idx, (image_stem, anns) in enumerate(data.items(), start=1):
        image_name = image_stem + ".jpg"
        if image_name not in image_index:
            alt_png = image_stem + ".png"
            alt_jpeg = image_stem + ".jpeg"
            if alt_png in image_index:
                image_name = alt_png
            elif alt_jpeg in image_index:
                image_name = alt_jpeg
            else:
                stats["missing_images"] += 1
                continue

        rel_img_path = image_index[image_name]
        abs_img_path = os.path.join(root, rel_img_path)
        img_w, img_h = get_image_size(abs_img_path)

        instances = []
        if not isinstance(anns, list):
            anns = []

        for ann in anns:
            stats["total_regions"] += 1

            text = safe_str(ann.get("transcription", ""))
            illegibility = bool(ann.get("illegibility", False))
            ignore = 1 if (illegibility or text == "###") else 0

            points = flatten_lsvt_points(ann.get("points", None))
            instance = validate_and_convert_instance(
                text=text,
                ignore=ignore,
                raw_points=points,
                image_w=img_w,
                image_h=img_h,
                stats=stats,
                source_name=f"LSVT:{image_stem}",
            )
            if instance is not None:
                instances.append(instance)

        record = {
            "img_path": rel_img_path,
            "height": int(img_h),
            "width": int(img_w),
            "instances": instances,
        }
        records.append(record)
        stats["images"] += 1

        if idx % 5000 == 0 or idx == total:
            print(f"[LSVT] {idx}/{total}")

    write_records(out_path, records, overwrite=overwrite)
    print(f"[SAVED] {out_path}")

    summary = {
        "images": int(stats["images"]),
        "regions": int(stats["total_regions"]),
        "valid_regions": int(stats["valid_instances"]),
        "ignored_regions": int(stats["ignored_instances"]),
        "non_ignored_regions": int(stats["non_ignored_instances"]),
        "missing_images": int(stats["missing_images"]),
        "skipped_all_minus_one": int(stats["skipped_all_minus_one"]),
        "skipped_invalid_points": int(stats["skipped_invalid_points"]),
        "skipped_too_few_points": int(stats["skipped_too_few_points"]),
        "skipped_degenerate_or_outside": int(stats["skipped_degenerate_or_outside"]),
        "clipped_bboxes": int(stats["clipped_bboxes"]),
    }

    print("LSVT summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def choose_rects_label_dir(rects_root: str, prefer_unicode: bool = False) -> str:
    gt_dir = os.path.join(rects_root, "gt")
    gt_unicode_dir = os.path.join(rects_root, "gt_unicode")

    if prefer_unicode:
        if os.path.isdir(gt_unicode_dir):
            return gt_unicode_dir
        if os.path.isdir(gt_dir):
            return gt_dir
    else:
        if os.path.isdir(gt_dir):
            return gt_dir
        if os.path.isdir(gt_unicode_dir):
            return gt_unicode_dir

    raise FileNotFoundError("Neither rects/gt nor rects/gt_unicode exists.")


def convert_rects(root: str, overwrite: bool, prefer_unicode: bool = False) -> Dict:
    rects_root = os.path.join(root, "rects")
    img_dir = os.path.join(rects_root, "img")
    label_dir = choose_rects_label_dir(rects_root, prefer_unicode=prefer_unicode)
    out_path = os.path.join(rects_root, "ocrdino_train.json")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"ReCTS image dir not found: {img_dir}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"ReCTS label dir not found: {label_dir}")

    print()
    print("=" * 10, "ReCTS TRAIN", "=" * 10)
    print(f"Using label dir: {label_dir}")

    label_files = sorted(
        [x for x in os.listdir(label_dir) if x.lower().endswith(".json")]
    )

    stats = Counter()
    records = []

    total = len(label_files)
    for idx, label_name in enumerate(label_files, start=1):
        label_path = os.path.join(label_dir, label_name)
        stem = os.path.splitext(label_name)[0]
        image_name = stem + ".jpg"
        image_path = os.path.join(img_dir, image_name)

        if not os.path.isfile(image_path):
            alt_png = os.path.join(img_dir, stem + ".png")
            alt_jpeg = os.path.join(img_dir, stem + ".jpeg")
            if os.path.isfile(alt_png):
                image_path = alt_png
                image_name = stem + ".png"
            elif os.path.isfile(alt_jpeg):
                image_path = alt_jpeg
                image_name = stem + ".jpeg"
            else:
                stats["missing_images"] += 1
                continue

        img_w, img_h = get_image_size(image_path)
        label_data = load_json(label_path)

        lines = label_data.get("lines", [])
        if not isinstance(lines, list):
            lines = []

        instances = []

        for line_idx, ann in enumerate(lines):
            stats["total_regions"] += 1

            text = safe_str(ann.get("transcription", ""))
            ignore = int(ann.get("ignore", 0))
            if text == "###":
                ignore = 1

            points = flatten_rects_points(ann.get("points", None))
            instance = validate_and_convert_instance(
                text=text,
                ignore=ignore,
                raw_points=points,
                image_w=img_w,
                image_h=img_h,
                stats=stats,
                source_name=f"ReCTS:{stem}:line{line_idx}",
            )
            if instance is not None:
                instances.append(instance)

        rel_img_path = os.path.relpath(image_path, root)
        record = {
            "img_path": rel_img_path,
            "height": int(img_h),
            "width": int(img_w),
            "instances": instances,
        }
        records.append(record)
        stats["images"] += 1

        if idx % 5000 == 0 or idx == total:
            print(f"[ReCTS] {idx}/{total}")

    write_records(out_path, records, overwrite=overwrite)
    print(f"[SAVED] {out_path}")

    summary = {
        "images": int(stats["images"]),
        "regions": int(stats["total_regions"]),
        "valid_regions": int(stats["valid_instances"]),
        "ignored_regions": int(stats["ignored_instances"]),
        "non_ignored_regions": int(stats["non_ignored_instances"]),
        "missing_images": int(stats["missing_images"]),
        "skipped_all_minus_one": int(stats["skipped_all_minus_one"]),
        "skipped_invalid_points": int(stats["skipped_invalid_points"]),
        "skipped_too_few_points": int(stats["skipped_too_few_points"]),
        "skipped_degenerate_or_outside": int(stats["skipped_degenerate_or_outside"]),
        "clipped_bboxes": int(stats["clipped_bboxes"]),
    }

    print("ReCTS summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Convert LSVT and ReCTS to OCRDINO format."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Dataset root, e.g. /root/autodl-tmp/dataset/chinese_dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output json files.",
    )
    parser.add_argument(
        "--rects-use-unicode",
        action="store_true",
        help="Use rects/gt_unicode instead of rects/gt. Default uses rects/gt.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Dataset root not found: {root}")

    print("=" * 80)
    print("Chinese OCRDINO Dataset Conversion")
    print("=" * 80)
    print(f"Dataset root: {root}")
    print("Original files will NOT be modified.")
    print()

    lsvt_summary = convert_lsvt(root=root, overwrite=args.overwrite)
    rects_summary = convert_rects(
        root=root,
        overwrite=args.overwrite,
        prefer_unicode=args.rects_use_unicode,
    )

    print()
    print("=" * 80)
    print("ALL CONVERSIONS DONE")
    print("=" * 80)
    print("LSVT output :", os.path.join(root, "LSVT", "ocrdino_train.json"))
    print("ReCTS output:", os.path.join(root, "rects", "ocrdino_train.json"))
    print()
    print("Final summaries:")
    print(json.dumps(
        {
            "LSVT": lsvt_summary,
            "ReCTS": rects_summary,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()