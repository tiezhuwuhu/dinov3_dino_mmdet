
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Read-only integrity checker for LSVT and ReCTS OCRDINO datasets."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check all converted Chinese OCRDINO annotations and images."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/dataset/chinese_dataset"
        ),
    )
    parser.add_argument(
        "--lsvt-json",
        default="LSVT/ocrdino_train.json",
    )
    parser.add_argument(
        "--rects-json",
        default="rects/ocrdino_train.json",
    )
    parser.add_argument(
        "--progress",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--check-mmdet",
        action="store_true",
        help="Also try building OCRDinoDataset through MMDetection.",
    )
    return parser.parse_args()


class Checker:
    def __init__(self, name, root, json_path, max_examples):
        self.name = name
        self.root = root
        self.json_path = json_path
        self.max_examples = max_examples

        self.stats = Counter()
        self.errors = Counter()
        self.warnings = Counter()
        self.examples = defaultdict(list)

    def issue(self, kind, where, detail, warning=False):
        counter = self.warnings if warning else self.errors
        counter[kind] += 1

        if len(self.examples[kind]) < self.max_examples:
            self.examples[kind].append({
                "location": where,
                "detail": detail,
            })

    def load_dataset(self):
        print(f"\n{'=' * 75}")
        print(f"CHECKING {self.name}")
        print(f"{'=' * 75}")
        print("JSON:", self.json_path)

        if not self.json_path.is_file():
            self.issue(
                "json_missing",
                str(self.json_path),
                "JSON file does not exist",
            )
            return None

        try:
            with self.json_path.open(
                "r",
                encoding="utf-8-sig",
            ) as file:
                data = json.load(file)

        except Exception as exc:
            self.issue(
                "json_read_failure",
                str(self.json_path),
                repr(exc),
            )
            return None

        print("[OK] JSON syntax parsed successfully")

        if isinstance(data, list):
            # Continue checking images, but mark the format incorrect.
            self.issue(
                "invalid_top_level_format",
                str(self.json_path),
                "Top level is a list. OCRDinoDataset expects "
                "{'metainfo': ..., 'data_list': [...]}.",
            )
            return data

        if not isinstance(data, dict):
            self.issue(
                "invalid_top_level_format",
                str(self.json_path),
                f"Unexpected root type: {type(data).__name__}",
            )
            return None

        meta = data.get("metainfo")

        if not isinstance(meta, dict):
            self.issue(
                "missing_metainfo",
                str(self.json_path),
                "Missing metainfo dictionary",
            )
        else:
            if meta.get("format_version") != "ocrdino_v1":
                self.issue(
                    "wrong_format_version",
                    str(self.json_path),
                    f"format_version={meta.get('format_version')!r}",
                )

            if list(meta.get("classes", [])) != ["text"]:
                self.issue(
                    "wrong_classes",
                    str(self.json_path),
                    f"classes={meta.get('classes')!r}",
                )

        records = data.get("data_list")

        if not isinstance(records, list):
            self.issue(
                "missing_data_list",
                str(self.json_path),
                "data_list must be a list",
            )
            return None

        return records

    def resolve_image_path(self, img_path):
        path = Path(img_path)

        if path.is_absolute():
            return path

        # Explicit expected convention:
        # img_path is relative to chinese_dataset/.
        return self.root / path

    def check_image(self, record, index):
        location = f"image[{index}]"

        if not isinstance(record, dict):
            self.issue(
                "invalid_image_record",
                location,
                "Image record is not a dictionary",
            )
            return None

        img_path = record.get("img_path")

        if not isinstance(img_path, str) or not img_path:
            self.issue(
                "missing_img_path",
                location,
                "img_path is missing or invalid",
            )
            return None

        actual_path = self.resolve_image_path(img_path)

        if not actual_path.is_file():
            self.issue(
                "image_missing",
                location,
                str(actual_path),
            )
            return None

        try:
            # im.load() forces actual image decoding.
            with Image.open(actual_path) as im:
                im.load()
                actual_w, actual_h = im.size

        except Exception as exc:
            self.issue(
                "image_decode_failure",
                location,
                f"{actual_path}: {exc}",
            )
            return None

        self.stats["images_readable"] += 1

        declared_w = record.get("width")
        declared_h = record.get("height")

        if (
            not isinstance(declared_w, int)
            or not isinstance(declared_h, int)
            or declared_w <= 0
            or declared_h <= 0
        ):
            self.issue(
                "invalid_declared_image_size",
                location,
                f"width={declared_w}, height={declared_h}",
            )
        elif (
            declared_w != actual_w
            or declared_h != actual_h
        ):
            self.issue(
                "image_size_mismatch",
                location,
                f"JSON=({declared_w},{declared_h}), "
                f"actual=({actual_w},{actual_h})",
            )

        return actual_w, actual_h

    @staticmethod
    def numeric_vector(value, length=None):
        if not isinstance(value, (list, tuple)):
            return None

        if length is not None and len(value) != length:
            return None

        try:
            numbers = [float(v) for v in value]
        except (TypeError, ValueError):
            return None

        if not all(math.isfinite(v) for v in numbers):
            return None

        return numbers

    def check_instance(self, inst, image_index, inst_index, size):
        location = f"image[{image_index}].instances[{inst_index}]"
        self.stats["instances_total"] += 1

        if not isinstance(inst, dict):
            self.issue(
                "invalid_instance",
                location,
                "Instance is not a dictionary",
            )
            return

        # --------------------------------------------------
        # OCRDINO field checks
        # --------------------------------------------------
        if "ignore_flag" not in inst:
            hint = (
                "Found 'ignore' instead of 'ignore_flag'"
                if "ignore" in inst
                else "ignore_flag is missing"
            )

            self.issue(
                "missing_ignore_flag",
                location,
                hint,
            )

        else:
            flag = inst["ignore_flag"]

            if (
                isinstance(flag, bool)
                or not isinstance(flag, int)
                or flag not in (0, 1)
            ):
                self.issue(
                    "invalid_ignore_flag",
                    location,
                    f"ignore_flag={flag!r}",
                )

            elif flag == 1:
                self.stats["instances_ignored"] += 1

            else:
                self.stats["instances_not_ignored"] += 1

        if inst.get("bbox_label") != 0:
            self.issue(
                "invalid_bbox_label",
                location,
                f"bbox_label={inst.get('bbox_label')!r}",
            )

        if not isinstance(inst.get("text"), str):
            self.issue(
                "invalid_text",
                location,
                f"text={inst.get('text')!r}",
            )

        if inst.get("text_type") not in ("text", "none"):
            self.issue(
                "invalid_text_type",
                location,
                f"text_type={inst.get('text_type')!r}",
            )

        # poly and order are part of our earlier unified format.
        # Missing them is reported separately as a warning.
        if "poly" not in inst:
            self.issue(
                "missing_poly",
                location,
                "Original polygon not preserved",
                warning=True,
            )

        if "order" not in inst:
            self.issue(
                "missing_order",
                location,
                "Original instance order not preserved",
                warning=True,
            )

        # --------------------------------------------------
        # bbox checks
        # --------------------------------------------------
        bbox = self.numeric_vector(
            inst.get("bbox"),
            length=4,
        )

        if bbox is None:
            self.issue(
                "invalid_bbox_format",
                location,
                f"bbox={inst.get('bbox')!r}",
            )
            return

        if all(v == -1 for v in bbox):
            self.issue(
                "minus_one_bbox",
                location,
                str(bbox),
            )
            return

        x1, y1, x2, y2 = bbox

        if x2 <= x1 or y2 <= y1:
            self.issue(
                "degenerate_bbox",
                location,
                str(bbox),
            )
            return

        if size is not None:
            width, height = size

            if (
                x1 < 0
                or y1 < 0
                or x2 > width
                or y2 > height
            ):
                self.issue(
                    "bbox_out_of_bounds",
                    location,
                    f"bbox={bbox}, image=({width},{height})",
                )
                return

        self.stats["bboxes_valid"] += 1

        # --------------------------------------------------
        # Optional polygon geometry checks
        # --------------------------------------------------
        if "poly" in inst:
            poly = self.numeric_vector(inst["poly"])

            if (
                poly is None
                or len(poly) < 8
                or len(poly) % 2 != 0
            ):
                self.issue(
                    "invalid_poly",
                    location,
                    f"poly={inst['poly']!r}",
                )
            elif all(v == -1 for v in poly):
                self.issue(
                    "minus_one_poly",
                    location,
                    str(poly),
                )

    def run(self, progress):
        records = self.load_dataset()

        if records is None:
            return self.make_report()

        print("Image records:", len(records))

        self.stats["images_total"] = len(records)

        seen_paths = set()

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                self.issue(
                    "invalid_image_record",
                    f"image[{index}]",
                    "Not a dictionary",
                )
                continue

            path = record.get("img_path")

            if isinstance(path, str):
                if path in seen_paths:
                    self.issue(
                        "duplicate_img_path",
                        f"image[{index}]",
                        path,
                    )
                seen_paths.add(path)

            size = self.check_image(record, index)

            instances = record.get("instances")

            if not isinstance(instances, list):
                self.issue(
                    "invalid_instances_list",
                    f"image[{index}]",
                    "instances must be a list",
                )
                continue

            for inst_index, inst in enumerate(instances):
                self.check_instance(
                    inst,
                    index,
                    inst_index,
                    size,
                )

            if (
                progress > 0
                and (
                    (index + 1) % progress == 0
                    or index + 1 == len(records)
                )
            ):
                print(
                    f"[{self.name}] "
                    f"{index + 1}/{len(records)} images, "
                    f"{self.stats['images_readable']} readable, "
                    f"{self.stats['instances_total']} instances"
                )

        return self.make_report()

    def make_report(self):
        result = {
            "dataset": self.name,
            "json_path": str(self.json_path),
            "statistics": dict(self.stats),
            "errors": dict(self.errors),
            "warnings": dict(self.warnings),
            "examples": dict(self.examples),
            "passed": sum(self.errors.values()) == 0,
        }

        print()
        print(f"RESULT: {self.name}")
        print("Readable images:", self.stats["images_readable"])
        print("Total images:", self.stats["images_total"])
        print("Valid boxes:", self.stats["bboxes_valid"])
        print("Total instances:", self.stats["instances_total"])
        print("Error count:", sum(self.errors.values()))
        print("Warning count:", sum(self.warnings.values()))

        for key, count in self.errors.items():
            print(f"  [ERROR] {key}: {count}")

        for key, count in self.warnings.items():
            print(f"  [WARN]  {key}: {count}")

        return result


def try_mmdet_load(root, relative_json):
    """Build the real dataset class only after schema checks pass."""
    from mmdet.registry import DATASETS
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=True)

    # Explicitly import the custom OCRDINO dataset module.
    import mmdet.datasets.ocr_dino_dataset  # noqa: F401

    dataset = DATASETS.build({
        "type": "OCRDinoDataset",
        "data_root": str(root),
        "ann_file": str(relative_json),
        "data_prefix": {"img_path": ""},
        "pipeline": [],
        "test_mode": True,
        "filter_cfg": None,
        "serialize_data": False,
    })

    if len(dataset) == 0:
        raise RuntimeError("OCRDinoDataset loaded zero images")

    first = dataset.get_data_info(0)

    print(
        "[MMDET OK]",
        relative_json,
        "images=",
        len(dataset),
        "first_image=",
        first.get("img_path"),
    )


def main():
    args = parse_args()

    root = args.root.expanduser().resolve()

    if not root.is_dir():
        raise NotADirectoryError(root)

    targets = {
        "LSVT": args.lsvt_json,
        "ReCTS": args.rects_json,
    }

    reports = {}
    all_passed = True

    print("=" * 75)
    print("Chinese OCRDINO full integrity check")
    print("=" * 75)
    print("Root:", root)
    print("Image decode: FULL")
    print("Source JSON and images: READ ONLY")

    for name, relative_json in targets.items():
        checker = Checker(
            name=name,
            root=root,
            json_path=root / relative_json,
            max_examples=args.max_examples,
        )

        report = checker.run(args.progress)

        if args.check_mmdet:
            if report["passed"]:
                try:
                    try_mmdet_load(root, relative_json)
                    report["mmdet_load"] = "passed"
                except Exception as exc:
                    report["mmdet_load"] = "failed"
                    checker.issue(
                        "mmdet_load_failure",
                        name,
                        repr(exc),
                    )
                    report = checker.make_report()
            else:
                report["mmdet_load"] = "not_run_schema_errors"

        reports[name] = report

        if not report["passed"]:
            all_passed = False

    output_path = root / "ocrdino_integrity_report.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            reports,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 75)
    print("FINAL SUMMARY")
    print("=" * 75)

    for name, report in reports.items():
        stats = report["statistics"]

        print(
            f"{name}: "
            f"images={stats.get('images_total', 0)}, "
            f"readable={stats.get('images_readable', 0)}, "
            f"instances={stats.get('instances_total', 0)}, "
            f"errors={sum(report['errors'].values())}, "
            f"warnings={sum(report['warnings'].values())}"
        )

    print()
    print("Saved report:", output_path)

    if all_passed:
        print("[PASS] All checked files and schemas passed")
    else:
        print("[FAIL] Problems found. Check the report.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
