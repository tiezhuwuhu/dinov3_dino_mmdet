#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/dinov3_dino_mmdet"
MMDET="${ROOT}/mmdetection"
CONFIG_DIR="${MMDET}/configs/dino"
WORK_ROOT="${ROOT}/work_dirs/stage3_grid_ext"

RESULT_CSV="${WORK_ROOT}/stage3_grid_ext_results.csv"
RESULT_MD="${WORK_ROOT}/stage3_grid_ext_results.md"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="${MMDET}:${PYTHONPATH:-}"

cd "${MMDET}"

LAMBDAS=("030" "040" "050" "060")
COSTS=("25" "30" "35" "40")

echo "============================================================"
echo "Point-DINO Stage 3 Extended Grid Test"
echo "============================================================"

for LTAG in "${LAMBDAS[@]}"; do
    for CTAG in "${COSTS[@]}"; do

        EXP_NAME="point_dino_stage3_lam${LTAG}_cost${CTAG}"
        CONFIG="${CONFIG_DIR}/${EXP_NAME}.py"
        EXP_DIR="${WORK_ROOT}/${EXP_NAME}"
        TEST_LOG="${EXP_DIR}/best_test.log"

        echo
        echo "============================================================"
        echo "Experiment: ${EXP_NAME}"
        echo "============================================================"

        if [[ ! -f "${CONFIG}" ]]; then
            echo "[ERROR] Config not found: ${CONFIG}"
            continue
        fi

        if [[ ! -d "${EXP_DIR}" ]]; then
            echo "[ERROR] Work dir not found: ${EXP_DIR}"
            continue
        fi

        CKPT=$(find "${EXP_DIR}" \
            -maxdepth 1 \
            -type f \
            -name 'best_point_f1@8px_epoch_*.pth' \
            | sort -V \
            | tail -n 1 || true)

        if [[ -z "${CKPT}" ]]; then
            echo "[ERROR] Best checkpoint not found."
            continue
        fi

        echo "Checkpoint: ${CKPT}"

        if [[ -f "${TEST_LOG}" ]] && \
           grep -q 'point/f1@4px:' "${TEST_LOG}" && \
           grep -q 'point/f1@8px:' "${TEST_LOG}"; then

            echo "[CACHE] Existing test result found."

        else

            python tools/test.py \
                "${CONFIG}" \
                "${CKPT}" \
                2>&1 | tee "${TEST_LOG}"
        fi

    done
done


python - <<'PY'
from pathlib import Path
import csv
import re

ROOT = Path("/root/autodl-tmp/dinov3_dino_mmdet")
WORK_ROOT = ROOT / "work_dirs" / "stage3_grid_ext"

lambdas = [
    ("030", 0.30),
    ("040", 0.40),
    ("050", 0.50),
    ("060", 0.60),
]

costs = [
    ("25", 25),
    ("30", 30),
    ("35", 35),
    ("40", 40),
]

metric_names = [
    "precision@4px",
    "recall@4px",
    "f1@4px",
    "mean_localization_error@4px",
    "precision@8px",
    "recall@8px",
    "f1@8px",
    "mean_localization_error@8px",
    "tp@4px",
    "fp@4px",
    "fn@4px",
    "tp@8px",
    "fp@8px",
    "fn@8px",
]

csv_path = WORK_ROOT / "stage3_grid_ext_results.csv"
md_path = WORK_ROOT / "stage3_grid_ext_results.md"


def extract_metrics(log_path):
    if not log_path.exists():
        return None

    text = log_path.read_text(encoding="utf-8", errors="ignore")

    lines = [
        line for line in text.splitlines()
        if "point/f1@4px:" in line
        and "point/f1@8px:" in line
    ]

    if not lines:
        return None

    line = lines[-1]
    result = {}

    for name in metric_names:
        pattern = rf"point/{re.escape(name)}:\s+([0-9.eE+-]+)"
        match = re.search(pattern, line)

        if match:
            value = float(match.group(1))

            if name.startswith(("tp@", "fp@", "fn@")):
                value = int(round(value))

            result[name] = value

    return result


matrix = {}
rows = []

for ltag, lam in lambdas:
    matrix[lam] = {}

    for ctag, cost in costs:
        exp_name = f"point_dino_stage3_lam{ltag}_cost{ctag}"
        exp_dir = WORK_ROOT / exp_name
        log_path = exp_dir / "best_test.log"

        metrics = extract_metrics(log_path)

        ckpts = sorted(
            exp_dir.glob("best_point_f1@8px_epoch_*.pth")
        )

        checkpoint = str(ckpts[-1]) if ckpts else ""

        row = {
            "lambda": lam,
            "PointL1Cost": cost,
            "checkpoint": checkpoint,
        }

        if metrics is not None:
            row.update(metrics)

        rows.append(row)
        matrix[lam][cost] = metrics


fieldnames = [
    "lambda",
    "PointL1Cost",
    "precision@4px",
    "recall@4px",
    "f1@4px",
    "mean_localization_error@4px",
    "precision@8px",
    "recall@8px",
    "f1@8px",
    "mean_localization_error@8px",
    "tp@4px",
    "fp@4px",
    "fn@4px",
    "tp@8px",
    "fp@8px",
    "fn@8px",
    "checkpoint",
]

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for row in rows:
        writer.writerow(row)


def build_matrix(metric, percent=False, digits=2):
    lines = []

    header = "| lambda / Cost | " + " | ".join(
        str(cost) for _, cost in costs
    ) + " |"

    separator = "|---:|" + "|".join(
        "---:" for _ in costs
    ) + "|"

    lines.append(header)
    lines.append(separator)

    for _, lam in lambdas:
        cells = []

        for _, cost in costs:
            m = matrix[lam][cost]

            if m is None or metric not in m:
                cells.append("-")
            else:
                value = m[metric]

                if percent:
                    value *= 100

                cells.append(f"{value:.{digits}f}")

        lines.append(
            f"| {lam:.2f} | " + " | ".join(cells) + " |"
        )

    return lines


md = []

md.append("# Point-DINO Stage 3 Extended Grid")
md.append("")
md.append(
    "Native 1024x768, Point DN=0.01, FocalLossCost=2."
)
md.append("")

md.append("## F1@4 (%)")
md.append("")
md.extend(build_matrix("f1@4px", percent=True, digits=2))

md.append("")
md.append("## F1@8 (%)")
md.append("")
md.extend(build_matrix("f1@8px", percent=True, digits=2))

md.append("")
md.append("## MLE@4 (px)")
md.append("")
md.extend(
    build_matrix(
        "mean_localization_error@4px",
        percent=False,
        digits=4,
    )
)

md.append("")
md.append("## MLE@8 (px)")
md.append("")
md.extend(
    build_matrix(
        "mean_localization_error@8px",
        percent=False,
        digits=4,
    )
)

valid = [
    row for row in rows
    if "f1@4px" in row and "f1@8px" in row
]

if valid:
    best4 = max(valid, key=lambda x: x["f1@4px"])
    best8 = max(valid, key=lambda x: x["f1@8px"])

    md.append("")
    md.append("## Best Results")
    md.append("")
    md.append(
        f"- Best F1@4: lambda={best4['lambda']:.2f}, "
        f"Cost={best4['PointL1Cost']}, "
        f"F1@4={best4['f1@4px'] * 100:.2f}%"
    )
    md.append(
        f"- Best F1@8: lambda={best8['lambda']:.2f}, "
        f"Cost={best8['PointL1Cost']}, "
        f"F1@8={best8['f1@8px'] * 100:.2f}%"
    )

md_path.write_text(
    "\n".join(md) + "\n",
    encoding="utf-8",
)


print()
print("=" * 90)
print("F1@4 / F1@8 (%)")
print("=" * 90)

print(
    f"{'lambda':>8}"
    + "".join(
        f"{('Cost=' + str(cost)):>20}"
        for _, cost in costs
    )
)

for _, lam in lambdas:
    line = f"{lam:>8.2f}"

    for _, cost in costs:
        m = matrix[lam][cost]

        if m is None:
            value = "N/A"
        else:
            value = (
                f"{m['f1@4px'] * 100:.2f}"
                f" / "
                f"{m['f1@8px'] * 100:.2f}"
            )

        line += f"{value:>20}"

    print(line)


if valid:
    print()
    print("Best F1@4:")
    print(
        f"lambda={best4['lambda']:.2f}, "
        f"Cost={best4['PointL1Cost']}, "
        f"F1@4={best4['f1@4px'] * 100:.2f}%"
    )

    print()
    print("Best F1@8:")
    print(
        f"lambda={best8['lambda']:.2f}, "
        f"Cost={best8['PointL1Cost']}, "
        f"F1@8={best8['f1@8px'] * 100:.2f}%"
    )

print()
print("Markdown:")
print(md_path)

print()
print("CSV:")
print(csv_path)
PY

echo
echo "============================================================"
echo "DONE"
echo "============================================================"
echo "Markdown: ${RESULT_MD}"
echo "CSV:      ${RESULT_CSV}"