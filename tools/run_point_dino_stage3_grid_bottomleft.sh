#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/dinov3_dino_mmdet"
MMDET="${ROOT}/mmdetection"
CONFIG_DIR="${MMDET}/configs/dino"
WORK_ROOT="${ROOT}/work_dirs/stage3_grid_bottomleft"

BASE_CONFIG="${CONFIG_DIR}/point_dino_r50_shanghaitech_stage2_dn001_euc01_native_12e.py"

RESULT_CSV="${WORK_ROOT}/stage3_grid_bottomleft_results.csv"
RESULT_MD="${WORK_ROOT}/stage3_grid_bottomleft_results.md"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="${MMDET}:${PYTHONPATH:-}"

cd "${MMDET}"
mkdir -p "${WORK_ROOT}"

LAMBDAS=(
    "0.3"
    "0.4"
    "0.5"
    "0.6"
)

LAMBDA_TAGS=(
    "030"
    "040"
    "050"
    "060"
)

COSTS=(
    "5"
    "10"
    "15"
    "20"
)

echo "============================================================"
echo "Point-DINO Stage 3 Bottom-Left Grid"
echo "============================================================"
echo "lambda = 0.30 0.40 0.50 0.60"
echo "cost   = 5 10 15 20"
echo "total  = 16 experiments"
echo "work   = ${WORK_ROOT}"
echo "============================================================"


# ============================================================
# Train + test
# ============================================================

for i in "${!LAMBDAS[@]}"; do

    LAMBDA="${LAMBDAS[$i]}"
    LTAG="${LAMBDA_TAGS[$i]}"

    for COST in "${COSTS[@]}"; do

        CTAG=$(printf "%02d" "${COST}")

        EXP_NAME="point_dino_stage3_lam${LTAG}_cost${CTAG}"

        CONFIG="${CONFIG_DIR}/${EXP_NAME}.py"
        WORK_DIR="${WORK_ROOT}/${EXP_NAME}"
        TEST_LOG="${WORK_DIR}/best_test.log"

        mkdir -p "${WORK_DIR}"

        echo
        echo "============================================================"
        echo "Experiment       : ${EXP_NAME}"
        echo "Euclidean lambda : ${LAMBDA}"
        echo "PointL1Cost      : ${COST}"
        echo "FocalLossCost    : 2.0"
        echo "Point DN         : 0.01"
        echo "============================================================"


        # ====================================================
        # Generate config
        # ====================================================

        cat > "${CONFIG}" <<EOF
_base_ = './point_dino_r50_shanghaitech_stage2_dn001_euc01_native_12e.py'

model = dict(
    bbox_head=dict(
        point_euclidean_weight=${LAMBDA}),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(
                    type='FocalLossCost',
                    weight=2.0),
                dict(
                    type='PointL1Cost',
                    weight=${COST}.0)
            ]
        )
    )
)

work_dir = '${WORK_DIR}'
EOF


        # ====================================================
        # Find existing best checkpoint
        # ====================================================

        BEST_CKPT=$(find "${WORK_DIR}" \
            -maxdepth 1 \
            -type f \
            -name 'best_point_f1@8px_epoch_*.pth' \
            2>/dev/null \
            | sort -V \
            | tail -n 1 || true)


        # ====================================================
        # Train if necessary
        # ====================================================

        if [[ -z "${BEST_CKPT}" ]]; then

            echo
            echo "[TRAIN] No best checkpoint found."
            echo "[TRAIN] Checking merged config..."

            python - <<PY
from mmengine.config import Config

cfg = Config.fromfile('${CONFIG}')

print()
print('===== CONFIG CHECK =====')
print('Experiment       : ${EXP_NAME}')
print('Euclidean lambda :', cfg.model.bbox_head.point_euclidean_weight)
print('Matcher          :', cfg.model.train_cfg.assigner.match_costs)
print('DN               :', cfg.model.dn_cfg)

print()
print('Train dataset:')
print('type     =', cfg.train_dataloader.dataset.type)
print('ann_file =', cfg.train_dataloader.dataset.ann_file)

print()
print('Train pipeline:')
for item in cfg.train_dataloader.dataset.pipeline:
    print(item)

print()
print('Test pipeline:')
for item in cfg.test_dataloader.dataset.pipeline:
    print(item)

cfg.dump('${WORK_DIR}/merged_config.py')
PY

            echo
            echo "[TRAIN] Starting training..."
            echo

            python tools/train.py \
                "${CONFIG}" \
                --work-dir "${WORK_DIR}" \
                2>&1 | tee "${WORK_DIR}/train.log"

            BEST_CKPT=$(find "${WORK_DIR}" \
                -maxdepth 1 \
                -type f \
                -name 'best_point_f1@8px_epoch_*.pth' \
                2>/dev/null \
                | sort -V \
                | tail -n 1 || true)

            if [[ -z "${BEST_CKPT}" ]]; then
                echo
                echo "[ERROR] Training finished but best checkpoint was not found."
                exit 1
            fi

        else

            echo
            echo "[SKIP TRAIN] Existing best checkpoint:"
            echo "${BEST_CKPT}"

        fi


        # ====================================================
        # Test best checkpoint
        # ====================================================

        if [[ -f "${TEST_LOG}" ]] && \
           grep -q 'point/f1@4px:' "${TEST_LOG}" && \
           grep -q 'point/f1@8px:' "${TEST_LOG}"; then

            echo
            echo "[SKIP TEST] Existing complete best_test.log found."

        else

            echo
            echo "[TEST] Best checkpoint:"
            echo "${BEST_CKPT}"
            echo

            python tools/test.py \
                "${CONFIG}" \
                "${BEST_CKPT}" \
                2>&1 | tee "${TEST_LOG}"

        fi

        echo
        echo "[DONE] ${EXP_NAME}"
        echo

    done
done


# ============================================================
# Parse all test logs
# ============================================================

python - <<'PY'
from pathlib import Path
import csv
import re

ROOT = Path("/root/autodl-tmp/dinov3_dino_mmdet")
WORK_ROOT = ROOT / "work_dirs" / "stage3_grid_bottomleft"

lambdas = [
    ("030", 0.30),
    ("040", 0.40),
    ("050", 0.50),
    ("060", 0.60),
]

costs = [
    ("05", 5),
    ("10", 10),
    ("15", 15),
    ("20", 20),
]

metric_names = [
    "precision@4px",
    "recall@4px",
    "f1@4px",
    "mean_localization_error@4px",
    "tp@4px",
    "fp@4px",
    "fn@4px",
    "precision@8px",
    "recall@8px",
    "f1@8px",
    "mean_localization_error@8px",
    "tp@8px",
    "fp@8px",
    "fn@8px",
]

csv_path = WORK_ROOT / "stage3_grid_bottomleft_results.csv"
md_path = WORK_ROOT / "stage3_grid_bottomleft_results.md"


def extract_metrics(log_path):
    if not log_path.exists():
        return None

    text = log_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

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
        else:
            print(
                f"[WARNING] Missing metrics: "
                f"lambda={lam}, cost={cost}"
            )

        matrix[lam][cost] = metrics
        rows.append(row)


# ============================================================
# CSV
# ============================================================

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

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in rows:
        writer.writerow(row)


# ============================================================
# Markdown helpers
# ============================================================

def table(metric, percent=False, digits=2):

    lines = []

    lines.append(
        "| lambda / Cost | "
        + " | ".join(str(cost) for _, cost in costs)
        + " |"
    )

    lines.append(
        "|---:|"
        + "|".join("---:" for _ in costs)
        + "|"
    )

    for _, lam in lambdas:

        cells = []

        for _, cost in costs:

            metrics = matrix[lam][cost]

            if metrics is None or metric not in metrics:
                cells.append("-")
                continue

            value = metrics[metric]

            if percent:
                value *= 100.0

            cells.append(
                f"{value:.{digits}f}"
            )

        lines.append(
            f"| {lam:.2f} | "
            + " | ".join(cells)
            + " |"
        )

    return lines


def combined_f1_table():

    lines = []

    lines.append(
        "| lambda / Cost | "
        + " | ".join(str(cost) for _, cost in costs)
        + " |"
    )

    lines.append(
        "|---:|"
        + "|".join("---:" for _ in costs)
        + "|"
    )

    for _, lam in lambdas:

        cells = []

        for _, cost in costs:

            metrics = matrix[lam][cost]

            if metrics is None:
                cells.append("-")
                continue

            f4 = metrics["f1@4px"] * 100.0
            f8 = metrics["f1@8px"] * 100.0

            cells.append(
                f"{f4:.2f} / {f8:.2f}"
            )

        lines.append(
            f"| {lam:.2f} | "
            + " | ".join(cells)
            + " |"
        )

    return lines


# ============================================================
# Markdown
# ============================================================

md = []

md.append("# Point-DINO Stage 3 Bottom-Left Grid")
md.append("")
md.append(
    "Native 1024x768, Point DN=0.01, FocalLossCost=2."
)
md.append("")
md.append(
    "Each cell is F1@4 / F1@8 (%)."
)
md.append("")

md.extend(combined_f1_table())

md.append("")
md.append("## F1@4 (%)")
md.append("")
md.extend(
    table(
        "f1@4px",
        percent=True,
        digits=2
    )
)

md.append("")
md.append("## F1@8 (%)")
md.append("")
md.extend(
    table(
        "f1@8px",
        percent=True,
        digits=2
    )
)

md.append("")
md.append("## MLE@4 (px)")
md.append("")
md.extend(
    table(
        "mean_localization_error@4px",
        percent=False,
        digits=4
    )
)

md.append("")
md.append("## MLE@8 (px)")
md.append("")
md.extend(
    table(
        "mean_localization_error@8px",
        percent=False,
        digits=4
    )
)


# ============================================================
# Best results
# ============================================================

valid_rows = [
    row for row in rows
    if "f1@4px" in row
    and "f1@8px" in row
]

if valid_rows:

    best4 = max(
        valid_rows,
        key=lambda x: x["f1@4px"]
    )

    best8 = max(
        valid_rows,
        key=lambda x: x["f1@8px"]
    )

    best_mle4 = min(
        valid_rows,
        key=lambda x: x["mean_localization_error@4px"]
    )

    best_mle8 = min(
        valid_rows,
        key=lambda x: x["mean_localization_error@8px"]
    )

    md.append("")
    md.append("## Best Results")
    md.append("")

    md.append(
        f"- Best F1@4: "
        f"lambda={best4['lambda']:.2f}, "
        f"Cost={best4['PointL1Cost']}, "
        f"F1@4={best4['f1@4px'] * 100:.2f}%"
    )

    md.append(
        f"- Best F1@8: "
        f"lambda={best8['lambda']:.2f}, "
        f"Cost={best8['PointL1Cost']}, "
        f"F1@8={best8['f1@8px'] * 100:.2f}%"
    )

    md.append(
        f"- Best MLE@4: "
        f"lambda={best_mle4['lambda']:.2f}, "
        f"Cost={best_mle4['PointL1Cost']}, "
        f"MLE@4="
        f"{best_mle4['mean_localization_error@4px']:.4f}"
    )

    md.append(
        f"- Best MLE@8: "
        f"lambda={best_mle8['lambda']:.2f}, "
        f"Cost={best_mle8['PointL1Cost']}, "
        f"MLE@8="
        f"{best_mle8['mean_localization_error@8px']:.4f}"
    )


md_path.write_text(
    "\n".join(md) + "\n",
    encoding="utf-8"
)


# ============================================================
# Terminal output
# ============================================================

print()
print("=" * 92)
print("Stage 3 Bottom-Left: F1@4 / F1@8 (%)")
print("=" * 92)

print(
    f"{'lambda':>8}"
    + "".join(
        f"{('Cost=' + str(cost)):>20}"
        for _, cost in costs
    )
)

for _, lam in lambdas:

    output = f"{lam:>8.2f}"

    for _, cost in costs:

        metrics = matrix[lam][cost]

        if metrics is None:
            cell = "N/A"

        else:
            cell = (
                f"{metrics['f1@4px'] * 100:.2f}"
                f" / "
                f"{metrics['f1@8px'] * 100:.2f}"
            )

        output += f"{cell:>20}"

    print(output)


if valid_rows:

    print()
    print("=" * 92)

    print(
        "Best F1@4: "
        f"lambda={best4['lambda']:.2f}, "
        f"Cost={best4['PointL1Cost']}, "
        f"{best4['f1@4px'] * 100:.2f}%"
    )

    print(
        "Best F1@8: "
        f"lambda={best8['lambda']:.2f}, "
        f"Cost={best8['PointL1Cost']}, "
        f"{best8['f1@8px'] * 100:.2f}%"
    )

    print(
        "Best MLE@4: "
        f"lambda={best_mle4['lambda']:.2f}, "
        f"Cost={best_mle4['PointL1Cost']}, "
        f"{best_mle4['mean_localization_error@4px']:.4f}"
    )

    print(
        "Best MLE@8: "
        f"lambda={best_mle8['lambda']:.2f}, "
        f"Cost={best_mle8['PointL1Cost']}, "
        f"{best_mle8['mean_localization_error@8px']:.4f}"
    )


print()
print("CSV:")
print(csv_path)

print()
print("Markdown:")
print(md_path)
PY


echo
echo "============================================================"
echo "BOTTOM-LEFT GRID FINISHED"
echo "============================================================"
echo
echo "Markdown:"
echo "${RESULT_MD}"
echo
echo "CSV:"
echo "${RESULT_CSV}"