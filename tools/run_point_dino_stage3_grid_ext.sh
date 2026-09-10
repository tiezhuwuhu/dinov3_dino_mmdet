#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/dinov3_dino_mmdet"
MMDET="${ROOT}/mmdetection"
CONFIG_DIR="${MMDET}/configs/dino"

# New extended Stage 3 grid
WORK_ROOT="${ROOT}/work_dirs/stage3_grid_ext"

# Native 1024x768 + DN=0.01 + Euclidean-capable config
BASE_CONFIG="${CONFIG_DIR}/point_dino_r50_shanghaitech_stage2_dn001_euc01_native_12e.py"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="${MMDET}:${PYTHONPATH:-}"

cd "${MMDET}"

mkdir -p "${WORK_ROOT}"


# ============================================================
# Extended Stage 3 grid
#
# lambda:
#   0.30 / 0.40 / 0.50 / 0.60
#
# PointL1Cost:
#   25 / 30 / 35 / 40
#
# Fixed:
#   Native input      = 1024x768
#   Point DN noise    = 0.01
#   FocalLossCost     = 2.0
# ============================================================

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
    "25"
    "30"
    "35"
    "40"
)


echo "============================================================"
echo "Point-DINO Stage 3 Extended Grid"
echo "============================================================"
echo "Lambda:"
echo "  0.30 0.40 0.50 0.60"
echo
echo "PointL1Cost:"
echo "  25 30 35 40"
echo
echo "Total experiments: 16"
echo
echo "Work root:"
echo "${WORK_ROOT}"
echo "============================================================"


for i in "${!LAMBDAS[@]}"; do

    LAMBDA="${LAMBDAS[$i]}"
    LTAG="${LAMBDA_TAGS[$i]}"

    for COST in "${COSTS[@]}"; do

        CTAG=$(printf "%02d" "${COST}")

        EXP_NAME="point_dino_stage3_lam${LTAG}_cost${CTAG}"
        CONFIG="${CONFIG_DIR}/${EXP_NAME}.py"
        WORK_DIR="${WORK_ROOT}/${EXP_NAME}"

        echo
        echo "============================================================"
        echo "Experiment: ${EXP_NAME}"
        echo "Euclidean lambda = ${LAMBDA}"
        echo "PointL1Cost      = ${COST}"
        echo "FocalLossCost    = 2.0"
        echo "Point DN         = 0.01"
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
        # Skip completed experiment
        # ====================================================

        EXISTING_BEST=$(find "${WORK_DIR}" \
            -maxdepth 1 \
            -type f \
            -name 'best_point_f1@8px_epoch_*.pth' \
            2>/dev/null \
            | sort -V \
            | tail -n 1 || true)

        if [[ -n "${EXISTING_BEST}" ]]; then
            echo
            echo "[SKIP] Best checkpoint already exists:"
            echo "${EXISTING_BEST}"
            continue
        fi


        mkdir -p "${WORK_DIR}"


        # ====================================================
        # Verify merged config
        # ====================================================

        python - <<PY
from mmengine.config import Config

cfg = Config.fromfile('${CONFIG}')

print()
print('===== CONFIG CHECK =====')
print('Experiment       : ${EXP_NAME}')
print('Euclidean lambda :', cfg.model.bbox_head.point_euclidean_weight)
print('Matcher          :', cfg.model.train_cfg.assigner.match_costs)
print('DN config        :', cfg.model.dn_cfg)

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


        # ====================================================
        # Train
        # ====================================================

        echo
        echo "Starting training..."
        echo

        python tools/train.py \
            "${CONFIG}" \
            --work-dir "${WORK_DIR}" \
            2>&1 | tee "${WORK_DIR}/train.log"

        echo
        echo "Finished: ${EXP_NAME}"
        echo

    done
done


echo
echo "============================================================"
echo "ALL EXTENDED STAGE 3 EXPERIMENTS FINISHED"
echo "============================================================"
echo
echo "Results:"
echo "${WORK_ROOT}"