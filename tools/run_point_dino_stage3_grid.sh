#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/dinov3_dino_mmdet"
MMDET="${ROOT}/mmdetection"
CONFIG_DIR="${MMDET}/configs/dino"
WORK_ROOT="${ROOT}/work_dirs/stage3_grid"

# 这里使用我们已经确认过的 native + DN=0.01 + Euclidean-capable 配置
BASE_CONFIG="${CONFIG_DIR}/point_dino_r50_shanghaitech_stage2_dn001_euc01_native_12e.py"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="${MMDET}:${PYTHONPATH:-}"

cd "${MMDET}"

mkdir -p "${WORK_ROOT}"

# ============================================================
# Stage 3 grid
#
# Euclidean lambda:
#   0.00 / 0.05 / 0.10 / 0.20
#
# Hungarian PointL1Cost:
#   5 / 10 / 15 / 20
#
# FocalLossCost always = 2.0
# Point DN noise always = 0.01
# Input always = native 1024x768
# ============================================================

LAMBDAS=(
    "0.0"
    "0.05"
    "0.1"
    "0.2"
)

LAMBDA_TAGS=(
    "000"
    "005"
    "010"
    "020"
)

COSTS=(
    "5"
    "10"
    "15"
    "20"
)

echo "============================================================"
echo "Point-DINO Stage 3: 4x4 Grid Ablation"
echo "============================================================"
echo "Base config:"
echo "${BASE_CONFIG}"
echo
echo "Work root:"
echo "${WORK_ROOT}"
echo
echo "Total experiments: 16"
echo "============================================================"
echo


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
        echo "lambda          = ${LAMBDA}"
        echo "PointL1Cost     = ${COST}"
        echo "FocalLossCost   = 2.0"
        echo "============================================================"

        # ----------------------------------------------------
        # 自动生成该实验的 config
        # ----------------------------------------------------

        cat > "${CONFIG}" <<EOF
_base_ = './point_dino_r50_shanghaitech_stage2_dn001_euc01_native_12e.py'

# ============================================================
# Stage 3 grid ablation
#
# Euclidean lambda = ${LAMBDA}
# PointL1Cost      = ${COST}
# FocalLossCost    = 2.0
# Point DN         = 0.01
# Input            = native 1024x768
# ============================================================

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

        # ----------------------------------------------------
        # 如果该组合已经完整训练并保存 best checkpoint，则跳过
        # 这样脚本中断后可以直接重新运行继续
        # ----------------------------------------------------

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
            echo
            continue
        fi

        mkdir -p "${WORK_DIR}"

        # ----------------------------------------------------
        # 保存最终 merged config，便于以后检查实验
        # ----------------------------------------------------

        python - <<PY
from mmengine.config import Config

cfg = Config.fromfile('${CONFIG}')

print()
print('===== CHECK CONFIG =====')
print('Experiment       : ${EXP_NAME}')
print('Euclidean lambda :', cfg.model.bbox_head.point_euclidean_weight)
print('Matcher          :', cfg.model.train_cfg.assigner.match_costs)
print('DN config        :', cfg.model.dn_cfg)

print()
print('Train pipeline:')
for x in cfg.train_dataloader.dataset.pipeline:
    print(x)

print()
print('Test pipeline:')
for x in cfg.test_dataloader.dataset.pipeline:
    print(x)

cfg.dump('${WORK_DIR}/merged_config.py')
PY

        # ----------------------------------------------------
        # 正式训练
        # ----------------------------------------------------

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
echo "ALL STAGE 3 GRID EXPERIMENTS FINISHED"
echo "============================================================"
echo
echo "Results:"
echo "${WORK_ROOT}"
echo