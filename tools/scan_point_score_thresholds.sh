#!/bin/bash

set -e

cd /root/autodl-tmp/dinov3_dino_mmdet/mmdetection

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH=/root/autodl-tmp/dinov3_dino_mmdet/mmdetection:$PYTHONPATH

CONFIG="configs/dino/point_dino_r50_shanghaitech_12e.py"

CHECKPOINT="/root/autodl-tmp/dinov3_dino_mmdet/work_dirs/point_dino_r50_shanghaitech_12e/best_point_f1_epoch_12.pth"

OUT_DIR="/root/autodl-tmp/dinov3_dino_mmdet/work_dirs/point_dino_threshold_scan"

mkdir -p "${OUT_DIR}"

for SCORE in 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
    echo
    echo "=================================================="
    echo "Testing score_threshold = ${SCORE}"
    echo "=================================================="

    python tools/test.py \
        "${CONFIG}" \
        "${CHECKPOINT}" \
        --cfg-options \
        test_evaluator.score_threshold=${SCORE} \
        2>&1 | tee "${OUT_DIR}/score_${SCORE}.log"

done

echo
echo "=================================================="
echo "Threshold scan finished"
echo "=================================================="