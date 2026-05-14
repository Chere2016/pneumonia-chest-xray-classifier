#!/bin/bash

# =============================================================================
# Experiment 6: Medical CNN — Linear Warmup + CosineAnnealingLR
# Fix: 10-epoch linear warmup prevents chaotic early-epoch gradients,
#      followed by smooth cosine decay for the remaining 90 epochs.
# =============================================================================

cd /home/falcon/student1/mscs/medical_classifier

mkdir -p logs

export CUDA_VISIBLE_DEVICES=1

nohup python3 src/train.py --config configs/train_config_exp6.yaml > logs/train_exp6.log 2>&1 &

echo "=============================================="
echo " Experiment 6 started on GPU 1 (A100 80GB)"
echo " Config  : configs/train_config_exp6.yaml"
echo " Log     : logs/train_exp6.log"
echo " Monitor : tail -f logs/train_exp6.log"
echo "=============================================="
