#!/bin/bash

# =============================================================================
# Experiment 4: Medical CNN — Stable Training (CosineAnnealingWarmRestarts)
# Key change: smooth cosine LR schedule eliminates validation loss spikes
# =============================================================================

cd /home/falcon/student1/mscs/medical_classifier

mkdir -p logs

export CUDA_VISIBLE_DEVICES=1

nohup python3 src/train.py --config configs/train_config_exp4.yaml > logs/train_exp4.log 2>&1 &

echo "=============================================="
echo " Experiment 4 started on GPU 1 (A100 80GB)"
echo " Config  : configs/train_config_exp4.yaml"
echo " Log     : logs/train_exp4.log"
echo " Monitor : tail -f logs/train_exp4.log"
echo "=============================================="
