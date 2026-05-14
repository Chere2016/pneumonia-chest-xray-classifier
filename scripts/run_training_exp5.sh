#!/bin/bash

# =============================================================================
# Experiment 5: Medical CNN — Stable Training (CosineAnnealingLR, no restarts)
# Fix: Single smooth cosine decay over all 60 epochs — zero spikes guaranteed
# =============================================================================

cd /home/falcon/student1/mscs/medical_classifier

mkdir -p logs

export CUDA_VISIBLE_DEVICES=1

nohup python3 src/train.py --config configs/train_config_exp5.yaml > logs/train_exp5.log 2>&1 &

echo "=============================================="
echo " Experiment 5 started on GPU 1 (A100 80GB)"
echo " Config  : configs/train_config_exp5.yaml"
echo " Log     : logs/train_exp5.log"
echo " Monitor : tail -f logs/train_exp5.log"
echo "=============================================="
