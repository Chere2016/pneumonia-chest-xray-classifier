#!/bin/bash

# =============================================================================
# Experiment 3: Medical CNN — Fix LR Death (Gentle Scheduler)
# Key change: scheduler_factor=0.5, patience=7, min_lr=1e-6
# These settings are now read from the YAML config (fixed in train.py)
# =============================================================================

cd /home/falcon/student1/mscs/medical_classifier

mkdir -p logs

export CUDA_VISIBLE_DEVICES=1

nohup python3 src/train.py --config configs/train_config_exp3.yaml > logs/train_exp3.log 2>&1 &

echo "=============================================="
echo " Experiment 3 started on GPU 1 (A100 80GB)"
echo " Config  : configs/train_config_exp3.yaml"
echo " Log     : logs/train_exp3.log"
echo " Monitor : tail -f logs/train_exp3.log"
echo "=============================================="
