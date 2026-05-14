#!/bin/bash

# =============================================================================
# Experiment 2: Medical CNN - Improved Recall
# Strategy: Start at optimal LR (0.0001), train for 50 epochs,
#            use stronger augmentation (ColorJitter).
# =============================================================================

# Navigate to the project root directory
cd /home/falcon/student1/mscs/medical_classifier

# Create logs directory if it doesn't exist
mkdir -p logs

# Set GPU to 1
export CUDA_VISIBLE_DEVICES=1

# Run Experiment 2 in the background
nohup python3 src/train.py --config configs/train_config_exp2.yaml > logs/train_exp2.log 2>&1 &

echo "=============================================="
echo " Experiment 2 started on GPU 1 (A100 80GB)"
echo " Config  : configs/train_config_exp2.yaml"
echo " Log     : logs/train_exp2.log"
echo " Monitor : tail -f logs/train_exp2.log"
echo "=============================================="
