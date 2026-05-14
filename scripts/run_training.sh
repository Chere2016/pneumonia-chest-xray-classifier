#!/bin/bash

# Navigate to the project root directory
cd /home/falcon/student1/mscs/medical_classifier

# Create logs directory if it doesn't exist
mkdir -p logs

# Set GPU to 1 as requested by the user
export CUDA_VISIBLE_DEVICES=1

# Run the training script in the background
nohup python3 src/train.py --config configs/train_config.yaml > logs/train.log 2>&1 &

echo "Training started in the background on GPU 1. Logs are being written to logs/train.log."
echo "You can monitor the output using: tail -f logs/train.log"
