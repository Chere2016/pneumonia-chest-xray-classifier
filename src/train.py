import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import yaml
import argparse
import wandb
import random
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from dataset import get_data_loaders
from model import MedicalCNN

def train_model(config_path):
    # --- 0. Set Random Seed ---
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    # --- 1. Load Configuration ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print("Loaded Configuration:", config)
    
    # --- 2. Initialize WandB ---
    wandb.init(
        project="medical-image-classifier", 
        name=config['experiment_name'],
        config=config
    )

    # --- 3. Hardware Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Training on device: {device}")

    # --- 4. Get Data ---
    print("Loading data...")
    train_loader, valid_loader, test_loader, classes, class_counts = get_data_loaders(
        data_dir=config['data_dir'], 
        batch_size=config['batch_size'],
        config=config
    )
    
    print(f"Data loaded! Classes: {classes}")
    print(f"Class distribution in train set: {class_counts}")
    
    # Calculate pos_weight for BCEWithLogitsLoss if not provided
    # Assuming class 0 is NORMAL and class 1 is PNEUMONIA
    if config.get('pos_weight') is None:
        pos_weight = torch.tensor([class_counts[0] / class_counts[1]], device=device)
    else:
        pos_weight = torch.tensor([config['pos_weight']], device=device)
        
    print(f"Using pos_weight for BCEWithLogitsLoss: {pos_weight.item():.4f}")

    # --- 5. Initialize Model, Loss Function, and Optimizer ---
    model = MedicalCNN().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    weight_decay = config.get('weight_decay', 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=weight_decay)

    scheduler_type = config.get('scheduler_type', 'plateau')
    if scheduler_type == 'cosine':
        # Warm restarts — LR cycles back up periodically
        T_0    = config.get('cosine_T0', 10)
        T_mult = config.get('cosine_Tmult', 2)
        min_lr = config.get('min_lr', 1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=min_lr
        )
        print(f"Scheduler: CosineAnnealingWarmRestarts (T_0={T_0}, T_mult={T_mult}, eta_min={min_lr})")
    elif scheduler_type == 'cosine_decay':
        # Single smooth cosine decay — no restarts, monotonically decreasing LR
        num_epochs_total = config.get('num_epochs', 60)
        min_lr = config.get('min_lr', 1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs_total, eta_min=min_lr
        )
        print(f"Scheduler: CosineAnnealingLR / single decay (T_max={num_epochs_total}, eta_min={min_lr})")
    elif scheduler_type == 'warmup_cosine':
        # Linear warmup → Cosine decay (no restarts)
        # Warmup: LR linearly grows from min_lr to learning_rate over warmup_epochs
        # Then:   LR smoothly decays from learning_rate to min_lr via cosine
        num_epochs_total = config.get('num_epochs', 100)
        warmup_epochs    = config.get('warmup_epochs', 10)
        min_lr           = config.get('min_lr', 1e-6)
        start_factor     = min_lr / config['learning_rate']  # e.g. 1e-6 / 1e-4 = 0.01
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs_total - warmup_epochs, eta_min=min_lr
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        print(f"Scheduler: Warmup({warmup_epochs} epochs) + CosineAnnealingLR (T_max={num_epochs_total - warmup_epochs}, eta_min={min_lr})")
    else:
        # ReduceLROnPlateau — validation-loss aware
        sched_patience = config.get('scheduler_patience', 3)
        sched_factor   = config.get('scheduler_factor', 0.1)
        min_lr         = config.get('min_lr', 1e-7)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=sched_factor,
            patience=sched_patience,
            min_lr=min_lr
        )
        print(f"Scheduler: ReduceLROnPlateau (factor={sched_factor}, patience={sched_patience}, min_lr={min_lr})")

    wandb.watch(model, criterion, log="all", log_freq=10)

    # --- 6. Training Loop ---
    num_epochs = config['num_epochs']
    best_val_auc = 0.0
    os.makedirs('models', exist_ok=True)
    save_path = f"models/{config['experiment_name']}_best_model.pth"

    for epoch in range(num_epochs):
        start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            images = images.to(device)
            # labels are LongTensor (0 or 1), need to be FloatTensor for BCEWithLogitsLoss
            labels = labels.to(device).float().unsqueeze(1) 

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        
        # --- VALIDATION PHASE ---
        model.eval()
        running_val_loss = 0.0
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]", leave=False):
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item() * images.size(0)
                
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        epoch_val_loss = running_val_loss / len(valid_loader.dataset)
        
        # Calculate Imbalanced Metrics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        accuracy = (all_preds == all_labels).mean() * 100
        f1 = f1_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0
            
        epoch_time = time.time() - start_time
        
        # Step scheduler
        if scheduler_type == 'cosine':
            scheduler.step(epoch + epoch_val_loss / num_epochs)  # fractional step for warm restarts
        elif scheduler_type in ('cosine_decay', 'warmup_cosine'):
            scheduler.step()  # both CosineAnnealingLR and SequentialLR step without argument
        else:
            scheduler.step(epoch_val_loss)  # ReduceLROnPlateau needs val loss
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | LR: {current_lr:.6f} | Train Loss: {epoch_loss:.4f} | Valid Loss: {epoch_val_loss:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Time: {epoch_time:.1f}s")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "valid_loss": epoch_val_loss,
            "valid_accuracy": accuracy,
            "valid_f1": f1,
            "valid_precision": precision,
            "valid_recall": recall,
            "valid_auc": auc,
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_time
        })

        # --- Save Best Model ---
        if auc > best_val_auc:
            print(f"🌟 Validation AUC improved from {best_val_auc:.4f} to {auc:.4f}! Saving best model...")
            best_val_auc = auc
            torch.save(model.state_dict(), save_path)

            # Save metrics to .txt
            metrics_path = save_path.replace('.pth', '_metrics.txt')
            with open(metrics_path, 'w') as f:
                f.write(f"Epoch: {epoch + 1}\n")
                f.write(f"Validation AUC-ROC: {auc:.4f}\n")
                f.write(f"Validation F1-Score: {f1:.4f}\n")
                f.write(f"Validation Precision: {precision:.4f}\n")
                f.write(f"Validation Recall: {recall:.4f}\n")
                f.write(f"Validation Accuracy: {accuracy:.2f}%\n")
                f.write(f"Validation Loss: {epoch_val_loss:.4f}\n")

            # Save confusion matrix image for best validation epoch
            cm_val = confusion_matrix(all_labels, all_preds)
            cm_img_path = save_path.replace('.pth', '_val_cm.png')
            _save_cm_image(cm_val, ['NORMAL', 'PNEUMONIA'],
                           title=f"Best Val CM (Epoch {epoch+1}, AUC={auc:.4f})",
                           save_path=cm_img_path)

    # --- 7. Finalize ---
    wandb.save(save_path)
    print(f"\nTraining Complete! Best model saved to: {save_path} with AUC-ROC: {best_val_auc:.4f}")
    wandb.finish()


def _save_cm_image(cm, classes, title, save_path):
    """Save a confusion matrix as a PNG image."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           ylabel='True Label', xlabel='Predicted Label',
           title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=16, fontweight='bold')
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Confusion matrix image saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Medical CNN")
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to YAML config')
    args = parser.parse_args()
    train_model(args.config)
