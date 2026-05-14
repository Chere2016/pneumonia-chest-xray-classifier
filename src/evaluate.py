import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import MedicalCNN
from dataset import get_data_loaders


def evaluate(model_path, data_dir, threshold=0.5, save_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEvaluating on device: {device}")
    print(f"Model  : {model_path}")
    print(f"Data   : {data_dir}")
    print(f"Threshold: {threshold}\n")

    # --- Load data (test set only) ---
    _, _, test_loader, classes, _ = get_data_loaders(data_dir=data_dir, batch_size=32)

    # --- Load model ---
    model = MedicalCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs >= threshold).astype(int)

    # --- Compute metrics ---
    auc       = roc_auc_score(all_labels, all_probs)
    f1        = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    accuracy  = accuracy_score(all_labels, all_preds) * 100
    cm        = confusion_matrix(all_labels, all_preds)

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # --- Print report ---
    print("=" * 50)
    print("         TEST SET EVALUATION REPORT")
    print("=" * 50)
    print(f"  Threshold  : {threshold}")
    print(f"  Total Imgs : {len(all_labels)}")
    print("-" * 50)
    print(f"  AUC-ROC    : {auc:.4f}")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}   (Sensitivity)")
    print(f"  Specificity: {specificity:.4f}  (True Negative Rate)")
    print(f"  Accuracy   : {accuracy:.2f}%")
    print("-" * 50)
    print(f"  Confusion Matrix:")
    print(f"              Pred NORMAL  Pred PNEUMONIA")
    print(f"  True NORMAL    {tn:5d}         {fp:5d}")
    print(f"  True PNEUMONIA {fn:5d}         {tp:5d}")
    print("-" * 50)
    print(f"\nFull Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # --- Save results to file ---
    out_path = model_path.replace('_best_model.pth', f'_test_eval_t{threshold}.txt')
    with open(out_path, 'w') as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Total Test Images: {len(all_labels)}\n\n")
        f.write(f"AUC-ROC    : {auc:.4f}\n")
        f.write(f"F1-Score   : {f1:.4f}\n")
        f.write(f"Precision  : {precision:.4f}\n")
        f.write(f"Recall     : {recall:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"Accuracy   : {accuracy:.2f}%\n\n")
        f.write(f"Confusion Matrix (TN FP / FN TP):\n")
        f.write(f"{tn} {fp}\n{fn} {tp}\n\n")
        f.write(classification_report(all_labels, all_preds, target_names=classes))

    print(f"\n✅ Results saved to: {out_path}")

    # --- Save confusion matrix as PNG ---
    cm_path = model_path.replace('_best_model.pth', f'_test_cm_t{threshold}.png')
    _save_confusion_matrix_image(cm=cm, classes=classes,
                                  title=f"Test Set Confusion Matrix (threshold={threshold})",
                                  save_path=cm_path)

    # --- Save sample prediction images ---
    if save_samples > 0:
        samples_path = model_path.replace('_best_model.pth', '_sample_predictions.png')
        _save_sample_predictions(
            model=model, device=device, data_dir=data_dir,
            classes=classes, threshold=threshold,
            n_samples=save_samples, save_path=samples_path
        )


def _save_sample_predictions(model, device, data_dir, classes, threshold, n_samples, save_path):
    """
    Pick n_samples images from the test set — a balanced mix of correct and
    incorrect predictions — and save them as an annotated grid.
    """
    import glob, random

    # ImageNet normalisation (same as training)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    infer_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Gather image paths from test set
    test_dir = os.path.join(data_dir, 'test')
    image_paths = []
    labels_list = []
    for class_idx, class_name in enumerate(classes):
        paths = glob.glob(os.path.join(test_dir, class_name, '*.jpeg')) + \
                glob.glob(os.path.join(test_dir, class_name, '*.jpg'))  + \
                glob.glob(os.path.join(test_dir, class_name, '*.png'))
        for p in paths:
            image_paths.append(p)
            labels_list.append(class_idx)

    # Shuffle and pick a subset to find interesting samples quickly
    combined = list(zip(image_paths, labels_list))
    random.seed(42)
    random.shuffle(combined)
    combined = combined[:200]  # scan first 200

    correct_samples   = []
    incorrect_samples = []

    model.eval()
    with torch.no_grad():
        for path, label in combined:
            img = Image.open(path).convert('RGB')
            tensor = infer_transform(img).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(tensor)).item()
            pred = int(prob >= threshold)

            entry = (path, label, pred, prob)
            if pred == label:
                correct_samples.append(entry)
            else:
                incorrect_samples.append(entry)

    # Build a balanced selection
    n_correct   = min(3, len(correct_samples))
    n_incorrect = min(n_samples - n_correct, len(incorrect_samples))
    selected = correct_samples[:n_correct] + incorrect_samples[:n_incorrect]
    # Pad with more correct predictions if not enough wrong ones
    if len(selected) < n_samples:
        selected += correct_samples[n_correct: n_samples - len(selected) + n_correct]
    selected = selected[:n_samples]

    # Plot
    fig, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 5))
    if len(selected) == 1:
        axes = [axes]

    fig.patch.set_facecolor('#1a1a2e')

    for ax, (path, true_label, pred_label, prob) in zip(axes, selected):
        img = Image.open(path).convert('RGB').resize((224, 224))
        ax.imshow(img, cmap='gray')
        ax.axis('off')

        true_name = classes[true_label]
        pred_name = classes[pred_label]
        correct   = (true_label == pred_label)

        confidence = prob if pred_label == 1 else 1 - prob
        color = '#00e676' if correct else '#ff1744'  # green if right, red if wrong

        title = (f"GT: {true_name}\n"
                 f"Pred: {pred_name}\n"
                 f"Conf: {confidence:.1%}")
        ax.set_title(title, fontsize=11, color=color, fontweight='bold',
                     pad=6, backgroundcolor='#1a1a2e')

    verdict = "✓ Correct" if correct else "✗ Wrong"  # noqa — last iteration
    fig.suptitle(
        "Exp5 — Sample Predictions on Held-Out Test Set",
        fontsize=13, color='white', fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print(f"✅ Sample predictions image saved to: {save_path}")


def _save_confusion_matrix_image(cm, classes, title, save_path):
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
    parser = argparse.ArgumentParser(description="Evaluate a trained MedicalCNN on the test set")
    parser.add_argument("--model",        required=True,  help="Path to the .pth model checkpoint")
    parser.add_argument("--data_dir",     required=True,  help="Path to data directory (must contain test/)")
    parser.add_argument("--threshold",    type=float, default=0.5, help="Decision threshold (default: 0.5)")
    parser.add_argument("--save_samples", type=int,   default=5,   help="Number of sample prediction images to save")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        data_dir=args.data_dir,
        threshold=args.threshold,
        save_samples=args.save_samples
    )
