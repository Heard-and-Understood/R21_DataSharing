"""
BERT Embeddings + Sklearn Logistic Regression for Pause Classification.

Two-stage approach:
1. Extract BERT [CLS] token embeddings and save as .npz files
2. Train sklearn Logistic Regression classifier on saved embeddings

This is a pipeline where BERT is only used to create the embeddings, not for classification.
Adapted from GBL pipeline script.

Required Inputs:
    - train/val/test CSV files with columns: pause_id, text, label

Output:
    - Stage 1: Saved embeddings (.npz files) and label mapping (JSON)
    - Stage 2: Trained sklearn model, classification reports, confusion matrices, predictions

Author: kirsten.bonson@uvm.edu
Last Updated: April 2026
"""

import json
import pickle
import random
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import BertTokenizer, BertModel

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

parser = argparse.ArgumentParser(
    description='BERT Embeddings + Sklearn Logistic Regression for Pause Classification'
)

# data args
parser.add_argument(
    '--data_dir',
    type=str,
    required=True,
    help='Directory containing train/val/test CSV files'
)
parser.add_argument(
    '--label_name',
    type=str,
    required=True,
    help='Label name used in CSV filenames (e.g., "is_def_cs")'
)

# mode selection
parser.add_argument(
    '--mode',
    type=str,
    default='both',
    choices=['extract', 'train', 'both'],
    help='Mode: extract embeddings, train classifier, or both (default: both)'
)

# device arguments (for embedding extraction only)
parser.add_argument(
    '--device',
    type=str,
    default='cuda',
    choices=['cuda', 'mps', 'cpu'],
    help='Device to use for embedding extraction (cuda for Linux, mps for Mac, cpu for CPU-only)'
)

# extraction arguments
parser.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help='Batch size for embedding extraction (default: 32)'
)
parser.add_argument(
    '--embeddings_dir',
    type=str,
    default='./embeddings',
    help='Directory to save/load embeddings'
)

# training arguments
parser.add_argument(
    '--max_iter',
    type=int,
    default=1000,
    help='Maximum iterations for Logistic Regression (default: 1000)'
)
parser.add_argument(
    '--use_class_weights',
    action='store_true',
    help='Use balanced class weights for Logistic Regression'
)
parser.add_argument(
    '--neg_class_weight',
    type=float,
    default=None,
    help='Custom weight for negative class (class 0). If provided with --pos_class_weight, overrides --use_class_weights'
)
parser.add_argument(
    '--pos_class_weight',
    type=float,
    default=None,
    help='Custom weight for positive class (class 1). If provided with --neg_class_weight, overrides --use_class_weights'
)

# output arguments
parser.add_argument(
    '--output_dir',
    type=str,
    default='./outputs/bert_sklearn',
    help='Directory to save model and results'
)

# run configurations
RANDOM_SEED = 42
BERT_MODEL_NAME = 'bert-base-uncased'
MAX_TEXT_LENGTH = 512

def seed_everything():
    """
    Set the seed for everything consistently across the board.
    """
    print(f'   ...PyTorch version: {torch.__version__}')
    print(f'   ...PyTorch built with CUDA: {torch.version.cuda if torch.version.cuda else "NO"}')
    print(f'   ...CUDA available (before init): {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   ...CUDA device count: {torch.cuda.device_count()}')
        print(f'   ...CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A"}')

    torch.manual_seed(RANDOM_SEED)
    try:
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        print(f'   ...CUDA seeds set successfully')
    except Exception as error:
        print(f'   ...WARNING: Failed to set CUDA seeds: {error}')

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)

    print(f'   ...CUDA available (after init): {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print('   ...using relaxed deterministic mode (CUDA operations may have minor non-determinism)')


def set_device(specified_device):
    """
    Determine if we can use requested device, default to CPU if not available.
    """
    available = True
    if not specified_device or specified_device == 'cpu':
        return torch.device("cpu")
    
    else:
        if specified_device == 'cuda':
            if not torch.cuda.is_available():
                available = False

        elif specified_device == 'mps':
            if not torch.backends.mps.is_available():
                available = False

        else:
            raise ValueError(f'Device {specified_device} not recognized, please check and try again!')
        
        if not available:
            print(f'\nWARNING: requested device {specified_device} not available, defaulting to CPU instead')
            return torch.device("cpu")
        
        else:
            return torch.device(specified_device)

def extract_bert_embeddings(
    csv_path: str,
    tokenizer: BertTokenizer,
    model: BertModel,
    device: torch.device,
    label_to_idx: Dict[str, int],
    label_col: str = 'label',
    batch_size: int = 32,
    max_length: int = MAX_TEXT_LENGTH
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract BERT [CLS] token embeddings for all samples in a CSV.
    
    Parameters:
        csv_path: Path to CSV file
        tokenizer: BERT tokenizer
        model: BERT model (should be in eval mode)
        device: Device to run on
        label_to_idx: Mapping from label names to indices
        label_col: Name of label column
        batch_size: Batch size for processing
        max_length: Maximum sequence length
    
    Returns:
        embeddings: numpy array of shape (n_samples, 768)
        labels: numpy array of shape (n_samples,)
        pause_ids: list of pause IDs
    """
    print(f"\nExtracting embeddings from: {csv_path}")
    
    # load CSV
    df = pd.read_csv(csv_path)
    
    # verify columns
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV. Available: {list(df.columns)}")
    if 'text' not in df.columns:
        raise ValueError(f"'text' column not found in CSV. Available: {list(df.columns)}")
    if 'pause_id' not in df.columns:
        raise ValueError(f"'pause_id' column not found in CSV. Available: {list(df.columns)}")
    
    print(f"  Found {len(df)} samples")
    print(f"  Label distribution:\n{df[label_col].value_counts()}")
    
    # prepare data
    texts = df['text'].tolist()
    labels = [label_to_idx[label] for label in df[label_col]]
    pause_ids = df['pause_id'].tolist()
    
    # extract embeddings in batches
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # tokenize batch
            encoding = tokenizer(
                batch_texts,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # move to device
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # get BERT outputs
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # extract [CLS] token embeddings (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
    
    # concatenate all batches
    embeddings = np.vstack(all_embeddings)
    labels = np.array(labels)
    
    print(f"  Extracted embeddings shape: {embeddings.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    return embeddings, labels, pause_ids

def save_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    pause_ids: List[str],
    output_path: Path
):
    """
    Save embeddings, labels, and pause_ids to a .npz file.
    """
    np.savez(
        output_path,
        embeddings=embeddings,
        labels=labels,
        pause_ids=pause_ids
    )
    print(f"  Saved embeddings to: {output_path}")

def load_embeddings(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load embeddings, labels, and pause_ids from a .npz file.
    """
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data['embeddings']
    labels = data['labels']
    pause_ids = data['pause_ids'].tolist()
    
    print(f"  Loaded embeddings from: {npz_path}")
    print(f"    Embeddings shape: {embeddings.shape}")
    print(f"    Labels shape: {labels.shape}")
    
    return embeddings, labels, pause_ids

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str,
    label_name: str = ''
):
    """
    Plot and save confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )

    title = f'Confusion Matrix ({label_name})' if label_name else 'Confusion Matrix'
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

def extract_all_embeddings(
    data_dir: Path,
    label_name: str,
    embeddings_dir: Path,
    device: torch.device,
    batch_size: int
):
    """
    Stage 1: Extract embeddings for train/val/test sets and save them.
    """
    print("\nExtracting BERT embeddings")
    
    # create embeddings directory
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # construct CSV paths
    train_csv = data_dir / f"{label_name}_training.csv"
    val_csv = data_dir / f"{label_name}_validation.csv"
    test_csv = data_dir / f"{label_name}_testing.csv"
    
    # verify files exist
    for csv_path in [train_csv, val_csv, test_csv]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Required CSV file not found: {csv_path}")
    
    print(f"\nFound all required CSV files:")
    print(f"  Training:   {train_csv}")
    print(f"  Validation: {val_csv}")
    print(f"  Test:       {test_csv}")
    
    # load tokenizer and model
    print(f"\nLoading BERT model: {BERT_MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertModel.from_pretrained(BERT_MODEL_NAME)
    model = model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    
    # create label mapping from training set
    print("\nCreating label mapping from training set...")
    train_df = pd.read_csv(train_csv)
    unique_labels = sorted(train_df[label_name].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    print(f"Label mapping: {label_to_idx}")
    
    # save label mapping
    label_mapping_path = embeddings_dir / f"{label_name}_label_mapping.json"
    with open(label_mapping_path, 'w') as f:
        json.dump({
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label
        }, f, indent=2)

    print(f"Saved label mapping to: {label_mapping_path}")
    
    # extract embeddings for each split
    for split_name, csv_path in [
        ('train', train_csv),
        ('val', val_csv),
        ('test', test_csv)
    ]:
        embeddings, labels, pause_ids = extract_bert_embeddings(
            csv_path=str(csv_path),
            tokenizer=tokenizer,
            model=model,
            device=device,
            label_to_idx=label_to_idx,
            label_col=label_name,
            batch_size=batch_size,
            max_length=MAX_TEXT_LENGTH
        )
        
        # save embeddings
        output_path = embeddings_dir / f"{label_name}_{split_name}_embeddings.npz"
        save_embeddings(embeddings, labels, pause_ids, output_path)
    
    print("\n   ...embeddings extracted and saved")

def train_classifier(
    embeddings_dir: Path,
    label_name: str,
    output_dir: Path,
    max_iter: int,
    use_class_weights: bool,
    data_dir: Path,
    neg_class_weight: float = None,
    pos_class_weight: float = None
):
    """
    Stage 2: Load embeddings and train sklearn Logistic Regression classifier.
    """
    # create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # load label mapping
    label_mapping_path = embeddings_dir / f"{label_name}_label_mapping.json"
    if not label_mapping_path.exists():
        raise FileNotFoundError(f"Label mapping not found: {label_mapping_path}")
    
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    
    label_to_idx = label_mapping['label_to_idx']
    idx_to_label = {int(k): v for k, v in label_mapping['idx_to_label'].items()}
    num_classes = len(label_to_idx)
    
    print(f"\nLabel mapping loaded: {num_classes} classes")
    print(f"  {label_to_idx}")
    
    # load embeddings
    print("\nLoading embeddings...")
    train_embeddings_path = embeddings_dir / f"{label_name}_train_embeddings.npz"
    val_embeddings_path = embeddings_dir / f"{label_name}_val_embeddings.npz"
    test_embeddings_path = embeddings_dir / f"{label_name}_test_embeddings.npz"
    
    X_train, y_train, train_pause_ids = load_embeddings(train_embeddings_path)
    X_val, y_val, val_pause_ids = load_embeddings(val_embeddings_path)
    X_test, y_test, test_pause_ids = load_embeddings(test_embeddings_path)
    
    # print class distribution
    print("\nClass distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label_idx, count in zip(unique, counts):
        label_name_str = idx_to_label[label_idx]
        pct = 100 * count / len(y_train)
        print(f"  Class {label_idx} ({label_name_str}): {count} samples ({pct:.1f}%)")
    
    # determine class weights
    class_weight_param = None
    if neg_class_weight is not None and pos_class_weight is not None:
        # custom weights provided
        class_weight_param = {0: neg_class_weight, 1: pos_class_weight}
        class_weight_str = f"custom {{0: {neg_class_weight}, 1: {pos_class_weight}}}"

    elif use_class_weights:
        # use balanced weights
        class_weight_param = 'balanced'
        class_weight_str = 'balanced'

    else:
        # no weights
        class_weight_param = None
        class_weight_str = 'None'
    
    # train logistic regression
    print("\nTraining Logistic Regression classifier...")
    print(f"  Max iterations: {max_iter}")
    print(f"  Class weights: {class_weight_str}")
    
    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight_param,
        random_state=RANDOM_SEED,
        verbose=1
    )
    
    clf.fit(X_train, y_train)
    
    print("\nTraining complete!")
    
    # evaluate on validation set
    print("\nRunning validation set evaluation")
    
    y_val_pred = clf.predict(X_val)
    val_acc = 100 * (y_val_pred == y_val).sum() / len(y_val)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    print(f"\nValidation Accuracy: {val_acc:.2f}%")
    print(f"Validation F1 (weighted): {val_f1:.4f}")
    
    # evaluate on test set
    print("\nRunning test set evaluation")
    
    y_test_pred = clf.predict(X_test)
    test_acc = 100 * (y_test_pred == y_test).sum() / len(y_test)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Test F1 (weighted): {test_f1:.4f}")
    
    # save model
    model_path = output_dir / f'best_model_{label_name}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

    print(f"\nModel saved to: {model_path}")
    
    # classification reports
    class_names = [str(idx_to_label[i]) for i in range(num_classes)]
    
    print("\nCreating validation classification report")
    val_report = classification_report(
        y_val,
        y_val_pred,
        target_names=class_names,
        digits=4
    )
    print(val_report)
    
    print("\nCreating test classification report")
    test_report = classification_report(
        y_test,
        y_test_pred,
        target_names=class_names,
        digits=4
    )
    print(test_report)
    
    # save classification reports
    with open(output_dir / f'classification_report_val_{label_name}.txt', 'w') as f:
        f.write(f"VALIDATION SET CLASSIFICATION REPORT ({label_name})\n")
        f.write("="*80 + "\n\n")
        f.write(val_report)
    
    with open(output_dir / f'classification_report_test_{label_name}.txt', 'w') as f:
        f.write(f"TEST SET CLASSIFICATION REPORT ({label_name})\n")
        f.write("="*80 + "\n\n")
        f.write(test_report)
    
    # plot confusion matrices
    plot_confusion_matrix(
        y_val,
        y_val_pred,
        class_names,
        output_dir / f'confusion_matrix_val_{label_name}.png',
        label_name=label_name
    )
    
    plot_confusion_matrix(
        y_test,
        y_test_pred,
        class_names,
        output_dir / f'confusion_matrix_test_{label_name}.png',
        label_name=label_name
    )
    
    # load original CSVs
    val_csv = data_dir / f"{label_name}_validation.csv"
    test_csv = data_dir / f"{label_name}_testing.csv"
    
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    # map predicted indices back to label names
    val_pred_labels = [idx_to_label[pred] for pred in y_val_pred]
    test_pred_labels = [idx_to_label[pred] for pred in y_test_pred]
    
    # add prediction column
    val_df['sklearn_predicted_label'] = val_pred_labels
    test_df['sklearn_predicted_label'] = test_pred_labels
    
    # save prediction CSVs
    val_pred_output = output_dir / f'validation_{label_name}_with_predictions.csv'
    test_pred_output = output_dir / f'testing_{label_name}_with_predictions.csv'
    
    val_df.to_csv(val_pred_output, index=False)
    test_df.to_csv(test_pred_output, index=False)
    
    print(f"\nSaved prediction CSVs:")
    print(f"  Validation: {val_pred_output}")
    print(f"  Testing:    {test_pred_output}")
    
    print("\nProcess complete!")
    
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"  - Model: best_model_{label_name}.pkl")
    print(f"  - Validation classification report: classification_report_val_{label_name}.txt")
    print(f"  - Test classification report: classification_report_test_{label_name}.txt")
    print(f"  - Validation confusion matrix: confusion_matrix_val_{label_name}.png")
    print(f"  - Test confusion matrix: confusion_matrix_test_{label_name}.png")
    print(f"  - Validation predictions CSV: validation_{label_name}_with_predictions.csv")
    print(f"  - Testing predictions CSV: testing_{label_name}_with_predictions.csv")


if __name__ == '__main__':
    args = parser.parse_args()
    
    # setup paths
    data_dir = Path(args.data_dir)
    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)
    label_name = args.label_name
    
    # seed everything
    print("\nInitializing environment")
    seed_everything()
    
    # run based on mode
    if args.mode in ['extract', 'both']:
        device = set_device(args.device)
        extract_all_embeddings(
            data_dir=data_dir,
            label_name=label_name,
            embeddings_dir=embeddings_dir,
            device=device,
            batch_size=args.batch_size
        )
    
    if args.mode in ['train', 'both']:
        train_classifier(
            embeddings_dir=embeddings_dir,
            label_name=label_name,
            output_dir=output_dir,
            max_iter=args.max_iter,
            use_class_weights=args.use_class_weights,
            data_dir=data_dir,
            neg_class_weight=args.neg_class_weight,
            pos_class_weight=args.pos_class_weight
        )
