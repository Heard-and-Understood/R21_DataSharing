"""
Optuna parameter search for mmbert_for_pause_classification.py

Searches over:
- Learning rate (1e-6 to 5e-5)
- Dropout (0.3 to 0.6)
- Positive class weight (1.0 to 5.0)
- Classification threshold (0.1 to 0.4)
- ResNet freezing (True/False)

Optimizes validation F1 for positive class while tracking precision and recall.

Usage:
    python parameter_search.py \
        --data_dir /path/to/data \
        --label_name is_definite_emotional_cs \
        --device mps \
        --n_trials 15

Author: kirsten.bonson@uvm.edu
Created: April 2026
"""


import sys
import torch
import optuna
import argparse
import torch.nn as nn

from pathlib import Path
from typing import Tuple
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from transformers import BertTokenizer, get_scheduler

from mmbert_for_pause_classification import (
    PauseDataset,
    MultiModalBERT,
    seed_everything,
    set_device,
    train_epoch,
    evaluate
)

RANDOM_SEED = 42
BATCH_SIZE = 8
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1
MAX_TEXT_LENGTH = 512
IMAGE_SIZE = 224
BERT_MODEL_NAME = 'bert-base-uncased'

TRAIN_DATASET = None
VAL_DATASET = None
DEVICE = None

parser = argparse.ArgumentParser(
    description='Optuna parameter search for Multi-Modal BERT'
)
parser.add_argument(
    '--data_dir',
    type=str,
    required=True,
    help='Directory containing train/val CSV files'
)
parser.add_argument(
    '--label_name',
    type=str,
    required=True,
    help='Label name used in CSV filenames'
)
parser.add_argument(
    '--device',
    type=str,
    default='mps',
    choices=['cuda', 'mps', 'cpu'],
    help='Device to use (default: mps)'
)
parser.add_argument(
    '--n_trials',
    type=int,
    default=15,
    help='Number of Optuna trials to run (default: 15)'
)


def load_datasets(train_csv: Path, val_csv: Path, label_col: str) -> Tuple:
    """
    Load datasets once before starting Optuna search.
    
    Returns:
        train_dataset, val_dataset
    """
    print("\nLoading datasets...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    train_dataset = PauseDataset(
        csv_path=str(train_csv),
        tokenizer=tokenizer,
        max_length=MAX_TEXT_LENGTH,
        label_col=label_col
    )
    
    val_dataset = PauseDataset(
        csv_path=str(val_csv),
        tokenizer=tokenizer,
        max_length=MAX_TEXT_LENGTH,
        label_to_idx=train_dataset.label_to_idx,
        label_col=label_col
    )
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    
    return train_dataset, val_dataset

def train_model_with_params(
    lr: float,
    dropout: float,
    pos_class_weight: float,
    threshold: float,
    freeze_resnet: bool,
    num_epochs: int = 5
) -> Tuple[float, float, float]:
    """
    Train model with given hyperparameters.
    
    Returns:
        best_val_f1, best_precision, best_recall for positive class
    """
    global TRAIN_DATASET, VAL_DATASET, DEVICE
    
    # seed for reproducibility
    seed_everything()
    
    # create dataloaders
    train_loader = DataLoader(
        TRAIN_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        VAL_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # initialize model with parameterized dropout
    model = MultiModalBERT(
        num_classes=2,
        freeze_bert=False,
        freeze_resnet=freeze_resnet,
        dropout=dropout
    )
    model = model.to(DEVICE)
    
    # set up loss with parameterized class weights
    class_weights = torch.tensor([1.0, pos_class_weight], dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # optimizer with parameterized learning rate
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY
    )
    
    # scheduler
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    
    scheduler = get_scheduler(
        'cosine',
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # training loop
    best_recall = 0.0
    best_precision = 0.0
    best_val_f1_pos = 0.0
    for epoch in range(1, num_epochs + 1):
        # run the training
        train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=DEVICE,
            epoch=epoch,
            pos_class_threshold=threshold
        )
        
        # validate
        _, _, val_preds, val_labels = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=DEVICE,
            split_name='Val',
            pos_class_threshold=threshold
        )
        
        # calculate metrics for positive class using classification_report
        report_dict = classification_report(
            val_labels,
            val_preds,
            labels=[0, 1],
            output_dict=True,
            zero_division=0
        )
        
        val_f1_pos = report_dict['1']['f1-score']
        val_precision_pos = report_dict['1']['precision']
        val_recall_pos = report_dict['1']['recall']
        
        # track best across epochs
        if val_f1_pos > best_val_f1_pos:
            best_val_f1_pos = val_f1_pos
            best_precision = val_precision_pos
            best_recall = val_recall_pos
    
    # cleanup to prevent memory leaks
    del model, optimizer, scheduler, criterion
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()

    elif DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    
    return best_val_f1_pos, best_precision, best_recall

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function.
    
    Returns validation F1 for positive class (to be maximized).
    """
    try:
        # provide ranges of hyperparameters
        lr = trial.suggest_float('lr', 1e-6, 5e-5, log=True)
        dropout = trial.suggest_float('dropout', 0.3, 0.6)
        pos_class_weight = trial.suggest_float('pos_class_weight', 1.0, 5.0)
        threshold = trial.suggest_float('threshold', 0.1, 0.4)
        freeze_resnet = trial.suggest_categorical('freeze_resnet', [True, False])
        
        print(f"\n{'='*80}")
        print(f"Trial {trial.number}: lr={lr:.2e}, dropout={dropout:.3f}, "
              f"pos_weight={pos_class_weight:.2f}, threshold={threshold:.3f}, "
              f"freeze_resnet={freeze_resnet}")
        print('='*80)
        
        # train and get metrics
        val_f1, val_precision, val_recall = train_model_with_params(
            lr=lr,
            dropout=dropout,
            pos_class_weight=pos_class_weight,
            threshold=threshold,
            freeze_resnet=freeze_resnet,
            num_epochs=5
        )
        
        # log additional metrics for analysis
        trial.set_user_attr('precision', val_precision)
        trial.set_user_attr('recall', val_recall)
        
        print(f"\nTrial {trial.number} Results:")
        print(f"  Val F1 (pos):        {val_f1:.4f}")
        print(f"  Val Precision (pos): {val_precision:.4f}")
        print(f"  Val Recall (pos):    {val_recall:.4f}")
        
        return val_f1
        
    except Exception as error:
        print(f"\nTrial {trial.number} FAILED: {error}")
        return 0.0

def run_mmbert_parameter_search(device: str, data_dir: Path, label_name: str, n_trials: int):
    """
    Run the Multi-Modal BERT parameter search.
    """
    global DEVICE, TRAIN_DATASET, VAL_DATASET
    
    # setup device
    DEVICE = set_device(device)
    print(f"Using device: {DEVICE}")
    
    # construct CSV paths
    data_dir = Path(data_dir)
    train_csv = data_dir / f'{label_name}_training.csv'
    val_csv = data_dir / f'{label_name}_validation.csv'
    
    # verify files exist
    if not train_csv.exists():
        print(f"ERROR: Training CSV not found: {train_csv}")
        sys.exit(1)

    if not val_csv.exists():
        print(f"ERROR: Validation CSV not found: {val_csv}")
        sys.exit(1)
    
    # load datasets once (before Optuna loop)
    TRAIN_DATASET, VAL_DATASET = load_datasets(
        train_csv=train_csv,
        val_csv=val_csv,
        label_col=label_name
    )
    
    # create Optuna study with SQLite persistence (local file)
    # - calculate n_startup_trials based on parameter space dimensionality
    # - we have 5 parameters: lr, dropout, pos_class_weight, threshold, freeze_resnet
    # - use 3x the number of parameters as a minimum for good coverage, or 70% of total trials
    num_parameters = 5
    n_startup_trials = min(
        max(3 * num_parameters, int(0.7 * n_trials)),
        n_trials
    )
    
    print("\nStarting Optuna parameter search")
    print(f"  Trials: {n_trials}")
    print(f"  Epochs per trial: 5")
    print(f"  Parameters to optimize: {num_parameters}")
    print(f"  Random startup trials: {n_startup_trials} (ensures exploration of parameter space)")
    print(f"  Bayesian optimization trials: {n_trials - n_startup_trials}")
    print(f"  Estimated time: ~{n_trials * 10}-{n_trials * 15} minutes")
    print(f"  Storage: optuna_mmbert_search.db (local SQLite file)")
    print()
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_SEED,
            n_startup_trials=n_startup_trials
        ),
        storage='sqlite:///optuna_mmbert_search.db',
        study_name=f'mmbert_{label_name}',
        load_if_exists=True
    )
    
    # run optimization
    study.optimize(objective, n_trials=n_trials)
    
    # report results
    print("\n   ...search complete")
    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best val F1 (positive class): {study.best_value:.4f}")
    print(f"  Precision: {study.best_trial.user_attrs['precision']:.4f}")
    print(f"  Recall:    {study.best_trial.user_attrs['recall']:.4f}")
    
    print("\nBest hyperparameters:")
    for param, value in study.best_params.items():
        if param == 'lr':
            print(f"  {param}: {value:.2e}")

        elif param in ['dropout', 'threshold', 'pos_class_weight']:
            print(f"  {param}: {value:.3f}")

        else:
            print(f"  {param}: {value}")
    
    
    # show top 5 trials
    print("\nTop 5 trials:")
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
    for i, trial in enumerate(sorted_trials[:5], 1):
        if trial.value is not None:
            precision = trial.user_attrs.get('precision', 0)
            recall = trial.user_attrs.get('recall', 0)
            print(f"\n{i}. Trial #{trial.number} - F1: {trial.value:.4f}, "
                  f"Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(f"   lr={trial.params['lr']:.2e}, dropout={trial.params['dropout']:.3f}, "
                  f"weight={trial.params['pos_class_weight']:.2f}, "
                  f"threshold={trial.params['threshold']:.3f}, "
                  f"freeze_resnet={trial.params['freeze_resnet']}")
    
    print("\nNOTE: to use these parameters...")
    print("\n1. Edit mmbert_for_pause_classification.py:")
    print(f"   - Line 159: LEARNING_RATE = {study.best_params['lr']:.2e}")
    print(f"   - Line 1124: class_weights = torch.tensor([1.0, {study.best_params['pos_class_weight']:.2f}], dtype=torch.float).to(device)")
    
    print("\n2. Run with these arguments:")
    print(f"   python mmbert_for_pause_classification.py \\")
    print(f"     --data_dir {data_dir} \\")
    print(f"     --label_name {label_name} \\")
    print(f"     --device {device} \\")
    print(f"     --use_class_weights \\")
    print(f"     --dropout {study.best_params['dropout']:.3f} \\")
    print(f"     --pos_class_threshold {study.best_params['threshold']:.3f} \\")
    if not study.best_params['freeze_resnet']:
        print(f"     (do NOT use --freeze_resnet)")
    else:
        print(f"     --freeze_resnet")
    print(f"     --save_model")
    
    print(f"\n3. SQLite database saved to: optuna_mmbert_search.db")
    print(f"   (can be deleted after reviewing results)")
    

if __name__ == '__main__':    
    args = parser.parse_args()

    run_mmbert_parameter_search(
        device=args.device,
        data_dir=args.data_dir,
        label_name=args.label_name,
        n_trials=args.n_trials
    )

