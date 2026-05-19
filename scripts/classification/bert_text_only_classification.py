"""
Text-Only BERT for Pause Classification. Classifies pause types using only pre-pause text 
transcriptions (no image/audio features).

Required Inputs:
    - train/val/test CSV files with columns: pause_id, text, label
    - Note: spectrogram_path column is optional and will be ignored if present

Output:
    - best model checkpoint
    - validation and test set classification reports
    - validation and test set confusion matrices
    - training curves

NOTE: This script requires pre-split train/val/test files.

NOTE: regarding the "freezing" optional run parameter...

For pause classification with ~6,000 samples:

1. Start with BERT frozen for a quick baseline:
   --freeze_bert

2. Then try fine-tuning BERT:
   (no freeze flag)

NOTE: adapted from mmbert_for_pause_classification.py

Author: kirsten.bonson@uvm.edu
Last Updated: April 2026
"""

import sys
import random
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader, Subset

from transformers import BertTokenizer, BertModel

parser = argparse.ArgumentParser(
    description='Text-Only BERT for Pause Classification'
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
    help='Label name used in CSV filenames (e.g., "is_def_cs" for files like is_def_cs_training.csv)'
)

# device arguments
parser.add_argument(
    '--device',
    type=str,
    default='cuda',
    choices=['cuda', 'mps', 'cpu'],
    help='Device to use for training (cuda for Linux, mps for Mac, cpu for CPU-only)'
)

# training arguments
parser.add_argument(
    '--freeze_bert',
    action='store_true',
    help='Freeze BERT parameters (feature extraction mode)'
)
parser.add_argument(
    '--num_workers',
    type=int,
    default=0,
    help='Number of dataloader workers (default: 0 for Mac/MPS, use 2-4 for CUDA/Linux)'
)
parser.add_argument(
    '--use_class_weights',
    action='store_true',
    help='Use class weights to balance imbalanced classes (recommended for imbalanced datasets)'
)
parser.add_argument(
    '--neg_class_weight',
    type=float,
    default=None,
    help='Custom weight for negative class (requires --pos_class_weight). If not provided with --use_class_weights, uses balanced weights.'
)
parser.add_argument(
    '--pos_class_weight',
    type=float,
    default=None,
    help='Custom weight for positive class (requires --neg_class_weight). If not provided with --use_class_weights, uses balanced weights.'
)
parser.add_argument(
    '--downsample_majority',
    action='store_true',
    help='Downsample the majority class in the training set to match the minority class (alternative to class weights)'
)
parser.add_argument(
    '--hybrid_balance',
    action='store_true',
    help='Hybrid balancing: downsample majority class to 3:1 ratio AND use class weights (best of both approaches)'
)
parser.add_argument(
    '--dropout',
    type=float,
    default=0.3,
    help='Dropout rate (default: 0.3; try higher like 0.5 for imbalanced classes)'
)
parser.add_argument(
    '--learning_rate',
    type=float,
    default=2e-5,
    help='Learning rate for optimizer (default: 2e-5; typical range: 1e-5 to 5e-5)'
)
parser.add_argument(
    '--pos_class_threshold',
    type=float,
    default=0.5,
    help='Classification threshold for positive class (default: 0.5, try lower like 0.2-0.3 for pos class minority)'
)
parser.add_argument(
    '--quick_test',
    action='store_true',
    help='Quick test mode: use only first 100 training samples and 2 epochs (for local testing)'
)
parser.add_argument(
    '--save_model',
    action='store_true',
    help='Save the best model checkpoint to disk (full model with BERT + head). By default, no model is saved.'
)

# output arguments
parser.add_argument(
    '--output_dir',
    type=str,
    default='./outputs/bert_text_only',
    help='Directory to save model and results'
)

# run configurations
RANDOM_SEED = 42
BERT_MODEL_NAME = 'bert-base-uncased'
MAX_TEXT_LENGTH = 512
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.1
EARLY_STOPPING_PATIENCE = 10


class PauseDataset(Dataset):
    """
    Dataset for pause classification with text only.
    
    Expected CSV format:
        - pause_id: Unique identifier for each pause
        - text: Pre-pause transcription text (30 seconds)
        - label: Pause category/type
    """
    def __init__(
        self,
        csv_path: str,
        tokenizer: BertTokenizer,
        max_length: int = MAX_TEXT_LENGTH,
        label_to_idx: Optional[Dict[str, int]] = None,
        label_col: str = 'label'
    ):
        """
        Parameters:
            csv_path (str) - path to CSV file with pause data
            tokenizer (BertTokenizer) - BERT tokenizer
            max_length (int) - maximum sequence length for text
            label_to_idx (Optional[Dict[str, int]]) - mapping from label names to indices (for test set)
            label_col (str) - name of the label column in the CSV (default: 'label')
        """
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_col = label_col
        
        # verify label column exists
        if label_col not in self.data.columns:
            raise ValueError(f"Label column '{label_col}' not found in CSV. Available columns: {list(self.data.columns)}")
        
        # create label mapping
        if label_to_idx is None:
            unique_labels = sorted(self.data[label_col].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            
        else:
            self.label_to_idx = label_to_idx
        
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        
        print(f"Loaded dataset with {len(self.data)} samples")
        print(f"Number of classes: {self.num_classes}")
        print(f"Label distribution (column: {label_col}):\n{self.data[label_col].value_counts()}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        
        # process text
        text = str(row['text'])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # get label
        label = self.label_to_idx[row[self.label_col]]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'pause_id': row['pause_id']
        }

class TextBERT(nn.Module):
    """
    Text-only BERT model for pause classification.
    """
    
    def __init__(
        self,
        num_classes: int,
        bert_model_name: str = BERT_MODEL_NAME,
        freeze_bert: bool = False,
        dropout: float = 0.3
    ):
        super(TextBERT, self).__init__()
        
        # text encoder: BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

            print("BERT parameters frozen")
        
        # optional projection layer to reduce dimensionality
        self.projection = nn.Linear(bert_hidden_size, 256)
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        
        # classification head
        self.classifier = nn.Linear(256, num_classes)
        
        # initialize weights for new layers
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights for projection and classifier layers
        """
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Parameters:
            input_ids - BERT input token ids [batch_size, seq_len]
            attention_mask - BERT attention mask [batch_size, seq_len]
        
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # text encoding
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # use [CLS] token representation
        text_features = bert_output.last_hidden_state[:, 0, :]
        
        # projection
        text_features = self.projection(text_features)
        text_features = self.dropout(text_features)
        
        # classification
        logits = self.classifier(text_features)
        
        return logits

def seed_everything():
    """
    Set the seed for everything consistently across the board.
    This also initializes CUDA if available, which is necessary in some
    Slurm/GPU environments where CUDA might not be detected until after initialization.
    Uses relaxed deterministic mode (allows minor non-determinism in CUDA operations)
    to avoid runtime errors with CUDA operations.
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
    Determine if we can use requested device, default to CPU if none is given or 
    recognized requested device is not available.
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

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    pos_class_threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Train for one epoch
    """
    model.train()
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]', disable=not sys.stderr.isatty())

    total = 0
    correct = 0
    total_loss = 0
    for batch in progress_bar:
        # move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # forward pass
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # backward pass
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # track metrics
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        predicted = (probs[:, 1] >= pos_class_threshold).long()

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': 100 * correct / total
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str = 'Val',
    pos_class_threshold: float = 0.5
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate the model
    """
    model.eval()
    progress_bar = tqdm(dataloader, desc=f'[{split_name}]', disable=not sys.stderr.isatty())
    
    total_loss = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for batch in progress_bar:
            # move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # track metrics
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            predicted = (probs[:, 1] >= pos_class_threshold).long()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = 100 * (all_predictions == all_labels).sum() / len(all_labels)
    
    return avg_loss, accuracy, all_predictions, all_labels

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

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    output_path: str,
    label_name: str = ''
):
    """
    Plot and save training curves
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    loss_title = f'Training and Validation Loss ({label_name})' if label_name else 'Training and Validation Loss'
    ax1.set_title(loss_title)
    ax1.legend()
    ax1.grid(True)
    
    # accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    acc_title = f'Training and Validation Accuracy ({label_name})' if label_name else 'Training and Validation Accuracy'
    ax2.set_title(acc_title)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {output_path}")

def create_downsampled_dataset(
    train_dataset: Dataset,
    idx_to_label: Dict[int, str],
    random_seed: int,
    pid_col: str = 'pid'
):
    """
    Downsample majority class(es) in the training set to match the smallest class.
    Uses participant-aware sampling to maintain balanced representation across participants.
    """
    # check if pid column exists
    if pid_col not in train_dataset.data.columns:
        print(f"\nERROR: Required column '{pid_col}' not found in training data.")
        print(f"Available columns: {list(train_dataset.data.columns)}")
        print(f"\nDownsampling requires participant-aware sampling to prevent data leakage.")
        print(f"Please regenerate your input CSVs with the '{pid_col}' column included.")
        sys.exit(1)
    
    print("\nNow downsampling majority class in training set (participant-aware)")
    
    class_indices = {}
    class_participant_indices = {}
    for idx in range(len(train_dataset)):
        sample = train_dataset[idx]
        label = sample['label'].item()
        
        # get participant ID from the dataset's dataframe
        pid = train_dataset.data.iloc[idx][pid_col]
        
        # group by class
        if label not in class_indices:
            class_indices[label] = []
            class_participant_indices[label] = {}

        class_indices[label].append(idx)
        
        # group by class and participant
        if pid not in class_participant_indices[label]:
            class_participant_indices[label][pid] = []

        class_participant_indices[label][pid].append(idx)
    
    # report original class distribution
    print("  Original class distribution in training set:")
    for label_idx in sorted(class_indices.keys()):
        label_name = str(idx_to_label[label_idx]) if label_idx in idx_to_label else str(label_idx)
        count = len(class_indices[label_idx])
        n_participants = len(class_participant_indices[label_idx])
        print(f"    Class {label_idx} ({label_name}): {count} samples from {n_participants} participants")
    
    # find smallest class size
    min_count = min(len(idxs) for idxs in class_indices.values())
    print(f"\n  Downsampling all classes to {min_count} samples each")
    print("  Using participant-aware sampling to maintain balanced representation")
    
    subset_indices = []
    random.seed(random_seed)
    for label_idx, indices in class_indices.items():
        label_name = str(idx_to_label[label_idx]) if label_idx in idx_to_label else str(label_idx)
        
        if len(indices) <= min_count:
            sampled = indices
            print(f"\n    Class {label_idx} ({label_name}): kept all {len(sampled)} samples (already <= target)")
        else:
            # perform participant-aware downsampling
            participant_indices = class_participant_indices[label_idx]
            n_participants = len(participant_indices)
            
            # calculate proportional allocation per participant
            total_samples = len(indices)
            target_samples = min_count
            
            # first pass: allocate proportionally
            participant_allocations = {}
            for pid, pid_indices in participant_indices.items():
                proportion = len(pid_indices) / total_samples
                allocation = int(np.round(proportion * target_samples))
                allocation = max(1, allocation) if len(pid_indices) > 0 else 0
                allocation = min(allocation, len(pid_indices))
                participant_allocations[pid] = allocation
            
            # adjust allocations to match target exactly
            current_total = sum(participant_allocations.values())
            
            if current_total != target_samples:
                sorted_pids = sorted(participant_indices.keys(), 
                                   key=lambda p: len(participant_indices[p]), 
                                   reverse=True)
                
                if current_total > target_samples:
                    diff = current_total - target_samples
                    for pid in sorted_pids:
                        if diff == 0:
                            break
                        if participant_allocations[pid] > 1:
                            reduction = min(diff, participant_allocations[pid] - 1)
                            participant_allocations[pid] -= reduction
                            diff -= reduction
                else:
                    diff = target_samples - current_total
                    for pid in sorted_pids:
                        if diff == 0:
                            break
                        max_possible = len(participant_indices[pid])
                        if participant_allocations[pid] < max_possible:
                            increase = min(diff, max_possible - participant_allocations[pid])
                            participant_allocations[pid] += increase
                            diff -= increase
            
            # sample from each participant according to allocation
            sampled = []
            for pid, allocation in participant_allocations.items():
                pid_indices = participant_indices[pid]
                if allocation > 0:
                    sampled_from_pid = random.sample(pid_indices, allocation)
                    sampled.extend(sampled_from_pid)
            
            # report participant distribution after downsampling
            n_participants_after = sum(1 for alloc in participant_allocations.values() if alloc > 0)
            samples_per_participant = [alloc for alloc in participant_allocations.values() if alloc > 0]
            
            print(f"\n    Class {label_idx} ({label_name}): {len(indices)} -> {len(sampled)} samples after downsampling")
            print(f"      Participants: {n_participants} -> {n_participants_after} represented")
            if samples_per_participant:
                print(f"      Samples per participant: min={min(samples_per_participant)}, "
                      f"max={max(samples_per_participant)}, "
                      f"mean={np.mean(samples_per_participant):.1f}")
        
        subset_indices.extend(sampled)
    
    # shuffle combined indices
    random.seed(random_seed)
    random.shuffle(subset_indices)
    
    subset = Subset(train_dataset, subset_indices)
    print(f"\n  Final downsampled training set size: {len(subset)} samples")
    
    return subset

def create_hybrid_downsampled_dataset(
    train_dataset: Dataset,
    idx_to_label: Dict[int, str],
    random_seed: int,
    ratio: float = 3.0,
    pid_col: str = 'pid'
):
    """
    Downsample majority class to a specified ratio relative to minority class.
    Used for hybrid balancing approach: downsample to reduce extreme imbalance,
    then apply class weights to further balance.
    
    Parameters:
        train_dataset (Dataset) - the full training dataset
        idx_to_label (Dict[int, str]) - dictionary mapping label indices to label names
        random_seed (int) - random seed for reproducibility
        ratio (float) - target ratio of majority to minority class (e.g., 3.0 for 3:1)
        pid_col (str) - name of participant ID column in the dataset (default: 'pid')
    
    Returns:
        Subset - a PyTorch Subset with majority class downsampled to the specified ratio
    """
    # check if pid column exists
    if pid_col not in train_dataset.data.columns:
        print(f"\nERROR: Required column '{pid_col}' not found in training data.")
        print(f"Available columns: {list(train_dataset.data.columns)}")
        print(f"\nHybrid balancing requires participant-aware sampling to prevent data leakage.")
        print(f"Please regenerate your input CSVs with the '{pid_col}' column included.")
        sys.exit(1)
    
    print(f"\nNow running hybrid downsampling: target ratio {ratio}:1 (majority:minority)")
    
    # group indices by class and participant
    class_indices = {}
    class_participant_indices = {}
    for idx in range(len(train_dataset)):
        sample = train_dataset[idx]
        label = sample['label'].item()
        
        # get participant ID from the dataset's dataframe
        pid = train_dataset.data.iloc[idx][pid_col]
        
        # group by class
        if label not in class_indices:
            class_indices[label] = []
            class_participant_indices[label] = {}

        class_indices[label].append(idx)
        
        # group by class and participant
        if pid not in class_participant_indices[label]:
            class_participant_indices[label][pid] = []

        class_participant_indices[label][pid].append(idx)
    
    # report original class distribution
    print("  Original class distribution in training set:")
    for label_idx in sorted(class_indices.keys()):
        label_name = str(idx_to_label[label_idx]) if label_idx in idx_to_label else str(label_idx)
        count = len(class_indices[label_idx])
        n_participants = len(class_participant_indices[label_idx])
        print(f"    Class {label_idx} ({label_name}): {count} samples from {n_participants} participants")
    
    # find minority class (smallest count)
    min_count = min(len(idxs) for idxs in class_indices.values())
    min_class_idx = min(class_indices.keys(), key=lambda k: len(class_indices[k]))
    
    # calculate target size for majority class
    target_majority_count = int(min_count * ratio)
    print(f"\n  Minority class size: {min_count}")
    print(f"  Target majority class size: {target_majority_count} ({ratio}:1 ratio)")
    print("  Using participant-aware sampling to maintain balanced representation")
    
    subset_indices = []
    random.seed(random_seed)
    for label_idx, indices in class_indices.items():
        label_name = str(idx_to_label[label_idx]) if label_idx in idx_to_label else str(label_idx)
        
        if label_idx == min_class_idx:
            # keep all minority class samples
            sampled = indices
            print(f"\n    Class {label_idx} ({label_name}): kept all {len(sampled)} samples (minority class)")
        
        elif len(indices) <= target_majority_count:
            # keep all samples if already at or below target
            sampled = indices
            print(f"\n    Class {label_idx} ({label_name}): kept all {len(sampled)} samples (already <= target)")
        
        else:
            # perform participant-aware downsampling
            participant_indices = class_participant_indices[label_idx]
            n_participants = len(participant_indices)
            
            # calculate proportional allocation per participant
            total_samples = len(indices)
            target_samples = target_majority_count
            
            # first pass: allocate proportionally
            participant_allocations = {}
            for pid, pid_indices in participant_indices.items():
                proportion = len(pid_indices) / total_samples
                allocation = int(np.round(proportion * target_samples))
                # ensure at least 1 sample per participant if they have samples
                allocation = max(1, allocation) if len(pid_indices) > 0 else 0
                # cap at actual number of samples for this participant
                allocation = min(allocation, len(pid_indices))
                participant_allocations[pid] = allocation
            
            # adjust allocations to match target exactly
            current_total = sum(participant_allocations.values())
            
            if current_total != target_samples:
                # sort participants by number of samples (descending)
                sorted_pids = sorted(participant_indices.keys(), 
                                   key=lambda p: len(participant_indices[p]), 
                                   reverse=True)
                
                if current_total > target_samples:
                    # reduce allocations from participants with most samples
                    diff = current_total - target_samples
                    for pid in sorted_pids:
                        if diff == 0:
                            break

                        if participant_allocations[pid] > 1:
                            reduction = min(diff, participant_allocations[pid] - 1)
                            participant_allocations[pid] -= reduction
                            diff -= reduction
                else:
                    # increase allocations for participants with room
                    diff = target_samples - current_total
                    for pid in sorted_pids:
                        if diff == 0:
                            break

                        max_possible = len(participant_indices[pid])
                        if participant_allocations[pid] < max_possible:
                            increase = min(diff, max_possible - participant_allocations[pid])
                            participant_allocations[pid] += increase
                            diff -= increase
            
            # sample from each participant according to allocation
            sampled = []
            for pid, allocation in participant_allocations.items():
                pid_indices = participant_indices[pid]
                if allocation > 0:
                    sampled_from_pid = random.sample(pid_indices, allocation)
                    sampled.extend(sampled_from_pid)
            
            # report participant distribution after downsampling
            n_participants_after = sum(1 for alloc in participant_allocations.values() if alloc > 0)
            samples_per_participant = [alloc for alloc in participant_allocations.values() if alloc > 0]
            
            print(f"\n    Class {label_idx} ({label_name}): {len(indices)} -> {len(sampled)} samples after downsampling")
            print(f"      Participants: {n_participants} -> {n_participants_after} represented")
            if samples_per_participant:
                print(f"      Samples per participant: min={min(samples_per_participant)}, "
                      f"max={max(samples_per_participant)}, "
                      f"mean={np.mean(samples_per_participant):.1f}")
        
        subset_indices.extend(sampled)
    
    # shuffle combined indices
    random.seed(random_seed)
    random.shuffle(subset_indices)
    
    subset = Subset(train_dataset, subset_indices)
    print(f"\n  Final downsampled training set size: {len(subset)} samples")
    
    return subset

def create_quick_test_subset(train_dataset, idx_to_label, random_seed, samples_per_class=50):
    """
    Create a balanced subset of the training dataset for quick testing.
    """
    print("\nQUICK TEST MODE: creating balanced subset")
    
    class_indices = {}
    for idx in range(len(train_dataset)):
        sample = train_dataset[idx]
        label = sample['label'].item()
        if label not in class_indices:
            class_indices[label] = []

        class_indices[label].append(idx)
    
    # randomly sample from each class
    print(f"  ...sampling up to {samples_per_class} samples per class")
    subset_indices = []
    for label_idx, indices in class_indices.items():
        label_name = str(idx_to_label[label_idx]) if label_idx in idx_to_label else str(label_idx)
        available = len(indices)
        n_samples = min(samples_per_class, available)
        
        random.seed(random_seed)
        sampled = random.sample(indices, n_samples)
        subset_indices.extend(sampled)
        
        print(f"      Class {label_idx} ({label_name}): {n_samples} samples (out of {available} available)")
    
    # shuffle the combined indices to mix classes
    random.seed(random_seed)
    random.shuffle(subset_indices)
    
    subset = Subset(train_dataset, subset_indices)
    
    print(f"  ...using {len(subset)} balanced training samples for local end-to-end test")
    
    return subset

def estimate_checkpoint_size_mb(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        save_model: bool = True
        ) -> float:
    """
    Estimate checkpoint file size in MB (model state + optimizer state + small metadata).
    Always saves full model when save_model is True.
    """
    state = model.state_dict()
    if save_model:
        model_bytes = sum(t.numel() * t.element_size() for t in state.values())

    else:
        model_bytes = sum(
            t.numel() * t.element_size() for k, t in state.items()
            if not k.startswith('bert.')
        )

    # AdamW stores exp_avg and exp_avg_sq per parameter (2 x model size)
    optimizer_param_count = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    optimizer_bytes = 2 * optimizer_param_count * 4
    total = (model_bytes + optimizer_bytes) * 1.05 + 50 * 1024

    return total / (1024 * 1024)

def check_output_dir_writable(output_dir: Path) -> None:
    """
    Verify that we can create and write a file in output_dir.
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / ".write_probe"

    try:
        probe.write_bytes(b"probe")
        probe.unlink()

    except OSError as error:
        raise RuntimeError(
            f"Cannot write to output directory: {output_dir}\n"
            f"  Error: {error}\n"
            f"  Check: (1) permissions on the directory, (2) disk quota, (3) filesystem read-only."
        ) from error

def run_pipeline(
    device,
    output_dir,
    train_csv,
    val_csv,
    test_csv,
    num_workers,
    freeze_bert,
    use_class_weights=False,
    neg_class_weight=None,
    pos_class_weight=None,
    label_col='label',
    quick_test=False,
    downsample_majority=False,
    hybrid_balance=False,
    save_model=False,
    dropout=0.3,
    learning_rate=2e-5,
    pos_class_threshold=0.5
):
    """
    Main process: training the model and evaluating it on the validation and test sets
    """
    print("\nBeginning training of Text-Only BERT for pause classification")

    print("~~~ Run Configuration ~~~")
    print(f"  Label:                  {label_col}")
    print(f"  Output dir:             {output_dir}")
    print(f"  Device:                 {device}")
    print()
    print(f"  -- Data --")
    print(f"  Train CSV:              {train_csv}")
    print(f"  Val CSV:                {val_csv}")
    print(f"  Test CSV:               {test_csv}")
    print()
    print(f"  -- Model --")
    print(f"  BERT model:             {BERT_MODEL_NAME}")
    print(f"  Freeze BERT:            {freeze_bert}")
    print(f"  Dropout:                {dropout}")
    print()
    print(f"  -- Training --")
    print(f"  Random seed:            {RANDOM_SEED}")
    print(f"  Batch size:             {BATCH_SIZE}")
    print(f"  Max epochs:             {NUM_EPOCHS}")
    print(f"  Early stop patience:    {EARLY_STOPPING_PATIENCE}")
    print(f"  Learning rate:          {learning_rate}")
    print(f"  Warmup ratio:           {WARMUP_RATIO}")
    print(f"  Weight decay:           {WEIGHT_DECAY}")
    print(f"  Max text length:        {MAX_TEXT_LENGTH}")
    print()
    print(f"  -- Class Imbalance --")
    print(f"  Use class weights:      {use_class_weights}")
    print(f"  Downsample majority:    {downsample_majority}")
    print(f"  Hybrid balance:         {hybrid_balance}")
    print(f"  Pos class threshold:    {pos_class_threshold}")
    print()
    print(f"  -- Other --")
    print(f"  Num workers:            {num_workers}")
    print(f"  Quick test mode:        {quick_test}")
    print(f"  Save model to disk:     {save_model}")
    
    print("\nInitializing environment")
    seed_everything()
    device = set_device(device)
    print(f"   ...using device: {device}")
    
    # create output directory and verify we can write to it
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    check_output_dir_writable(output_dir)
    print(f"   ...output directory writable: {output_dir}")
    if save_model:
        print("   ...best model will be saved to disk (full model: BERT + head)")
    else:
        print("   ...models will NOT be saved to disk (use --save_model to save best model)")
    
    # initialize tokenizer
    print("\nLoading tokenizer")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    # load datasets
    print("\nLoading datasets")
    print("  NOTE: Using pre-split train/val/test files")
    
    train_dataset = PauseDataset(
        csv_path=train_csv,
        tokenizer=tokenizer,
        max_length=MAX_TEXT_LENGTH,
        label_col=label_col,
    )
    
    # load validation set with same label mapping as training set
    val_dataset = PauseDataset(
        csv_path=val_csv,
        tokenizer=tokenizer,
        max_length=MAX_TEXT_LENGTH,
        label_to_idx=train_dataset.label_to_idx,
        label_col=label_col,
    )
    
    # load test set with same label mapping as training set
    test_dataset = PauseDataset(
        csv_path=test_csv,
        tokenizer=tokenizer,
        max_length=MAX_TEXT_LENGTH,
        label_to_idx=train_dataset.label_to_idx,
        label_col=label_col,
    )
    
    # get label information from training dataset
    label_to_idx = train_dataset.label_to_idx
    num_classes = train_dataset.num_classes
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    # optional downsampling of majority class(es) in the training set
    if hybrid_balance and not quick_test:
        # hybrid approach: downsample to 3:1 ratio AND use class weights
        train_dataset = create_hybrid_downsampled_dataset(
            train_dataset=train_dataset,
            idx_to_label=idx_to_label,
            random_seed=RANDOM_SEED,
            ratio=3.0,
            pid_col='pid'
        )
    elif downsample_majority and not quick_test:
        train_dataset = create_downsampled_dataset(
            train_dataset=train_dataset,
            idx_to_label=idx_to_label,
            random_seed=RANDOM_SEED,
            pid_col='pid'
        )
    
    # quick test mode: create balanced subset
    if quick_test:
        train_dataset = create_quick_test_subset(
            train_dataset=train_dataset,
            idx_to_label=idx_to_label,
            random_seed=RANDOM_SEED,
            samples_per_class=50
        )
    
    # convert class names to strings
    class_names = [str(idx_to_label[i]) for i in range(num_classes)]
    
    # create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if str(device) != 'cpu' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if str(device) != 'cpu' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if str(device) != 'cpu' else False
    )
    
    # initialize model
    print("\nInitializing model...")
    model = TextBERT(
        num_classes=num_classes,
        freeze_bert=freeze_bert,
        dropout=dropout
    )
    model = model.to(device)
    
    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # loss function (with optional class weights for imbalanced classes)
    if use_class_weights or hybrid_balance:
        print("\nCalculating class weights for imbalanced classes...")
        # get all labels from training dataset
        train_labels = []
        for idx in range(len(train_dataset)):
            sample = train_dataset[idx]
            train_labels.append(sample['label'].item())
        
        train_labels = torch.tensor(train_labels)
        class_counts = torch.bincount(train_labels, minlength=num_classes).float()
        total_samples = len(train_labels)
        
        # check if custom class weights are provided
        if neg_class_weight is not None and pos_class_weight is not None:
            # use custom weights provided by user
            class_weights = torch.tensor([neg_class_weight, pos_class_weight], dtype=torch.float)
            print(f"  Using custom class weights: [neg={neg_class_weight}, pos={pos_class_weight}]")

        else:
            # calculate inverse frequency weights (balanced)
            # weight for each class = total_samples / (num_classes * count_of_class)
            class_weights = total_samples / (num_classes * class_counts)
            # normalize so weights sum to num_classes (optional, but helps with stability)
            class_weights = class_weights / class_weights.sum() * num_classes
            print(f"  Using balanced class weights based on class distribution")
        
        print(f"  Class distribution in training set:")
        for idx in range(num_classes):
            label_name = idx_to_label[idx]
            count = int(class_counts[idx].item())
            weight = class_weights[idx].item()
            pct = 100 * count / total_samples
            print(f"    Class {idx} ({label_name}): {count} samples ({pct:.1f}%) -> weight: {weight:.3f}")
        
        # move weights to device
        class_weights = class_weights.to(device)
        
        # loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"\nUsing weighted CrossEntropyLoss to handle class imbalance")

    else:
        criterion = nn.CrossEntropyLoss()
        print("\nUsing standard CrossEntropyLoss (no class weighting)")
    
    # optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY
    )
    
    # determine number of epochs
    if quick_test:
        num_epochs = 5
        print(f"\nQUICK TEST MODE: using {num_epochs} epochs for local end-to-end test")

    else:
        num_epochs = NUM_EPOCHS
    
    print(f"\nTraining for {num_epochs} epochs")
    
    # learning rate scheduler
    num_training_steps = len(train_loader) * num_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=num_training_steps,
        pct_start=WARMUP_RATIO,
        anneal_strategy='cos'
    )
    
    # report estimated checkpoint size if saving to disk
    if save_model:
        est_mb = estimate_checkpoint_size_mb(model, optimizer, save_model=True)
        print(f"\nEstimated checkpoint size (best_model_*.pt): ~{est_mb:.1f} MB  (ensure this much free space in output_dir)")
    
    # training loop
    print("\n   ~~~ Starting training! ~~~")
    
    val_accs = []
    train_accs = []
    val_losses = []
    train_losses = []
    patience_counter = 0
    best_val_loss = float('inf')
    best_model_state = None  # track best model in memory
    best_epoch = 0
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # train
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            epoch,
            pos_class_threshold
        )
        
        # validate
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model,
            val_loader,
            criterion,
            device,
            split_name='Val',
            pos_class_threshold=pos_class_threshold
        )
        
        # store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # calculate F1 score
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        print(f"  Val F1 (weighted): {val_f1:.4f}")
        
        # early stopping and model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # always save best model state in memory
            import copy
            best_model_state = copy.deepcopy(model.state_dict())
            
            # optionally save to disk if requested
            if save_model:
                output_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = (output_dir / f'best_model_{label_col}.pt').resolve()
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'label_to_idx': label_to_idx
                }

                try:
                    with open(checkpoint_path, 'wb') as f:
                        torch.save(checkpoint_data, f)
                    print(f"  ...new best model saved to disk (val_loss: {val_loss:.4f})")

                except OSError as error:
                    raise RuntimeError(
                        f"Failed to save checkpoint to {checkpoint_path}: {error}. "
                        "Check permissions, disk quota, and that the directory is writable."
                    ) from error
                
            else:
                print(f"  ...new best model (val_loss: {val_loss:.4f}) [in memory, not saved to disk]")

        else:
            patience_counter += 1
            print(f"  ...no improvement for {patience_counter} epoch(s)")
            
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
    
    # final evaluation
    print("\nRunning final evaluation")

    # load best model from memory
    if best_model_state is None:
        print("WARNING: No best model found (training may have failed). Using final model state.")
        best_epoch = num_epochs

    else:
        print(f"Loading best model from memory (Epoch {best_epoch})...")
        model.load_state_dict(best_model_state)
    
    # evaluate on validation set
    val_loss, val_acc, val_preds, val_labels = evaluate(
        model,
        val_loader,
        criterion,
        device,
        split_name='Val',
        pos_class_threshold=pos_class_threshold
    )
    
    print(f"\nBest model performance (Epoch {best_epoch}):")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val Acc:  {val_acc:.2f}%")
    
    # evaluate on test set
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model,
        test_loader,
        criterion,
        device,
        split_name='Test',
        pos_class_threshold=pos_class_threshold
    )
    
    # create prediction CSVs with original data + predictions
    print("\nSaving prediction CSVs")
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    # map predicted indices back to label names
    val_pred_labels = [idx_to_label[pred] for pred in val_preds]
    test_pred_labels = [idx_to_label[pred] for pred in test_preds]
    
    # add prediction column
    val_df['bert_predicted_label'] = val_pred_labels
    test_df['bert_predicted_label'] = test_pred_labels
    
    # save prediction CSVs
    val_pred_output = output_dir / f'validation_{label_col}_with_predictions.csv'
    test_pred_output = output_dir / f'testing_{label_col}_with_predictions.csv'
    
    val_df.to_csv(val_pred_output, index=False)
    test_df.to_csv(test_pred_output, index=False)
    
    print(f"\nSaved prediction CSVs:")
    print(f"  Validation: {val_pred_output}")
    print(f"  Testing:    {test_pred_output}")
    
    print(f"\n  Test Loss: {test_loss:.4f}")
    print(f"  Test Acc:  {test_acc:.2f}%")
    
    # calculate F1 scores
    val_f1 = f1_score(val_labels, val_preds, average='weighted')
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    print(f"  Val F1 (weighted):  {val_f1:.4f}")
    print(f"  Test F1 (weighted): {test_f1:.4f}")
    
    # classification reports
    print("\nMaking validation classification report")
    val_report = classification_report(
        val_labels,
        val_preds,
        target_names=class_names,
        digits=4
    )
    print(val_report)
    
    print("\nMaking test classification report")
    test_report = classification_report(
        test_labels,
        test_preds,
        target_names=class_names,
        digits=4
    )
    print(test_report)
    
    # save classification reports
    with open(output_dir / f'classification_report_val_{label_col}.txt', 'w') as f:
        f.write(f"VALIDATION SET CLASSIFICATION REPORT ({label_col})\n")
        f.write("="*80 + "\n\n")
        f.write(val_report)
    
    with open(output_dir / f'classification_report_test_{label_col}.txt', 'w') as f:
        f.write(f"TEST SET CLASSIFICATION REPORT ({label_col})\n")
        f.write("="*80 + "\n\n")
        f.write(test_report)
    
    # plot confusion matrices
    plot_confusion_matrix(
        val_labels,
        val_preds,
        class_names,
        output_dir / f'confusion_matrix_val_{label_col}.png',
        label_name=label_col
    )
    
    plot_confusion_matrix(
        test_labels,
        test_preds,
        class_names,
        output_dir / f'confusion_matrix_test_{label_col}.png',
        label_name=label_col
    )
    
    # plot training curves
    plot_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        output_dir / f'training_curves_{label_col}.png',
        label_name=label_col
    )
    
    print("\nProcess complete!")
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"  - Best model: best_model_{label_col}.pt")
    print(f"  - Validation classification report: classification_report_val_{label_col}.txt")
    print(f"  - Test classification report: classification_report_test_{label_col}.txt")
    print(f"  - Validation confusion matrix: confusion_matrix_val_{label_col}.png")
    print(f"  - Test confusion matrix: confusion_matrix_test_{label_col}.png")
    print(f"  - Training curves: training_curves_{label_col}.png")
    print(f"  - Validation predictions CSV: validation_{label_col}_with_predictions.csv")
    print(f"  - Testing predictions CSV: testing_{label_col}_with_predictions.csv")


if __name__ == '__main__':
    args = parser.parse_args()
    
    # construct CSV file paths based on naming convention
    data_dir = Path(args.data_dir)
    label_name = args.label_name
    
    train_csv = data_dir / f"{label_name}_training.csv"
    val_csv = data_dir / f"{label_name}_validation.csv"
    test_csv = data_dir / f"{label_name}_testing.csv"
    
    # validate that all required files exist
    missing_files = []
    if not train_csv.exists():
        missing_files.append(str(train_csv))
    if not val_csv.exists():
        missing_files.append(str(val_csv))
    if not test_csv.exists():
        missing_files.append(str(test_csv))
    
    if missing_files:
        print("ERROR: The following required CSV files were not found:")
        for f in missing_files:
            print(f"  - {f}")
        print(f"\nExpected files in directory: {data_dir}")
        print(f"  - {train_csv.name}")
        print(f"  - {val_csv.name}")
        print(f"  - {test_csv.name}")
        sys.exit(1)
    
    print(f"\nFound all required CSV files:")
    print(f"  Training:   {train_csv}")
    print(f"  Validation: {val_csv}")
    print(f"  Test:       {test_csv}")
    
    device = args.device    
    output_dir = args.output_dir
    num_workers = args.num_workers
    freeze_bert = args.freeze_bert
    use_class_weights = args.use_class_weights
    label_col = args.label_name
    quick_test = args.quick_test
    downsample_majority = args.downsample_majority
    hybrid_balance = args.hybrid_balance
    save_model = args.save_model
    dropout = args.dropout
    learning_rate = args.learning_rate
    pos_class_threshold = args.pos_class_threshold
    neg_class_weight = args.neg_class_weight
    pos_class_weight = args.pos_class_weight
    
    # validation: custom class weights require both neg and pos to be specified
    if (neg_class_weight is None) != (pos_class_weight is None):
        print("\nERROR: Both --neg_class_weight and --pos_class_weight must be provided together.")
        print("       Cannot specify only one weight without the other.")
        sys.exit(1)
    
    # validation: custom class weights require --use_class_weights or --hybrid_balance
    if (neg_class_weight is not None or pos_class_weight is not None):
        if not use_class_weights and not hybrid_balance:
            print("\nERROR: Custom class weights (--neg_class_weight, --pos_class_weight)")
            print("       require either --use_class_weights or --hybrid_balance to be set.")
            sys.exit(1)
    
    # basic safety: avoid conflicting imbalance-handling strategies
    conflicting_strategies = sum([use_class_weights, downsample_majority, hybrid_balance])
    if conflicting_strategies > 1:
        print("\nERROR: Only ONE of the following can be set at a time:")
        print("       --use_class_weights")
        print("       --downsample_majority")
        print("       --hybrid_balance")
        print("\n       --hybrid_balance combines downsampling (3:1 ratio) + class weights")
        print("       Please choose only ONE imbalance handling strategy.")
        sys.exit(1)

    run_pipeline(
        device,
        output_dir,
        str(train_csv),
        str(val_csv),
        str(test_csv),
        num_workers,
        freeze_bert,
        use_class_weights=use_class_weights,
        neg_class_weight=neg_class_weight,
        pos_class_weight=pos_class_weight,
        label_col=label_col,
        quick_test=quick_test,
        downsample_majority=downsample_majority,
        hybrid_balance=hybrid_balance,
        save_model=save_model,
        dropout=dropout,
        learning_rate=learning_rate,
        pos_class_threshold=pos_class_threshold
    )
