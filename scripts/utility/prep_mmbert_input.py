"""
Data Preparation Script for Multi-Modal BERT for R21 Pause Classification

This script helps prep pause data into the format required by the multi-modal BERT classifier.

Input: 
    - existing pause records table/CSV
    - directory containing spectrogram images

Output: 
    - formatted CSV with columns: pause_id, pid, text, spectrogram_path, label
    - optionally, train/val/test split files

NOTE: the output of this script can be used directly with the multi-modal BERT classifier.
By using it in this way, you can skip the data preparation step there and go straight to training.

Author: kirsten.bonson@uvm.edu
Created: April 2026
"""

import os
import sys
import argparse
import warnings
import pandas as pd

from typing import Tuple
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

RANDOM_SEED = 42

parser = argparse.ArgumentParser(
    description='Prepare pause data for multi-modal classification'
)

# required arguments
parser.add_argument(
    '--input_csv',
    type=str,
    required=True,
    help='Path to input CSV with pause records'
)
parser.add_argument(
    '--spectrogram_dir',
    type=str,
    required=True,
    help='Directory containing spectrogram images'
)
parser.add_argument(
    '--output_dir',
    type=str,
    required=True,
    help='Directory to save formatted CSV(s) (filenames auto-generated from label name)'
)
parser.add_argument(
    '--label_col',
    type=str,
    required=True,
    help='Name of label column'
)

# optional arguments
parser.add_argument(
    '--drop_incomplete_rows',
    action='store_true',
    help='Drop rows missing either text or spectrogram (both are required for MMBERT)'
)
parser.add_argument(
    '--remove_speaker_labels',
    action='store_true',
    help='Remove speaker labels (e.g., "SPEAKER_00:") from text sequences'
)

# train/val/test split options
parser.add_argument(
    '--create_split',
    action='store_true',
    help='Also create train/val/test split files'
)
parser.add_argument(
    '--ratios',
    type=str,
    default='0.7,0.15,0.15',
    help='Comma-separated ratios for train,val,test (as decimals 0.0-1.0 or percentages 0-100, default: 0.7,0.15,0.15)'
)


def prepare_pause_data(
    input_csv,
    spectrogram_dir,
    output_csv,
    pause_id_col='record_id',
    pid_col='pid',
    text_before_col='text_before_pause',
    text_after_col='text_after_pause',
    label_col='label',
    spectrogram_extension='.png',
    drop_incomplete_rows=False,
    remove_speaker_labels=False
):
    """
    Prepare pause data for multi-modal classification.
    
    Parameters:
        input_csv (str) - path to input CSV with pause records
        spectrogram_dir (str) - directory containing spectrogram images (parent directory)
        output_csv (str) - path to save formatted CSV
        pause_id_col (str) - name of pause ID column in input CSV
        pid_col (str) - name of participant ID column in input CSV (for subdirectory lookup)
        text_before_col (str) - name of text before pause column in input CSV
        text_after_col (str) - name of text after pause column in input CSV
        label_col (str) - name of label column in input CSV
        spectrogram_extension (str) - file extension for spectrograms
        drop_incomplete_rows (boolean) - drop rows missing either text or spectrogram
        remove_speaker_labels (boolean) - remove speaker labels (e.g., "SPEAKER_00:") from text
    """
    
    print("\nPreparing data for multi-modal pause classification")

    print(f"\nLoading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"   ...loaded {len(df)} records")
    print(f"   ...columns: {list(df.columns)}")
    
    # verify required columns exist
    required_cols = [pause_id_col, text_before_col, text_after_col, label_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"ERROR: input CSV is missing required columns: {missing_cols}")
    
    # UPDATE: only consider records with consert_pause_class values (only those that match have this)
    print(f'\nFiltering records to consider only those with consert_pause_class values (i.e. CONSERT-matched pauses)')
    df = df[~df.consert_pause_class.isna()]
    print(f'   ...a total of {len(df)} records have CONSERT pause matches')
    
    # filter out records with missing labels
    n_before_filter = len(df)
    df = df[df[label_col].notna()].copy()
    n_after_filter = len(df)
    n_filtered = n_before_filter - n_after_filter
    
    if n_filtered > 0:
        print(f"\nWARNING: {n_filtered} records were missing requested label ({label_col}) and have been filtered out")
        print(f"  ...processing {n_after_filter} records with label (out of {n_before_filter} total)")
    else:
        print(f"\nAll {n_after_filter} records have label")
    
    if n_after_filter == 0:
        raise ValueError("ERROR: No records remain after filtering for missing label!")
    
    # check if PID column exists (needed for nested directory structure)
    has_pid_col = pid_col in df.columns
    if not has_pid_col:
        print(f"WARNING: PID column '{pid_col}' not found. Will look for spectrograms in flat directory structure.")
    
    # combine text before and after pause into a single text string
    print("\nCombining text_before_pause and text_after_pause into single text sequence")
    df['text_combined'] = (
        df[text_before_col].fillna('').astype(str) + ' ' + 
        df[text_after_col].fillna('').astype(str)
    ).str.strip()
    
    # remove speaker labels if requested
    if remove_speaker_labels:
        import re
        print("  ...removing speaker labels from text")
        # pattern matches speaker labels like "SPEAKER_00:", "SPEAKER_01:", etc.
        speaker_pattern = r'\bSPEAKER_\d{2}:\s*'
        df['text_combined'] = df['text_combined'].apply(
            lambda text: re.sub(speaker_pattern, '', text).strip()
        )
    
    # count how many had missing text in either column
    missing_before = df[text_before_col].isna().sum()
    missing_after = df[text_after_col].isna().sum()
    missing_both = (df[text_before_col].isna() & df[text_after_col].isna()).sum()
    if missing_before > 0 or missing_after > 0:
        print(f"  ...{missing_before} rows missing text_before_pause")
        print(f"  ...{missing_after} rows missing text_after_pause")
        print(f"  ...{missing_both} rows missing both (will have empty text)")
    
    # convert spectrogram directory to Path
    spec_dir = Path(spectrogram_dir)
    if not spec_dir.exists():
        raise ValueError(f"ERROR: spectrogram directory does not exist: {spectrogram_dir}")
    
    print(f"\nUsing spectrogram directory: {spectrogram_dir}")
    
    # create spectrogram_path column
    print("\nMapping spectrograms to pause records")
    missing_spectrograms = []
    spectrogram_paths = []
    found_in_nested = 0
    found_in_flat = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        pause_id = row[pause_id_col]
        # build expected spectrogram filename:
        # pid_[pid]_pauseid_[pause_id]_spectrogram.png
        expected_path = None
        
        # try nested directory structure first (if PID column exists)
        if has_pid_col:
            pid = str(row[pid_col]).zfill(3)  # Ensure 3-digit format with leading zeros
            spec_name = f"pid_{pid}_pauseid_{pause_id}_spectrogram{spectrogram_extension}"
            nested_path = os.path.join(spectrogram_dir, pid, spec_name)
            if os.path.exists(nested_path):
                expected_path = nested_path
                found_in_nested += 1

            else:
                # fall back to flat structure
                flat_path = os.path.join(spectrogram_dir, spec_name)
                if os.path.exists(flat_path):
                    expected_path = flat_path
                    found_in_flat += 1
        else:
            # only try flat structure if no PID column
            flat_path = os.path.join(spectrogram_dir, spec_name)
            if os.path.exists(flat_path):
                expected_path = flat_path
                found_in_flat += 1
        
        if expected_path is None:
            missing_spectrograms.append(pause_id)
            spectrogram_paths.append("")

        else:
            # convert to absolute path for use in other scripts
            abs_path = os.path.abspath(expected_path)
            spectrogram_paths.append(abs_path)
    
    # report where spectrograms were found
    if has_pid_col:
        print(f"  ...found {found_in_nested} spectrograms in PID subdirectories")
        if found_in_flat > 0:
            print(f"  ...found {found_in_flat} spectrograms in flat directory structure")
    
    df['spectrogram_path'] = spectrogram_paths
    
    # report missing spectrograms
    if missing_spectrograms:
        print(f"\nWARNING: {len(missing_spectrograms)} spectrograms not found")
        print(f"   ...first 10 missing: {missing_spectrograms[:10]}")
    
    # select and rename columns
    # preserve original label values exactly as they appear in the input CSV
    # (no transformation - use values directly from the input column)
    # use the original label column name in the output
    output_df = pd.DataFrame({
        'pause_id': df[pause_id_col],
        'text': df['text_combined'],
        'spectrogram_path': df['spectrogram_path'],
    })
    
    # add PID column if available
    if has_pid_col:
        output_df['pid'] = df[pid_col]
    
    output_df[label_col] = df[label_col]  # preserve original label column name and values
    
    print("\nStarting data quality checks")
    
    # check for missing text (fill with empty string for now)
    missing_text = output_df['text'].isna().sum()
    if missing_text > 0:
        print(f"WARNING: {missing_text} records have missing text")
        print("  ...filling with empty string")
        output_df['text'] = output_df['text'].fillna("")
    
    # count rows with empty text (after filling NaN)
    empty_text = (output_df['text'].str.strip() == "").sum()
    
    # count rows with missing spectrograms
    missing_spec = (output_df['spectrogram_path'] == "").sum()
    
    # identify incomplete rows (missing either text or spectrogram)
    incomplete_mask = (
        (output_df['text'].str.strip() == "") | 
        (output_df['spectrogram_path'] == "")
    )
    n_incomplete = incomplete_mask.sum()
    
    if n_incomplete > 0:
        print(f"\nWARNING: {n_incomplete} incomplete rows found:")
        print(f"  ...{empty_text} rows with empty text")
        print(f"  ...{missing_spec} rows with missing spectrograms")
        
        # option to drop incomplete rows
        if drop_incomplete_rows:
            output_df = output_df[~incomplete_mask].copy()
            print(f"  ...dropped {n_incomplete} incomplete records")
            print(f"  ...remaining records: {len(output_df)}")

        else:
            print(f"  ...keeping all rows (use --drop_incomplete_rows to filter them out)")
    
    # verify no missing labels (shouldn't happen since we filtered earlier, but double-check)
    missing_labels = output_df[label_col].isna().sum()
    if missing_labels > 0:
        print(f"WARNING: {missing_labels} records still have missing labels (unexpected after filtering)")
        print("  ...filtering these out as well")
        output_df = output_df[output_df[label_col].notna()].copy()
    
    # check text lengths
    text_lengths = output_df['text'].str.len()
    print(f"\nText Statistics:")
    print(f"  Mean length: {text_lengths.mean():.1f} characters")
    print(f"  Median length: {text_lengths.median():.1f} characters")
    print(f"  Min length: {text_lengths.min()} characters")
    print(f"  Max length: {text_lengths.max()} characters")
    
    # approximate token count (rough estimate: 4 chars per token)
    approx_tokens = text_lengths / 4
    over_512 = (approx_tokens > 512).sum()
    if over_512 > 0:
        print(f"\n  INFO: ~{over_512} texts may exceed BERT's 512 token limit")
        print(f"    (These will be automatically truncated during training)")
    
    # label distribution
    print(f"\nLabel Distribution (column: {label_col}):")
    label_counts = output_df[label_col].value_counts()
    for label, count in label_counts.items():
        percentage = 100 * count / len(output_df)
        label_str = str(label)  # convert to string in case label is numeric
        print(f"  {label_str:20s}: {count:5d} ({percentage:5.1f}%)")
    
    # check for class imbalance
    max_count = label_counts.max()
    min_count = label_counts.min()
    imbalance_ratio = max_count / min_count
    if imbalance_ratio > 5:
        print(f"\n  WARNING: Class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
        print(f"    ...consider using class weights or data augmentation")
    
    # save output
    print(f"\nSaving formatted data to: {output_csv}")
    
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_df.to_csv(output_csv, index=False)
    print(f"Saved to: {output_csv}")
    print(f"Final record count: {len(output_df)}")
    
    print("\nData preparation complete")
    print(f"You can now use this file with:")
    print(f"  python mm_bert_for_pause_classification.py --train_csv {output_csv}")

def normalize_ratio(ratio: float) -> float:
    """
    Normalize ratio to decimal format (0.0-1.0).
    If ratio > 1, assume it's a percentage and divide by 100.
    
    Parameters:
        ratio (float) - Ratio as decimal (0.0-1.0) or percentage (0-100)
    
    Returns:
        Normalized ratio as decimal (0.0-1.0)
    """
    if ratio > 1.0:
        return ratio / 100.0
    
    return ratio

def parse_ratios(ratios_str: str) -> Tuple[float, float, float]:
    """
    Parse comma-separated ratios string into three float values.
    
    Parameters:
        ratios_str (str) - Comma-separated string like "0.7,0.15,0.15" or "50,25,25"
    
    Returns:
        Tuple[float, float, float] - (train_ratio, val_ratio, test_ratio) as normalized decimals
    """
    parts = [s.strip() for s in ratios_str.split(',')]

    if len(parts) != 3:
        raise ValueError(
            f"Ratios must be exactly 3 comma-separated values, got: {ratios_str}"
        )
    
    try:
        train_ratio = normalize_ratio(float(parts[0]))
        val_ratio = normalize_ratio(float(parts[1]))
        test_ratio = normalize_ratio(float(parts[2]))

    except ValueError as error:
        raise Exception(
            f"ERROR: invalid ratio format {ratios_str}. \
                It must be a comma-separated string like '0.7,0.15,0.15' or '50,25,25'"
        ) from error
    
    return train_ratio, val_ratio, test_ratio

def generate_split_filenames(base_csv_path):
    """
    Generate output filenames for train, validation, and test splits
    based on the base CSV path.
    
    Parameters:
        base_csv_path (str) - Path to the base CSV file
    
    Returns:
        Tuple of (train_output, val_output, test_output) paths
    """
    base_path = Path(base_csv_path)
    base_dir = base_path.parent
    base_name = base_path.stem  # filename without extension
    extension = base_path.suffix  # .csv
    
    train_output = str(base_dir / f"{base_name}_training{extension}")
    val_output = str(base_dir / f"{base_name}_validation{extension}")
    test_output = str(base_dir / f"{base_name}_testing{extension}")
    
    return train_output, val_output, test_output

def split_by_participant(
    df: pd.DataFrame,
    pid_col: str,
    label_col: str,
    val_ratio: float,
    test_ratio: float,
    random_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by participant to prevent data leakage.
    Attempts to stratify by participant's dominant label.
    
    Parameters:
        df (pd.DataFrame) - dataframe to split
        pid_col (str) - name of participant ID column
        label_col (str) - name of label column
        val_ratio (float) - fraction for validation set
        test_ratio (float) - fraction for test set
        random_seed (int) - random seed for reproducibility
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # get unique participants
    unique_pids = df[pid_col].unique()
    n_participants = len(unique_pids)
    
    print(f"  Splitting {n_participants} unique participants")
    
    # for each participant, determine their dominant label for stratification
    pid_dominant_labels = {}
    for pid in unique_pids:
        pid_data = df[df[pid_col] == pid]
        # get the most common label for this participant
        dominant_label = pid_data[label_col].mode()[0]
        pid_dominant_labels[pid] = dominant_label
    
    # create a dataframe of PIDs with their dominant labels
    pid_df = pd.DataFrame({
        'pid': list(pid_dominant_labels.keys()),
        'dominant_label': list(pid_dominant_labels.values())
    })
    
    # try stratified split on participants by their dominant label
    temp_size = val_ratio + test_ratio
    
    try:
        # first split: train vs (val + test)
        train_pids, temp_pids = train_test_split(
            pid_df['pid'].values,
            test_size=temp_size,
            random_state=random_seed,
            stratify=pid_df['dominant_label'].values
        )
        
        # get dominant labels for temp_pids for second split
        temp_pid_labels = pid_df[pid_df['pid'].isin(temp_pids)]['dominant_label'].values
        
        # second split: val vs test
        val_relative_ratio = val_ratio / temp_size
        val_pids, test_pids = train_test_split(
            temp_pids,
            test_size=(1 - val_relative_ratio),
            random_state=random_seed,
            stratify=temp_pid_labels
        )
        
        print(f"  Using stratified split by participant's dominant label")
        
    except ValueError as error:
        print(f"  WARNING: Stratified split failed ({error}), likely due to insufficient samples per class")
        print(f"  ...falling back to random split without stratification (but still with PID handling!)")
        
        # fall back to random split without stratification
        train_pids, temp_pids = train_test_split(
            pid_df['pid'].values,
            test_size=temp_size,
            random_state=random_seed
        )
        
        val_relative_ratio = val_ratio / temp_size
        val_pids, test_pids = train_test_split(
            temp_pids,
            test_size=(1 - val_relative_ratio),
            random_state=random_seed
        )
    
    # create dataframes by filtering on PIDs
    train_df = df[df[pid_col].isin(train_pids)].copy()
    val_df = df[df[pid_col].isin(val_pids)].copy()
    test_df = df[df[pid_col].isin(test_pids)].copy()
    
    return train_df, val_df, test_df

def create_train_val_test_split(
    input_csv: str,
    train_output: str,
    val_output: str,
    test_output: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    label_col: str = 'label',
    random_seed: int = 42,
    pid_col: str = 'pid'
):
    """
    Split formatted data into train, validation, and test sets by participant.
    
    Splits by participant to prevent data leakage - ensures each participant 
    appears in only one split (train, val, OR test).
    
    Parameters:
        input_csv (str) - path to formatted CSV
        train_output (str) - path to save training CSV
        val_output (str) - path to save validation CSV
        test_output (str) - path to save test CSV
        train_ratio (float) - fraction of data for training (as decimal or percentage)
        val_ratio (float) - fraction of data for validation (as decimal or percentage)
        test_ratio (float) - fraction of data for testing (as decimal or percentage)
        label_col (str) - name of label column in the CSV (default: 'label')
        random_seed (int) - random seed for reproducibility
        pid_col (str) - name of participant ID column (default: 'pid', REQUIRED)
    
    Raises:
        ValueError - if pid_col is not found in the data
    """
    # normalize ratios (handle both decimal and percentage formats)
    train_ratio = normalize_ratio(train_ratio)
    val_ratio = normalize_ratio(val_ratio)
    test_ratio = normalize_ratio(test_ratio)
    
    # validate ratios sum to approximately 1.0 (allow small floating point errors)
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(
            f"Ratios must sum to 1.0 (or 100%), but got: "
            f"train={train_ratio:.3f}, val={val_ratio:.3f}, test={test_ratio:.3f} "
            f"(sum={total_ratio:.3f})"
        )
    
    print(f"\nSplitting data into train/val/test sets...")
    print(f"  Ratios: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")
    
    df = pd.read_csv(input_csv)
    
    # require PID column for participant-level splitting
    if pid_col not in df.columns:
        raise ValueError(
            f"ERROR: Participant column '{pid_col}' not found in data. "
            f"Available columns: {list(df.columns)}. "
            f"Participant-level splitting is required to prevent data leakage."
        )
    
    print(f"  Splitting by participant (column: '{pid_col}') to prevent data leakage")
    train_df, val_df, test_df = split_by_participant(
        df,
        pid_col,
        label_col,
        val_ratio,
        test_ratio,
        random_seed
    )
    
    # validate and report participant-level statistics
    train_pids = set(train_df[pid_col].unique())
    val_pids = set(val_df[pid_col].unique())
    test_pids = set(test_df[pid_col].unique())
    
    print(f"\n  Participant distribution:")
    print(f"    Train: {len(train_pids)} unique participants")
    print(f"    Val:   {len(val_pids)} unique participants")
    print(f"    Test:  {len(test_pids)} unique participants")
    
    # verify no overlap
    train_val_overlap = train_pids & val_pids
    train_test_overlap = train_pids & test_pids
    val_test_overlap = val_pids & test_pids
    
    total_overlap = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
    
    if total_overlap > 0:
        print(f"\n  WARNING: Participant overlap detected!")
        print(f"    Train-Val overlap: {len(train_val_overlap)} participants")
        print(f"    Train-Test overlap: {len(train_test_overlap)} participants")
        print(f"    Val-Test overlap: {len(val_test_overlap)} participants")
    else:
        print(f"    No participant overlap - data leakage prevented!")
    
    # save splits
    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)
    test_df.to_csv(test_output, index=False)
    
    print(f"\n  Train set: {len(train_df)} samples → {train_output}")
    print(f"  Val set:   {len(val_df)} samples → {val_output}")
    print(f"  Test set:  {len(test_df)} samples → {test_output}")
    
    # show label distribution
    print(f"\n  Train label distribution ({label_col}):")
    for label, count in train_df[label_col].value_counts().items():
        pct = 100 * count / len(train_df)
        label_str = str(label)  # convert to string in case label is numeric
        print(f"    {label_str}: {count} ({pct:.1f}%)")
    
    print(f"\n  Val label distribution ({label_col}):")
    for label, count in val_df[label_col].value_counts().items():
        pct = 100 * count / len(val_df)
        label_str = str(label)  # convert to string in case label is numeric
        print(f"    {label_str}: {count} ({pct:.1f}%)")
    
    print(f"\n  Test label distribution ({label_col}):")
    for label, count in test_df[label_col].value_counts().items():
        pct = 100 * count / len(test_df)
        label_str = str(label)  # convert to string in case label is numeric
        print(f"    {label_str}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    args = parser.parse_args()

    input_csv = args.input_csv
    spectrogram_dir = args.spectrogram_dir
    output_dir = args.output_dir
    label_col = args.label_col
    create_split = args.create_split
    ratios_str = args.ratios
    drop_incomplete_rows = args.drop_incomplete_rows
    remove_speaker_labels = args.remove_speaker_labels
    
    # create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # generate output CSV filename based on label column name
    output_csv = str(output_path / f"{label_col}.csv")
    
    # prepare data (using default column names for now)
    prepare_pause_data(
        input_csv,
        spectrogram_dir,
        output_csv,
        pause_id_col='record_id',
        pid_col='pid',
        text_before_col='text_before_pause',
        text_after_col='text_after_pause',
        label_col=label_col,
        spectrogram_extension='.png',
        drop_incomplete_rows=drop_incomplete_rows,
        remove_speaker_labels=remove_speaker_labels
    )
    
    # create train/val/test split if requested
    if create_split:
        try:
            train_ratio, val_ratio, test_ratio = parse_ratios(ratios_str)

        except ValueError as error:
            print(f"ERROR: {error}")
            sys.exit(1)
        
        train_output, val_output, test_output = generate_split_filenames(output_csv)
        
        create_train_val_test_split(
            output_csv,
            train_output,
            val_output,
            test_output,
            train_ratio,
            val_ratio,
            test_ratio,
            label_col=label_col,
            pid_col='pid'
        )