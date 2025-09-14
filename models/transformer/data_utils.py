# data_utils.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import random 

# --- add near the top if not present ---
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

def _rank() -> int:
    return int(os.environ.get("RANK", "0"))

# --- add inside DataProcessor class (methods) ---

    def _worker_init_fn(self, worker_id: int, base_seed: int):
        """
        Deterministic seeding per worker and per rank.
        """
        seed = base_seed + 1000 * _rank() + worker_id
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def make_dataloader(self, dataset, batch_size: int, num_workers: int,
                        shuffle: bool, base_seed: int, sampler=None):
        """
        Centralized DataLoader builder with pinned memory and deterministic worker seeding.
        If `sampler` is provided, `shuffle` is ignored.
        """
        return DataLoader(
            dataset,
            batch_size=max(1, batch_size),
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=lambda wid: self._worker_init_fn(wid, base_seed),
        )

    def get_dataloaders(self, batch_size: int, num_workers: int,
                        ddp: bool = False, seed: int = 42):
        """
        Convenience constructor for train/val/test loaders. If `ddp=True`,
        attaches DistributedSampler to each split.
        """
        # Build datasets from the sequences/masks/labels already populated
        train_ds = TripDataset(self.train_sequences, self.train_labels, self.train_masks)
        val_ds   = TripDataset(self.val_sequences,   self.val_labels,   self.val_masks)
        test_ds  = TripDataset(self.test_sequences,  self.test_labels,  self.test_masks)

        if ddp and _world_size() > 1:
            train_samp = DistributedSampler(train_ds, shuffle=True,  drop_last=False)
            val_samp   = DistributedSampler(val_ds,   shuffle=False, drop_last=False)
            test_samp  = DistributedSampler(test_ds,  shuffle=False, drop_last=False)
        else:
            train_samp = val_samp = test_samp = None

        train_loader = self.make_dataloader(train_ds, batch_size, num_workers,
                                            shuffle=True, base_seed=seed, sampler=train_samp)
        val_loader   = self.make_dataloader(val_ds,   batch_size, num_workers,
                                            shuffle=False, base_seed=seed, sampler=val_samp)
        test_loader  = self.make_dataloader(test_ds,  batch_size, num_workers,
                                            shuffle=False, base_seed=seed, sampler=test_samp)

        return {"train": train_loader, "val": val_loader, "test": test_loader}

class TripDataset(Dataset):
    """
    Handles storing sequences and labels (and/or masks, lengths, etc.).
    """
    def __init__(self, sequences, labels, masks=None):
        self.sequences = sequences
        self.labels = labels
        
        # If masks exist, store them; otherwise store None or handle differently
        self.masks = masks

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.masks is not None:
            mask = torch.tensor(self.masks[idx], dtype=torch.bool)
            return sequence, mask, label
        else:
            return sequence, label


class DataProcessor:
    """
    Handles reading CSVs, chunking, train/val/test splitting by traj_id,
    partial fitting of scaler, label encoding, and generating sliding-window sequences.
    """
    def __init__(
        self,
        data_path,
        feature_columns,
        target_column,
        traj_id_column='traj_id',
        test_size=0.15,
        val_size=0.15,
        random_state=42,
        chunksize=10**6,
        window_size=100,
        stride=50
    ):
        self.data_path = data_path
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.traj_id_column = traj_id_column
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.chunksize = chunksize
        self.window_size = window_size
        self.stride = stride
        
        # Scaler & encoder
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Store sets of IDs
        self.unique_traj_ids = set()
        self.train_ids = set()
        self.val_ids = set()
        self.test_ids = set()

        # Sequence placeholders
        self.train_sequences = []
        self.train_labels = []
        self.train_masks = []

        self.val_sequences = []
        self.val_labels = []
        self.val_masks = []

        self.test_sequences = []
        self.test_labels = []
        self.test_masks = []

        self.skipped_sequences = 0

    def get_unique_traj_ids(self):
        """Gather unique trajectory IDs in a first pass."""
        print("Extracting unique trajectory IDs...")
        for chunk in tqdm(
            pd.read_csv(self.data_path, usecols=[self.traj_id_column], chunksize=self.chunksize),
            desc="Reading traj_ids"
        ):
            self.unique_traj_ids.update(chunk[self.traj_id_column].unique())
        print(f"Total unique traj_ids found: {len(self.unique_traj_ids)}")

    def split_traj_ids(self):
        """Split unique traj_ids into train/val/test based on self.test_size and self.val_size."""
        print("Splitting traj_ids into train, validation, and test sets...")
        traj_ids = sorted(list(self.unique_traj_ids))  # Ensure consistent ordering

        train_ids, temp_ids = train_test_split(
            traj_ids,
            test_size=(self.val_size + self.test_size),
            random_state=self.random_state,
            shuffle=True
        )
        val_ratio_adjusted = self.val_size / (self.val_size + self.test_size)
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=(1 - val_ratio_adjusted),
            random_state=self.random_state,
            shuffle=True
        )
        self.train_ids = set(train_ids)
        self.val_ids = set(val_ids)
        self.test_ids = set(test_ids)

        print(f"Train: {len(self.train_ids)}, Val: {len(self.val_ids)}, Test: {len(self.test_ids)}")

    def preprocess_label_encoder(self):
        """Fit label encoder on all labels from the entire dataset in chunks."""
        print("Fitting label encoder on target column...")
        labels = set()
        for chunk in tqdm(
            pd.read_csv(self.data_path, usecols=[self.target_column], chunksize=self.chunksize),
            desc="Reading labels for encoding"
        ):
            labels.update(chunk[self.target_column].unique())
        self.label_encoder.fit(list(labels))
        print(f"Classes found: {self.label_encoder.classes_}")

    def normalize_features_first_pass(self):
        """Partially fit the StandardScaler on training data only, to avoid data leakage."""
        print("Fitting scaler on training features...")
        cols = [self.traj_id_column] + self.feature_columns
        for chunk in tqdm(
            pd.read_csv(self.data_path, usecols=cols, chunksize=self.chunksize),
            desc="Reading training features for scaling"
        ):
            train_chunk = chunk[chunk[self.traj_id_column].isin(self.train_ids)].copy()
            if not train_chunk.empty:
                self.scaler.partial_fit(train_chunk[self.feature_columns])
        print("Scaler fitting completed.")

    def create_sliding_windows(self, sequence, label):
        """Split the given sequence into overlapping windows with stride and store masks if needed."""
        windows = []
        seq_len = sequence.shape[0]

        if seq_len == 0:
            self.skipped_sequences += 1
            return windows

        for start in range(0, seq_len, self.stride):
            end = start + self.window_size
            window_seq = sequence[start:end]
            actual_len = window_seq.shape[0]

            # If shorter than window_size, pad and build mask
            if actual_len < self.window_size:
                padded_window = np.zeros((self.window_size, window_seq.shape[1]))
                padded_window[:actual_len] = window_seq
                window_seq = padded_window

                window_mask = np.zeros((self.window_size,), dtype=bool)
                window_mask[actual_len:] = True
            else:
                window_mask = np.zeros((self.window_size,), dtype=bool)

            windows.append((window_seq, window_mask, label))

            if end >= seq_len:
                break

        return windows

    def create_sequences(self):
        """Second pass over CSV to transform data (scale/encode) and build windows."""
        print("Creating sequences in sliding windows...")
        for chunk in tqdm(pd.read_csv(self.data_path, chunksize=self.chunksize), desc="Processing chunks"):
            train_chunk = chunk[chunk[self.traj_id_column].isin(self.train_ids)].copy()
            val_chunk   = chunk[chunk[self.traj_id_column].isin(self.val_ids)].copy()
            test_chunk  = chunk[chunk[self.traj_id_column].isin(self.test_ids)].copy()

            for split, df in zip(['train', 'val', 'test'], [train_chunk, val_chunk, test_chunk]):
                if df.empty:
                    continue

                # Encode labels
                df['label_encoded'] = self.label_encoder.transform(df[self.target_column])

                # Scale features
                df[self.feature_columns] = self.scaler.transform(df[self.feature_columns])

                grouped = df.groupby(self.traj_id_column)
                for _, group in grouped:
                    seq = group[self.feature_columns].values
                    label = group['label_encoded'].iloc[0]
                    windows = self.create_sliding_windows(seq, label)

                    if split == 'train':
                        self.train_sequences.extend([w[0] for w in windows])
                        self.train_masks.extend([w[1] for w in windows])
                        self.train_labels.extend([w[2] for w in windows])
                    elif split == 'val':
                        self.val_sequences.extend([w[0] for w in windows])
                        self.val_masks.extend([w[1] for w in windows])
                        self.val_labels.extend([w[2] for w in windows])
                    else:
                        self.test_sequences.extend([w[0] for w in windows])
                        self.test_masks.extend([w[1] for w in windows])
                        self.test_labels.extend([w[2] for w in windows])

        print(f"Train: {len(self.train_sequences)}, Val: {len(self.val_sequences)}, Test: {len(self.test_sequences)}")
        print(f"Skipped sequences (zero length): {self.skipped_sequences}")

    def load_and_process_data(self):
        """
        Wrapper that calls everything:
        1) get_unique_traj_ids
        2) split_traj_ids
        3) preprocess_label_encoder
        4) normalize_features_first_pass
        5) create_sequences
        """
        self.get_unique_traj_ids()
        self.split_traj_ids()
        self.preprocess_label_encoder()
        self.normalize_features_first_pass()
        self.create_sequences()

    def save_preprocessors(self, scaler_path, label_encoder_path):
        import joblib
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, label_encoder_path)
        print(f"Scaler saved to {scaler_path}")
        print(f"Label encoder saved to {label_encoder_path}")
