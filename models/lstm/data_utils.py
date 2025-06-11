import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm
import random

# ------------------ TripDataset Class ------------------ #
class TripDataset(Dataset):
    def __init__(self, sequences, labels, lengths, masks):
        """
        Stores the final, fixed-length sub-sequences and their metadata.

        Args:
            sequences (list of np.ndarray): Each is shape (200, num_features).
            labels (list of int): Class labels.
            lengths (list of int): Actual valid length of each sub-sequence.
            masks (list of np.ndarray): Binary mask of shape (200,) for each sub-sequence.
        """
        self.sequences = sequences
        self.labels = labels
        self.lengths = lengths
        self.masks = masks

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)  # (200, num_features)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        length = self.lengths[idx]
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)          # (200,)
        return sequence, label, length, mask

    @staticmethod
    def collate_fn(batch):
        """
        batch is a list of tuples (sequence, label, length, mask).
        """
        sequences, labels, lengths, masks = zip(*batch)

        # If you're guaranteeing everything is 200, you can do this:
        padded_sequences = torch.stack(sequences, dim=0)  # (batch_size, 200, num_features)

        labels = torch.stack(labels)
        lengths = torch.tensor(lengths, dtype=torch.long)
        masks = torch.stack(masks, dim=0)  # (batch_size, 200)

        return padded_sequences, labels, lengths, masks


# ------------------ DataHandler Class ------------------ #
class DataHandler:
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
        seed=42  # For DataLoader and potential extra uses
    ):
        self.data_path = data_path
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.traj_id_column = traj_id_column
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.chunksize = chunksize
        self.seed = seed  # Store seed

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Storage for final splits
        self.train_sequences = []
        self.train_labels = []
        self.train_lengths = []
        self.train_masks = []

        self.val_sequences = []
        self.val_labels = []
        self.val_lengths = []
        self.val_masks = []

        self.test_sequences = []
        self.test_labels = []
        self.test_lengths = []
        self.test_masks = []

        self.unique_traj_ids = set()

    # -------------- Helper function for splitting sub-sequences -------------- #
    def split_into_subsequences(self, arr, chunk_size=200, stride=150):
        subsequences = []
        seq_len = len(arr)
        for start in range(0, seq_len, stride):
            end = start + chunk_size
            sub_arr = arr[start:end]
            actual_len = len(sub_arr)
            if actual_len < chunk_size:
                pad_size = chunk_size - actual_len
                sub_arr = np.pad(
                    sub_arr,
                    ((0, pad_size), (0, 0)),
                    mode='constant', constant_values=0
                )
            mask = np.zeros((chunk_size,), dtype=np.float32)
            mask[:actual_len] = 1.0
            subsequences.append((sub_arr, actual_len, mask))
            if end >= seq_len:
                break
        return subsequences


    # -------------- Everything else -------------- #
    def gather_ids_and_labels(self):
        print("Gathering unique trajectory IDs and labels in one pass...")
        traj_ids = set()
        labels = set()

        for chunk in tqdm(
            pd.read_csv(self.data_path, usecols=[self.traj_id_column, self.target_column],
                        chunksize=self.chunksize),
            desc="Reading IDs & labels"
        ):
            traj_ids.update(chunk[self.traj_id_column].unique())
            labels.update(chunk[self.target_column].unique())

        self.unique_traj_ids = traj_ids
        self.label_encoder.fit(sorted(list(labels)))  # Sort labels for consistency

        print(f"Total unique traj_ids found: {len(self.unique_traj_ids)}")
        print(f"Classes found: {self.label_encoder.classes_}")

    def split_traj_ids(self):
        print("Splitting traj_ids into train, val, and test sets...")
        traj_ids = sorted(list(self.unique_traj_ids))  # Sort for consistency
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

    def normalize_features_first_pass(self):
        print("Partial fitting scaler on training features...")
        usecols = [self.traj_id_column] + self.feature_columns

        for chunk in tqdm(pd.read_csv(self.data_path, usecols=usecols, chunksize=self.chunksize),
                          desc="Reading training features for scaling"):
            train_chunk = chunk[chunk[self.traj_id_column].isin(self.train_ids)]
            if not train_chunk.empty:
                self.scaler.partial_fit(train_chunk[self.feature_columns])
        print("Scaler fitting completed.")

    def create_sequences(self):
        print("Creating trip sequences (fixed length=200, stride=150) from CSV in chunks...")

        for chunk in tqdm(pd.read_csv(self.data_path, chunksize=self.chunksize),
                          desc="Processing & transforming chunks"):

            train_chunk = chunk[chunk[self.traj_id_column].isin(self.train_ids)]
            val_chunk = chunk[chunk[self.traj_id_column].isin(self.val_ids)]
            test_chunk = chunk[chunk[self.traj_id_column].isin(self.test_ids)]

            for split, data_chunk in zip(['train', 'val', 'test'], [train_chunk, val_chunk, test_chunk]):
                if data_chunk.empty:
                    continue

                # Sort by traj_id (and possibly by index if needed) for consistent grouping
                data_chunk = data_chunk.sort_values(by=self.traj_id_column)

                data_chunk['label_encoded'] = self.label_encoder.transform(data_chunk[self.target_column])
                data_chunk[self.feature_columns] = self.scaler.transform(data_chunk[self.feature_columns])

                # group by trajectory with sorted groups
                grouped = data_chunk.groupby(self.traj_id_column, sort=True)
                for _, group in grouped:
                    arr = group[self.feature_columns].values  # shape: (seq_len, num_features)
                    label = group['label_encoded'].iloc[0]

                    # Split into sub-sequences
                    subseqs = self.split_into_subsequences(arr, chunk_size=200, stride=150)

                    for (sub_arr, actual_len, mask) in subseqs:
                        if split == 'train':
                            self.train_sequences.append(sub_arr)
                            self.train_labels.append(label)
                            self.train_lengths.append(actual_len)
                            self.train_masks.append(mask)
                        elif split == 'val':
                            self.val_sequences.append(sub_arr)
                            self.val_labels.append(label)
                            self.val_lengths.append(actual_len)
                            self.val_masks.append(mask)
                        else:  # test
                            self.test_sequences.append(sub_arr)
                            self.test_labels.append(label)
                            self.test_lengths.append(actual_len)
                            self.test_masks.append(mask)

        # print(f"Finished creating sub-sequences -> "
        #       f"Train: {len(self.train_sequences)}, "
        #       f"Val: {len(self.val_sequences)}, "
        #       f"Test: {len(self.test_sequences)}")

    def load_and_process_data(self):
        self.gather_ids_and_labels()
        self.split_traj_ids()
        self.normalize_features_first_pass()
        self.create_sequences()

    def save_preprocessors(self, scaler_path, label_encoder_path):
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, label_encoder_path)
        print(f"Scaler saved to {scaler_path}")
        print(f"Label Encoder saved to {label_encoder_path}")

    def get_dataloaders(self, batch_size, num_workers):
        print("Creating DataLoaders...")

        # Define a generator with a fixed seed for shuffling
        g = torch.Generator()
        g.manual_seed(self.seed)

        def worker_init_fn(worker_id):
            # Each worker gets a deterministically different seed
            worker_seed = self.seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            # For extra caution if using torch random in workers:
            torch.manual_seed(worker_seed)

        # Build dataset objects
        train_dataset = TripDataset(
            self.train_sequences,
            self.train_labels,
            self.train_lengths,
            self.train_masks
        )
        val_dataset = TripDataset(
            self.val_sequences,
            self.val_labels,
            self.val_lengths,
            self.val_masks
        )
        test_dataset = TripDataset(
            self.test_sequences,
            self.test_labels,
            self.test_lengths,
            self.test_masks
        )

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,            # Shuffling with the generator below
            num_workers=num_workers,
            collate_fn=TripDataset.collate_fn,
            generator=g,             # Ensures deterministic shuffling
            worker_init_fn=worker_init_fn,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=TripDataset.collate_fn,
            worker_init_fn=worker_init_fn,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=TripDataset.collate_fn,
            worker_init_fn=worker_init_fn,
            pin_memory=True
        )

        print("DataLoaders created.")
        return {'train': train_loader, 'val': val_loader, 'test': test_loader}
