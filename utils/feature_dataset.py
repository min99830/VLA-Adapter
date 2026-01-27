import glob
import os
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class FeatureDataset(Dataset):
    """
    Dataset for loading extracted VLA features from .npz files.
    """

    def __init__(self, feature_dir, device="cpu", flatten=False):
        """
        Args:
            feature_dir (str): Directory containing .npz feature files.
            device (str): Device to move tensors to.
            flatten (bool): If True, __getitem__ returns individual samples (row-wise)
                            instead of the whole batch stored in the .npz file.
                            Note: This requires pre-scanning files to determine total length, which might be slow.
        """
        self.feature_dir = feature_dir
        self.device = device
        self.flatten = flatten

        # Sort files numerically by batch index and then by rank
        # Assumes format: batch_{int}_rank_{int}.npz
        files = glob.glob(os.path.join(feature_dir, "*.npz"))

        def extract_indices(filename):
            name = os.path.basename(filename)
            batch_match = re.search(r"batch_(\d+)_", name)
            rank_match = re.search(r"rank_(\d+)\.", name)
            batch_idx = int(batch_match.group(1)) if batch_match else -1
            rank_idx = int(rank_match.group(1)) if rank_match else -1
            return (batch_idx, rank_idx)

        self.files = sorted(files, key=extract_indices)

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz files found in {feature_dir}")

        print(f"Found {len(self.files)} feature files in {feature_dir}")

        self.sample_map = []  # [(file_idx, row_idx), ...]
        if self.flatten:
            print("Scanning files to flatten dataset index...")
            for i, f in enumerate(self.files):
                # We need to peek at the file to know how many samples it has
                # Loading 'input_ids' is usually small enough
                try:
                    with np.load(f, allow_pickle=True) as data:
                        n_samples = data["input_ids"].shape[0]
                        for r in range(n_samples):
                            self.sample_map.append((i, r))
                except Exception as e:
                    print(f"Error reading {f}: {e}")
            print(f"Total samples found: {len(self.sample_map)}")

    def __len__(self):
        if self.flatten:
            return len(self.sample_map)
        return len(self.files)

    def __getitem__(self, idx):
        if self.flatten:
            file_idx, row_idx = self.sample_map[idx]
            data = np.load(self.files[file_idx], allow_pickle=True)

            # Helper to extract row and convert
            def get_item(key):
                if key not in data:
                    return None
                val = data[key]
                # If it's an array/tensor, take the row
                if hasattr(val, "ndim") and val.ndim > 0 and val.shape[0] > row_idx:
                    item = val[row_idx]
                    if isinstance(item, np.ndarray):
                        return torch.from_numpy(item).to(self.device)
                    return item
                return val

            sample = {
                "task_hidden_states": get_item("task_hidden_states"),
                "action_hidden_states": get_item("action_hidden_states"),
                "text_hidden_states": get_item("text_hidden_states"),
                "predicted_actions": get_item("predicted_actions"),
                "ground_truth_actions": get_item("ground_truth_actions"),
                "input_ids": get_item("input_ids"),
            }
            # Add other keys if present (for eval features)
            for k in ["observations", "steps", "success"]:
                if k in data:
                    sample[k] = get_item(k)

            return sample

        else:
            # Return whole batch
            data = np.load(self.files[idx], allow_pickle=True)

            # Convert to torch tensors
            sample = {
                "task_hidden_states": (
                    torch.from_numpy(data["task_hidden_states"]).to(self.device)
                    if "task_hidden_states" in data
                    else None
                ),
                "action_hidden_states": (
                    torch.from_numpy(data["action_hidden_states"]).to(self.device)
                    if "action_hidden_states" in data
                    else None
                ),
                "text_hidden_states": (
                    torch.from_numpy(data["text_hidden_states"]).to(self.device)
                    if "text_hidden_states" in data
                    else None
                ),
                # Fallback
                "hidden_states": (
                    torch.from_numpy(data["hidden_states"]).to(self.device) if "hidden_states" in data else None
                ),
                "text": torch.from_numpy(data["text"]).to(self.device) if "text" in data else None,
                "predicted_actions": torch.from_numpy(data["predicted_actions"]).to(self.device),
                "ground_truth_actions": torch.from_numpy(data["ground_truth_actions"]).to(self.device),
                "input_ids": torch.from_numpy(data["input_ids"]).to(self.device),
            }

            # Optional: Handle extra fields if they exist (e.g. from eval features)
            for k in ["observations", "steps", "success"]:
                if k in data:
                    sample[k] = data[k]

            return sample


def get_feature_dataloader(feature_dir, batch_size=1, shuffle=False, num_workers=0, device="cpu", flatten=True):
    """
    Utility to create a DataLoader for features.

    Args:
        flatten (bool): If True, the dataset will be indexed by sample, and the DataLoader
                        will re-batch them according to `batch_size`.
                        If False, the DataLoader yields the original saved chunks (one chunk per 'batch'),
                        ignoring the `batch_size` parameter (effectively batch_size=1 chunk).
    """
    dataset = FeatureDataset(feature_dir, device=device, flatten=flatten)

    # If not flattening, we are loading pre-batched files. usually we want batch_size=1 (1 file per step)
    # If flattening, we construct new batches.
    return DataLoader(dataset, batch_size=batch_size if flatten else 1, shuffle=shuffle, num_workers=num_workers)
