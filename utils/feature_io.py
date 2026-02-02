import logging
import os
import tarfile
import io
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import BoundedSemaphore

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Global executor for async IO
# max_workers=1 is usually sufficient to offload the main thread without creating too much contention,
# but can be increased if writing to parallel filesystems.
io_executor = ThreadPoolExecutor(max_workers=1)

# Prevents memory buildup if producer (GPU) is faster than consumer (Disk)
# Limit pending writes to 32 batches.
write_semaphore = BoundedSemaphore(value=32)


class WebDatasetShardWriter:
    def __init__(self, output_dir: Path, max_size: int = 500 * 1024 * 1024, filename_pattern: str = "shard_{:06d}.tar"):
        self.output_dir = Path(output_dir)
        self.max_size = max_size
        self.filename_pattern = filename_pattern
        
        self.shard_count = 0
        self.current_tar = None
        self.current_tar_path = None
        self.current_size = 0
        
        os.makedirs(self.output_dir, exist_ok=True)
        self._open_new_shard()

    def _open_new_shard(self):
        if self.current_tar:
            self.current_tar.close()
        
        self.current_tar_path = self.output_dir / self.filename_pattern.format(self.shard_count)
        self.current_tar = tarfile.open(self.current_tar_path, "w")
        self.current_size = 0
        self.shard_count += 1
        
    def write(self, sample_key: str, data: dict):
        # Serialize data to .npz in memory
        with io.BytesIO() as bio:
            np.savez_compressed(bio, **data)
            bio.seek(0)
            file_bytes = bio.getvalue()
            
        file_size = len(file_bytes)
        
        # Check if we need to rotate shard
        if self.current_size + file_size > self.max_size:
            self._open_new_shard()
            
        # Add to tar
        tar_info = tarfile.TarInfo(name=f"{sample_key}.npz")
        tar_info.size = file_size
        tar_info.mtime = time.time()
        
        self.current_tar.addfile(tar_info, io.BytesIO(file_bytes))
        self.current_size += file_size

    def close(self):
        if self.current_tar:
            self.current_tar.close()


def _save_npz_task(save_path, **kwargs):
    """Helper function to run in background thread."""
    try:
        np.savez_compressed(save_path, **kwargs)
    except Exception as e:
        logger.error(f"Failed to save {save_path}: {e}")


def _release_semaphore(future):
    """Callback to release semaphore when task is done."""
    write_semaphore.release()


def save_eval_features(local_log_dir, task_suite_name, task_description, episode_features, success):
    """
    Saves features collected during an evaluation episode.

    Args:
        local_log_dir (str): Base directory for logs.
        task_suite_name (str): Name of the task suite.
        task_description (str): Description of the specific task.
        episode_features (dict): Dictionary containing lists of 'observations', 'predicted_actions', 'hidden_states', 'steps'.
        success (bool): Whether the episode was successful.
    """
    # Create directory for features
    features_dir = os.path.join(local_log_dir, "features", task_suite_name, task_description.replace(" ", "_"))
    os.makedirs(features_dir, exist_ok=True)

    # Use timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_path = os.path.join(features_dir, f"episode_{timestamp}.npz")

    # Eval features are usually saved at the end of an episode, so blocking is less critical,
    # but we can make it async too for consistency if desired.
    # For now, keeping it sync or using the executor is fine.
    # Let's use the executor to be safe.

    write_semaphore.acquire()
    future = io_executor.submit(
        _save_npz_task,
        save_path,
        observations=episode_features["observations"],
        predicted_actions=episode_features["predicted_actions"],
        hidden_states=episode_features["hidden_states"],
        steps=episode_features["steps"],
        success=success,
    )
    future.add_done_callback(_release_semaphore)
    logger.info(f"Saved features to {save_path}")


def save_train_features(
    save_dir,
    batch_idx,
    device_id,
    hidden_states=None,
    predicted_actions=None,
    ground_truth_actions=None,
    input_ids=None,
    **kwargs,
):
    """
    Saves features collected during a training/validation forward pass.

    Args:
        save_dir (Path): Directory to save the features.
        batch_idx (int): Index of the current batch.
        device_id (int): ID of the device (rank).
        hidden_states (torch.Tensor, optional): Hidden states from the model.
        predicted_actions (torch.Tensor, optional): Actions predicted by the model.
        ground_truth_actions (torch.Tensor, optional): Ground truth actions.
        input_ids (torch.Tensor, optional): Input IDs from the batch.
        **kwargs: Additional features to save.
    """
    os.makedirs(save_dir, exist_ok=True)

    # We use batch_idx and device_id for uniqueness
    filename = f"batch_{batch_idx}_rank_{device_id}.npz"
    save_path = save_dir / filename

    def to_numpy(v):
        if v is None:
            return None
        if hasattr(v, "detach"):
            # Convert bfloat16 to float32 for numpy compatibility
            return v.detach().cpu().float().numpy() if v.dtype == torch.bfloat16 else v.detach().cpu().numpy()
        return v

    save_dict = {}
    if hidden_states is not None:
        save_dict["hidden_states"] = to_numpy(hidden_states)
    if predicted_actions is not None:
        save_dict["predicted_actions"] = to_numpy(predicted_actions)
    if ground_truth_actions is not None:
        save_dict["ground_truth_actions"] = to_numpy(ground_truth_actions)
    if input_ids is not None:
        save_dict["input_ids"] = to_numpy(input_ids)

    # Process extra features
    for k, v in kwargs.items():
        if v is not None:
            save_dict[k] = to_numpy(v)

    # Offload the heavy compression and disk write to a background thread
    write_semaphore.acquire()
    future = io_executor.submit(_save_npz_task, save_path, **save_dict)
    future.add_done_callback(_release_semaphore)
