"""
convert_features_to_wds.py

Converts existing .npz feature datasets (old format) to WebDataset format (.tar shards).
"""

import os
import glob
import numpy as np
import draccus
from dataclasses import dataclass
from pathlib import Path
import tqdm
from utils.feature_io import WebDatasetShardWriter

@dataclass
class ConvertConfig:
    input_dir: str  # Path to directory containing .npz files (e.g. runs/RUN_ID/features)
    output_dir: str # Path to output directory (e.g. runs/RUN_ID/features_wds)
    max_shard_size: int = 500 * 1024 * 1024  # 500 MB

def convert_folder(input_dir: Path, output_dir: Path, max_shard_size: int = 500 * 1024 * 1024):
    print(f"Converting features from {input_dir} to {output_dir}")
    
    # Find all .npz files
    # Old format: batch_X_rank_Y.npz directly in input_dir
    files = sorted(glob.glob(os.path.join(input_dir, "*.npz")))
    if not files:
        print(f"No .npz files found in {input_dir}")
        return

    print(f"Found {len(files)} files to convert.")
    
    # Determine layers from first file
    first_data = np.load(files[0])
    if "task_hidden_states" not in first_data:
        print("Error: 'task_hidden_states' not found in .npz file.")
        return
        
    # Shape check: (batch, layers, patches, dim) or similar
    task_states = first_data["task_hidden_states"]
    num_layers = task_states.shape[1] if task_states.ndim == 4 else 1
    
    print(f"Detected {num_layers} layers.")
    
    # Initialize writers
    writers = {}
    for i in range(num_layers):
        layer_dir = output_dir / f"layer_{i}"
        writers[i] = WebDatasetShardWriter(layer_dir, max_size=max_shard_size)
        
    global_sample_count = 0
    
    for file_path in tqdm.tqdm(files, desc="Converting"):
        try:
            data = np.load(file_path)
            
            # Extract common data
            input_ids = data["input_ids"]
            gt_actions = data["ground_truth_actions"]
            pred_actions = data.get("predicted_actions", None)
            
            # Extract layer data
            task_hidden = data["task_hidden_states"]     # (B, L, N, D)
            action_hidden = data["action_hidden_states"] # (B, L, N, D)
            text_hidden = data["text_hidden_states"]     # (B, L, N, D)
            
            batch_size = input_ids.shape[0]
            
            for b in range(batch_size):
                sample_key = f"{global_sample_count:09d}"
                
                record_base = {
                    "input_ids": input_ids[b],
                    "ground_truth_actions": gt_actions[b],
                }
                if pred_actions is not None and len(pred_actions) > 0:
                    record_base["predicted_actions"] = pred_actions[b]
                    
                for l in range(num_layers):
                    # Check dimensions
                    # If ndim=4, take layer slice
                    t_state = task_hidden[b, l] if task_hidden.ndim == 4 else task_hidden[b]
                    a_state = action_hidden[b, l] if action_hidden.ndim == 4 else action_hidden[b]
                    txt_state = text_hidden[b, l] if text_hidden.ndim == 4 else text_hidden[b]
                    
                    record = record_base.copy()
                    record["task_hidden_states"] = t_state
                    record["action_hidden_states"] = a_state
                    record["text_hidden_states"] = txt_state
                    
                    writers[l].write(sample_key, record)
                
                global_sample_count += 1
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    # Close all writers
    for writer in writers.values():
        writer.close()
        
    print("Conversion complete.")
    
@draccus.wrap()
def convert(cfg: ConvertConfig) -> None:
    convert_folder(Path(cfg.input_dir), Path(cfg.output_dir), cfg.max_shard_size)

if __name__ == "__main__":
    convert()
