import glob
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from transformers import AutoProcessor
except ImportError:
    OpenVLAForActionPrediction = None


class ActivationsStore:
    def __init__(
        self,
        model,
        cfg: dict,
    ):
        self.cfg = cfg
        self.device = cfg["device"]
        self.model_batch_size = cfg["model_batch_size"]
        self.num_batches_in_buffer = cfg["num_batches_in_buffer"]
        self.is_dataset_on_disk = cfg.get("is_dataset_on_disk", False)
        self.verified_features = False

        if self.is_dataset_on_disk:
            self.dataset_path = cfg["dataset_path"]
            
            # Check for WebDataset (.tar) or original (.npz)
            self.tar_files = sorted(glob.glob(os.path.join(self.dataset_path, "**", "*.tar"), recursive=True))
            if not self.tar_files:
                # Try finding one level deeper if dataset_path is root and not layer specific
                # But typically cfg['layer'] should define the path?
                # Let's assume dataset_path might be root, and we need layer_{layer}
                layer_path = os.path.join(self.dataset_path, f"layer_{cfg.get('layer', 0)}")
                self.tar_files = sorted(glob.glob(os.path.join(layer_path, "*.tar")))
            
            if self.tar_files:
                print(f"Found {len(self.tar_files)} WebDataset shards. Using 'datasets' streaming.")
                from datasets import load_dataset
                self.is_webdataset = True
                
                # Load dataset
                # We need to handle the case where we might want to shuffle
                ds = load_dataset("webdataset", data_files=self.tar_files, split="train", streaming=True)
                # Shuffle with a buffer
                self.dataset = iter(ds.shuffle(buffer_size=cfg.get("shuffle_buffer_size", 1000)))
                
            else:
                self.is_webdataset = False
                self.files = sorted(glob.glob(os.path.join(self.dataset_path, "*.npz")))
                if not self.files:
                    raise ValueError(f"No .tar or .npz files found in {self.dataset_path}")
                print(f"Found {len(self.files)} feature files.")
                self.file_idx = 0
                # np.random.shuffle(self.files)
        else:
            self.model = model
            if OpenVLAForActionPrediction is not None and isinstance(model, OpenVLAForActionPrediction):
                self.is_openvla = True
                print("Initializing ActivationsStore for OpenVLA...")
                
                # OpenVLA Dataset Setup
                processor = AutoProcessor.from_pretrained(cfg["model_name"], trust_remote_code=True)
                action_tokenizer = ActionTokenizer(processor.tokenizer)
                
                batch_transform = RLDSBatchTransform(
                    action_tokenizer,
                    processor.tokenizer,
                    image_transform=processor.image_processor.apply_transform,
                    prompt_builder_fn=PurePromptBuilder,
                )
                
                # Use dataset path from cfg or default
                data_root_dir = cfg.get("data_root_dir", "datasets/rlds")
                dataset_name = cfg.get("dataset_name", "libero_spatial_no_noops")
                
                print(f"Loading RLDS dataset: {dataset_name} from {data_root_dir}")
                dataset = RLDSDataset(
                    data_root_dir,
                    dataset_name,
                    batch_transform,
                    resize_resolution=(224, 224), # Standard OpenVLA resolution
                    shuffle_buffer_size=cfg.get("shuffle_buffer_size", 1000),
                    image_aug=False,
                )

                collator = PaddedCollatorForActionPrediction(
                    processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
                )
                
                self.vla_dataloader = DataLoader(
                    dataset,
                    batch_size=self.model_batch_size,
                    sampler=None,
                    collate_fn=collator,
                    num_workers=4,
                    prefetch_factor=2,
                    pin_memory=True,
                )
                self.vla_iterator = iter(self.vla_dataloader)
                self.hook_point = f"layers.{cfg['layer']}" # Placeholder, actual access via hidden_states
                
            else:
                self.is_openvla = False
                from datasets import load_dataset

                self.dataset = iter(load_dataset(cfg["dataset_path"], split="train", streaming=True))
                self.hook_point = cfg["hook_point"]
                self.context_size = min(cfg["seq_len"], model.cfg.n_ctx)
                self.tokens_column = self._get_tokens_column()
                self.tokenizer = model.tokenizer

        self.activation_buffer = self._fill_buffer()
        self.dataloader = self._get_dataloader()
        self.dataloader_iter = iter(self.dataloader)

    def _get_tokens_column(self):
        # Only relevant for text dataset
        if self.is_dataset_on_disk:
            return None 
            
        sample = next(self.dataset)
        if "tokens" in sample:
            return "tokens"
        elif "input_ids" in sample:
            return "input_ids"
        elif "text" in sample:
            return "text"
        else:
            raise ValueError("Dataset must have a 'tokens', 'input_ids', or 'text' column.")

    def get_batch_tokens(self):
        # Not used for OpenVLA
        all_tokens = []
        while len(all_tokens) < self.model_batch_size * self.context_size:
            batch = next(self.dataset)
            if self.tokens_column == "text":
                tokens = self.model.to_tokens(
                    batch["text"], truncate=True, move_to_device=True, prepend_bos=True
                ).squeeze(0)
            else:
                tokens = batch[self.tokens_column]
            all_tokens.extend(tokens)
        token_tensor = torch.tensor(all_tokens, dtype=torch.long, device=self.device)[
            : self.model_batch_size * self.context_size
        ]
        return token_tensor.view(self.model_batch_size, self.context_size)

    def get_activations(self, batch):
        if self.is_dataset_on_disk:
             raise NotImplementedError("Use _load_next_file_activations for disk datasets")
             
        if getattr(self, "is_openvla", False):
            # OpenVLA Feature Extraction
            with torch.no_grad():
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                pixel_values = batch["pixel_values"].to(torch.bfloat16).to(self.device)
                
                # Run model
                # We need to use autocast for bfloat16/mixed precision usually
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        output_hidden_states=True,
                    )
                
                # Extract hidden states from specified layer
                # output.hidden_states is a tuple of (batch, seq_len, dim)
                # Index 0 is embeddings, 1 is layer 1, etc.
                # Assuming cfg['layer'] is 0-indexed corresponding to transformer layers.
                # Usually hidden_states[i] is output of layer i (or i-1 depending on norm location).
                # Let's assume hidden_states[1:] are the layers.
                
                layer_idx = self.cfg["layer"]
                # OpenVLA/Llama structure: hidden_states contains (embeddings, layer_0, ..., layer_N)
                # So layer_idx 0 (first layer) is at index 1.
                
                target_states = output.hidden_states[layer_idx + 1] # (batch, seq, dim)
                
                # We might want to filter tokens (e.g., only action tokens, or all tokens)
                # For now, let's take all tokens.
                activations = target_states
                
                # Verify features
                if not self.verified_features:
                    print(f"\n[Feature Verification]")
                    print(f"Layer: {layer_idx}")
                    print(f"Shape: {activations.shape}")
                    print(f"Mean: {activations.float().mean().item():.4f}")
                    print(f"Std: {activations.float().std().item():.4f}")
                    print(f"Min: {activations.float().min().item():.4f}")
                    print(f"Max: {activations.float().max().item():.4f}")
                    print("-" * 20 + "\n")
                    self.verified_features = True
                
                return activations.reshape(-1, self.cfg["act_size"])

        else:
            # TransformerLens
            batch_tokens = batch # Input is tokens
            with torch.no_grad():
                _, cache = self.model.run_with_cache(
                    batch_tokens,
                    names_filter=[self.hook_point],
                    stop_at_layer=self.cfg["layer"] + 1,
                )
            return cache[self.hook_point]

    def _load_next_file_activations(self):
        # Only used for .npz files
        if self.file_idx >= len(self.files):
            self.file_idx = 0
            np.random.shuffle(self.files)

        file_path = self.files[self.file_idx]
        self.file_idx += 1

        try:
            data = np.load(file_path)
            # Shape: (batch, layers, tokens, dim)

            # Select layer
            layer_idx = self.cfg["layer"]
            target_feature_name = self.cfg.get("target_features", "task_hidden_states")

            hidden_states = data[target_feature_name]
            # Old format check: (batch, layers, tokens, dim)
            if hidden_states.ndim == 4:
                if layer_idx >= hidden_states.shape[1]:
                    raise ValueError(f"Requested layer {layer_idx} but file only has {hidden_states.shape[1]} layers.")
                activations = hidden_states[:, layer_idx, :, :]  # (batch, tokens, dim)
            else:
                # Assume it might be (batch, tokens, dim) if pre-filtered?
                activations = hidden_states

            # Flatten to (batch*tokens, dim)
            activations = activations.reshape(-1, activations.shape[-1])

            return torch.from_numpy(activations).to(self.device, dtype=self.cfg["dtype"])

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def _fill_buffer(self):
        all_activations = []
        current_size = 0
        target_size = self.cfg["batch_size"] * self.num_batches_in_buffer  # Heuristic, or use cfg settings

        # For disk loading, we just load files until we have enough
        if self.is_dataset_on_disk:
            if self.is_webdataset:
                # Consume from stream until we have enough activations
                while len(all_activations) * self.cfg.get("act_size", 4096) < target_size * 2048: # Estimation
                    try:
                        sample = next(self.dataset)
                        # 'npz' dict is automatically unpacked by datasets if structure matches
                        # sample is a dict. keys depend on what's in npz.
                        # If using 'datasets' with 'webdataset' script, sample['npz'] contains data
                        
                        data = sample['npz'] if 'npz' in sample else sample
                        target_feature_name = self.cfg.get("target_features", "task_hidden_states")
                        
                        if target_feature_name not in data:
                            continue
                            
                        # data[key] is list of lists (if loaded by datasets) or numpy array
                        acts = np.array(data[target_feature_name])
                        
                        # Shape: (tokens, dim) or (batch, tokens, dim)?
                        # In my writer: (patches, dim) per record. But record is 1 sample from batch.
                        # Writer: record["task_hidden_states"] = task_state (numpy)
                        # So it should be (tokens, dim).
                        
                        if acts.ndim == 2:
                            acts = acts.reshape(-1, acts.shape[-1])
                        
                        tensor_acts = torch.from_numpy(acts).to(self.device, dtype=self.cfg["dtype"])
                        all_activations.append(tensor_acts)
                        
                        if len(all_activations) > self.num_batches_in_buffer * 10: # Safety break
                            break
                            
                    except StopIteration:
                        # Restart iterator
                        print("Dataset exhausted, restarting iterator.")
                        from datasets import load_dataset
                        ds = load_dataset("webdataset", data_files=self.tar_files, split="train", streaming=True)
                        self.dataset = iter(ds.shuffle(buffer_size=self.cfg.get("shuffle_buffer_size", 1000)))
                        continue
            else:
                while len(all_activations) < self.num_batches_in_buffer:  # Load at least 'num_batches_in_buffer' files
                    acts = self._load_next_file_activations()
                    if acts is not None:
                        all_activations.append(acts)
        else:
            if getattr(self, "is_openvla", False):
                # OpenVLA logic
                while len(all_activations) * self.cfg.get("act_size", 4096) < target_size: # Approximate check
                     try:
                        batch = next(self.vla_iterator)
                        acts = self.get_activations(batch)
                        all_activations.append(acts)
                        
                        if len(all_activations) * self.cfg.get("act_size", 4096) > target_size:
                             break
                     except StopIteration:
                        print("Dataset iterator exhausted, restarting.")
                        self.vla_iterator = iter(self.vla_dataloader)
                        
            else:
                for _ in range(self.num_batches_in_buffer):
                    batch_tokens = self.get_batch_tokens()
                    activations = self.get_activations(batch_tokens).reshape(-1, self.cfg["act_size"])
                    all_activations.append(activations)

        return torch.cat(all_activations, dim=0)

    def _get_dataloader(self):
        return DataLoader(TensorDataset(self.activation_buffer), batch_size=self.cfg["batch_size"], shuffle=True)

    def next_batch(self):
        try:
            return next(self.dataloader_iter)[0]
        except (StopIteration, AttributeError):
            self.activation_buffer = self._fill_buffer()
            self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            return next(self.dataloader_iter)[0]
