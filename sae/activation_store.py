import glob
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Import constants directly
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
)
from utils.feature_dataset import FeatureDataset

try:
    from transformers import AutoProcessor
    from transformers.modeling_outputs import CausalLMOutputWithPast

    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
except ImportError:
    OpenVLAForActionPrediction = None


# Local implementation of mask functions to fix cumsum error AND shape mismatch
def get_current_action_mask(token_ids):
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids)

    # We need to return a mask with exactly NUM_ACTIONS_CHUNK True values per sequence
    # to match the model's action_queries shape (when using Libero constants).

    mask = torch.zeros_like(token_ids, dtype=torch.bool)

    # Iterate over batch to handle each sequence
    # This is a bit slow but safe given the mismatch issues
    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(0)

    for i in range(token_ids.shape[0]):
        # Find start of action sequence (first non-ignore token)
        # We assume actions are at the end.
        # token_ids is (seq_len)

        # Find indices where it's NOT ignore index
        valid_indices = (token_ids[i] != IGNORE_INDEX).nonzero(as_tuple=True)[0]

        if len(valid_indices) > 0:
            start_idx = valid_indices[0]
            # Mark NUM_ACTIONS_CHUNK tokens starting from start_idx
            # We assume sequence is long enough. If not, clip to end.
            end_idx = min(start_idx + NUM_ACTIONS_CHUNK, token_ids.shape[1])
            mask[i, start_idx:end_idx] = True

            # If we couldn't fit 8 tokens (e.g. at end of seq), we might still have a mismatch.
            # But RLDSBatchTransform usually pads to 64, so it should be fine.
        else:
            # Fallback if no valid tokens found: just mark the last N tokens?
            # This avoids 0-size mask
            mask[i, -NUM_ACTIONS_CHUNK:] = True

    return mask


def get_next_actions_mask(token_ids):
    # This usually masks "future" actions.
    # If we redefined current to be the chunk of 8, next might be empty or everything after.
    # For SAE extraction on 'action' or 'text', we want consistency.

    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids)

    # Use standard logic for "next" or just inverse?
    # extract_features.py uses: action_mask = current | next
    # If current covers the whole action chunk (8), then next should probably be empty or everything after.

    # Let's keep the logic consistent with current mask: everything AFTER the chunk.

    mask = torch.zeros_like(token_ids, dtype=torch.bool)
    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(0)

    for i in range(token_ids.shape[0]):
        valid_indices = (token_ids[i] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
        if len(valid_indices) > 0:
            start_idx = valid_indices[0]
            chunk_end = min(start_idx + NUM_ACTIONS_CHUNK, token_ids.shape[1])
            # Next actions are everything after the chunk
            mask[i, chunk_end:] = True

    return mask


class ActivationsStore:
    """
    ActivationsStore dedicated to OpenVLA.
    Supports two modes:
    1. Online: Extracts features from a loaded model using RLDS dataset.
    2. Offline: Loads pre-extracted features from .npz files using FeatureDataset.
    """

    def __init__(
        self,
        model,  # Used for Online mode
        cfg: dict,
        proprio_projector=None,
    ):
        self.cfg = cfg
        self.device = cfg["device"]
        self.is_offline = cfg.get("is_offline", False)
        self.target_feature = cfg.get("target_feature", "task_hidden_states")
        self.verified_features = False

        if self.is_offline:
            print(f"Initializing ActivationsStore in OFFLINE mode (FeatureDataset)...")
            print(f"Dataset path: {cfg['dataset_path']}")
            self.dataset = FeatureDataset(cfg["dataset_path"], device="cpu", flatten=True)
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=cfg["batch_size"],
                shuffle=True,
                num_workers=cfg.get("num_workers", 4),
                pin_memory=True if self.device != "cpu" else False,
            )
            self.dataloader_iter = iter(self.dataloader)
        else:
            if proprio_projector is None:
                raise ValueError("proprio_projector must be provided for Online mode.")

            self.proprio_projector = proprio_projector
            print(f"Initializing ActivationsStore in ONLINE mode (Model-based)...")
            self.model = model
            self.model_batch_size = cfg.get("model_batch_size", 1)

            # OpenVLA Dataset Setup
            processor = AutoProcessor.from_pretrained(cfg["model_name"], trust_remote_code=True)
            action_tokenizer = ActionTokenizer(processor.tokenizer)

            batch_transform = RLDSBatchTransform(
                action_tokenizer,
                processor.tokenizer,
                image_transform=processor.image_processor.apply_transform,
                prompt_builder_fn=PurePromptBuilder,
                use_wrist_image=cfg.get("num_images_in_input", 1) > 1,
                use_proprio=True,
                use_minivlm=True,
            )

            data_root_dir = cfg.get("data_root_dir", "data/libero")
            dataset_name = cfg.get("dataset_name", "libero_spatial_no_noops")

            print(f"Loading RLDS dataset: {dataset_name} from {data_root_dir}")
            dataset = RLDSDataset(
                data_root_dir,
                dataset_name,
                batch_transform,
                resize_resolution=(224, 224),
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

            # For buffer-based yield
            self.num_batches_in_buffer = cfg.get("num_batches_in_buffer", 10)
            self.activation_buffer = self._fill_online_buffer()
            self.activation_dataloader = DataLoader(
                TensorDataset(self.activation_buffer), batch_size=cfg["batch_size"], shuffle=True
            )
            self.activation_dataloader_iter = iter(self.activation_dataloader)

    def _get_online_activations(self, batch):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            pixel_values = batch["pixel_values"].to(torch.bfloat16).to(self.device)
            labels = batch["labels"].to(self.device) if "labels" in batch else input_ids.clone()
            proprio = batch["proprio"].to(self.device).to(torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                    output_hidden_states=True,
                    proprio=proprio,
                    proprio_projector=self.proprio_projector,
                    use_film=False,
                )

            layer_idx = self.cfg["layer"]
            item = output.hidden_states[layer_idx + 1]  # (batch, seq, dim)

            if self.target_feature == "all":
                activations = item
            else:
                num_patches = (
                    self.model.vision_backbone.get_num_patches() * self.model.vision_backbone.get_num_images_in_input()
                )
                task_states = item[:, :num_patches]
                text_states = item[:, num_patches:-1]

                if "task" in self.target_feature:
                    activations = task_states
                elif "action" in self.target_feature or "text" in self.target_feature:
                    ground_truth_token_ids = labels[:, 1:].to(self.device)
                    current_action_mask = get_current_action_mask(ground_truth_token_ids)
                    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
                    action_mask = current_action_mask | next_actions_mask

                    if "action" in self.target_feature:
                        activations = text_states[action_mask]
                    else:  # text
                        activations = text_states[~action_mask]
                else:
                    activations = item

            if not self.verified_features:
                self._verify(activations)

            return activations.reshape(-1, activations.shape[-1])

    def _fill_online_buffer(self):
        all_activations = []
        target_size = self.cfg["batch_size"] * self.num_batches_in_buffer
        while len(all_activations) * 1024 < target_size:  # Heuristic
            try:
                batch = next(self.vla_iterator)
                all_activations.append(self._get_online_activations(batch))
            except StopIteration:
                self.vla_iterator = iter(self.vla_dataloader)
        return torch.cat(all_activations, dim=0)

    def _verify(self, activations):
        print(f"\n[Feature Verification]")
        print(f"Target: {self.target_feature}")
        print(f"Shape: {activations.shape}")
        print(f"Mean: {activations.float().mean().item():.4f}")
        print(f"Std: {activations.float().std().item():.4f}")
        print("-" * 20 + "\n")
        self.verified_features = True

    def next_batch(self):
        if self.is_offline:
            try:
                batch_data = next(self.dataloader_iter)
            except StopIteration:
                self.dataloader_iter = iter(self.dataloader)
                batch_data = next(self.dataloader_iter)

            activations = batch_data[self.target_feature].to(self.device, dtype=self.cfg.get("dtype", torch.float32))
            if activations.ndim == 3:
                activations = activations.reshape(-1, activations.shape[-1])
            if not self.verified_features:
                self._verify(activations)
            return activations
        else:
            try:
                return next(self.activation_dataloader_iter)[0]
            except StopIteration:
                self.activation_buffer = self._fill_online_buffer()
                self.activation_dataloader = DataLoader(
                    TensorDataset(self.activation_buffer), batch_size=self.cfg["batch_size"], shuffle=True
                )
                self.activation_dataloader_iter = iter(self.activation_dataloader)
                return next(self.activation_dataloader_iter)[0]
