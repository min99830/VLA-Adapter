"""
extract_features.py

Extracts features from a trained VLA model using a dataset.
"""

import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Type

import draccus
import torch
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import snapshot_download
from peft import PeftModel, get_peft_model, LoraConfig
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from prismatic.models import load, load_vla
from prismatic.models.action_heads import L1RegressionActionHead, SimpleMLPActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    NUM_TOKENS,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from utils.feature_io import save_train_features

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="Length of IterableDataset")

@dataclass
class ExtractConfig:
    # fmt: off
    config_file_path: str = "openvla/openvla-7b"     # Path to necessary config files of LA-Adapter
    vlm_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)
    use_minivlm: bool = False                        # 
    resum_vla_path: str = "openvla/openvla-7b"       # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of dataset
    run_root_dir: Path = Path("runs")                # Path to directory to store features
    shuffle_buffer_size: int = 1                     # Set to 1 to disable shuffling for matched extraction

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, uses L1 regression action head
    use_simple_mlp_head: bool = False                # If True, uses simple MLP action head
    use_diffusion: bool = False                      # If True, uses diffusion (for structure compatibility)
    use_film: bool = False                           # If True, uses FiLM
    num_images_in_input: int = 1                     # Number of images in the VLA input
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input

    # Extraction configuration
    batch_size: int = 8                              # Batch size per device
    max_steps: int = 200000                          # Max number of steps to extract
    image_aug: bool = False                          # If True, uses image augmentations (usually False for extraction)
    
    # LoRA / Checkpoint
    use_lora: bool = False                           # If True, loads LoRA adapter
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    
    # Full Finetune structure compatibility
    use_fz: bool = False

    # revision version
    use_pro_version: bool = True
    phase: str = "Extraction"
    save_features: bool = True                       # Always True for this script
    # fmt: on

def get_run_id(cfg) -> str:
    """Generates an identifier string for the extraction run."""
    run_id = (
        f"EXTRACT+{cfg.config_file_path.split('/')[-1]}+{cfg.dataset_name}"
        f"--{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}"
    )
    return run_id


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        # try finding without step if generic
        checkpoint_path = os.path.join(path, f"{module_name}--checkpoint.pt")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found.")
        return {}
        
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    
    # Remove DDP prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: ExtractConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
) -> nn.Module:
    module = module_class(**module_args)
    
    # Try to load checkpoint if resum_vla_path looks like a checkpoint directory
    # For extraction, we assume weights are in the path provided or standard locations
    if os.path.isdir(cfg.resum_vla_path):
        # Try to find a checkpoint file
        # This is a simplification; users might need to point to specific checkpoints
        state_dict = load_checkpoint(module_name, cfg.resum_vla_path, 0) # Step 0 as placeholder
        if state_dict:
            module.load_state_dict(state_dict, strict=False)
            print(f"Loaded {module_name} from {cfg.resum_vla_path}")

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)
    return module


@draccus.wrap()
def extract_features(cfg: ExtractConfig) -> None:
    print(f"Extracting features using model `{cfg.config_file_path}` on `{cfg.dataset_name}`")

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)
    
    feature_save_dir = run_dir / "features"

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Model Setup logic similar to finetune.py
    if model_is_on_hf_hub(cfg.config_file_path):
        vla_download_path = snapshot_download(repo_id=cfg.config_file_path)
        cfg.config_file_path = vla_download_path
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if distributed_state.is_main_process:
        update_auto_map(cfg.config_file_path)
        check_model_logic_mismatch(cfg.config_file_path)

    # Load processor and VLA
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    processor = AutoProcessor.from_pretrained(cfg.config_file_path, trust_remote_code=True)

    if cfg.use_minivlm and not os.path.isdir(cfg.vlm_path):
        hf_token = ""
        if "prism-qwen25-extra-dinosiglip-224px-0_5b" in cfg.vlm_path:
             vlm = load(cfg.vlm_path, hf_token=hf_token, load_for_training=True)
        else:
             vlm = load_vla(cfg.vlm_path, hf_token=hf_token, load_for_training=True)
             
        config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16).to(device_id)
        
        # Mapping logic from finetune.py
        replace_map = [
            ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
            ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
            ("llm_backbone.llm", "language_model"),
            ("projector.projector.0", "projector.fc1"),
            ("projector.projector.2", "projector.fc2"),
            ("projector.projector.4", "projector.fc3"),
            ("gamma", "scale_factor"),
        ]

        def rename_state_dict_keys(state_dict, replace_map):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                for old, new in replace_map:
                    if old in new_k:
                        new_k = new_k.replace(old, new)
                new_state_dict[new_k] = v
            return new_state_dict

        old_state_dict = vlm.state_dict()
        RAW_STATE_DICT = rename_state_dict_keys(old_state_dict, replace_map)
        vla.load_state_dict(RAW_STATE_DICT, strict=False)
        del old_state_dict

    else:
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.config_file_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=False,
        ).to(device_id)

    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # Load LoRA if specified
    if cfg.use_lora:
        # If we are extracting from a LoRA checkpoint, we need to load it
        # Assuming cfg.resum_vla_path contains the adapter
        adapter_path = Path(cfg.resum_vla_path) / "lora_adapter"
        if adapter_path.exists():
            print(f"Loading LoRA adapter from {adapter_path}")
            vla = PeftModel.from_pretrained(vla, adapter_path)
        else:
            print("Warning: use_lora is True but adapter path not found. Using base model or initialized LoRA.")
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=2 * cfg.lora_rank,
                lora_dropout=cfg.lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)

    # FiLM setup
    if cfg.use_film:
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        if os.path.isdir(cfg.resum_vla_path):
            state_dict = load_checkpoint("vision_backbone", cfg.resum_vla_path, 0)
            if state_dict:
                vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Additional Projectors/Heads
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.llm_dim if hasattr(vla, "llm_dim") else vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
            to_bf16=True,
        )

    action_head = None
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.llm_dim if hasattr(vla, "llm_dim") else vla.module.llm_dim,
                "hidden_dim": vla.llm_dim if hasattr(vla, "llm_dim") else vla.module.llm_dim,
                "action_dim": ACTION_DIM,
                "use_pro_version": cfg.use_pro_version,
            },
            to_bf16=True,
        )
    elif cfg.use_simple_mlp_head:
        action_head = init_module(
            SimpleMLPActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.llm_dim if hasattr(vla, "llm_dim") else vla.module.llm_dim,
                "hidden_dim": vla.llm_dim if hasattr(vla, "llm_dim") else vla.module.llm_dim,
                "action_dim": ACTION_DIM,
            },
            to_bf16=True,
        )

    # Dataset Setup
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    use_wrist_image = cfg.num_images_in_input > 1
    
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        use_minivlm=cfg.use_minivlm,
    )
    dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes) if hasattr(vla, "config") else (224, 224),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    # Extraction Loop
    vla.eval()
    if action_head: action_head.eval()
    if proprio_projector: proprio_projector.eval()
    
    num_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
    if isinstance(vla, PeftModel):
         # access underlying model if wrapped
         pass

    print(f"Starting extraction... Saving to {feature_save_dir}")
    
    # Use len(dataloader) if possible, otherwise max_steps
    try:
        total_batches = min(len(dataloader), cfg.max_steps)
    except:
        total_batches = cfg.max_steps

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, total=total_batches, desc="Extracting")):
            if batch_idx >= cfg.max_steps:
                break
                
            # Prepare inputs
            input_ids = batch["input_ids"].to(device_id)
            attention_mask = batch["attention_mask"].to(device_id)
            pixel_values = batch["pixel_values"].to(torch.bfloat16).to(device_id)
            
            # Prepare labels if they are not in batch or are incorrect for mask gen
            if "labels" in batch and batch["labels"] is not None:
                labels = batch["labels"].to(device_id)
            else:
                # If labels are missing, fallback to input_ids but this might fail mask logic
                # unless they actually contain action tokens in the right range.
                labels = input_ids.clone()
            
            ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)

            # Prepare proprio
            proprio = None
            if cfg.use_proprio and "proprio" in batch and batch["proprio"] is not None:
                proprio = batch["proprio"].to(device_id).to(torch.bfloat16)

            # Forward Pass
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                    output_hidden_states=True,
                    proprio=proprio,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    use_film=cfg.use_film,
                )

            # Extract Features Logic (Copied/Adapted from finetune.py)
            ground_truth_token_ids = labels[:, 1:].to(device_id)
            current_action_mask = get_current_action_mask(ground_truth_token_ids)
            next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
            
            multi_layer_hidden_states = []
            for item in output.hidden_states[0:]:
                text_hidden_states = item[:, num_patches:-1]
                batch_size_curr = input_ids.shape[0]
                
                actions_hidden_states = (
                    text_hidden_states[current_action_mask | next_actions_mask]
                    .reshape(batch_size_curr, 1, NUM_TOKENS, -1)
                    .to(torch.bfloat16)
                )
                task_latten_states = item[:, :num_patches].reshape(batch_size_curr, 1, num_patches, -1)
                all_hidden_states = torch.cat((task_latten_states, actions_hidden_states), 2)
                multi_layer_hidden_states.append(all_hidden_states)
            
            multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim=1)

            # Predict Actions (if head exists)
            predicted_actions = None
            if action_head:
                predicted_actions = action_head.predict_action(
                    multi_layer_hidden_states,
                    proprio=batch["proprio"].to(device_id).to(torch.bfloat16) if cfg.use_proprio and batch["proprio"] is not None else None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    phase=cfg.phase,
                )
            
            # Save Features
            save_train_features(
                save_dir=feature_save_dir,
                batch_idx=batch_idx,
                device_id=device_id,
                hidden_states=multi_layer_hidden_states,
                predicted_actions=predicted_actions if predicted_actions is not None else torch.tensor([]),
                ground_truth_actions=ground_truth_actions,
                input_ids=batch["input_ids"]
            )
            
if __name__ == "__main__":
    extract_features()
