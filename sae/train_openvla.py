import os
import sys
import warnings

import torch

# Add project root to path
sys.path.append(os.getcwd())

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)

from sae.activation_store import ActivationsStore
from sae.config import get_default_cfg
from sae.sae import BatchTopKSAE, JumpReLUSAE, TopKSAE, VanillaSAE
from sae.training import train_sae

# Import OpenVLA classes and setup patching for ONLINE mode
try:
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import (
        PrismaticImageProcessor,
        PrismaticProcessor,
    )
    from prismatic.models.projectors import ProprioProjector
    from prismatic.vla.constants import PROPRIO_DIM

except ImportError as e:
    print(f"Warning: Could not import some Prismatic modules. Online mode may fail. Error: {e}")


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


def load_vla_adapter(cfg) -> tuple[OpenVLAForActionPrediction, ProprioProjector]:
    # Register OpenVLA classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    print(f"Loading OpenVLA model from {cfg['model_name']}...")
    model = AutoModelForVision2Seq.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        trust_remote_code=False,
    ).to(cfg["device"])
    model.eval()
    model.vision_backbone.set_num_images_in_input(cfg.get("num_images_in_input", 2))

    proprio_projector = ProprioProjector(model.llm_dim, PROPRIO_DIM)
    state_dict = load_checkpoint("proprio_projector", cfg["model_name"], step=0, device=cfg["device"])
    proprio_projector.load_state_dict(state_dict, strict=False)

    return model, proprio_projector

def get_arguments():
    import argparse

    parser = argparse.ArgumentParser(description="Grid Search for SAE Hyperparameters")
    parser.add_argument(
        "--topk",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--expansion_factor",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--target_feature",
        type=str,
        default="text",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=12,
    )
    args = parser.parse_args()
    return args


def run_actual_train():
    cfg = get_default_cfg()
    
    args = get_arguments()

    # --- Mode Toggle ---
    cfg["is_offline"] = False

    # --- Configuration ---
    cfg["model_name"] = "./pretrained_models/LIBERO-Spatial-Pro"
    # cfg["dataset_path"] = "runs/EXTRACT+LIBERO-Spatial-Pro+libero_spatial_no_noops--2026_01_27-17_45_43/features"
    cfg["data_root_dir"] = "data/libero"
    cfg["dataset_name"] = "libero_spatial_no_noops"
    cfg["target_feature"] = args.target_feature
    cfg["layer"] = args.layer
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["batch_size"] = 4096
    cfg["model_batch_size"] = 8
    cfg["sae_type"] = "topk"
    cfg["top_k"] = args.topk
    cfg["act_size"] = 896
    cfg["dict_size"] = 896 * args.expansion_factor
    cfg["num_tokens"] = int(1e8)
    cfg["num_images_in_input"] = 2
    
    cfg["name"] = f"VLA-adapter_{cfg['sae_type']}_layer{cfg['layer']}_feat{cfg['target_feature']}_topk{cfg['top_k']}_exp{args.expansion_factor}"

    print(f"Starting OpenVLA SAE training (Mode: {'OFFLINE' if cfg['is_offline'] else 'ONLINE'})...")
    print(f"Target Feature: {cfg['target_feature']} at Layer {cfg['layer']}")

    model = None
    if not cfg["is_offline"]:
        # Register OpenVLA classes

        print(f"Loading OpenVLA model from {cfg['model_name']}...")
        try:
            model, proprio_projector = load_vla_adapter(cfg)

        except Exception as e:
            print(f"Failed to load model: {e}")
            return

    # Activation Store
    try:
        activations_store = ActivationsStore(
            model, cfg, proprio_projector=proprio_projector if not cfg["is_offline"] else None
        )
    except Exception as e:
        print(f"Failed to initialize ActivationsStore: {e}")
        import traceback

        traceback.print_exc()
        return

    # SAE Init
    if cfg["sae_type"] == "vanilla":
        sae = VanillaSAE(cfg)
    elif cfg["sae_type"] == "topk":
        sae = TopKSAE(cfg)
    elif cfg["sae_type"] == "batchtopk":
        sae = BatchTopKSAE(cfg)
    elif cfg["sae_type"] == "jumprelu":
        sae = JumpReLUSAE(cfg)
    else:
        raise ValueError(f"Unknown SAE type: {cfg['sae_type']}")

    sae = sae.to(cfg["device"])

    # Train
    print("Starting training loop...")
    train_sae(sae, activations_store, model, cfg)


if __name__ == "__main__":
    run_actual_train()
