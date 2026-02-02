import os
import sys
import torch
import warnings

# Add project root to path
sys.path.append(os.getcwd())

from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor, AutoImageProcessor
from sae.config import get_default_cfg
from sae.training import train_sae
from sae.sae import BatchTopKSAE, JumpReLUSAE, TopKSAE, VanillaSAE
from sae.activation_store import ActivationsStore

# Import OpenVLA classes
try:
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
except ImportError as e:
    print(f"Error importing Prismatic/OpenVLA modules: {e}")
    sys.exit(1)

def train():
    # Register OpenVLA classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Config
    cfg = get_default_cfg()
    
    # Overrides for OpenVLA SAE training
    cfg["model_name"] = "./pretrained_models/LIBERO-Spatial-Pro"
    cfg["is_dataset_on_disk"] = False # Use online feature extraction
    
    # Dataset config (Adjust these paths as per your environment)
    cfg["data_root_dir"] = "datasets/rlds" 
    cfg["dataset_name"] = "libero_spatial_no_noops" 
    
    cfg["layer"] = 12 
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["sae_type"] = "topk" 
    cfg["batch_size"] = 4096 # SAE batch size
    cfg["model_batch_size"] = 4 # OpenVLA batch size (adjust based on VRAM)
    
    print(f"Loading OpenVLA model from {cfg['model_name']}...")
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            cfg["model_name"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(cfg["device"])
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Initializing SAE training for layer {cfg['layer']}...")
    print(f"Dataset: {cfg['dataset_name']} at {cfg['data_root_dir']}")
    
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

    # Activation Store
    print("Initializing ActivationsStore...")
    # This will trigger the OpenVLA dataset loading logic we added to activation_store.py
    try:
        activations_store = ActivationsStore(model, cfg)
    except Exception as e:
        print(f"Failed to initialize ActivationsStore: {e}")
        print("Please ensure your RLDS datasets are correctly placed in 'datasets/rlds' or update cfg['data_root_dir'].")
        return

    # Train
    print("Starting training...")
    # We rely on ActivationsStore to verify features on the first batch
    train_sae(sae, activations_store, model, cfg)

if __name__ == "__main__":
    train()
