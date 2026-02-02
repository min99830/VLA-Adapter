# %%
from .activation_store import ActivationsStore
from .config import get_default_cfg, post_init_cfg
from .training import train_sae
from .sae import BatchTopKSAE, JumpReLUSAE, TopKSAE, VanillaSAE
import torch
import os

def run_training():
    os.environ["WANDB_MODE"] = "offline"
    # Load default config (which is already adapted for VLA features in config.py)
    cfg = get_default_cfg()
    
    # Optional: Override some settings here if needed
    # cfg["l1_coeff"] = 0.0
    # cfg["top_k"] = 32
    
    # Ensure post_init is called (get_default_cfg calls it, but if we change things we might need to call it again? 
    # Actually get_default_cfg calls it. If we modify keys used in post_init, we should re-call it or set them manually.
    # But cfg["name"] is already set. Let's just trust get_default_cfg for now.
    
    print(f"Running training for {cfg['name']}")
    print(f"Dataset path: {cfg['dataset_path']}")
    
    # Model is None for disk-based features
    model = None
    
    # If we weren't using disk features, we'd load the model here:
    if not cfg.get("is_dataset_on_disk", False):
        try:
            from transformer_lens import HookedTransformer
            model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
        except ImportError:
            print("TransformerLens not installed, but required for model-based training.")
            return

    # Initialize SAE
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

    # Initialize Activation Store
    activations_store = ActivationsStore(model, cfg)

    # Train
    train_sae(sae, activations_store, model, cfg)

if __name__ == "__main__":
    run_training()