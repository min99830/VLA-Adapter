import torch


# helper to replace utils.get_act_name if needed
def get_act_name(name, layer):
    return f"blocks.{layer}.{name}"


def get_default_cfg():
    default_cfg = {
        "seed": 49,
        "batch_size": 4096,
        "lr": 3e-4,
        "num_tokens": int(1e9),
        "l1_coeff": 0,
        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 100000,
        "seq_len": 128,
        "dtype": torch.float32,
        "model_name": "openvla-7b",
        "site": "resid_pre",
        "layer": 12,
        "act_size": 896,
        "dict_size": 896 * 32,
        "device": "cuda:0",
        "model_batch_size": 8,
        "num_batches_in_buffer": 10,
        "dataset_path": "path/to/feature",
        "is_feature_dataset": True,
        "wandb_project": "sparse_autoencoders",
        "input_unit_norm": True,
        "perf_log_freq": 1000,
        "sae_type": "topk",
        "checkpoint_freq": 10000,
        "n_batches_to_dead": 5,
        # (Batch)TopKSAE specific
        "top_k": 32,
        "top_k_aux": 512,
        "aux_penalty": 1 / 32,
        # for jumprelu
        "bandwidth": 0.001,
    }
    default_cfg = post_init_cfg(default_cfg)
    return default_cfg


def post_init_cfg(cfg):
    if not cfg.get("is_feature_dataset", False):
        cfg["hook_point"] = get_act_name(cfg["site"], cfg["layer"])
    else:
        cfg["hook_point"] = f"layer_{cfg['layer']}"

    cfg["name"] = (
        f"{cfg['model_name']}_{cfg['hook_point']}_{cfg['dict_size']}_{cfg['sae_type']}_{cfg['top_k']}_{cfg['lr']}"
    )
    return cfg
