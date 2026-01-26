import os
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def save_eval_features(
    local_log_dir, 
    task_suite_name, 
    task_description, 
    episode_features, 
    success
):
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
    
    np.savez_compressed(
        save_path, 
        observations=episode_features["observations"],
        predicted_actions=episode_features["predicted_actions"],
        hidden_states=episode_features["hidden_states"],
        steps=episode_features["steps"],
        success=success
    )
    logger.info(f"Saved features to {save_path}")


def save_train_features(
    save_dir, 
    batch_idx, 
    device_id, 
    hidden_states, 
    predicted_actions, 
    ground_truth_actions, 
    input_ids
):
    """
    Saves features collected during a training/validation forward pass.

    Args:
        save_dir (Path): Directory to save the features.
        batch_idx (int): Index of the current batch.
        device_id (int): ID of the device (rank).
        hidden_states (torch.Tensor): Hidden states from the model.
        predicted_actions (torch.Tensor): Actions predicted by the model.
        ground_truth_actions (torch.Tensor): Ground truth actions.
        input_ids (torch.Tensor): Input IDs from the batch.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # We use batch_idx and device_id for uniqueness
    filename = f"batch_{batch_idx}_rank_{device_id}.npz"
    save_path = save_dir / filename
    
    # Detach and move to CPU if they are tensors
    if hasattr(hidden_states, 'detach'):
        hidden_states_np = hidden_states.detach().cpu().float().numpy()
    else:
        hidden_states_np = hidden_states

    if hasattr(predicted_actions, 'detach'):
        predicted_actions_np = predicted_actions.detach().cpu().float().numpy()
    else:
        predicted_actions_np = predicted_actions

    if hasattr(ground_truth_actions, 'detach'):
        ground_truth_actions_np = ground_truth_actions.detach().cpu().float().numpy()
    else:
        ground_truth_actions_np = ground_truth_actions
        
    if hasattr(input_ids, 'detach'):
        input_ids_np = input_ids.detach().cpu().numpy()
    else:
        input_ids_np = input_ids
    
    np.savez_compressed(
        save_path,
        hidden_states=hidden_states_np,
        predicted_actions=predicted_actions_np,
        ground_truth_actions=ground_truth_actions_np,
        input_ids=input_ids_np,
    )
