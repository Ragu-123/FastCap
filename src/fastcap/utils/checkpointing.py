 
# enhanced-fastcap/src/fastcap/utils/checkpointing.py

import os
import shutil
import torch

def save_checkpoint(state, is_best, checkpoint_dir='checkpoints', filename='checkpoint.pth.tar'):
    """
    Saves the model and training state to a checkpoint file.

    Args:
        state (dict): A dictionary containing model and training information.
                      Expected keys: 'epoch', 'state_dict', 'optimizer', 'best_score'.
        is_best (bool): If True, saves a separate copy as 'best_checkpoint.pth.tar'.
        checkpoint_dir (str): The directory where checkpoints will be saved.
        filename (str): The name of the checkpoint file.
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found. Creating '{checkpoint_dir}'")
        os.makedirs(checkpoint_dir)
        
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to '{filepath}' (Epoch {state['epoch']})")
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_checkpoint.pth.tar')
        shutil.copyfile(filepath, best_filepath)
        print(f"Best model saved to '{best_filepath}' (Epoch {state['epoch']}, Score: {state['best_score']:.4f})")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Loads a model and training state from a checkpoint file.

    Args:
        checkpoint_path (str): The path to the checkpoint file.
        model (nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state into.
        scheduler (torch.optim.lr_scheduler, optional): The LR scheduler to load the state into.

    Returns:
        int: The epoch number to start training from.
        float: The best score achieved so far.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint file not found at '{checkpoint_path}'. Starting from scratch.")
        return 0, 0.0 # Start from epoch 0 with a score of 0.0

    print(f"Loading checkpoint from '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) # Load to CPU first
    
    # Load model state
    # Handle potential DataParallel prefix 'module.'
    state_dict = checkpoint['state_dict']
    if all(key.startswith('module.') for key in state_dict):
        print("DataParallel prefix 'module.' detected. Removing it.")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    # Load optimizer state
    if optimizer is not None and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Optimizer state loaded successfully.")
        except ValueError as e:
            print(f"Warning: Could not load optimizer state. It might be from a different optimizer. Error: {e}")

    # Load scheduler state
    if scheduler is not None and 'scheduler' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("Scheduler state loaded successfully.")
        except ValueError as e:
            print(f"Warning: Could not load scheduler state. Error: {e}")
            
    start_epoch = checkpoint.get('epoch', 0) + 1 # Resume from the next epoch
    best_score = checkpoint.get('best_score', 0.0)
    
    print(f"Checkpoint loaded. Resuming training from epoch {start_epoch} with best score {best_score:.4f}")
    
    return start_epoch, best_score

# Example usage:
if __name__ == '__main__':
    # 1. Setup a mock model and optimizer
    mock_model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )
    mock_optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)
    
    checkpoint_directory = './tmp_checkpoints'
    
    # --- Test Saving ---
    print("--- Testing Checkpoint Saving ---")
    state_to_save = {
        'epoch': 10,
        'state_dict': mock_model.state_dict(),
        'optimizer': mock_optimizer.state_dict(),
        'best_score': 0.95
    }
    # Save a regular checkpoint
    save_checkpoint(state_to_save, is_best=False, checkpoint_dir=checkpoint_directory)
    # Save a new best checkpoint
    state_to_save['epoch'] = 11
    save_checkpoint(state_to_save, is_best=True, checkpoint_dir=checkpoint_directory)
    
    assert os.path.exists(os.path.join(checkpoint_directory, 'checkpoint.pth.tar'))
    assert os.path.exists(os.path.join(checkpoint_directory, 'best_checkpoint.pth.tar'))
    print("\nCheckpoint files created successfully.\n")
    
    # --- Test Loading ---
    print("--- Testing Checkpoint Loading ---")
    new_model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    
    # Load from the 'best' checkpoint
    start_epoch, best_score = load_checkpoint(
        os.path.join(checkpoint_directory, 'best_checkpoint.pth.tar'),
        new_model,
        new_optimizer
    )
    
    # Verify that the states are loaded correctly
    assert start_epoch == 12 # Should be 11 (saved epoch) + 1
    assert best_score == 0.95
    # Check if model weights are identical
    assert torch.equal(
        mock_model.state_dict()['0.weight'], 
        new_model.state_dict()['0.weight']
    )
    print("\nCheckpoint loading successful and state is consistent.")

    # --- Clean up the temporary directory ---
    shutil.rmtree(checkpoint_directory)
    print(f"\nCleaned up temporary directory: '{checkpoint_directory}'")
