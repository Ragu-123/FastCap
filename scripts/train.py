 
# enhanced-fastcap/src/fastcap/scripts/train.py

import argparse
import yaml
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Assuming a simple project structure where 'src' is in the python path
# In a real project, you might need to adjust sys.path or install the package
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastcap.model import EnhancedFastCap
from fastcap.utils.checkpointing import save_checkpoint, load_checkpoint
from fastcap.utils.metrics import CaptionMetrics
# from data.datasets import CaptionDataset # This would be your actual dataset class

# --- Mock Dataset for Demonstration ---
class MockCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100, vocab_size=1000):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)
        caption = torch.randint(1, self.vocab_size, (20,))
        padding_mask = (caption == 0)
        return image, caption, padding_mask
# ------------------------------------

def train_one_epoch(model, dataloader, optimizer, device, epoch, num_epochs):
    model.train()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
    total_loss = 0.0

    for i, (images, captions, padding_masks) in enumerate(progress_bar):
        images = images.to(device)
        captions = captions.to(device)
        padding_masks = padding_masks.to(device)

        optimizer.zero_grad()
        
        # The model's forward pass in training mode returns a dictionary of losses
        loss_dict = model(images, captions, padding_masks)
        loss = loss_dict['total_loss']
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{total_loss / (i + 1):.4f}")

    return total_loss / len(dataloader)


def validate(model, dataloader, metrics_calculator, device):
    model.eval()
    progress_bar = tqdm(dataloader, desc="Validating")
    
    hypotheses = {}
    
    with torch.no_grad():
        for i, (images, _, _) in enumerate(progress_bar):
            images = images.to(device)
            
            # The model's forward pass in eval mode returns generated token IDs
            generated_ids = model(images)
            
            # For this mock example, we'll generate fake text captions
            # In a real scenario, you'd use your tokenizer to decode `generated_ids`
            for j, ids in enumerate(generated_ids):
                img_id = i * dataloader.batch_size + j
                hypotheses[img_id] = [f"generated caption for image {img_id}"]

    # In a real scenario, the metrics calculator would be initialized with real references
    scores = metrics_calculator.compute_scores(hypotheses)
    return scores


def main(config_path):
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Setup Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config['training']['seed'])
    print(f"Using device: {device}")

    # 3. Load Data
    # Replace with your actual dataset and dataloaders
    train_dataset = MockCaptionDataset(num_samples=100, vocab_size=config['model']['vocab_size'])
    val_dataset = MockCaptionDataset(num_samples=20, vocab_size=config['model']['vocab_size'])
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])
    
    # 4. Initialize Model, Optimizer, and Scheduler
    model = EnhancedFastCap(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['lr_step_size'], gamma=0.1)

    # 5. Load Checkpoint if available
    start_epoch = 0
    best_cider = 0.0
    if config['training']['resume_checkpoint']:
        start_epoch, best_cider = load_checkpoint(config['training']['resume_checkpoint'], model, optimizer, scheduler)

    # 6. Initialize Metrics Calculator
    # In a real implementation, load your ground truth validation captions here
    mock_references = {i: [f"reference caption for image {i}"] for i in range(len(val_dataset))}
    metrics_calculator = CaptionMetrics(mock_references)

    # 7. Training and Validation Loop
    print("Starting training...")
    for epoch in range(start_epoch, config['training']['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, config['training']['num_epochs'])
        print(f"Epoch {epoch+1} Average Training Loss: {train_loss:.4f}")

        scores = validate(model, val_loader, metrics_calculator, device)
        current_cider = scores.get("CIDEr", 0.0)
        print(f"Epoch {epoch+1} Validation CIDEr: {current_cider:.4f}")

        scheduler.step()

        # 8. Save Checkpoint
        is_best = current_cider > best_cider
        if is_best:
            best_cider = current_cider
        
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_score': best_cider,
            'config': config
        }, is_best, checkpoint_dir=config['training']['checkpoint_dir'])

    print("Training complete.")
    print(f"Best CIDEr score achieved: {best_cider:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the EnhancedFastCap Model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration YAML file.")
    args = parser.parse_args()
    
    # Create a dummy config if it doesn't exist for the example to run
    if not os.path.exists(args.config):
        print(f"Config file not found. Creating a dummy config at '{args.config}' for demonstration.")
        dummy_config = {
            'model': {'vocab_size': 1000, 'embed_dim': 256},
            'vision': {'embed_dims': [64, 128, 256], 'depths': [2, 2, 2]},
            'cmfa': {'projection_dim': 128},
            'moe': {'num_experts': 4, 'num_layers': 2, 'load_balance_alpha': 0.01},
            'icmr': {'max_iterations': 3},
            'training': {
                'seed': 42,
                'batch_size': 8,
                'num_epochs': 10,
                'learning_rate': 1e-4,
                'lr_step_size': 5,
                'checkpoint_dir': './checkpoints',
                'resume_checkpoint': None
            }
        }
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, 'w') as f:
            yaml.dump(dummy_config, f)

    main(args.config)
