# scripts/train.py (Corrected and Final Version)

import argparse
import yaml
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the project root to the Python path to allow for correct module imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.datasets import CocoKarpathyDataset
from fastcap.model import EnhancedFastCap, adapt_backbone
from fastcap.utils.checkpointing import save_checkpoint, load_checkpoint
from fastcap.utils.metrics import CaptionMetrics

def train_one_epoch(model, dataloader, optimizer, device, epoch, num_epochs):
    """
    Runs a single epoch of training.
    """
    model.train()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
    total_loss = 0.0

    for i, (images, captions, padding_masks) in enumerate(progress_bar):
        images = images.to(device)
        captions = captions.to(device)
        padding_masks = padding_masks.to(device)

        optimizer.zero_grad()
        
        loss_dict = model(images, captions, padding_masks)
        loss = loss_dict['total_loss']
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{total_loss / (i + 1):.4f}")

    return total_loss / len(dataloader)


def validate(model, dataloader, metrics_calculator, device, tokenizer):
    """
    Runs validation and computes metrics.
    """
    model.eval()
    progress_bar = tqdm(dataloader, desc="Validating")
    
    hypotheses = {}
    
    with torch.no_grad():
        for i, (images, _, _) in enumerate(progress_bar):
            images = images.to(device)
            
            generated_ids = model(images)
            
            for j, ids in enumerate(generated_ids):
                # The dataloader provides samples in order, so we can use a simple index as the image ID
                img_id = i * dataloader.batch_size + j
                # Decode the generated token IDs into a text caption
                caption = tokenizer.decode(ids, skip_special_tokens=True)
                hypotheses[img_id] = [caption]

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

    # 3. Load Real Data
    print("Building tokenizer...")
    tokenizer = CocoKarpathyDataset.build_tokenizer()
    pad_id = tokenizer.pad_token_id
    
    print("Loading actual train dataset...")
    train_dataset = CocoKarpathyDataset(
        split='train', 
        tokenizer=tokenizer,
        pad_token_id=pad_id,
        subset_percentage=config['training'].get('subset_percentage', 1.0)
    )
    print("Loading actual validation dataset...")
    val_dataset = CocoKarpathyDataset(
        split='validation', 
        tokenizer=tokenizer,
        pad_token_id=pad_id,
        subset_percentage=config['training'].get('subset_percentage', 1.0)
    )
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=CocoKarpathyDataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], collate_fn=CocoKarpathyDataset.collate_fn)
    
    # 4. Initialize Model, Optimizer, and Scheduler
    
    # CRITICAL FIX: Apply the runtime patch to the backbone's forward pass
    adapt_backbone()
    
    model = EnhancedFastCap(config, tokenizer=tokenizer).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['lr_step_size'], gamma=config['training']['lr_gamma'])

    # 5. Load Checkpoint if available
    start_epoch = 0
    best_cider = 0.0
    if config['training'].get('resume_checkpoint'):
        start_epoch, best_cider = load_checkpoint(config['training']['resume_checkpoint'], model, optimizer, scheduler)

    # 6. Initialize Metrics Calculator with Real References
    print("Loading ground truth references for validation...")
    val_references = {i: val_dataset.hf_dataset[i]['sentences'] for i in range(len(val_dataset))}
    metrics_calculator = CaptionMetrics(val_references)

    # 7. Training and Validation Loop
    print("Starting training...")
    for epoch in range(start_epoch, config['training']['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, config['training']['num_epochs'])
        print(f"Epoch {epoch+1} Average Training Loss: {train_loss:.4f}")

        scores = validate(model, val_loader, metrics_calculator, device, tokenizer)
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
    
    main(args.config)
