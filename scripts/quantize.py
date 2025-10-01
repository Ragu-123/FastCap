 
# enhanced-fastcap/src/fastcap/scripts/quantize.py

import argparse
import yaml
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastcap.model import EnhancedFastCap
from fastcap.quantization.aqat import AQATModule
from fastcap.utils.checkpointing import load_checkpoint

# --- Mock Dataset for Demonstration ---
# In a real scenario, this would be a subset of your actual training data.
class MockCalibrationDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=50):
        self.num_samples = num_samples
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)
        # For sensitivity analysis, we need a target for the loss function
        caption = torch.randint(1, 1000, (20,))
        return image, caption
# ------------------------------------

def main(args):
    # 1. Load Configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Setup Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Load Full-Precision Model
    model = EnhancedFastCap(config).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    print("Successfully loaded full-precision model from checkpoint.")

    # 4. Load Calibration Data
    # This data is used to analyze which layers are most sensitive to quantization.
    calibration_dataset = MockCalibrationDataset()
    calibration_loader = DataLoader(calibration_dataset, batch_size=config['training']['batch_size'])
    
    # Define a loss function for sensitivity analysis
    # This should be the same as your main captioning loss
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    # 5. Initialize and Run AQAT Pipeline
    print("\nInitializing AQAT Module...")
    aqat_module = AQATModule(model)

    print("--- 1. Computing Layer Sensitivities using calibration data ---")
    # This is a mock forward function for the purpose of calibration
    def calibration_forward(calib_model, inputs, targets):
        # We need logits from the training decoder for a stable loss
        vision_features = calib_model.vision_backbone(inputs)
        # We pass dummy captions to satisfy the training decoder's signature
        dummy_captions = targets[:, :-1]
        logits, _, _ = calib_model.training_decoder(dummy_captions, vision_features)
        return logits.reshape(-1, config['model']['vocab_size']), targets[:, 1:].reshape(-1)

    aqat_module.compute_layer_sensitivities(calibration_loader, calibration_forward)
    
    print("\n--- 2. Performing Adaptive Bit Allocation ---")
    aqat_module.adaptive_bit_allocation()

    # 6. Apply Final Quantization
    print("\n--- 3. Applying final quantization to the model ---")
    # In a real QAT scenario, you would retrain for a few epochs here.
    # For post-training quantization, we apply it directly.
    aqat_module.apply_quantization_to_model()
    print("Model weights have been quantized.")

    # 7. Save the Quantized Model
    quantized_state = {
        'state_dict': model.state_dict(),
        'quantization_params': {
            'bit_widths': aqat_module.bit_widths,
            'scales_offsets': aqat_module.quantization_params.state_dict()
        },
        'config': config
    }
    torch.save(quantized_state, args.output_path)
    print(f"\nSuccessfully saved quantized model to '{args.output_path}'")
    print("This model is now ready for efficient deployment.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quantize the EnhancedFastCap Model using AQAT.")
    parser.add_argument('--config', type=str, required=True, help="Path to the model's configuration YAML file.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained full-precision model checkpoint.")
    parser.add_argument('--output-path', type=str, required=True, help="Path to save the final quantized model.")
    args = parser.parse_args()
    
    # Create dummy files if they don't exist for the example to run
    if not os.path.exists(args.config):
        print(f"Config file not found. Creating a dummy config at '{args.config}' for demonstration.")
        dummy_config = {
            'model': {'vocab_size': 1000, 'embed_dim': 256},
            'vision': {'embed_dims': [64, 128, 256], 'depths': [2, 2, 2]},
            'cmfa': {'projection_dim': 128},
            'moe': {'num_experts': 4, 'num_layers': 2, 'load_balance_alpha': 0.01},
            'icmr': {'max_iterations': 3},
            'training': { 'batch_size': 8 } # Need batch_size for dataloader
        }
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, 'w') as f:
            yaml.dump(dummy_config, f)
            
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint file not found. Creating a dummy checkpoint at '{args.checkpoint}' for demonstration.")
        # Create a dummy model and save its state
        dummy_model = EnhancedFastCap(yaml.safe_load(open(args.config)))
        dummy_state = {'state_dict': dummy_model.state_dict()}
        os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
        torch.save(dummy_state, args.checkpoint)

    main(args)
