 
# enhanced-fastcap/src/fastcap/scripts/inference.py

import argparse
import yaml
import os
import torch
from torchvision import transforms
from PIL import Image

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastcap.model import EnhancedFastCap
from fastcap.quantization.aqat import AQATModule

# --- Mock Tokenizer for Demonstration ---
class MockTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        # In a real scenario, this would convert IDs back to words
        tokens = [f"word_{tid.item()}" for tid in token_ids if tid > 0] # 0 is padding
        return " ".join(tokens)
# ------------------------------------

def load_model_for_inference(checkpoint_path, device):
    """
    Loads a model from a checkpoint for inference, handling both standard
    and quantized models.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at '{checkpoint_path}'")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = EnhancedFastCap(config).to(device)
    
    # Handle potential DataParallel prefix 'module.'
    state_dict = checkpoint['state_dict']
    if all(key.startswith('module.') for key in state_dict):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # If this is a quantized model, apply the quantization parameters
    if 'quantization_params' in checkpoint:
        print("Quantized model detected. Applying quantization parameters.")
        aqat_module = AQATModule(model)
        aqat_module.bit_widths = checkpoint['quantization_params']['bit_widths']
        aqat_module.quantization_params.load_state_dict(
            checkpoint['quantization_params']['scales_offsets']
        )
        aqat_module.apply_quantization_to_model()

    model.eval()
    return model

def preprocess_image(image_path, device):
    """
    Loads an image from a path and applies the necessary transformations.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at '{image_path}'")
        
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def main(args):
    # 1. Setup Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model
    print(f"Loading model from '{args.checkpoint}'...")
    model = load_model_for_inference(args.checkpoint, device)
    print("Model loaded successfully.")

    # 3. Preprocess Image
    print(f"Loading and preprocessing image from '{args.image}'...")
    image_tensor = preprocess_image(args.image, device)
    
    # 4. Generate Caption
    print("Generating caption...")
    with torch.no_grad():
        # The model's forward pass in eval mode uses the efficient ICMR decoder
        generated_ids = model(image_tensor)

    # 5. Decode and Print Caption
    # Replace with your actual tokenizer
    tokenizer = MockTokenizer()
    caption = tokenizer.decode(generated_ids.squeeze(0))
    
    print("\n--- Generated Caption ---")
    print(caption)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a caption for an image using EnhancedFastCap.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained model checkpoint (.pth.tar).")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()
    
    # Create dummy files if they don't exist for the example to run
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint file not found. Creating a dummy checkpoint at '{args.checkpoint}' for demonstration.")
        dummy_config = {
            'model': {'vocab_size': 1000, 'embed_dim': 256},
            'vision': {'embed_dims': [64, 128, 256], 'depths': [2, 2, 2]},
            'cmfa': {'projection_dim': 128},
            'moe': {'num_experts': 4, 'num_layers': 2, 'load_balance_alpha': 0.01},
            'icmr': {'max_iterations': 3},
        }
        dummy_model = EnhancedFastCap(dummy_config)
        dummy_state = {'state_dict': dummy_model.state_dict(), 'config': dummy_config}
        os.makedirs(os.path.dirname(args.checkpoint) or '.', exist_ok=True)
        torch.save(dummy_state, args.checkpoint)

    if not os.path.exists(args.image):
        print(f"Image file not found. Creating a dummy image at '{args.image}' for demonstration.")
        dummy_image = Image.new('RGB', (224, 224), color='red')
        os.makedirs(os.path.dirname(args.image) or '.', exist_ok=True)
        dummy_image.save(args.image)

    main(args)
