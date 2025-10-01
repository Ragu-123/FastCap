 
# enhanced-fastcap/src/fastcap/scripts/export.py

import argparse
import yaml
import os
import torch

# It's good practice to reuse code from other scripts
from inference import load_model_for_inference

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastcap.model import EnhancedFastCap

def main(args):
    # 1. Setup Environment
    device = torch.device("cpu") # Exporting is typically done on CPU
    print(f"Using device: {device}")

    # 2. Load Model for Inference
    print(f"Loading model from '{args.checkpoint}'...")
    model = load_model_for_inference(args.checkpoint, device)
    print("Model loaded successfully.")
    
    # 3. Create a Dummy Input Tensor
    # The ONNX exporter needs a sample input to trace the model's execution path.
    # The shape must match the model's expected input shape.
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    print(f"Created dummy input tensor with shape: {dummy_input.shape}")

    # 4. Export the Model to ONNX
    # The inference path of the model will be traced.
    print(f"Exporting model to ONNX format at '{args.output_path}'...")
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    try:
        torch.onnx.export(
            model,
            dummy_input,
            args.output_path,
            input_names=['input_image'],   # The model's input names
            output_names=['generated_ids'], # The model's output names
            dynamic_axes={'input_image' : {0 : 'batch_size'}, # Allow for variable batch size
                          'generated_ids' : {0 : 'batch_size'}},
            opset_version=11 # A commonly used ONNX opset version
        )
        print("\nModel successfully exported to ONNX.")
        print(f"You can now use '{args.output_path}' with an ONNX-compatible runtime.")
    except Exception as e:
        print(f"\nAn error occurred during ONNX export: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export the EnhancedFastCap Model to ONNX format.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained model checkpoint (.pth.tar).")
    parser.add_argument('--output-path', type=str, required=True, help="Path to save the final ONNX model.")
    args = parser.parse_args()
    
    # Create a dummy checkpoint if it doesn't exist for the example to run
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

    main(args)
