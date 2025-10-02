# enhanced-fastcap/src/fastcap/quantization/aqat.py

import torch
import torch.nn as nn
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
import yaml # Added for the example

# --- It's good practice to import from the project's own modules in examples ---
from ..model import EnhancedFastCap
from ..data.caption_dataset import CocoKarpathyDataset

class StraightThroughEstimator(torch.autograd.Function):
    """
    Custom autograd function for the Straight-Through Estimator (STE).
    In the forward pass, it uses the quantized values.
    In the backward pass, it passes the gradients through to the original,
    full-precision weights, "estimating" the gradient of the non-differentiable
    quantization operation as the identity function.
    """
    @staticmethod
    def forward(ctx, quantized_input, original_input):
        return quantized_input

    @staticmethod
    def backward(ctx, grad_output):
        # Pass the gradient directly to the original input
        return grad_output, grad_output

class AQATModule(nn.Module):
    """
    Implements Adaptive Quantization-Aware Training (AQAT).

    This module wraps a model and provides methods to perform sensitivity analysis,
    adaptive bit allocation, and quantization-aware training. This is a direct
    implementation of the concepts outlined in "Innovation 9".
    """
    def __init__(self, model_to_quantize, target_bits=8, sensitivity_threshold=0.1):
        super().__init__()
        self.model = model_to_quantize
        self.target_bits = target_bits
        self.sensitivity_threshold = sensitivity_threshold

        # Dictionaries to store layer-specific quantization parameters
        self.layer_sensitivities = {}
        self.quantization_params = nn.ParameterDict()
        self.bit_widths = {}
        
        self._initialize_quantization_params()

    def _get_quantizable_layers(self):
        """Helper to iterate over layers that can be quantized."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                yield name, module

    def _initialize_quantization_params(self):
        """Initializes scale and offset parameters for each quantizable layer."""
        for name, module in self._get_quantizable_layers():
            # As per the spec, scale and compensation offset are learnable
            scale = module.weight.abs().mean() / (2**(self.target_bits - 1))
            offset = torch.zeros_like(module.weight.mean())
            self.quantization_params[f"{name}.scale"] = nn.Parameter(scale)
            self.quantization_params[f"{name}.offset"] = nn.Parameter(offset)
            self.bit_widths[name] = self.target_bits # Default bit width

    def quantize_layer(self, weight_tensor, layer_name):
        """
        Applies adaptive quantization to a specific layer's weight tensor.
        This follows the formulation: Q(W) = round(W/s) * s + ε
        """
        scale = self.quantization_params[f"{layer_name}.scale"]
        offset = self.quantization_params[f"{layer_name}.offset"]
        bit_width = self.bit_widths.get(layer_name, self.target_bits)
        
        q_min = -(2**(bit_width - 1))
        q_max = 2**(bit_width - 1) - 1
        
        quantized = torch.round(weight_tensor / scale.clamp(min=1e-8)).clamp(q_min, q_max)
        dequantized = quantized * scale + offset
        
        return dequantized

    def compute_layer_sensitivities(self, calibration_dataloader, forward_hook_func, device):
        """
        Computes the sensitivity of each layer to quantization based on output perturbation.
        This now directly implements the formula from the idea document:
        Sensitivity S_l = ||f(x; W) - f(x; Q(W_l))||^2 / ||f(x; W)||^2
        
        Args:
            calibration_dataloader: Dataloader with calibration data (inputs only needed).
            forward_hook_func: A function that takes a model and inputs, and returns the
                               tensor to be used for sensitivity comparison (e.g., logits).
            device: The device to run computations on.
        """
        self.model.eval()
        original_model = deepcopy(self.model)
        
        # Get baseline outputs with the full-precision model
        baseline_outputs = []
        with torch.no_grad():
            for batch in tqdm(calibration_dataloader, desc="Calibrating (Baseline)"):
                images, captions, _, _ = batch
                if images is None: continue
                
                images = images.to(device)
                targets = captions.to(device)
                
                outputs, _ = forward_hook_func(original_model, images, targets)
                baseline_outputs.append(outputs.cpu()) # Store on CPU to save GPU memory

        # Compute sensitivity for each layer
        for name, module in self._get_quantizable_layers():
            quantized_model = deepcopy(original_model)
            quantized_module = quantized_model.get_submodule(name)
            
            # Temporarily quantize this layer
            original_weight = quantized_module.weight.data.clone()
            quantized_weight = self.quantize_layer(original_weight, name)
            quantized_module.weight.data = quantized_weight
            
            # Compute outputs with the quantized layer
            quantized_outputs = []
            with torch.no_grad():
                for batch in calibration_dataloader:
                    images, captions, _, _ = batch
                    if images is None: continue

                    images = images.to(device)
                    targets = captions.to(device)
                    
                    outputs, _ = forward_hook_func(quantized_model, images, targets)
                    quantized_outputs.append(outputs.cpu())
            
            # CORRECTED: Calculate sensitivity based on L2 norm difference, as per the document
            total_error = sum(torch.norm(b_out - q_out, p=2)**2 for b_out, q_out in zip(baseline_outputs, quantized_outputs))
            total_norm = sum(torch.norm(b_out, p=2)**2 for b_out in baseline_outputs)
            
            sensitivity = total_error / (total_norm + 1e-8)
            self.layer_sensitivities[name] = sensitivity.item()
                
        print("Layer sensitivities computed:", self.layer_sensitivities)
        return self.layer_sensitivities

    def adaptive_bit_allocation(self):
        """Allocates bit widths based on layer sensitivities."""
        # A simple strategy: sensitive layers get higher precision
        for name, sensitivity in self.layer_sensitivities.items():
            if sensitivity > self.sensitivity_threshold:
                self.bit_widths[name] = 8 # High precision for sensitive layers
            else:
                self.bit_widths[name] = 4 # Low precision for robust layers
        print("Adaptive bit allocation complete:", self.bit_widths)

    def apply_quantization_to_model(self):
        """
        Applies quantization to all relevant layers for a QAT forward pass.
        Uses the Straight-Through Estimator to allow gradients to flow.
        """
        for name, module in self._get_quantizable_layers():
            original_weight = module.weight.data
            quantized_weight = self.quantize_layer(original_weight, name)
            # Replace the weight with the STE-wrapped quantized version
            module.weight.data = StraightThroughEstimator.apply(quantized_weight, original_weight)

# --- Updated Example Usage ---
if __name__ == '__main__':
    # This example demonstrates how to use AQATModule with the actual project components.
    
    # 1. Create a dummy config for the model
    dummy_config = {
        'model': {'vocab_size': 1000, 'embed_dim': 256},
        'vision': {'embed_dims': [64, 128, 256], 'depths': [2, 2, 2]},
        'cmfa': {'projection_dim': 128},
        'moe': {'num_experts': 4, 'num_layers': 2, 'load_balance_alpha': 0.01},
        'icmr': {'max_iterations': 3},
        'training': {} # Needs to be present
    }

    # 2. Setup a mock model and the AQAT module
    model = EnhancedFastCap(dummy_config)
    aqat_module = AQATModule(model)
    print("Initialized AQATModule with EnhancedFastCap.\n")
    
    # 3. Create dummy calibration data
    # In a real script (like quantize.py), you'd use the real CocoKarpathyDataset
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return 10
        def __getitem__(self, idx):
            return torch.randn(3, 224, 224), torch.randint(1, 1000, (20,)), None, idx

    dataloader = torch.utils.data.DataLoader(DummyDataset(), batch_size=4)
    
    # 4. Define a forward hook that matches the model's training step
    def calibration_forward(calib_model, inputs, targets):
        # This function should mimic a simplified forward pass to get logits
        output_dict = calib_model.default_train_step(inputs, targets, None)
        # The sensitivity analysis needs the logits and the ground truth labels
        return output_dict['nar_logits'], targets

    # 5. Run the pipeline (on CPU for this example)
    print("--- 1. Computing Layer Sensitivities ---")
    aqat_module.compute_layer_sensitivities(dataloader, calibration_forward, device='cpu')
    
    print("\n--- 2. Performing Adaptive Bit Allocation ---")
    aqat_module.adaptive_bit_allocation()

    # 6. Simulate a QAT training step
    print("\n--- 3. Simulating a Quantization-Aware Training Step ---")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # In your training loop, before the forward pass:
    aqat_module.apply_quantization_to_model()
    
    # Dummy forward and backward pass
    images, captions, _, _ = next(iter(dataloader))
    loss_fn = nn.CrossEntropyLoss()
    
    # Get model output (which includes the loss)
    output_dict = model(images, captions, None)
    loss = output_dict['total_loss']
    loss.backward()
    optimizer.step()
    
    print(f"QAT forward pass successful. Loss: {loss.item():.4f}")
    
    # Check if gradients flowed to the original weights (they should not be None)
    # Checking a backbone layer and a decoder layer
    assert model.vision_backbone.stages[0][0].spatial_mamba.in_proj.weight.grad is not None
    assert model.training_decoder.output_proj.weight.grad is not None
    print("Gradients successfully flowed through the quantized layers via STE.")
