# enhanced-fastcap/src/fastcap/quantization/aqat.py

import torch
import torch.nn as nn
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

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

    def compute_layer_sensitivities(self, calibration_dataloader, forward_hook_func):
        """
        Computes the sensitivity of each layer to quantization based on output perturbation.
        This now directly implements the formula from the idea document:
        Sensitivity S_l = ||f(x; W) - f(x; Q(W_l))||^2 / ||f(x; W)||^2
        
        Args:
            calibration_dataloader: Dataloader with calibration data (inputs only needed).
            forward_hook_func: A function that takes a model and inputs, and returns the
                               tensor to be used for sensitivity comparison (e.g., logits).
        """
        self.model.eval()
        original_model = deepcopy(self.model)
        device = next(self.model.parameters()).device

        # Get baseline outputs with the full-precision model
        baseline_outputs = []
        with torch.no_grad():
            for batch in tqdm(calibration_dataloader, desc="Calibrating (Baseline)"):
                inputs, _ = batch # We don't need targets for this calculation
                inputs = inputs.to(device)
                outputs = forward_hook_func(original_model, inputs)
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
                    inputs, _ = batch
                    inputs = inputs.to(device)
                    outputs = forward_hook_func(quantized_model, inputs)
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

# Example usage:
if __name__ == '__main__':
    # 1. Define a simple mock model to quantize
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(16 * 30 * 30, 100) # Simplified dimensions
            self.fc2 = nn.Linear(100, 10) # Sensitive layer
        
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    # 2. Setup
    model = MockModel()
    aqat_module = AQATModule(model)
    print("Initialized AQATModule.\n")
    
    # Create dummy calibration data (note: targets are not used by the new method)
    calibration_data = [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))) for _ in range(5)]
    
    # Define a forward hook function for the mock model
    def mock_forward_hook(hook_model, inputs):
        return hook_model(inputs)

    # 3. Run the AQAT pipeline
    print("--- 1. Computing Layer Sensitivities ---")
    aqat_module.compute_layer_sensitivities(calibration_data, mock_forward_hook)
    
    print("\n--- 2. Performing Adaptive Bit Allocation ---")
    aqat_module.adaptive_bit_allocation()

    # 4. Simulate a QAT training step
    print("\n--- 3. Simulating a Quantization-Aware Training Step ---")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # In your training loop, before the forward pass:
    aqat_module.apply_quantization_to_model()
    
    # Dummy forward and backward pass
    inputs, targets = calibration_data[0]
    loss_fn = nn.CrossEntropyLoss()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    
    print(f"QAT forward pass successful. Loss: {loss.item():.4f}")
    # Check if gradients flowed to the original weights (they should not be None)
    assert model.fc2.weight.grad is not None
    print("Gradients successfully flowed through the quantized layers via STE.")

