# enhanced-fastcap/src/fastcap/generation/dlag.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicLengthPredictor(nn.Module):
    """
    Predicts the optimal caption length based on global vision features.
    
    This module implements the length prediction formulation from "Innovation 7,"
    using a multi-layer perceptron (MLP) to map vision features to a scalar
    caption length.
    """
    def __init__(self, vision_dim, max_length=30, min_length=5, dropout=0.1):
        super().__init__()
        self.max_length = max_length
        self.min_length = min_length
        self.vision_dim = vision_dim
        
        # MLP for predicting a normalized length value [0, 1]
        self.length_predictor = nn.Sequential(
            nn.Linear(vision_dim, vision_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(vision_dim // 2, vision_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(vision_dim // 4, 1),
            nn.Sigmoid() # Squashes output to the range (0, 1)
        )
        
    def forward(self, global_vision_features):
        """
        Args:
            global_vision_features (torch.Tensor): Pooled vision features, shape (B, D)
        
        Returns:
            torch.Tensor: Predicted integer lengths for each item in the batch, shape (B,)
        """
        # Predict normalized length [0, 1]
        length_normalized = self.length_predictor(global_vision_features).squeeze(-1)
        
        # Scale and shift to the desired range [min_length, max_length]
        predicted_length_float = self.min_length + length_normalized * (self.max_length - self.min_length)
        
        # Round to the nearest integer to get the final predicted length
        predicted_length_int = torch.round(predicted_length_float).long()
        
        return predicted_length_int


class DynamicPositionEncoding(nn.Module):
    """
    Generates position encodings that are dynamically scaled by sequence length.
    
    This is the core of the DLAG innovation, implementing the theoretically optimal
    length-aware position encoding: PE_DLAG(pos, L) = PE(pos) * f(pos/L) * g(L).
    """
    def __init__(self, d_model=256, max_length=50, l_ref=15.0):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.l_ref = l_ref # Average caption length for stable scaling

        # Standard sinusoidal position encoding components
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, predicted_lengths):
        """
        Generates dynamic position encodings for a batch based on predicted lengths.

        Args:
            predicted_lengths (torch.Tensor): Predicted lengths for each batch item, shape (B,)
        
        Returns:
            torch.Tensor: The dynamic position encodings, shape (B, max_len_in_batch, D)
        """
        # --- CORE CORRECTION: Vectorized implementation to remove the inefficient for-loop ---
        batch_size = predicted_lengths.size(0)
        batch_max_length = predicted_lengths.max().item()
        device = predicted_lengths.device

        # Create positions tensor for the entire batch: (B, max_len_in_batch)
        positions = torch.arange(0, batch_max_length, device=device).float().unsqueeze(0).repeat(batch_size, 1)
        
        # Prepare lengths for broadcasting: (B, 1)
        L = predicted_lengths.float().unsqueeze(1)

        # 1. Position Normalization: r = pos / L
        r = positions / (L + 1e-8) # Add epsilon for stability

        # 2. Position Normalization Function: f(r) = sqrt(2) * sin(pi*r/2)
        f_r = math.sqrt(2) * torch.sin(math.pi * r / 2)
        
        # 3. Length Scaling Function: g(L) = log(1 + L/L_ref)
        g_L = torch.log(1 + L / self.l_ref)

        # 4. Base Sinusoidal Encoding
        # (B, max_len, 1) * (D/2) -> (B, max_len, D/2)
        sin_enc = torch.sin(positions.unsqueeze(-1) * self.div_term)
        cos_enc = torch.cos(positions.unsqueeze(-1) * self.div_term)
        
        # 5. Combine into final PE tensor and apply dynamic scaling
        pe = torch.zeros(batch_size, batch_max_length, self.d_model, device=device)
        pe[..., 0::2] = sin_enc
        pe[..., 1::2] = cos_enc
        
        # Apply scaling factors f(r) and g(L)
        # (B, max_len, D) * (B, max_len, 1) * (B, 1, 1) -> (B, max_len, D)
        pe = pe * f_r.unsqueeze(-1) * g_L.unsqueeze(-1)
        
        # 6. Mask out positions beyond the predicted length for each item
        mask = torch.arange(batch_max_length, device=device)[None, :] >= predicted_lengths[:, None]
        pe.masked_fill_(mask.unsqueeze(-1), 0.0)

        return pe


class DLAGModule(nn.Module):
    """
    A wrapper module for Dynamic Length-Aware Generation components.
    This module can be easily integrated into the main model to provide
    length prediction and dynamic positional encodings.
    """
    def __init__(self, vision_dim=256, embed_dim=256, max_length=30, min_length=5):
        super().__init__()
        self.length_predictor = DynamicLengthPredictor(vision_dim, max_length, min_length)
        self.position_encoder = DynamicPositionEncoding(embed_dim, max_length)

    def forward(self, vision_features):
        """
        Takes vision features, predicts lengths, and generates corresponding encodings.

        Args:
            vision_features (torch.Tensor): Vision features, shape (B, N, D)
        
        Returns:
            A dictionary containing predicted lengths and positional encodings.
        """
        # Predict length from global (pooled) vision features
        global_vision_features = vision_features.mean(dim=1)
        predicted_lengths = self.length_predictor(global_vision_features)
        
        # Generate position encodings based on predicted lengths
        position_encodings = self.position_encoder(predicted_lengths)

        return {
            "predicted_lengths": predicted_lengths,
            "position_encodings": position_encodings
        }

# Example usage:
if __name__ == '__main__':
    batch_size = 4
    vision_dim = 256
    vision_seq_len = 49 # e.g., 7x7 patches

    # Instantiate the DLAG module
    dlag_module = DLAGModule(vision_dim=vision_dim, embed_dim=vision_dim)
    print("Initialized DLAGModule.\n")

    # Create dummy vision features
    vision_features = torch.randn(batch_size, vision_seq_len, vision_dim)
    print(f"Input vision features shape: {vision_features.shape}\n")

    # --- Forward pass ---
    output = dlag_module(vision_features)
    pred_lengths = output["predicted_lengths"]
    pos_encodings = output["position_encodings"]
    
    print("--- DLAG Module Output ---")
    print(f"Predicted lengths for the batch: {pred_lengths.tolist()}")
    print(f"Shape of generated positional encodings: {pos_encodings.shape}")
    
    # The sequence length of the encodings should match the longest predicted length in the batch
    assert pos_encodings.shape[0] == batch_size
    assert pos_encodings.shape[1] == pred_lengths.max().item()
    assert pos_encodings.shape[2] == vision_dim
    
    print("\nOutput shapes are correct and dynamically sized to the batch.")
