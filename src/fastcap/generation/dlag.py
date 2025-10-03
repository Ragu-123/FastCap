# src/fastcap/generation/dlag.py (Corrected and Final Version)

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, target_len):
        # Return positional encodings up to the target length
        return self.pe[:target_len].transpose(0, 1) # (1, target_len, D)

class DLAGModule(nn.Module):
    """
    Dynamic Length-Aware Generation (DLAG) Module.
    This module predicts the length of the caption and provides length-aware
    positional encodings.
    """
    def __init__(self, vision_dim, embed_dim, max_len=100):
        super().__init__()
        self.length_predictor = nn.Sequential(
            nn.Linear(vision_dim, 256),
            nn.ReLU(),
            nn.Linear(256, max_len)
        )
        self.position_encoder = PositionalEncoding(embed_dim, max_len)

    def forward(self, vision_features, target_len=None):
        """
        Args:
            vision_features (torch.Tensor): Vision features from the backbone.
            target_len (int, optional): The target sequence length for training.
                                        If None, length is predicted for inference.
        """
        # Pool the vision features to get a global representation
        global_vision_features = vision_features.mean(dim=1)
        
        # Predict the distribution over possible lengths
        length_logits = self.length_predictor(global_vision_features)
        
        if self.training:
            assert target_len is not None, "target_len must be provided during training."
            # During training, we generate PEs for the padded length of the batch
            position_encodings = self.position_encoder(vision_features, target_len)
            return {
                "length_logits": length_logits,
                "position_encodings": position_encodings
            }
        else:
            # During inference, we predict the most likely length
            predicted_lengths = torch.argmax(length_logits, dim=1)
            # Generate PEs for the single predicted length
            # Note: This logic assumes batch size of 1 for simplicity in inference
            position_encodings = self.position_encoder(vision_features, predicted_lengths[0].item())
            return {
                "predicted_lengths": predicted_lengths,
                "position_encodings": position_encodings
            }