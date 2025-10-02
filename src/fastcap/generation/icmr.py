# src/fastcap/generation/icmr.py (Corrected and Final Version)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..decoder.attention import RankAugmentedLinearAttention
from .dlag import DynamicLengthPredictor

class TransformerDecoderLayer(nn.Module):
    """
    A helper class for a single layer of the Transformer decoder.
    This is used within the ICMR process for each refinement step.
    It leverages RALA for efficient attention computation.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = RankAugmentedLinearAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attn = RankAugmentedLinearAttention(embed_dim, num_heads, dropout=dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, vision_features):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x)))
        x = self.norm2(x + self.dropout(self.cross_attn(query=x, key=vision_features, value=vision_features)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class ICMRDecoder(nn.Module):
    """
    Implements the Iterative Conditional Masked Refinement (ICMR) decoder,
    corrected for logical consistency and to align with the innovation's design.
    """
    def __init__(self, vocab_size, embed_dim=256, vision_dim=256, num_layers=6, num_heads=8, max_len=50, max_iterations=3, dropout=0.1, mask_token_id=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_iterations = max_iterations
        
        # CORRECTED: Use a configurable mask_token_id for robustness
        if mask_token_id is None:
            raise ValueError("mask_token_id must be provided to ICMRDecoder.")
        self.mask_token_id = mask_token_id

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))

        # Synergy: Use the Dynamic Length Predictor from DLAG (Innovation 7)
        self.length_predictor = DynamicLengthPredictor(vision_dim)
        
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        
        self.output_projection = nn.Linear(embed_dim, vocab_size)

        # CORRECTED: This confidence head is now used in the refinement step.
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, confidences, iteration, tau_0=0.8, alpha=0.5):
        """
        Generates the confidence-based mask for the next refinement iteration.
        Tokens with confidence *below* the adaptive threshold will be masked.
        """
        # Adaptive threshold: τ_k = τ_0 * e^(-αk)
        tau_k = tau_0 * math.exp(-alpha * iteration)
        # Returns a boolean mask where True indicates a token to be re-predicted
        return confidences < tau_k

    def refinement_step(self, tokens, vision_features):
        """
        Performs a single refinement step of the ICMR process.
        Returns logits, predicted tokens, and learned confidences.
        """
        B, S = tokens.shape
        token_embeds = self.token_embedding(tokens)
        pos_embeds = self.positional_embedding[:, :S]
        
        x = self.dropout(token_embeds + pos_embeds)
        
        for layer in self.decoder_layers:
            x = layer(x, vision_features)
        
        # Get logits from the final hidden states
        logits = self.output_projection(x)
        
        # CORRECTED: Use the confidence_head to get learned confidence scores
        # Squeeze to remove the last dimension of size 1
        confidences = self.confidence_head(x).squeeze(-1)
        
        pred_tokens = torch.argmax(logits, dim=-1)
        
        return logits, pred_tokens, confidences

    def forward(self, vision_features):
        """
        Forward pass for the ICMR Decoder, with corrected iterative logic.
        """
        B, _, D = vision_features.shape
        device = vision_features.device
        
        # 1. Predict caption lengths dynamically
        global_vision_features = vision_features.mean(dim=1)
        predicted_lengths = self.length_predictor(global_vision_features)
        max_len_in_batch = predicted_lengths.max().item()

        # 2. Initialize all tokens to [MASK]
        current_tokens = torch.full(
            (B, max_len_in_batch), self.mask_token_id, dtype=torch.long, device=device
        )

        # 3. CORRECTED: Iterative Refinement Loop
        for k in range(self.max_iterations):
            # Predict a full sequence of tokens and their confidences from the current input
            _, pred_tokens, confidences = self.refinement_step(current_tokens, vision_features)

            # On the final iteration, accept all predicted tokens and exit the loop
            if k == self.max_iterations - 1:
                current_tokens = pred_tokens
                break

            # For intermediate iterations:
            # Generate a mask based on the confidences of the tokens we just predicted.
            # True means low confidence, so we will mask it for the next round.
            mask_for_next_iter = self.generate_mask(confidences, iteration=k)
            
            # Create the input for the next iteration:
            # - Keep the high-confidence tokens we just predicted.
            # - Re-mask the low-confidence tokens.
            current_tokens = torch.where(mask_for_next_iter, self.mask_token_id, pred_tokens)
        
        # 4. Create a final mask to zero out padding based on predicted lengths
        final_mask = torch.arange(max_len_in_batch, device=device)[None, :] >= predicted_lengths[:, None]
        current_tokens.masked_fill_(final_mask, 0) # Use 0 for padding token

        return current_tokens
