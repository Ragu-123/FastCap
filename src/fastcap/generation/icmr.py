# enhanced-fastcap/src/fastcap/generation/icmr.py

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
    Implements the Iterative Conditional Masked Refinement (ICMR) decoder.
    
    This non-autoregressive decoder generates a caption in parallel and then iteratively
    refines it over a fixed number of steps. This approach is based on "Innovation 4,"
    using a confidence-based masking strategy to decide which tokens to refine.
    """
    def __init__(self, vocab_size, embed_dim=256, vision_dim=256, num_layers=6, num_heads=8, max_len=50, max_iterations=3, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_iterations = max_iterations
        
        # Special tokens
        self.mask_token_id = vocab_size - 1 # Convention: last token in vocab is [MASK]

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))

        # Synergy: Use the Dynamic Length Predictor from DLAG (Innovation 7)
        self.length_predictor = DynamicLengthPredictor(vision_dim)
        
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        
        self.output_projection = nn.Linear(embed_dim, vocab_size)

        # Confidence head to predict token-level confidence scores
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, confidences, iteration, tau_0=0.8, alpha=0.5):
        """
        Generates the confidence-based mask for the next refinement iteration.
        Tokens with confidence *below* the adaptive threshold will be masked.
        
        Args:
            confidences (torch.Tensor): Confidence scores for each token, shape (B, S)
            iteration (int): The current refinement iteration number.
            tau_0 (float): The initial confidence threshold.
            alpha (float): The decay rate for the threshold.

        Returns:
            torch.Tensor: A boolean mask where True indicates a token to be masked.
        """
        # Adaptive threshold: τ_k = τ_0 * e^(-αk)
        tau_k = tau_0 * math.exp(-alpha * iteration)
        # True for tokens with confidence < threshold (i.e., tokens to be re-predicted)
        return confidences < tau_k

    def refinement_step(self, tokens, vision_features):
        """
        Performs a single refinement step of the ICMR process.
        """
        B, S = tokens.shape
        token_embeds = self.token_embedding(tokens)
        pos_embeds = self.positional_embedding[:, :S]
        
        x = self.dropout(token_embeds + pos_embeds)
        
        for layer in self.decoder_layers:
            x = layer(x, vision_features)
            
        logits = self.output_projection(x)
        
        # Get token predictions and their max probabilities (for the next round's confidence)
        pred_tokens = torch.argmax(logits, dim=-1)
        pred_probs = torch.max(F.softmax(logits, dim=-1), dim=-1).values
        
        return pred_tokens, pred_probs

    def forward(self, vision_features):
        """
        Forward pass for the ICMR Decoder.

        Args:
            vision_features (torch.Tensor): Vision features from backbone, shape (B, N, D)
        
        Returns:
            torch.Tensor: The final refined caption tokens, shape (B, S)
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
        # Confidence is initially 0 for all tokens
        token_probs = torch.zeros(B, max_len_in_batch, device=device)

        # 3. Iterative Refinement Loop
        for k in range(self.max_iterations):
            refined_tokens, new_probs = self.refinement_step(current_tokens, vision_features)

            if k < self.max_iterations - 1:
                # Determine which tokens to keep vs. which to re-mask for the next iteration
                mask_for_next_iter = self.generate_mask(token_probs, iteration=k)
                
                # CORRECTED LOGIC:
                # Keep high-confidence tokens from the current refinement.
                # Explicitly set low-confidence tokens to [MASK] for the next iteration.
                next_input_tokens = torch.where(mask_for_next_iter, self.mask_token_id, refined_tokens)
                
                # Update the probabilities: reset masked tokens to 0, keep others.
                token_probs = torch.where(mask_for_next_iter, 0.0, new_probs)
                
                # Set the input for the next loop
                current_tokens = next_input_tokens
            else:
                # On the final iteration, accept all refined tokens
                current_tokens = refined_tokens
        
        # Create a final mask to ignore padding based on predicted lengths
        # This ensures the output tensor is clean.
        final_mask = torch.arange(max_len_in_batch, device=device)[None, :] >= predicted_lengths[:, None]
        current_tokens.masked_fill_(final_mask, 0) # Use 0 for padding

        return current_tokens

# Example usage:
if __name__ == '__main__':
    batch_size = 4
    vision_dim = 256
    vision_seq_len = 49
    vocab_size = 10000

    # Instantiate the ICMR decoder
    icmr_decoder = ICMRDecoder(vocab_size=vocab_size, embed_dim=vision_dim, vision_dim=vision_dim)
    print("Initialized ICMRDecoder module.\n")

    # Create dummy vision features
    vision_features = torch.randn(batch_size, vision_seq_len, vision_dim)
    print(f"Input vision features shape: {vision_features.shape}\n")

    # --- Forward pass for inference ---
    # The decoder handles length prediction and iterative generation internally
    final_tokens = icmr_decoder(vision_features)
    
    print("--- ICMR Decoder Output ---")
    print(f"Final generated tokens shape: {final_tokens.shape}")
    print("Example output tokens for the batch:")
    print(final_tokens)

    assert final_tokens.shape[0] == batch_size
    assert final_tokens.dim() == 2
    print("\nOutput shape is correct.")

