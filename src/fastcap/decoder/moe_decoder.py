# src/fastcap/decoder/moe_decoder.py (Corrected and Final Version)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Each expert is a Transformer layer, which uses our custom attention
from .attention import RankAugmentedLinearAttention

class Expert(nn.Module):
    """
    A single expert module, implemented as a standard Transformer decoder layer.
    It uses Rank-Augmented Linear Attention for both self-attention and cross-attention
    to maintain high efficiency.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = RankAugmentedLinearAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attn = RankAugmentedLinearAttention(embed_dim, num_heads, dropout=dropout)
        
        # Feed-forward network
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

    def forward(self, x, vision_features, self_attn_mask=None):
        """
        Args:
            x (torch.Tensor): Input tensor from previous layer (Batch, SeqLen, Dim)
            vision_features (torch.Tensor): Vision features from backbone (Batch, ImgFeatLen, Dim)
            self_attn_mask (torch.Tensor, optional): Causal mask for self-attention.
        """
        # Self-attention block with the provided mask
        attn_output = self.self_attn(x, x, x, key_padding_mask=self_attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention block (attending to vision features)
        attn_output = self.cross_attn(query=x, key=vision_features, value=vision_features)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward block
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x


class GatingNetwork(nn.Module):
    """
    A lightweight gating network that dynamically selects experts based on global image features.
    """
    def __init__(self, vision_dim, num_experts):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, 128)
        self.fc2 = nn.Linear(128, num_experts)

    def forward(self, vision_features):
        # Use the global representation of the image (e.g., pooled features)
        global_vision_features = vision_features.mean(dim=1) # (B, SeqLen, Dim) -> (B, Dim)
        x = F.relu(self.fc1(global_vision_features))
        # Return raw logits for loss calculation and softmax for weighting
        logits = self.fc2(x)
        weights = F.softmax(logits, dim=-1)
        return weights, logits


class MoEDecoder(nn.Module):
    """
    Implements the Mixture of Expert (MoE) Decoder, updated to integrate with
    DLAG and SCR.
    """
    def __init__(self, vocab_size, embed_dim=256, num_experts=4, num_layers=4, num_heads=8, max_len=50, dropout=0.1, load_balance_alpha=0.01):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.load_balance_alpha = load_balance_alpha # Coefficient for the load balancing loss
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # This static positional embedding serves as a fallback if dynamic PEs are not provided.
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        
        self.gating_network = GatingNetwork(embed_dim, num_experts)
        
        # A stack of MoE layers
        self.layers = nn.ModuleList([
            nn.ModuleList([Expert(embed_dim, num_heads, dropout) for _ in range(num_experts)])
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Attribute to store the last hidden states for SCR compatibility
        self.last_hidden_states = None

    def forward(self, input_ids, vision_features, position_encodings=None):
        """
        Forward pass for the MoE Decoder.

        Args:
            input_ids (torch.Tensor): Input token ids, shape (Batch, SeqLen)
            vision_features (torch.Tensor): Vision features from backbone, shape (Batch, ImgFeatLen, Dim)
            position_encodings (torch.Tensor, optional): Dynamically generated position encodings from DLAG.
        
        Returns:
            - logits (torch.Tensor): Logits for the next token, shape (Batch, SeqLen, VocabSize)
            - aux_loss (torch.Tensor): Auxiliary load balancing loss for training.
            - expert_weights (torch.Tensor): The weights assigned to each expert by the gate.
        """
        B, S = input_ids.shape
        device = input_ids.device
        
        # 1. Get token embeddings
        x = self.token_embedding(input_ids) * (self.embed_dim ** 0.5)
        
        # 2. Add position encodings (dynamic from DLAG if provided, else static)
        if position_encodings is not None:
            x = x + position_encodings[:, :S]
        else:
            x = x + self.positional_embedding[:, :S]
        x = self.dropout(x)
        
        # 3. Get expert weights from the gating network
        expert_weights, gate_logits = self.gating_network(vision_features)

        # 4. Create a causal mask for decoder self-attention
        causal_mask = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)

        # 5. Pass through MoE layers
        for layer_experts in self.layers:
            expert_outputs = [expert(x, vision_features, self_attn_mask=causal_mask) for expert in layer_experts]
            expert_outputs_stacked = torch.stack(expert_outputs, dim=-1) # (B, S, D, E)
            
            # Combine expert outputs using a weighted sum
            x = torch.sum(expert_outputs_stacked * expert_weights.view(B, 1, 1, self.num_experts), dim=-1)

        # 6. Calculate the auxiliary load balancing loss
        gate_probs_mean = torch.mean(F.softmax(gate_logits, dim=-1), dim=0)
        gate_weights_mean = torch.mean(expert_weights, dim=0)
        load_balancing_loss = self.num_experts * torch.sum(gate_probs_mean * gate_weights_mean)
        aux_loss = self.load_balance_alpha * load_balancing_loss

        # 7. Final normalization and output projection
        x = self.norm(x)
        
        # 8. Store the hidden states before the final projection for SCR
        self.last_hidden_states = x
        
        logits = self.output_proj(x)
        
        return logits, aux_loss, expert_weights
    
    def get_last_hidden_states(self):
        """
        Exposes the final hidden states of the decoder for use in the SCR loss.
        """
        return self.last_hidden_states
