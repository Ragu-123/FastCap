# src/fastcap/decoder/moe_decoder.py (Corrected and Final Version)

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import RankAugmentedLinearAttention

class Expert(nn.Module):
    """A single expert module, implemented as a standard Transformer decoder layer."""
    def __init__(self, embed_dim, vision_dim, num_heads, dropout=0.1):
        super().__init__()
        # Self-attention uses the main embedding dimension for all inputs
        self.self_attn = RankAugmentedLinearAttention(embed_dim, num_heads, dropout=dropout)
        
        # --- CRITICAL FIX: Initialize cross-attention with the correct vision dimension ---
        self.cross_attn = RankAugmentedLinearAttention(embed_dim, num_heads, kv_embed_dim=vision_dim, dropout=dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim), nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, vision_features, key_padding_mask=None):
        attn_output = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # This call now works because cross_attn was initialized correctly
        attn_output = self.cross_attn(query=x, key=vision_features, value=vision_features)
        x = self.norm2(x + self.dropout(attn_output))
        
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x


class GatingNetwork(nn.Module):
    """A lightweight gating network that dynamically selects experts."""
    def __init__(self, vision_dim, num_experts):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, 128)
        self.fc2 = nn.Linear(128, num_experts)

    def forward(self, vision_features):
        global_vision_features = vision_features.mean(dim=1)
        x = F.relu(self.fc1(global_vision_features))
        logits = self.fc2(x)
        weights = F.softmax(logits, dim=-1)
        return weights, logits


class MoEDecoder(nn.Module):
    """Implements the Mixture of Expert (MoE) Decoder."""
    def __init__(self, vocab_size, embed_dim, vision_dim, num_experts, num_layers, load_balance_alpha, num_heads=8, max_len=50, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.load_balance_alpha = load_balance_alpha
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.gating_network = GatingNetwork(vision_dim, num_experts)
        
        # --- CRITICAL FIX: Pass the vision_dim to the Experts ---
        self.layers = nn.ModuleList([
            nn.ModuleList([Expert(embed_dim, vision_dim, num_heads, dropout) for _ in range(num_experts)])
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.last_hidden_states = None

    def forward(self, input_ids, vision_features, position_encodings=None, key_padding_mask=None):
        B, S = input_ids.shape
        x = self.token_embedding(input_ids) * (self.embed_dim ** 0.5)
        
        if position_encodings is not None:
            x += position_encodings[:, :S]
        else:
            x += self.positional_embedding[:, :S]
        x = self.dropout(x)
        
        expert_weights, gate_logits = self.gating_network(vision_features)

        for layer_experts in self.layers:
            expert_outputs = [expert(x, vision_features, key_padding_mask=key_padding_mask) for expert in layer_experts]
            expert_outputs_stacked = torch.stack(expert_outputs, dim=-1)
            x = torch.sum(expert_outputs_stacked * expert_weights.view(B, 1, 1, self.num_experts), dim=-1)

        gate_probs_mean = torch.mean(F.softmax(gate_logits, dim=-1), dim=0)
        gate_weights_mean = torch.mean(expert_weights, dim=0)
        load_balancing_loss = self.num_experts * torch.sum(gate_probs_mean * gate_weights_mean)
        aux_loss = self.load_balance_alpha * load_balancing_loss

        x = self.norm(x)
        self.last_hidden_states = x
        logits = self.output_proj(x)
        
        return logits, aux_loss, expert_weights
    
    def get_last_hidden_states(self):
        return self.last_hidden_states