# src/fastcap/decoder/attention.py (Corrected and Final Version)

import torch
import torch.nn as nn
import torch.nn.functional as F

class RankAugmentedLinearAttention(nn.Module):
    """
    Implements Rank-Augmented Linear Attention, now with support for separate
    key/value dimensions to enable robust cross-modal attention.
    """
    def __init__(self, embed_dim, num_heads, kv_embed_dim=None, rank=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # If no separate key/value dimension is provided, assume it's the same as the query dimension
        if kv_embed_dim is None:
            kv_embed_dim = embed_dim
        
        self.kv_embed_dim = kv_embed_dim
        
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # --- CRITICAL FIX: Use the correct dimensions for projection layers ---
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(kv_embed_dim, embed_dim) # Key can have a different input dimension
        self.v_proj = nn.Linear(kv_embed_dim, embed_dim) # Value can have a different input dimension
        
        self.q_lora = nn.Sequential(nn.Linear(embed_dim, rank, bias=False), nn.Linear(rank, embed_dim, bias=False))
        self.k_lora = nn.Sequential(nn.Linear(kv_embed_dim, rank, bias=False), nn.Linear(rank, embed_dim, bias=False))
        self.v_lora = nn.Sequential(nn.Linear(kv_embed_dim, rank, bias=False), nn.Linear(rank, embed_dim, bias=False))

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        B, S_q, _ = query.shape
        B, S_k, _ = key.shape
        
        # Project and augment query, key, and value
        q = self.q_proj(query) + self.q_lora(query)
        k = self.k_proj(key) + self.k_lora(key)
        v = self.v_proj(value) + self.v_lora(value)
        
        # Reshape for multi-head attention
        q = q.view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply padding mask to keys
        if key_padding_mask is not None:
            k = k.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(-1), float('-inf'))

        # Linear attention mechanism
        k_norm = F.softmax(k, dim=-2)
        q_norm = F.softmax(q, dim=-1)
        
        context_v = torch.matmul(k_norm.transpose(-1, -2), v)
        output = torch.matmul(q_norm, context_v)
        
        # Reshape and project output
        output = output.transpose(1, 2).reshape(B, S_q, self.embed_dim)
        output = self.out_proj(output)
        
        return self.dropout(output)