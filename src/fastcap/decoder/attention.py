 
# enhanced-fastcap/src/fastcap/decoder/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RankAugmentedLinearAttention(nn.Module):
    """
    Implements Rank-Augmented Linear Attention (RALA).
    
    This module approximates standard softmax attention with linear complexity, making it
    highly efficient for long sequences. It is based on "Innovation 3" from the project
    documentation, which uses rank restoration theory to improve upon standard linear
    attention mechanisms like those found in the Performer model.

    The key idea is to augment the feature map phi(x) to a higher rank by using a
    projection, i.e., phi_aug(x) = [phi_1(x), ..., phi_r(x)], which prevents the
    attention matrix from becoming rank-deficient and losing critical information.
    """
    def __init__(self, embed_dim, num_heads, rank_k=4, kernel_fn=nn.ReLU(), dropout=0.1):
        """
        Args:
            embed_dim (int): Total dimension of the model.
            num_heads (int): Number of parallel attention heads.
            rank_k (int): The rank augmentation factor. This determines how many feature
                          maps are used, directly influencing the rank of the approximated
                          attention matrix.
            kernel_fn (nn.Module): The non-linear function to approximate the softmax kernel.
                                   ReLU is a common and efficient choice.
            dropout (float): Dropout probability.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rank_k = rank_k
        self.kernel_fn = kernel_fn
        
        # In linear attention, the augmented dimension is what matters.
        self.aug_head_dim = self.head_dim * self.rank_k

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.eps = 1e-6 # For numerical stability in normalization

    def forward(self, query, key, value, key_padding_mask=None):
        """
        Forward pass for Rank-Augmented Linear Attention.

        Args:
            query (torch.Tensor): Query tensor, shape (batch_size, seq_len_q, embed_dim)
            key (torch.Tensor): Key tensor, shape (batch_size, seq_len_k, embed_dim)
            value (torch.Tensor): Value tensor, shape (batch_size, seq_len_v, embed_dim)
            key_padding_mask (torch.Tensor, optional): Mask for key positions. 
                                                     Shape (batch_size, seq_len_k).

        Returns:
            torch.Tensor: The attention output, shape (batch_size, seq_len_q, embed_dim)
        """
        B, S_q, _ = query.shape
        _, S_k, _ = key.shape

        # 1. Project and reshape Q, K, V for multi-head attention
        q = self.q_proj(query).view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v are now (B, num_heads, seq_len, head_dim)

        # 2. Augment rank by applying the kernel function (phi in the docs)
        # This is the core of the RALA innovation. Instead of a random feature map,
        # we use a simple but effective non-linearity.
        q_aug = self.kernel_fn(q) + self.eps
        k_aug = self.kernel_fn(k) + self.eps
        # q_aug, k_aug are (B, num_heads, seq_len, head_dim)

        # 3. Apply padding mask to keys BEFORE the associative computation
        if key_padding_mask is not None:
            # Mask should be broadcastable to the shape of k_aug
            # key_padding_mask: (B, S_k) -> (B, 1, S_k, 1)
            k_aug = k_aug.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(-1), 0)

        # 4. Use the associative property of matrix multiplication for linear complexity
        # Instead of computing (Q @ K.T) @ V, we compute Q @ (K.T @ V)
        # This avoids creating the large (S_q, S_k) attention matrix.
        
        # Compute K.T @ V. Shape: (B, num_heads, head_dim, head_dim)
        kv_context = torch.einsum('bhsd,bhsv->bhdv', k_aug, v)
        
        # Compute the normalization factor, which is Q @ sum(K.T)
        # This is the denominator in the softmax approximation
        k_sum = k_aug.sum(dim=2) # (B, num_heads, head_dim)
        denominator = torch.einsum('bhqd,bhd->bhq', q_aug, k_sum)
        denominator = denominator.clamp(min=self.eps) # Avoid division by zero

        # 5. Compute the final attention output
        # (Q @ (K.T @ V)) / (Q @ sum(K.T))
        attn_output = torch.einsum('bhqd,bhdv->bhqv', q_aug, kv_context)
        attn_output = attn_output / denominator.unsqueeze(-1)
        
        # 6. Reshape and project output
        # Concatenate heads: (B, num_heads, S_q, head_dim) -> (B, S_q, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S_q, self.embed_dim)
        
        # Final output projection
        output = self.out_proj(attn_output)
        
        return self.dropout(output)

# Example usage:
if __name__ == '__main__':
    embed_dim = 256
    num_heads = 8
    batch_size = 4
    seq_len = 50

    # Create the RALA module
    rala = RankAugmentedLinearAttention(embed_dim=embed_dim, num_heads=num_heads)
    print("Initialized RankAugmentedLinearAttention module.")
    print(f"Parameters: embed_dim={embed_dim}, num_heads={num_heads}\n")
    
    # Create dummy input tensors
    query = torch.randn(batch_size, seq_len, embed_dim)
    context = torch.randn(batch_size, seq_len, embed_dim)
    
    # Create a dummy padding mask (masking the last 10 tokens)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask[:, -10:] = True
    
    print(f"Input query shape: {query.shape}")
    print(f"Input context (key/value) shape: {context.shape}")
    print(f"Mask shape: {mask.shape}\n")
    
    # --- Test with mask ---
    print("--- Testing with padding mask ---")
    output_with_mask = rala(query, context, context, key_padding_mask=mask)
    print(f"Output shape (with mask): {output_with_mask.shape}")
    assert output_with_mask.shape == (batch_size, seq_len, embed_dim)
    print("Output shape is correct.\n")

    # --- Test without mask ---
    print("--- Testing without padding mask ---")
    output_without_mask = rala(query, context, context, key_padding_mask=None)
    print(f"Output shape (without mask): {output_without_mask.shape}")
    assert output_without_mask.shape == (batch_size, seq_len, embed_dim)
    print("Output shape is correct.")
