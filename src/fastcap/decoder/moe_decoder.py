# enhanced-fastcap/src/fastcap/decoder/moe_decoder.py

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
        # --- CORRECTION: Correctly apply the causal self-attention mask ---
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
    A lightweight gating network that dynamically selects experts.
    As described in "Innovation 2," this is a shallow MLP that takes global vision
    features and outputs a softmax distribution over the experts.
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
    Implements the Mixture of Expert (MoE) Decoder.

    This module combines a pool of specialized 'expert' decoders with a gating
    network. At each step, it dynamically weights the outputs of the experts based
    on the input image's features, allowing for adaptive and efficient caption generation.
    It also includes a load balancing loss to ensure experts are utilized effectively.
    """
    def __init__(self, vocab_size, embed_dim=256, num_experts=4, num_layers=4, num_heads=8, max_len=50, dropout=0.1, load_balance_alpha=0.01):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.load_balance_alpha = load_balance_alpha # Coefficient for the load balancing loss
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # This parameter is now removed as dynamic PEs are passed in.
        # self.positional_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        
        self.gating_network = GatingNetwork(embed_dim, num_experts)
        
        # A stack of MoE layers
        self.layers = nn.ModuleList([
            nn.ModuleList([Expert(embed_dim, num_heads, dropout) for _ in range(num_experts)])
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.last_hidden_states = None # To store for SCRModule if needed

    def forward(self, input_ids, vision_features, position_encodings):
        """
        Forward pass for the MoE Decoder.

        Args:
            input_ids (torch.Tensor): Input token ids, shape (B, S)
            vision_features (torch.Tensor): Vision features from backbone, shape (B, N, D)
            position_encodings (torch.Tensor): Dynamic position encodings from DLAG module, shape (B, S, D)
        
        Returns:
            - logits (torch.Tensor): Logits for the next token, shape (B, S, V)
            - aux_loss (torch.Tensor): Auxiliary load balancing loss for training.
            - expert_weights (torch.Tensor): The weights assigned to each expert by the gate.
        """
        B, S = input_ids.shape
        device = input_ids.device
        
        # 1. Get embeddings and add dynamic position encodings
        x = self.token_embedding(input_ids) * (self.embed_dim ** 0.5)
        x = x + position_encodings
        x = self.dropout(x)
        
        # 2. Get expert weights from the gating network
        expert_weights, gate_logits = self.gating_network(vision_features) # (B, E), (B, E)

        # 3. Create the causal self-attention mask
        self_attn_mask = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)

        # 4. Pass through MoE layers
        for layer_experts in self.layers:
            expert_outputs = [expert(x, vision_features, self_attn_mask) for expert in layer_experts]
            expert_outputs_stacked = torch.stack(expert_outputs, dim=-1) # (B, S, D, E)
            
            # 5. Combine expert outputs using a weighted sum
            x = torch.sum(expert_outputs_stacked * expert_weights.view(B, 1, 1, self.num_experts), dim=-1)

        # 6. Calculate the auxiliary load balancing loss
        gate_probs_mean = torch.mean(F.softmax(gate_logits, dim=-1), dim=0)
        gate_weights_mean = torch.mean(expert_weights, dim=0)
        load_balancing_loss = self.num_experts * torch.sum(gate_probs_mean * gate_weights_mean)
        aux_loss = self.load_balance_alpha * load_balancing_loss

        # 7. Final normalization and output projection
        x = self.norm(x)
        self.last_hidden_states = x # Store for SCR loss calculation
        logits = self.output_proj(x)
        
        return logits, aux_loss, expert_weights
        
    def get_last_hidden_states(self):
        """
        Helper method to retrieve the hidden states for the SCRModule.
        """
        return self.last_hidden_states

# Example usage block restored for clarity and testing
if __name__ == '__main__':
    vocab_size = 10000
    embed_dim = 256
    num_experts = 4
    num_layers = 2
    batch_size = 4
    seq_len = 30
    img_feat_len = 49 # e.g., 7x7 grid

    # Create the MoE Decoder module
    moe_decoder = MoEDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_experts=num_experts,
        num_layers=num_layers
    )
    print("Initialized MoEDecoder module.")
    print(f"Parameters: embed_dim={embed_dim}, num_experts={num_experts}, num_layers={num_layers}\n")

    # Create dummy input tensors
    input_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    vision_feats = torch.randn(batch_size, img_feat_len, embed_dim)
    # Create dummy position encodings (normally from DLAG)
    pos_encodings = torch.randn(batch_size, seq_len, embed_dim)
    
    print(f"Input tokens shape: {input_tokens.shape}")
    print(f"Input vision features shape: {vision_feats.shape}")
    print(f"Input position encodings shape: {pos_encodings.shape}\n")

    # --- Forward pass ---
    logits, aux_loss, weights = moe_decoder(input_tokens, vision_feats, pos_encodings)

    print("--- Forward Pass Results ---")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expert weights shape: {weights.shape}")
    print(f"Auxiliary Load Balancing Loss: {aux_loss.item():.4f}")
    print(f"Example weights for first batch item: {weights[0].detach().numpy().round(2)}")
    
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert weights.shape == (batch_size, num_experts)
    assert aux_loss.requires_grad
    print("\nOutput shapes and loss calculation are correct.")
