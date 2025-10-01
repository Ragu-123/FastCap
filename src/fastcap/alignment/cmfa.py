 
# enhanced-fastcap/src/fastcap/alignment/cmfa.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """
    Implements the bidirectional cross-modal attention mechanism.
    
    This module is a key component of CMFA, designed for fine-grained feature
    alignment. It allows vision features to attend to text features and vice-versa,
    creating more contextually aware representations before the contrastive loss.
    This is based on the "Cross-Modal Attention Mechanism" section in Innovation 8.
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.vision_to_text_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.text_to_vision_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, vision_features, text_features, text_padding_mask=None):
        """
        Args:
            vision_features (torch.Tensor): Vision features, shape (B, Nv, D)
            text_features (torch.Tensor): Text features, shape (B, Nt, D)
            text_padding_mask (torch.Tensor, optional): Mask for text padding, shape (B, Nt)
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Aligned vision and text features.
        """
        # Vision attends to Text
        attended_vision, _ = self.vision_to_text_attention(
            query=vision_features,
            key=text_features,
            value=text_features,
            key_padding_mask=text_padding_mask
        )
        aligned_vision = self.norm1(vision_features + attended_vision)

        # Text attends to Vision
        attended_text, _ = self.text_to_vision_attention(
            query=text_features,
            key=vision_features,
            value=vision_features
        )
        aligned_text = self.norm2(text_features + attended_text)

        return aligned_vision, aligned_text


class CrossModalFeatureAlignment(nn.Module):
    """
    Implements the main Cross-Modal Feature Alignment (CMFA) module.

    This module aligns vision and language feature spaces using a bidirectional
    contrastive loss (InfoNCE). It's a crucial training-time component that
    enhances the model's cross-modal understanding, as detailed in "Innovation 8".
    """
    def __init__(self, vision_dim=256, text_dim=256, projection_dim=128, num_heads=8):
        super().__init__()
        
        # Projectors to map vision and text features to a common space
        self.vision_projector = nn.Sequential(
            nn.Linear(vision_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

        # Fine-grained alignment using cross-attention
        self.cross_attention = CrossModalAttention(projection_dim, num_heads)
        
        # Learnable temperature parameter for the contrastive loss
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, vision_features, text_features, text_padding_mask=None):
        """
        Args:
            vision_features (torch.Tensor): Vision features from backbone, shape (B, Nv, Dv)
            text_features (torch.Tensor): Text features from a text encoder, shape (B, Nt, Dt)
            text_padding_mask (torch.Tensor, optional): Mask for text padding, shape (B, Nt)

        Returns:
            Dict containing the total alignment loss and the aligned features.
        """
        # 1. Project features into the common embedding space
        vision_proj = self.vision_projector(vision_features) # (B, Nv, D_proj)
        text_proj = self.text_projector(text_features)     # (B, Nt, D_proj)

        # 2. Perform fine-grained alignment with cross-attention
        aligned_vision, aligned_text = self.cross_attention(
            vision_proj, text_proj, text_padding_mask
        )

        # 3. Pool features to get a single vector representation for each modality
        # For vision, we average across spatial/patch features.
        vision_pooled = aligned_vision.mean(dim=1)
        
        # For text, we use masked average pooling to ignore padding tokens.
        if text_padding_mask is not None:
            # Invert mask: 0 for padding, 1 for real tokens
            mask = ~text_padding_mask.unsqueeze(-1)
            text_lengths = mask.sum(dim=1)
            text_pooled = (aligned_text * mask).sum(dim=1) / text_lengths.clamp(min=1)
        else:
            text_pooled = aligned_text.mean(dim=1)

        # 4. Normalize features for the contrastive loss (L2 normalization)
        vision_norm = F.normalize(vision_pooled, p=2, dim=-1)
        text_norm = F.normalize(text_pooled, p=2, dim=-1)

        # 5. Compute the similarity matrix
        # The core of contrastive learning: sim(v, t)
        similarity_matrix = torch.matmul(vision_norm, text_norm.t()) / self.temperature
        
        # 6. Compute the bidirectional InfoNCE loss
        batch_size = vision_features.size(0)
        labels = torch.arange(batch_size, device=vision_features.device) # Positive pairs are on the diagonal

        # L_CMFA = L(v→t) + L(t→v)
        loss_v2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2v = F.cross_entropy(similarity_matrix.t(), labels)
        total_loss = (loss_v2t + loss_t2v) / 2

        return {
            'alignment_loss': total_loss,
            'aligned_vision': aligned_vision, # Return sequence for the decoder
            'aligned_text': aligned_text,
            'similarity_matrix': similarity_matrix
        }

# Example usage:
if __name__ == '__main__':
    # A simple text encoder for demonstration purposes
    class SimpleTextEncoder(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_layers=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        def forward(self, x):
            return self.transformer(self.embedding(x))

    # Parameters
    batch_size = 8
    vision_seq_len = 49  # e.g., 7x7 grid of patches
    text_seq_len = 30
    vision_dim = 256
    text_dim = 256
    vocab_size = 1000

    # Instantiate modules
    cmfa_module = CrossModalFeatureAlignment(vision_dim, text_dim)
    text_encoder = SimpleTextEncoder(vocab_size, text_dim)

    print("Initialized CrossModalFeatureAlignment module.\n")

    # Create dummy data
    vision_feats = torch.randn(batch_size, vision_seq_len, vision_dim)
    text_tokens = torch.randint(1, vocab_size, (batch_size, text_seq_len)) # 0 is padding
    
    # Create a dummy padding mask (last 10 tokens are padding)
    padding_mask = torch.zeros(batch_size, text_seq_len, dtype=torch.bool)
    padding_mask[:, -10:] = True
    text_tokens.masked_fill_(padding_mask, 0)
    
    # Get text features
    text_feats = text_encoder(text_tokens)

    print(f"Input Vision Features Shape: {vision_feats.shape}")
    print(f"Input Text Features Shape: {text_feats.shape}\n")

    # --- Forward pass through CMFA ---
    output = cmfa_module(vision_feats, text_feats, text_padding_mask=padding_mask)

    print("--- CMFA Module Output ---")
    print(f"Alignment Loss: {output['alignment_loss'].item():.4f}")
    print(f"Aligned Vision Shape: {output['aligned_vision'].shape}")
    print(f"Aligned Text Shape: {output['aligned_text'].shape}")
    print(f"Similarity Matrix Shape: {output['similarity_matrix'].shape}\n")

    assert output['alignment_loss'].requires_grad
    assert output['aligned_vision'].shape[2] == 128 # projection_dim
    assert output['similarity_matrix'].shape == (batch_size, batch_size)
    print("Output shapes and loss calculation are correct.")
