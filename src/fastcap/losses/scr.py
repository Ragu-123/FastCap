 
# enhanced-fastcap/src/fastcap/losses/scr.py

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: `sentence-transformers` is not installed. SCR's semantic loss will not work.")
    print("Please install it: `pip install -U sentence-transformers`")
    SentenceTransformer = None

class SemanticEncoder(nn.Module):
    """
    A wrapper for a pre-trained SentenceBERT model to compute semantic embeddings.
    This is a core component for calculating the semantic consistency loss.
    The model is frozen by default as we only use it for inference.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cpu'):
        super().__init__()
        if SentenceTransformer is None:
            raise ImportError("SentenceTransformer library not found.")
        self.model = SentenceTransformer(model_name, device=device)
        # Freeze the model parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, sentences):
        # The model expects a list of strings
        with torch.no_grad():
            embeddings = self.model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
        return embeddings

class FeatureAligner(nn.Module):
    """
    A simple module to align the hidden state dimensions of the NAR and AR decoders.
    As specified in the SCR design document.
    """
    def __init__(self, nar_dim, ar_dim):
        super().__init__()
        self.projection = nn.Linear(nar_dim, ar_dim)
        self.layer_norm = nn.LayerNorm(ar_dim)

    def forward(self, nar_hidden_states):
        return self.layer_norm(self.projection(nar_hidden_states))


class SCRModule(nn.Module):
    """
    Implements the Semantic Consistency Regularization (SCR) loss.

    This module enforces consistency between a non-autoregressive (NAR) and an
    autoregressive (AR) decoder during training. The loss is a weighted sum of
    output-level, feature-level, and semantic-level consistency losses, as detailed
    in "Innovation 5".
    """
    def __init__(self, nar_dim, ar_dim, semantic_device='cpu', tokenizer=None):
        super().__init__()
        if tokenizer is None:
            raise ValueError("SCRModule requires a tokenizer to compute semantic loss.")
        
        self.tokenizer = tokenizer
        
        # Helper modules
        self.feature_aligner = FeatureAligner(nar_dim, ar_dim) if nar_dim != ar_dim else nn.Identity()
        self.semantic_encoder = SemanticEncoder(device=semantic_device)

        # Learnable weights for the three loss components, as per the spec
        self.lambda_output = nn.Parameter(torch.tensor(1.0))
        self.lambda_feature = nn.Parameter(torch.tensor(0.5))
        self.lambda_semantic = nn.Parameter(torch.tensor(0.3))

    def _logits_to_text(self, logits):
        """Helper to convert batch of logits to a list of text sentences."""
        pred_token_ids = torch.argmax(logits, dim=-1)
        return self.tokenizer.batch_decode(pred_token_ids, skip_special_tokens=True)

    def forward(self, nar_outputs, ar_outputs):
        """
        Calculates the total SCR loss.

        Args:
            nar_outputs (dict): A dictionary from the NAR decoder containing:
                                - 'logits': (B, S, V)
                                - 'hidden_states': (B, S, D_nar)
            ar_outputs (dict): A dictionary from the AR decoder containing:
                               - 'logits': (B, S, V)
                               - 'hidden_states': (B, S, D_ar)
        
        Returns:
            dict: A dictionary containing the total loss and its components.
        """
        nar_logits, nar_hidden = nar_outputs['logits'], nar_outputs['hidden_states']
        ar_logits, ar_hidden = ar_outputs['logits'], ar_outputs['hidden_states']

        # The AR path provides a stable target, so we detach its outputs from the graph
        ar_logits_detached = ar_logits.detach()
        ar_hidden_detached = ar_hidden.detach()

        # 1. Output Consistency Loss (L_output)
        nar_probs = F.softmax(nar_logits, dim=-1)
        ar_probs = F.softmax(ar_logits_detached, dim=-1)
        loss_output = F.mse_loss(nar_probs, ar_probs)

        # 2. Feature Consistency Loss (L_feature)
        nar_hidden_aligned = self.feature_aligner(nar_hidden)
        loss_feature = F.mse_loss(nar_hidden_aligned, ar_hidden_detached)

        # 3. Semantic Consistency Loss (L_semantic)
        nar_captions = self._logits_to_text(nar_logits)
        ar_captions = self._logits_to_text(ar_logits_detached)
        
        nar_embeddings = self.semantic_encoder(nar_captions)
        ar_embeddings = self.semantic_encoder(ar_captions)
        
        # loss = 1 - cosine_similarity
        loss_semantic = 1 - F.cosine_similarity(nar_embeddings, ar_embeddings, dim=-1).mean()

        # 4. Weighted combination of all losses
        total_loss = (self.lambda_output * loss_output +
                      self.lambda_feature * loss_feature +
                      self.lambda_semantic * loss_semantic)
                      
        return {
            'total_loss': total_loss,
            'loss_output': loss_output,
            'loss_feature': loss_feature,
            'loss_semantic': loss_semantic
        }

# Example usage:
if __name__ == '__main__':
    # A mock tokenizer for demonstration
    class MockTokenizer:
        def batch_decode(self, token_ids, skip_special_tokens=True):
            return [f"caption for sample {i}" for i in range(token_ids.shape[0])]

    # Parameters
    batch_size = 4
    seq_len = 20
    vocab_size = 1000
    nar_dim = 256
    ar_dim = 512 # Assume different hidden dims to test the aligner

    # Instantiate the SCR module
    scr_module = SCRModule(nar_dim=nar_dim, ar_dim=ar_dim, tokenizer=MockTokenizer())
    print("Initialized SCRModule.\n")
    
    # Create dummy outputs from NAR and AR decoders
    nar_outputs = {
        'logits': torch.randn(batch_size, seq_len, vocab_size),
        'hidden_states': torch.randn(batch_size, seq_len, nar_dim)
    }
    ar_outputs = {
        'logits': torch.randn(batch_size, seq_len, vocab_size),
        'hidden_states': torch.randn(batch_size, seq_len, ar_dim)
    }

    print(f"NAR logits shape: {nar_outputs['logits'].shape}")
    print(f"AR hidden states shape: {ar_outputs['hidden_states'].shape}\n")

    # --- Calculate SCR Loss ---
    loss_dict = scr_module(nar_outputs, ar_outputs)

    print("--- SCR Module Output ---")
    print(f"Total SCR Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"  - Output Loss: {loss_dict['loss_output'].item():.4f} (Weight: {scr_module.lambda_output.item():.2f})")
    print(f"  - Feature Loss: {loss_dict['loss_feature'].item():.4f} (Weight: {scr_module.lambda_feature.item():.2f})")
    print(f"  - Semantic Loss: {loss_dict['loss_semantic'].item():.4f} (Weight: {scr_module.lambda_semantic.item():.2f})")
    
    # Check that the total loss requires gradients for backpropagation
    assert loss_dict['total_loss'].requires_grad
    print("\nLoss calculation is correct and gradients are attached.")
