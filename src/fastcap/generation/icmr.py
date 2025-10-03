# src/fastcap/generation/icmr.py (Corrected and Final Version)

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..decoder.moe_decoder import MoEDecoder 
from .dlag import DLAGModule

class ICMRDecoder(nn.Module):
    """
    Implements the Iterative Conditional Masked Refinement (ICMR) strategy.
    """
    def __init__(self, config, vocab_size, embed_dim, vision_dim, mask_token_id):
        super().__init__()
        self.max_iterations = config['icmr']['max_iterations']
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size

        # --- CORRECTED INITIALIZATION ---
        # The internal NAR decoder now receives all the necessary config parameters.
        self.nar_decoder = MoEDecoder(
            vocab_size=vocab_size, 
            embed_dim=embed_dim,
            vision_dim=vision_dim, # Pass the correct vision dimension
            num_experts=config['moe']['num_experts'],
            num_layers=config['moe']['num_layers'],
            load_balance_alpha=config['moe']['load_balance_alpha']
        )
        
        self.length_predictor = DLAGModule(vision_dim=vision_dim, embed_dim=embed_dim)

    def forward(self, vision_features):
        B = vision_features.shape[0]
        device = vision_features.device

        with torch.no_grad():
            dlag_output = self.length_predictor(vision_features)
            predicted_lengths = dlag_output["predicted_lengths"]
            max_len = predicted_lengths.max().item()

        tokens = torch.full((B, max_len), self.mask_token_id, dtype=torch.long, device=device)

        for i in range(self.max_iterations):
            logits, _, _ = self.nar_decoder(tokens, vision_features)
            predicted_tokens = torch.argmax(logits, dim=-1)
            
            probs = F.softmax(logits, dim=-1)
            token_probs = probs.gather(-1, predicted_tokens.unsqueeze(-1)).squeeze(-1)

            mask_ratio = (1.0 - (i / self.max_iterations)) 
            num_to_mask = int(mask_ratio * max_len)
            
            noise = torch.rand_like(token_probs) * 1e-5
            _, indices_to_mask = torch.topk(token_probs + noise, k=num_to_mask, largest=False, dim=-1)
            
            mask = torch.zeros_like(tokens, dtype=torch.bool)
            mask.scatter_(-1, indices_to_mask, True)

            tokens[mask] = self.mask_token_id
            tokens = torch.where(mask, tokens, predicted_tokens)

        return tokens