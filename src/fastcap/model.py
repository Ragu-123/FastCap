# enhanced-fastcap/src/fastcap/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import all innovation modules
from .backbone.spatial_mamba import SpatialMambaBackbone
from .decoder.moe_decoder import MoEDecoder
from .generation.icmr import ICMRDecoder
from .alignment.cmfa import CrossModalFeatureAlignment
from .losses.scr import SCRModule
from .training.pmtd import PMTDModule

class SimpleTextEncoder(nn.Module):
    """
    A simple text encoder required for training the CMFA module.
    It converts token IDs into feature representations.
    """
    def __init__(self, vocab_size, embed_dim, num_layers=2, num_heads=8, max_len=50, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids, padding_mask=None):
        B, S = token_ids.shape
        x = self.embedding(token_ids) + self.positional_embedding[:, :S]
        x = self.transformer_encoder(self.dropout(x), src_key_padding_mask=padding_mask)
        return x

class ARDecoder(nn.Module):
    """
    A simple Autoregressive (AR) decoder used as a consistency target for SCR training.
    """
    def __init__(self, vocab_size, embed_dim, num_layers=4, num_heads=8, max_len=50, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vision_features, captions):
        B, S = captions.shape
        caption_embeds = self.embedding(captions) + self.positional_embedding[:, :S]
        
        # Create a causal mask for the decoder
        causal_mask = torch.triu(torch.ones(S, S, device=captions.device) * float('-inf'), diagonal=1)
        
        hidden_states = self.transformer_decoder(
            tgt=self.dropout(caption_embeds),
            memory=vision_features,
            tgt_mask=causal_mask
        )
        logits = self.output_proj(hidden_states)
        return {'logits': logits, 'hidden_states': hidden_states}


class EnhancedFastCap(nn.Module):
    """
    The main Enhanced FastCap model, updated to handle multiple training modes.
    """
    def __init__(self, config, tokenizer=None, teacher_models=None):
        super().__init__()
        self.config = config
        self.training_mode = config['training'].get('mode', 'default')

        # --- Base Modules (Innovations 1, 2, 3, 4, 7, 8) ---
        self.vision_backbone = SpatialMambaBackbone(
            embed_dims=config['vision']['embed_dims'], depths=config['vision']['depths']
        )
        self.cmfa_module = CrossModalFeatureAlignment(
            vision_dim=config['vision']['embed_dims'][-1], text_dim=config['model']['embed_dim'],
            projection_dim=config['cmfa']['projection_dim']
        )
        self.text_encoder = SimpleTextEncoder(
            vocab_size=config['model']['vocab_size'], embed_dim=config['model']['embed_dim']
        )
        self.training_decoder = MoEDecoder( # This is our NAR decoder
            vocab_size=config['model']['vocab_size'], embed_dim=config['model']['embed_dim'],
            num_experts=config['moe']['num_experts'], num_layers=config['moe']['num_layers'],
            load_balance_alpha=config['moe']['load_balance_alpha']
        )
        self.inference_decoder = ICMRDecoder(
            vocab_size=config['model']['vocab_size'], embed_dim=config['model']['embed_dim'],
            vision_dim=config['vision']['embed_dims'][-1],
            max_iterations=config['icmr']['max_iterations']
        )

        # --- Conditional Modules for Advanced Training (Innovations 5, 6) ---
        if self.training_mode == 'scr':
            print("Initializing model in SCR training mode.")
            self.ar_decoder = ARDecoder(
                vocab_size=config['model']['vocab_size'], embed_dim=config['model']['embed_dim']
            )
            self.scr_module = SCRModule(
                nar_dim=config['model']['embed_dim'], ar_dim=config['model']['embed_dim'],
                tokenizer=tokenizer
            )

        if self.training_mode == 'pmtd':
            print("Initializing model in PMTD training mode.")
            self.pmtd_module = PMTDModule(
                student_model=self, # The module will call the student's base forward pass
                teacher_models=teacher_models,
                student_dim=config['model']['embed_dim'],
                teacher_dims=config['distillation']['teacher_dims'],
                temperature=config['distillation']['temperature'],
                task_loss_weight=config['distillation']['task_loss_weight']
            )

    def forward(self, images, captions=None, padding_mask=None, training_progress=None):
        # The forward pass now handles different logic based on the training mode.
        if self.training:
            assert captions is not None, "Captions must be provided during training."
            
            # The PMTD module wraps the entire training logic, so we call it first.
            if self.training_mode == 'pmtd':
                return self.pmtd_module(images, captions, training_progress)
            
            # --- Default and SCR training path ---
            return self.default_train_step(images, captions, padding_mask)
        else:
            # --- Inference Path ---
            return self.inference_step(images)

    def default_train_step(self, images, captions, padding_mask):
        # This function contains the logic for both 'default' and 'scr' training modes.
        vision_features_seq = self.vision_backbone(images, return_sequence=True)
        text_features_seq = self.text_encoder(captions, padding_mask=padding_mask)
        
        cmfa_output = self.cmfa_module(vision_features_seq, text_features_seq, padding_mask)
        alignment_loss = cmfa_output['alignment_loss']
        aligned_vision_features = cmfa_output['aligned_vision']

        decoder_input = captions[:, :-1]
        nar_logits, moe_aux_loss, _ = self.training_decoder(
            input_ids=decoder_input, vision_features=aligned_vision_features
        )
        
        caption_loss = F.cross_entropy(
            nar_logits.reshape(-1, self.config['model']['vocab_size']),
            captions[:, 1:].reshape(-1), ignore_index=0
        )
        
        total_loss = caption_loss + alignment_loss + moe_aux_loss
        
        # If in SCR mode, calculate and add the consistency loss
        if self.training_mode == 'scr':
            nar_hidden_states = self.training_decoder.get_last_hidden_states() # Assumes MoEDecoder exposes this
            ar_outputs = self.ar_decoder(aligned_vision_features, decoder_input)
            
            nar_outputs_dict = {'logits': nar_logits, 'hidden_states': nar_hidden_states}
            
            scr_loss_dict = self.scr_module(nar_outputs_dict, ar_outputs)
            total_loss += scr_loss_dict['total_loss']
            
            return {
                'caption_loss': caption_loss, 'alignment_loss': alignment_loss,
                'moe_aux_loss': moe_aux_loss, 'scr_loss': scr_loss_dict['total_loss'],
                'total_loss': total_loss
            }
            
        return {
            'caption_loss': caption_loss, 'alignment_loss': alignment_loss,
            'moe_aux_loss': moe_aux_loss, 'total_loss': total_loss
        }

    def inference_step(self, images):
        vision_features_seq = self.vision_backbone(images, return_sequence=True)
        generated_tokens = self.inference_decoder(vision_features_seq)
        return generated_tokens

# A small adaptation to the SpatialMambaBackbone to return sequence features
def adapt_backbone():
    def forward_adapted(self, x, return_sequence=False):
        # ... (implementation from previous model.py version) ...
        x = self.patch_embed(x)
        # ... stages ...
        x_seq = self.final_norm(x)
        if return_sequence:
            return x_seq.flatten(1, 2)
        # ... pooling and projection ...
        return self.feature_proj(x_seq.mean(dim=(1,2)))
    SpatialMambaBackbone.forward = forward_adapted

# Note: For this modular design to work, the train.py script would need to be
# aware of the 'pmtd' mode and pass in teacher models during model initialization.
# The MoEDecoder would also need a method like `get_last_hidden_states()` for SCR.

