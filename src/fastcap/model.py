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
from .generation.dlag import DLAGModule

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
    The main Enhanced FastCap model, updated to handle multiple training modes and
    corrected for logical consistency in the training process.
    """
    def __init__(self, config, tokenizer=None, teacher_models=None):
        super().__init__()
        self.config = config
        self.training_mode = config['training'].get('mode', 'default')

        # --- Base Modules (Innovations 1, 2, 4, 7, 8) ---
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
        # Innovation 7: Dynamic Length-Aware Generation Module
        self.dlag_module = DLAGModule(
            vision_dim=config['vision']['embed_dims'][-1], embed_dim=config['model']['embed_dim']
        )
        # Innovation 2: Mixture of Expert Decoder (Non-Autoregressive)
        self.training_decoder = MoEDecoder(
            vocab_size=config['model']['vocab_size'], embed_dim=config['model']['embed_dim'],
            num_experts=config['moe']['num_experts'], num_layers=config['moe']['num_layers'],
            load_balance_alpha=config['moe']['load_balance_alpha']
        )
        # Innovation 4: Iterative Conditional Masked Refinement (for inference)
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
                student_model=self,
                teacher_models=teacher_models,
                student_dim=config['model']['embed_dim'],
                teacher_dims=config['distillation']['teacher_dims']
            )

    def forward(self, images, captions=None, padding_mask=None, training_progress=None):
        # Main forward pass routes to the correct logic based on mode
        if self.training:
            assert captions is not None, "Captions must be provided during training."
            
            # If using PMTD, it wraps the entire training step
            if self.training_mode == 'pmtd':
                return self.pmtd_module(images, captions, training_progress)
            
            # Otherwise, use the default training step (which also handles SCR)
            return self.default_train_step(images, captions, padding_mask)
        else:
            # Inference path is separate and uses the ICMR decoder
            return self.inference_step(images)

    def default_train_step(self, images, captions, padding_mask):
        """
        Defines the training logic for 'default' and 'scr' modes, using the
        corrected non-autoregressive approach.
        """
        # 1. Extract and align vision and text features
        vision_features_seq = self.vision_backbone(images) # Shape: (B, N, D)
        text_features_seq = self.text_encoder(captions, padding_mask=padding_mask)
        
        cmfa_output = self.cmfa_module(vision_features_seq, text_features_seq, padding_mask)
        alignment_loss = cmfa_output['alignment_loss']
        aligned_vision_features = cmfa_output['aligned_vision']

        # 2. **CORE CORRECTION**: Implement true Non-Autoregressive (NAR) training
        # Predict caption lengths and generate dynamic position encodings using DLAG
        dlag_output = self.dlag_module(vision_features_seq)
        position_encodings = dlag_output["position_encodings"]

        # For NAR training, the input is a fully masked sequence of the target length.
        # The model must predict the entire caption from the visual context alone.
        nar_input_ids = torch.full_like(captions, self.config['model']['vocab_size'] - 1) # Use a [MASK] token ID

        # 3. Pass the masked input and vision features to the training decoder (MoE)
        nar_logits, moe_aux_loss, _ = self.training_decoder(
            input_ids=nar_input_ids,
            vision_features=aligned_vision_features,
            position_encodings=position_encodings # Provide the dynamic PEs
        )
        
        # 4. Calculate the primary captioning loss
        # The loss is calculated over the entire sequence in parallel.
        caption_loss = F.cross_entropy(
            nar_logits.permute(0, 2, 1), # Reshape to (B, VocabSize, SeqLen)
            captions,                   # Target is the ground truth (B, SeqLen)
            ignore_index=0              # Assuming 0 is the padding token
        )
        
        # Combine the losses
        total_loss = caption_loss + alignment_loss + moe_aux_loss
        
        # 5. If in SCR mode, compute and add the consistency loss
        if self.training_mode == 'scr':
            # We need the hidden states from the NAR decoder for feature consistency
            nar_hidden_states = self.training_decoder.get_last_hidden_states() # Assumes MoEDecoder has this method
            
            # Get outputs from the auxiliary AR decoder
            ar_outputs = self.ar_decoder(aligned_vision_features, captions)
            
            nar_outputs_dict = {'logits': nar_logits, 'hidden_states': nar_hidden_states}
            
            # Compute SCR loss
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
        """
        Defines the inference logic, which uses the efficient ICMR decoder.
        """
        vision_features_seq = self.vision_backbone(images)
        generated_tokens = self.inference_decoder(vision_features_seq)
        return generated_tokens
