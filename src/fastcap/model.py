# src/fastcap/model.py (Corrected and Final Version)

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.spatial_mamba import SpatialMambaBackbone
from .decoder.moe_decoder import MoEDecoder
from .generation.icmr import ICMRDecoder
from .alignment.cmfa import CrossModalFeatureAlignment
from .losses.scr import SCRModule
from .training.pmtd import PMTDModule
from .generation.dlag import DLAGModule
from einops import rearrange

def adapt_backbone():
    def forward_adapted(self, x):
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        x_seq = self.final_norm(x)
        return x_seq.reshape(x_seq.shape[0], -1, x_seq.shape[-1])

    SpatialMambaBackbone.forward = forward_adapted
    print("Backbone's forward pass has been successfully patched at runtime.")


class SimpleTextEncoder(nn.Module):
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
        causal_mask = torch.triu(torch.ones(S, S, device=captions.device) * float('-inf'), diagonal=1)
        hidden_states = self.transformer_decoder(
            tgt=self.dropout(caption_embeds),
            memory=vision_features,
            tgt_mask=causal_mask
        )
        logits = self.output_proj(hidden_states)
        return {'logits': logits, 'hidden_states': hidden_states}


class EnhancedFastCap(nn.Module):
    def __init__(self, config, tokenizer=None, teacher_models=None):
        super().__init__()
        self.config = config
        self.training_mode = config['training'].get('mode', 'default')
        self.vocab_size = config['model']['vocab_size']
        self.mask_token_id = self.vocab_size - 1

        self.vision_backbone = SpatialMambaBackbone(
            embed_dim=config['vision']['embed_dims'][0], depths=config['vision']['depths']
        )
        adapt_backbone()
        
        self.cmfa_module = CrossModalFeatureAlignment(
            vision_dim=config['vision']['embed_dims'][-1], text_dim=config['model']['embed_dim'],
            projection_dim=config['cmfa']['projection_dim']
        )
        self.text_encoder = SimpleTextEncoder(
            vocab_size=self.vocab_size, embed_dim=config['model']['embed_dim']
        )
        self.dlag_module = DLAGModule(
            vision_dim=config['vision']['embed_dims'][-1], embed_dim=config['model']['embed_dim']
        )
        
        self.training_decoder = MoEDecoder(
            vocab_size=self.vocab_size, 
            embed_dim=config['model']['embed_dim'],
            vision_dim=config['cmfa']['projection_dim'],
            num_experts=config['moe']['num_experts'], 
            num_layers=config['moe']['num_layers'],
            load_balance_alpha=config['moe']['load_balance_alpha']
        )
        
        self.inference_decoder = ICMRDecoder(
            config=config,
            vocab_size=self.vocab_size, 
            embed_dim=config['model']['embed_dim'],
            vision_dim=config['vision']['embed_dims'][-1],
            mask_token_id=self.mask_token_id
        )

        if self.training_mode == 'scr':
            self.ar_decoder = ARDecoder(vocab_size=self.vocab_size, embed_dim=config['model']['embed_dim'])
            self.scr_module = SCRModule(nar_dim=config['model']['embed_dim'], ar_dim=config['model']['embed_dim'], tokenizer=tokenizer)
        if self.training_mode == 'pmtd':
            self.pmtd_module = PMTDModule(student_model=self, teacher_models=teacher_models, student_dim=config['model']['embed_dim'], teacher_dims=config['distillation']['teacher_dims'])

    def forward(self, images, captions=None, padding_mask=None, training_progress=None):
        if self.training:
            return self.default_train_step(images, captions, padding_mask)
        else:
            return self.inference_step(images)

    def default_train_step(self, images, captions, padding_mask):
        vision_features_seq = self.vision_backbone(images)
        
        text_features_seq = self.text_encoder(captions, padding_mask=padding_mask)
        cmfa_output = self.cmfa_module(vision_features_seq, text_features_seq, padding_mask)
        alignment_loss = cmfa_output['alignment_loss']
        aligned_vision_features = cmfa_output['aligned_vision']

        target_seq_len = captions.shape[1] 
        dlag_output = self.dlag_module(vision_features_seq, target_len=target_seq_len)
        position_encodings = dlag_output["position_encodings"]
        
        nar_input_ids = torch.full_like(captions, fill_value=self.mask_token_id)
        
        # --- CRITICAL FIX: Pass the padding_mask to the decoder ---
        nar_logits, moe_aux_loss, _ = self.training_decoder(
            input_ids=nar_input_ids,
            vision_features=aligned_vision_features,
            position_encodings=position_encodings,
            key_padding_mask=padding_mask # This line fixes the error
        )

        caption_loss = F.cross_entropy(
            nar_logits.permute(0, 2, 1),
            captions,
            ignore_index=0
        )
        
        total_loss = caption_loss + alignment_loss + moe_aux_loss
        
        if self.training_mode == 'scr':
            nar_hidden_states = self.training_decoder.get_last_hidden_states()
            ar_outputs = self.ar_decoder(aligned_vision_features, captions)
            nar_outputs_dict = {'logits': nar_logits, 'hidden_states': nar_hidden_states}
            scr_loss_dict = self.scr_module(nar_outputs_dict, ar_outputs)
            total_loss += scr_loss_dict['total_loss']
            return {
                'total_loss': total_loss, 'caption_loss': caption_loss, 
                'alignment_loss': alignment_loss, 'moe_aux_loss': moe_aux_loss, 
                'scr_loss': scr_loss_dict['total_loss']
            }
            
        return {
            'total_loss': total_loss, 'caption_loss': caption_loss, 
            'alignment_loss': alignment_loss, 'moe_aux_loss': moe_aux_loss
        }

    def inference_step(self, images):
        vision_features_seq = self.vision_backbone(images)
        generated_tokens = self.inference_decoder(vision_features_seq)
        return generated_tokens