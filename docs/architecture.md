# FastCap: System Architecture

The FastCap model is a highly efficient, non-autoregressive image captioning system designed for rapid inference on standard CPUs. Its architecture is a synergistic combination of several innovative components that work together to produce high-quality captions with minimal computational overhead.

This document provides a detailed breakdown of each major component, explaining its role and how it is implemented in the codebase.



## Overall Architecture Diagram
![architecture](https://github.com/user-attachments/assets/c062f694-ee66-4896-a1e5-b5b1d239b148)
The system is composed of a vision backbone that extracts features from an image, and two distinct decoding paths: one for efficient training and one for ultra-fast inference.

* **Vision Backbone:** A `SpatialMambaBackbone` processes the input image.
* **Training Path:** For training, vision features are aligned with text features using a `CrossModalFeatureAlignment` module. A `MoEDecoder` then generates logits for calculating losses.
* **Inference Path:** For inference, the `ICMRDecoder` takes the vision features directly and generates a complete caption in a non-autoregressive manner.

---

## 1. Vision Backbone: SpatialMambaBackbone

The foundation of the model is the Spatial-Mamba Vision Backbone, a novel vision encoder that uses State Space Models (SSMs) to achieve linear time complexity, O(n), which is significantly more efficient than the O(n²) complexity of standard Transformers.

**File:** `src/fastcap/backbone/spatial_mamba.py`

**Core Idea:** It processes image patches sequentially using a 4-direction cross-scan pattern (up, down, left, right) to capture comprehensive spatial relationships without the need for a global attention mechanism.

```python
# From: src/fastcap/backbone/spatial_mamba.py

class SpatialMambaBackbone(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.patch_embed = PatchEmbedding(...)
        self.stages = nn.ModuleList([
            # ... multiple SpatialMambaBlocks ...
        ])
        # ... final projection layers ...

    def forward(self, x):
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        # ... pooling and projection ...
        return final_features
```

---

## 2. Training Decoder: MoEDecoder

During training, FastCap uses a Mixture of Expert (MoE) Decoder to generate captions. This is the first MoE architecture specifically designed for image captioning.

**File:** `src/fastcap/decoder/moe_decoder.py`

**Core Idea:** Instead of a single large decoder, the MoE decoder uses a pool of smaller, specialized "expert" networks. A lightweight gating network dynamically selects which experts to use for a given image, enabling conditional computation for greater efficiency.

```python
# From: src/fastcap/decoder/moe_decoder.py

class MoEDecoder(nn.Module):
    def __init__(self, ..., num_experts=4):
        super().__init__()
        self.gating_network = GatingNetwork(...)
        self.layers = nn.ModuleList([
            nn.ModuleList([Expert(...) for _ in range(num_experts)])
        ])

    def forward(self, input_ids, vision_features):
        # Get weights from the gating network
        expert_weights, _ = self.gating_network(vision_features)
        
        # Combine expert outputs with a weighted sum
        for layer_experts in self.layers:
            expert_outputs = [expert(x, vision_features) for expert in layer_experts]
            # ... weighted sum logic ...
        return logits, aux_loss, expert_weights
```

The experts themselves use Rank-Augmented Linear Attention (RALA) (`src/fastcap/decoder/attention.py`), an efficient attention mechanism that preserves quality while maintaining linear complexity.

---

## 3. Inference Decoder: ICMRDecoder

For inference, the model switches to the Iterative Conditional Masked Refinement (ICMR) decoder. This is a non-autoregressive decoder that generates the entire caption in parallel and then refines it over a few steps.

**File:** `src/fastcap/generation/icmr.py`

**Core Idea:**

* It first predicts the optimal caption length using the Dynamic Length-Aware Generation (DLAG) module (`src/fastcap/generation/dlag.py`).
* It initializes a sequence of `[MASK]` tokens of the predicted length.
* It iteratively refines the sequence, at each step re-masking only the tokens with the lowest confidence scores. This allows the model to correct its own errors in parallel.

```python
# From: src/fastcap/generation/icmr.py

class ICMRDecoder(nn.Module):
    def forward(self, vision_features):
        # 1. Predict caption length
        predicted_lengths = self.length_predictor(global_vision_features)
        
        # 2. Initialize with [MASK] tokens
        current_tokens = torch.full((B, max_len_in_batch), self.mask_token_id, ...)
        
        # 3. Iterative Refinement Loop
        for k in range(self.max_iterations):
            refined_tokens, new_probs = self.refinement_step(current_tokens, vision_features)
            ![architecture](https://github.com/user-attachments/assets/3ebaa776-d365-430f-b4cb-3fc798cd9371)

            if k < self.max_iterations - 1:
                # Re-mask low-confidence tokens for the next iteration
                mask = self.generate_mask(token_probs, iteration=k)
                current_tokens = torch.where(mask, self.mask_token_id, refined_tokens)
                # ... update probabilities ...
            else:
                current_tokens = refined_tokens
        
        return current_tokens
```


 
