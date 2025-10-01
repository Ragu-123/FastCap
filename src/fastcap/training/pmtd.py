 
# enhanced-fastcap/src/fastcap/training/pmtd.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAligner(nn.Module):
    """
    A simple module to align the hidden state dimensions between a teacher
    and the student model if they differ.
    """
    def __init__(self, teacher_dim, student_dim):
        super().__init__()
        self.projection = nn.Linear(teacher_dim, student_dim)
        self.layer_norm = nn.LayerNorm(student_dim)

    def forward(self, teacher_hidden_states):
        return self.layer_norm(self.projection(teacher_hidden_states))

class PMTDModule(nn.Module):
    """
    Implements Progressive Multi-Teacher Distillation (PMTD).

    This module orchestrates the knowledge transfer from multiple teacher models
    to a single student model. It uses a curriculum-based weighting scheme to
    progressively shift focus from simpler to more complex teachers during
    training, as detailed in "Innovation 6".
    """
    def __init__(self, student_model, teacher_models, student_dim, teacher_dims, temperature=4.0, task_loss_weight=0.3):
        super().__init__()
        if not teacher_models or not teacher_dims:
            raise ValueError("teacher_models and teacher_dims cannot be empty.")
        
        self.student = student_model
        # Teachers should be in evaluation mode and their gradients turned off
        self.teachers = nn.ModuleList(teacher_models)
        for teacher in self.teachers:
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
        
        self.num_teachers = len(teacher_models)
        self.temperature = temperature
        self.task_loss_weight = task_loss_weight

        # Learnable curriculum parameters (alpha, beta, tau) as per the spec
        self.alpha = nn.Parameter(torch.ones(self.num_teachers))
        self.beta = nn.Parameter(torch.ones(self.num_teachers) * 0.1)
        # Initialize tau to be spread across the training process [0, 1]
        self.tau = nn.Parameter(torch.linspace(0.1, 0.9, self.num_teachers))

        # Feature aligners for each teacher
        self.feature_aligners = nn.ModuleList([
            FeatureAligner(td, student_dim) if td != student_dim else nn.Identity()
            for td in teacher_dims
        ])

    def compute_teacher_weights(self, training_progress):
        """
        Computes time-varying teacher weights based on the curriculum schedule.
        Formula: w_i(t) = softmax(alpha_i * sigmoid(beta_i * (t - tau_i)))

        Args:
            training_progress (float): A value between 0.0 and 1.0 indicating
                                       the current progress in the training process.
        """
        t = training_progress
        # The core curriculum formula
        influences = self.alpha * torch.sigmoid(self.beta * 10 * (t - self.tau)) # Scale beta for faster transition
        weights = F.softmax(influences, dim=0)
        return weights

    def _compute_distillation_loss(self, student_logits, student_features, teacher_logits, teacher_features):
        """Calculates output and feature distillation losses for one teacher."""
        # 1. Output-level distillation (KL Divergence on soft targets)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KLDivLoss expects log-probs as input and probs as target
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        # Scale loss by temperature squared, as is common practice
        kl_loss *= (self.temperature ** 2)

        # 2. Feature-level distillation (MSE on hidden states)
        feature_loss = F.mse_loss(student_features, teacher_features)

        return kl_loss, feature_loss

    def forward(self, vision_features, target_captions, training_progress):
        """
        Performs a forward pass for PMTD.

        Args:
            vision_features (torch.Tensor): Input from the vision backbone.
            target_captions (torch.Tensor): Ground truth caption tokens for the task loss.
            training_progress (float): Current training progress (e.g., current_step / total_steps).

        Returns:
            A dictionary containing the final combined loss and its components.
        """
        # Get student outputs (assuming the student model returns a dict)
        student_outputs = self.student(vision_features, target_captions)
        student_logits = student_outputs['logits']
        student_features = student_outputs['hidden_states']
        
        # Get teacher weights for the current training step
        teacher_weights = self.compute_teacher_weights(training_progress)

        total_distillation_loss = 0.0
        loss_components = {}
        
        with torch.no_grad(): # Ensure teachers are not trained
            for i, teacher in enumerate(self.teachers):
                teacher_outputs = teacher(vision_features, target_captions)
                teacher_logits = teacher_outputs['logits']
                teacher_features = teacher_outputs['hidden_states']
                
                # Align teacher features to student's dimension
                aligned_teacher_features = self.feature_aligners[i](teacher_features)
                
                # Compute distillation losses for this teacher
                kl_loss, feat_loss = self._compute_distillation_loss(
                    student_logits, student_features, teacher_logits, aligned_teacher_features
                )
                
                # Combine and weight the losses for this teacher
                teacher_distill_loss = (0.7 * kl_loss) + (0.3 * feat_loss)
                total_distillation_loss += teacher_weights[i] * teacher_distill_loss
                
                loss_components[f'teacher_{i}_kl_loss'] = kl_loss.item()
                loss_components[f'teacher_{i}_feat_loss'] = feat_loss.item()
                loss_components[f'teacher_{i}_weight'] = teacher_weights[i].item()

        # Compute the primary task loss (Cross-Entropy on ground truth)
        task_loss = F.cross_entropy(
            student_logits.reshape(-1, student_logits.size(-1)),
            target_captions.reshape(-1),
            ignore_index=0 # Assuming 0 is the padding token
        )

        # Final combined loss
        final_loss = self.task_loss_weight * task_loss + (1 - self.task_loss_weight) * total_distillation_loss
        
        loss_components['task_loss'] = task_loss
        loss_components['total_distill_loss'] = total_distillation_loss
        loss_components['final_loss'] = final_loss

        return loss_components

# Example usage:
if __name__ == '__main__':
    # Mock models for demonstration
    class MockCaptioner(nn.Module):
        def __init__(self, hidden_dim, vocab_size):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.fc_in = nn.Linear(256, hidden_dim)
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        def forward(self, vision_feats, captions):
            # A simplified process for demonstration
            vision_feats_proj = self.fc_in(vision_feats.mean(dim=1)).unsqueeze(1)
            hidden_states, _ = self.gru(vision_feats_proj.repeat(1, captions.size(1), 1))
            logits = self.fc_out(hidden_states)
            return {'logits': logits, 'hidden_states': hidden_states}

    # Parameters
    student_dim, vocab_size = 256, 1000
    teacher_dims = [128, 256, 512] # Teachers with different complexities

    # Create student and teacher models
    student_model = MockCaptioner(student_dim, vocab_size)
    teacher_models = [MockCaptioner(dim, vocab_size) for dim in teacher_dims]
    print(f"Initialized Student (dim={student_dim}) and {len(teacher_models)} Teachers (dims={teacher_dims}).\n")

    # Instantiate PMTD module
    pmtd_module = PMTDModule(
        student_model=student_model,
        teacher_models=teacher_models,
        student_dim=student_dim,
        teacher_dims=teacher_dims
    )
    
    # Dummy inputs
    vision_features = torch.randn(4, 49, 256)
    target_captions = torch.randint(1, vocab_size, (4, 20))

    # --- Simulate training progress ---
    print("--- Simulating Training Progress ---")
    for step in [0.0, 0.3, 0.6, 0.9, 1.0]:
        print(f"\nTraining Progress: {step*100:.0f}%")
        
        # In a real training loop, you would get this from (current_step / total_steps)
        losses = pmtd_module(vision_features, target_captions, training_progress=step)
        
        weights = [v for k, v in losses.items() if 'weight' in k]
        print(f"Teacher Weights: {[round(w, 2) for w in weights]}")
        print(f"Final Combined Loss: {losses['final_loss'].item():.4f}")
        
        # Check that loss requires grad for backpropagation
        assert losses['final_loss'].requires_grad
        
    print("\nPMTD module successfully calculates losses and adapts teacher weights.")
