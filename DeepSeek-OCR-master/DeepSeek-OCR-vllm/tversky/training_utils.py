import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Dict, Any, List, Optional
import json
import os


def create_tversky_optimizer(
    model: nn.Module,
    base_lr: float = 1e-4,
    tversky_lr_multiplier: float = 0.1,
    weight_decay: float = 0.01
) -> torch.optim.Optimizer:
    """
    Create optimizer with different learning rates for Tversky parameters.
    """
    
    tversky_params = []
    other_params = []
    
    tversky_param_names = {'alpha_raw', 'beta_raw', 'gamma', 'feature_bank', 'prototype_bank'}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        param_name = name.split('.')[-1]
        if param_name in tversky_param_names:
            tversky_params.append(param)
        else:
            other_params.append(param)
            
    param_groups = [
        {'params': other_params, 'lr': base_lr, 'weight_decay': weight_decay},
        {'params': tversky_params, 'lr': base_lr * tversky_lr_multiplier, 'weight_decay': 0.0}
    ]
    
    return AdamW(param_groups)


def get_tversky_regularization_loss(
    model: nn.Module,
    diversity_weight: float = 0.01,
    sparsity_weight: float = 0.001
) -> torch.Tensor:
    """
    Compute regularization losses for Tversky layers.
    """
    
    device = next(model.parameters()).device
    diversity_loss = torch.tensor(0.0, device=device)
    sparsity_loss = torch.tensor(0.0, device=device)
    num_tversky_layers = 0
    
    for module in model.modules():
        if hasattr(module, 'prototype_bank'):
            num_tversky_layers += 1
            
            prototypes = module.prototype_bank
            prototypes_norm = prototypes / (prototypes.norm(dim=1, keepdim=True) + 1e-8)
            similarity_matrix = prototypes_norm @ prototypes_norm.T
            
            mask = 1 - torch.eye(similarity_matrix.shape[0], device=device)
            diversity_loss = diversity_loss + (similarity_matrix * mask).abs().mean()
            
            sparsity_loss = sparsity_loss + prototypes.abs().mean()
            
    if num_tversky_layers > 0:
        diversity_loss = diversity_loss / num_tversky_layers
        sparsity_loss = sparsity_loss / num_tversky_layers
        
    return diversity_weight * diversity_loss + sparsity_weight * sparsity_loss


def analyze_tversky_parameters(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """Analyze learned Tversky parameters."""
    
    analysis = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'alpha') and hasattr(module, 'beta'):
            layer_analysis = {
                'alpha': module.alpha.item() if hasattr(module.alpha, 'item') else float(module.alpha),
                'beta': module.beta.item() if hasattr(module.beta, 'item') else float(module.beta),
                'gamma': module.gamma.item() if hasattr(module.gamma, 'item') else float(module.gamma)
            }
            
            if hasattr(module, 'prototype_bank'):
                prototypes = module.prototype_bank.data
                layer_analysis['prototype_sparsity'] = (prototypes.abs() < 0.01).float().mean().item()
                layer_analysis['prototype_norm_mean'] = prototypes.norm(dim=1).mean().item()
                layer_analysis['prototype_norm_std'] = prototypes.norm(dim=1).std().item()
                
            if hasattr(module, 'feature_bank') and not module.shared_features:
                features = module.feature_bank.data
                layer_analysis['feature_sparsity'] = (features.abs() < 0.01).float().mean().item()
                layer_analysis['feature_norm_mean'] = features.norm(dim=0).mean().item()
                
            analysis[name] = layer_analysis
                
    return analysis


def monitor_tversky_health(model):
    """Monitor Tversky layer health with memory optimization"""
    warnings = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'prototype_bank'):
            # FIXED: Move to CPU and process in chunks to avoid OOM
            with torch.no_grad():
                protos = module.prototype_bank.data.cpu()  # Move to CPU
                
                # Check for dead prototypes
                proto_norms = protos.norm(dim=-1)
                dead_protos = (proto_norms < 0.01).sum().item()
                if dead_protos > 0:
                    warnings.append(f"{name}: {dead_protos} dead prototypes")
                
                # FIXED: Only check similarity if not too large
                num_protos = protos.shape[0]
                if num_protos < 1000:  # Only for manageable sizes
                    protos_norm = protos / (protos.norm(dim=-1, keepdim=True) + 1e-8)
                    
                    # Compute in chunks to avoid OOM
                    chunk_size = 100
                    max_sim = 0
                    for i in range(0, num_protos, chunk_size):
                        chunk = protos_norm[i:i+chunk_size]
                        sim_chunk = torch.mm(chunk, protos_norm.T)
                        sim_chunk.fill_diagonal_(0)
                        max_sim = max(max_sim, sim_chunk.max().item())
                    
                    if max_sim > 0.95:
                        warnings.append(f"{name}: High prototype similarity: {max_sim:.3f}")
                else:
                    warnings.append(f"{name}: Skipping similarity check (too many prototypes: {num_protos})")
        
        if hasattr(module, 'alpha'):
            alpha = module.alpha.item() if module.alpha.numel() == 1 else module.alpha.mean().item()
            beta = module.beta.item() if module.beta.numel() == 1 else module.beta.mean().item()
            
            if alpha < 0 or beta < 0:
                warnings.append(f"{name}: Negative Tversky params (α={alpha:.3f}, β={beta:.3f})")
    
    return warnings


class TverskyTrainingConfig:
    """Configuration for training with Tversky layers."""
    
    def __init__(
        self,
        # Tversky architecture
        num_features: int = 512,
        conversion_strategy: str = 'lm_head_only',
        feature_reduction: str = 'sum',
        feature_activation: str = 'softplus',  # New: 'relu', 'softplus', 'abs'
        use_smooth_min: bool = True,           # New: better gradient flow
        smooth_min_temperature: float = 0.5,   # New: temperature for smooth min
        
        # Tversky initialization
        init_alpha: float = 0.5,
        init_beta: float = 0.5,
        init_gamma: float = 10.0,
        
        # Training
        base_lr: float = 1e-4,
        tversky_lr_multiplier: float = 0.1,
        warmup_steps: int = 1000,
        max_epochs: int = 50,
        
        # Regularization
        diversity_weight: float = 0.01,
        sparsity_weight: float = 0.001,
        
        # Gradual unfreezing
        freeze_tversky_epochs: int = 0,
        
        # Model config (can be overridden at runtime)
        hidden_size: Optional[int] = None,   # Changed: None means auto-detect
        vocab_size: Optional[int] = None,    # Changed: None means auto-detect
    ):
        self.num_features = num_features
        self.conversion_strategy = conversion_strategy
        self.feature_reduction = feature_reduction
        self.feature_activation = feature_activation
        self.use_smooth_min = use_smooth_min
        self.smooth_min_temperature = smooth_min_temperature
        self.init_alpha = init_alpha
        self.init_beta = init_beta
        self.init_gamma = init_gamma
        self.base_lr = base_lr
        self.tversky_lr_multiplier = tversky_lr_multiplier
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.diversity_weight = diversity_weight
        self.sparsity_weight = sparsity_weight
        self.freeze_tversky_epochs = freeze_tversky_epochs
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TverskyTrainingConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__init__.__code__.co_varnames})
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, path: str) -> 'TverskyTrainingConfig':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# Pre-defined configurations
SINHALA_OCR_TVERSKY_CONFIG = TverskyTrainingConfig(
    num_features=512,
    conversion_strategy='lm_head_only',
    feature_reduction='sum',
    feature_activation='softplus',   # Better gradient flow than relu
    use_smooth_min=True,
    smooth_min_temperature=0.5,
    init_alpha=0.3,
    init_beta=0.7,
    init_gamma=15.0,
    base_lr=5e-5,
    tversky_lr_multiplier=0.05,
    warmup_steps=2000,
    max_epochs=50,
    diversity_weight=0.02,
    sparsity_weight=0.001,
    freeze_tversky_epochs=1,
    hidden_size=None,  # Auto-detect from model
    vocab_size=None    # Auto-detect from model
)

AGGRESSIVE_TVERSKY_CONFIG = TverskyTrainingConfig(
    num_features=256,
    conversion_strategy='attention_output',
    feature_reduction='sum',
    feature_activation='softplus',
    use_smooth_min=True,
    smooth_min_temperature=0.3,  # Lower temp = closer to hard min
    init_alpha=0.5,
    init_beta=0.5,
    init_gamma=10.0,
    base_lr=1e-4,
    tversky_lr_multiplier=0.1,
    warmup_steps=1000,
    max_epochs=30,
    diversity_weight=0.05,
    sparsity_weight=0.005,
    freeze_tversky_epochs=2,
    hidden_size=None,
    vocab_size=None
)

CONSERVATIVE_TVERSKY_CONFIG = TverskyTrainingConfig(
    num_features=1024,
    conversion_strategy='lm_head_only',
    feature_reduction='sum',
    feature_activation='softplus',
    use_smooth_min=True,
    smooth_min_temperature=1.0,  # Higher temp = smoother
    init_alpha=0.5,
    init_beta=0.5,
    init_gamma=20.0,
    base_lr=2e-5,
    tversky_lr_multiplier=0.02,
    warmup_steps=5000,
    max_epochs=100,
    diversity_weight=0.01,
    sparsity_weight=0.0005,
    freeze_tversky_epochs=0,
    hidden_size=None,
    vocab_size=None
)