import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Dict, Any
import math
import warnings


def smooth_minimum(x: torch.Tensor, y: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Smooth approximation to minimum using softmin.
    Provides better gradient flow than torch.minimum().
    
    As temperature -> 0, approaches hard minimum.
    Higher temperature = smoother gradients but less accurate minimum.
    """
    stacked = torch.stack([x, y], dim=-1)
    weights = F.softmax(-stacked / temperature, dim=-1)
    return (stacked * weights).sum(dim=-1)


class TverskyProjection(nn.Module):
    """
    Tversky Projection Layer - Replaces nn.Linear with psychologically-plausible
    similarity computation based on common and distinctive features.
    
    Tversky Similarity: S(a,b) = f(A∩B) / [f(A∩B) + α·f(A-B) + β·f(B-A)]
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_features: int = 256,
        feature_reduction: Literal['sum', 'mean', 'max'] = 'sum',
        learnable_asymmetry: bool = True,
        init_alpha: float = 1.0,
        init_beta: float = 1.0,
        init_gamma: float = 1.0,
        bias: bool = True,
        shared_feature_bank: Optional[nn.Parameter] = None,
        eps: float = 1e-8,
        feature_activation: Literal['relu', 'softplus', 'abs'] = 'softplus',
        use_smooth_min: bool = True,
        smooth_min_temperature: float = 0.5
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_features = num_features
        self.feature_reduction = feature_reduction
        self.eps = eps
        self.feature_activation = feature_activation
        self.use_smooth_min = use_smooth_min
        self.smooth_min_temperature = smooth_min_temperature
        
        # Feature Bank: W_F ∈ R^{d_in × K}
        if shared_feature_bank is not None:
            self.feature_bank = shared_feature_bank
            self.shared_features = True
        else:
            self.feature_bank = nn.Parameter(torch.empty(in_features, num_features))
            self.shared_features = False
            
        # Prototype Bank: W_P ∈ R^{d_out × K}
        self.prototype_bank = nn.Parameter(torch.empty(out_features, num_features))
        
        # Tversky parameters
        if learnable_asymmetry:
            self.alpha_raw = nn.Parameter(torch.tensor(math.log(math.exp(init_alpha) - 1)))
            self.beta_raw = nn.Parameter(torch.tensor(math.log(math.exp(init_beta) - 1)))
        else:
            self.register_buffer('alpha_raw', torch.tensor(math.log(math.exp(init_alpha) - 1)))
            self.register_buffer('beta_raw', torch.tensor(math.log(math.exp(init_beta) - 1)))
            
        self.gamma = nn.Parameter(torch.tensor(init_gamma))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize feature and prototype banks with improved strategies."""
        if not self.shared_features:
            # Use kaiming for feature bank since it's followed by activation
            nn.init.kaiming_uniform_(self.feature_bank, a=math.sqrt(5))
        
        # For large output dimensions (e.g., vocab), use scaled initialization
        # to prevent vanishing/exploding gradients
        if self.out_features > 10000:
            # Scaled initialization for large vocabularies
            std = 0.02  # Similar to BERT/GPT initialization
            nn.init.normal_(self.prototype_bank, mean=0.0, std=std)
        else:
            nn.init.xavier_uniform_(self.prototype_bank)
        
    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.alpha_raw)
    
    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.beta_raw)
        
    def compute_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute non-negative feature activations.
        
        Uses configurable activation to ensure non-negativity while
        preserving gradient flow.
        """
        features = x @ self.feature_bank
        
        if self.feature_activation == 'relu':
            # Simple but causes information loss for negative values
            return F.relu(features)
        elif self.feature_activation == 'softplus':
            # Smooth approximation to ReLU, better gradient flow
            return F.softplus(features)
        elif self.feature_activation == 'abs':
            # Preserves magnitude, symmetric around zero
            return torch.abs(features)
        else:
            return F.softplus(features)  # Default fallback
    
    def tversky_similarity(
        self, 
        x_features: torch.Tensor, 
        p_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Tversky similarity with optional smooth minimum.
        
        Smooth minimum provides better gradient flow during training
        compared to hard minimum which has sparse gradients.
        
        Args:
            x_features: Input features [..., num_features]
            p_features: Prototype features [num_outputs, num_features]
        
        Returns:
            Similarity scores [..., num_outputs]
        """
        # Save original shape for later
        orig_shape = x_features.shape[:-1]  # [...] without features dim
        num_features = x_features.shape[-1]
        num_outputs = p_features.shape[0]
        
        # Flatten x_features to 2D for efficient computation: [N, num_features]
        x_flat = x_features.reshape(-1, num_features)
        
        # Expand for broadcasting: [N, 1, num_features] vs [1, num_outputs, num_features]
        x_expanded = x_flat.unsqueeze(1)  # [N, 1, num_features]
        p_expanded = p_features.unsqueeze(0)  # [1, num_outputs, num_features]
        
        # Common features: min(x_f, p_f)
        if self.use_smooth_min:
            # Smooth minimum for better gradients
            common = smooth_minimum(x_expanded, p_expanded, self.smooth_min_temperature)
        else:
            common = torch.minimum(x_expanded, p_expanded)
        
        # Distinctive features (ReLU is correct here - we want exact zero for non-distinctive)
        x_distinctive = F.relu(x_expanded - p_expanded)
        p_distinctive = F.relu(p_expanded - x_expanded)
        
        if self.feature_reduction == 'sum':
            common_score = common.sum(dim=-1)
            x_dist_score = x_distinctive.sum(dim=-1)
            p_dist_score = p_distinctive.sum(dim=-1)
        elif self.feature_reduction == 'mean':
            common_score = common.mean(dim=-1)
            x_dist_score = x_distinctive.mean(dim=-1)
            p_dist_score = p_distinctive.mean(dim=-1)
        else:
            common_score = common.max(dim=-1)[0]
            x_dist_score = x_distinctive.max(dim=-1)[0]
            p_dist_score = p_distinctive.max(dim=-1)[0]
            
        numerator = common_score
        denominator = common_score + self.alpha * x_dist_score + self.beta * p_dist_score + self.eps
        
        similarity = self.gamma * (numerator / denominator)  # [N, num_outputs]
        
        # Reshape back to original shape: [..., num_outputs]
        output_shape = orig_shape + (num_outputs,)
        return similarity.reshape(output_shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_features = self.compute_feature_activations(x)
        # Use same activation for prototypes as for input features
        if self.feature_activation == 'relu':
            p_features = F.relu(self.prototype_bank)
        elif self.feature_activation == 'softplus':
            p_features = F.softplus(self.prototype_bank)
        else:  # abs
            p_features = torch.abs(self.prototype_bank)
        output = self.tversky_similarity(x_features, p_features)
        
        if self.bias is not None:
            output = output + self.bias
            
        return output
    
    def get_interpretable_features(self, x: torch.Tensor) -> Dict[str, Any]:
        x_features = self.compute_feature_activations(x)
        # Use same activation for prototypes
        if self.feature_activation == 'relu':
            p_features = F.relu(self.prototype_bank)
        elif self.feature_activation == 'softplus':
            p_features = F.softplus(self.prototype_bank)
        else:
            p_features = torch.abs(self.prototype_bank)
        
        x_expanded = x_features.unsqueeze(-2)
        
        return {
            'input_features': x_features,
            'prototype_features': p_features,
            'common_features': torch.minimum(x_expanded, p_features),
            'input_distinctive': F.relu(x_expanded - p_features),
            'prototype_distinctive': F.relu(p_features - x_expanded),
            'alpha': self.alpha.item(),
            'beta': self.beta.item(),
            'gamma': self.gamma.item()
        }
    
    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'num_features={self.num_features}, reduction={self.feature_reduction}, '
                f'activation={self.feature_activation}, smooth_min={self.use_smooth_min}')


class TverskyLMHead(nn.Module):
    """
    Language Model Head using Tversky Projection for OCR.
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_features: int = 512,
        init_from_linear: Optional[nn.Linear] = None,
        feature_activation: str = 'softplus',
        use_smooth_min: bool = True,
        smooth_min_temperature: float = 0.5,
        init_alpha: float = 0.5,
        init_beta: float = 0.5,
        init_gamma: float = 10.0
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_features = num_features
        
        self.tversky_proj = TverskyProjection(
            in_features=hidden_size,
            out_features=vocab_size,
            num_features=num_features,
            feature_reduction='sum',
            learnable_asymmetry=True,
            init_alpha=init_alpha,
            init_beta=init_beta,
            init_gamma=init_gamma,
            bias=False,
            feature_activation=feature_activation,
            use_smooth_min=use_smooth_min,
            smooth_min_temperature=smooth_min_temperature
        )
        
        if init_from_linear is not None:
            self._init_from_linear(init_from_linear)
            
    def _init_from_linear(self, linear: nn.Linear):
        """Initialize from pretrained linear layer using SVD decomposition."""
        with torch.no_grad():
            W = linear.weight.data  # (vocab_size, hidden_size)
            
            min_dim = min(self.hidden_size, self.vocab_size)
            
            if self.num_features < min_dim:
                # SVD decomposition for dimensionality reduction
                try:
                    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                    k = self.num_features
                    
                    # Ensure numerical stability
                    S_sqrt = torch.sqrt(S[:k].clamp(min=1e-8))
                    
                    feature_init = Vh[:k].T * S_sqrt  # (hidden_size, k)
                    prototype_init = U[:, :k] * S_sqrt  # (vocab_size, k)
                    
                    self.tversky_proj.feature_bank.copy_(feature_init)
                    self.tversky_proj.prototype_bank.copy_(prototype_init)
                    
                except RuntimeError as e:
                    warnings.warn(
                        f"SVD initialization failed: {e}. Using default initialization."
                    )
            else:
                # num_features >= min_dim: can't use SVD for reduction
                warnings.warn(
                    f"num_features ({self.num_features}) >= min(hidden_size, vocab_size) ({min_dim}). "
                    f"Using default initialization instead of SVD."
                )
                # Use random projection from linear weights
                # Project to num_features dimensions
                random_proj = torch.randn(min_dim, self.num_features, device=W.device)
                random_proj = random_proj / random_proj.norm(dim=0, keepdim=True)
                
                # Feature bank: random orthogonal projection
                self.tversky_proj.feature_bank.copy_(
                    W.T[:, :min_dim] @ random_proj / math.sqrt(min_dim)
                )
                # Prototype bank: identity-like mapping
                nn.init.normal_(self.tversky_proj.prototype_bank, std=0.02)
                
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.tversky_proj(hidden_states)
    
    def get_character_analysis(
        self, 
        hidden_states: torch.Tensor,
        top_k: int = 5
    ) -> Dict[str, Any]:
        analysis = self.tversky_proj.get_interpretable_features(hidden_states)
        
        logits = self.forward(hidden_states)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = probs.topk(top_k, dim=-1)
        
        analysis['top_predictions'] = top_indices
        analysis['top_probabilities'] = top_probs
        
        return analysis


class TverskyAttentionOutput(nn.Module):
    """
    Replace attention output projection with Tversky layer.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_features: int = 256,
        shared_bank: Optional[nn.Parameter] = None,
        feature_activation: str = 'softplus',
        use_smooth_min: bool = True
    ):
        super().__init__()
        
        self.tversky_proj = TverskyProjection(
            in_features=hidden_size,
            out_features=hidden_size,
            num_features=num_features,
            shared_feature_bank=shared_bank,
            init_gamma=5.0,
            feature_activation=feature_activation,
            use_smooth_min=use_smooth_min
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tversky_proj(x)


class TverskyMoEExpert(nn.Module):
    """
    MoE Expert using Tversky projections.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_features: int = 256,
        activation: str = 'silu',
        shared_bank: Optional[nn.Parameter] = None
    ):
        super().__init__()
        
        self.up_proj = TverskyProjection(
            in_features=hidden_size,
            out_features=intermediate_size,
            num_features=num_features,
            shared_feature_bank=shared_bank
        )
        
        self.gate_proj = TverskyProjection(
            in_features=hidden_size,
            out_features=intermediate_size,
            num_features=num_features,
            shared_feature_bank=shared_bank
        )
        
        self.down_proj = TverskyProjection(
            in_features=intermediate_size,
            out_features=hidden_size,
            num_features=num_features
        )
        
        self.activation = nn.SiLU() if activation == 'silu' else nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))