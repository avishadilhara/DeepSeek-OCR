import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List

from .layers import TverskyProjection, TverskyLMHead, TverskyAttentionOutput, TverskyMoEExpert
from .feature_banks import SharedFeatureBankManager, SinhalaFeatureBank


def convert_decoder_to_tversky(
    decoder_model: nn.Module,
    config: Dict[str, Any],
    strategy: str = 'lm_head_only'
) -> nn.Module:
    """
    Convert DeepSeek-OCR decoder to use Tversky projections.
    
    Args:
        decoder_model: Original DeepSeek MoE decoder
        config: Model configuration dict
        strategy: Conversion strategy
            - 'lm_head_only': Only convert final LM head (safest)
            - 'attention_output': Also convert attention output projections
            - 'full': Convert all applicable linear layers
            
    Returns:
        Modified decoder with Tversky layers
    """
    
    hidden_size = config.get('hidden_size', 2048)
    vocab_size = config.get('vocab_size', 102400)
    num_features = config.get('tversky_num_features', 512)
    
    feature_bank_manager = SinhalaFeatureBank(
        input_dim=hidden_size,
        num_features=num_features
    )
    
    feature_bank_manager.register_to_module(decoder_model)
    
    # Get additional Tversky parameters from config
    feature_activation = config.get('feature_activation', 'softplus')
    use_smooth_min = config.get('use_smooth_min', True)
    smooth_min_temperature = config.get('smooth_min_temperature', 0.5)
    init_alpha = config.get('init_alpha', 0.5)
    init_beta = config.get('init_beta', 0.5)
    init_gamma = config.get('init_gamma', 10.0)
    
    # Strategy 1: LM Head Only
    if hasattr(decoder_model, 'lm_head'):
        old_lm_head = decoder_model.lm_head
        
        new_lm_head = TverskyLMHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_features=num_features,
            init_from_linear=old_lm_head if isinstance(old_lm_head, nn.Linear) else None,
            feature_activation=feature_activation,
            use_smooth_min=use_smooth_min,
            smooth_min_temperature=smooth_min_temperature,
            init_alpha=init_alpha,
            init_beta=init_beta,
            init_gamma=init_gamma
        )
        
        decoder_model.lm_head = new_lm_head
        
        old_params = sum(p.numel() for p in old_lm_head.parameters())
        new_params = sum(p.numel() for p in new_lm_head.parameters())
        print(f"Converted LM head to Tversky projection")
        print(f"  Original params: {old_params:,}")
        print(f"  Tversky params: {new_params:,}")
        print(f"  Reduction: {(1 - new_params/old_params)*100:.1f}%")
    
    if strategy == 'lm_head_only':
        return decoder_model
        
    # Strategy 2: Attention output projections
    if strategy in ['attention_output', 'full']:
        shared_bank = feature_bank_manager.get_default_bank()
        converted_count = 0
        
        for name, module in list(decoder_model.named_modules()):
            if 'o_proj' in name.lower() and isinstance(module, nn.Linear):
                try:
                    parent_name = '.'.join(name.split('.')[:-1])
                    attr_name = name.split('.')[-1]
                    
                    if parent_name:
                        parent = decoder_model.get_submodule(parent_name)
                    else:
                        parent = decoder_model
                    
                    new_proj = TverskyAttentionOutput(
                        hidden_size=module.in_features,
                        num_features=num_features // 2,
                        shared_bank=shared_bank,
                        feature_activation=feature_activation,
                        use_smooth_min=use_smooth_min
                    )
                    
                    setattr(parent, attr_name, new_proj)
                    converted_count += 1
                except Exception as e:
                    print(f"Could not convert {name}: {e}")
                    
        print(f"Converted {converted_count} attention output projections")
                    
    # Strategy 3: Full conversion
    if strategy == 'full':
        print("Warning: Full conversion is experimental")
        # Additional conversions can be added here
        pass
        
    return decoder_model


def get_tversky_layers(model: nn.Module) -> List[nn.Module]:
    """Get all Tversky layers in a model."""
    tversky_layers = []
    
    for module in model.modules():
        if isinstance(module, (TverskyProjection, TverskyLMHead, 
                               TverskyAttentionOutput, TverskyMoEExpert)):
            tversky_layers.append(module)
            
    return tversky_layers


def count_tversky_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters in Tversky layers vs other layers."""
    tversky_params = 0
    other_params = 0
    
    tversky_param_names = {'alpha_raw', 'beta_raw', 'gamma', 'feature_bank', 'prototype_bank'}
    
    for name, param in model.named_parameters():
        param_name = name.split('.')[-1]
        if param_name in tversky_param_names:
            tversky_params += param.numel()
        else:
            other_params += param.numel()
            
    return {
        'tversky_params': tversky_params,
        'other_params': other_params,
        'total_params': tversky_params + other_params,
        'tversky_ratio': tversky_params / (tversky_params + other_params) if (tversky_params + other_params) > 0 else 0
    }


class DeepSeekOCRWithTversky(nn.Module):
    """
    Complete DeepSeek-OCR model with Tversky-enhanced decoder.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        tversky_config: Dict[str, Any]
    ):
        super().__init__()
        
        self.encoder = encoder
        
        self.decoder = convert_decoder_to_tversky(
            decoder,
            tversky_config,
            strategy=tversky_config.get('conversion_strategy', 'lm_head_only')
        )
        
        self.tversky_config = tversky_config
        
    def forward(
        self,
        images: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        vision_tokens = self.encoder(images)
        
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=vision_tokens,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to vision tokens."""
        return self.encoder(images)
    
    def get_feature_analysis(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        position: int = -1
    ) -> Dict[str, Any]:
        """
        Get interpretable feature analysis for a prediction.
        """
        with torch.no_grad():
            vision_tokens = self.encoder(images)
            
            if hasattr(self.decoder, 'get_hidden_states'):
                hidden_states = self.decoder.get_hidden_states(
                    input_ids=input_ids,
                    encoder_hidden_states=vision_tokens
                )
            else:
                outputs = self.decoder(
                    input_ids=input_ids,
                    encoder_hidden_states=vision_tokens,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
                
            if hidden_states is not None and hasattr(self.decoder, 'lm_head'):
                if hasattr(self.decoder.lm_head, 'get_character_analysis'):
                    return self.decoder.lm_head.get_character_analysis(
                        hidden_states[:, position:position+1, :]
                    )
                
        return {}
    
    def get_parameter_stats(self) -> Dict[str, Any]:
        """Get parameter statistics for the model."""
        return count_tversky_parameters(self)


class TverskyModelWrapper:
    """
    Wrapper for easy integration with existing DeepSeek-OCR pipeline.
    Provides utilities for saving/loading Tversky weights separately.
    """
    
    def __init__(
        self,
        base_model_path: str,
        tversky_config: Optional[Dict[str, Any]] = None
    ):
        self.base_model_path = base_model_path
        self.tversky_config = tversky_config or {
            'hidden_size': 2048,
            'vocab_size': 102400,
            'tversky_num_features': 512,
            'conversion_strategy': 'lm_head_only'
        }
        self.model = None
        
    def load_and_convert(self, base_model: nn.Module, device: str = 'cuda') -> nn.Module:
        """
        Convert a loaded base model to use Tversky layers.
        
        Args:
            base_model: Already loaded DeepSeek-OCR model
            device: Target device
            
        Returns:
            Model with Tversky layers
        """
        self.model = convert_decoder_to_tversky(
            decoder_model=base_model,
            config=self.tversky_config,
            strategy=self.tversky_config.get('conversion_strategy', 'lm_head_only')
        )
        self.model = self.model.to(device)
        return self.model
        
    def save_tversky_weights(self, path: str):
        """Save only the Tversky-specific weights."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_and_convert first.")
            
        tversky_state = {}
        tversky_keys = {'alpha_raw', 'beta_raw', 'gamma', 'feature_bank', 'prototype_bank'}
        
        for name, param in self.model.named_parameters():
            if any(k in name for k in tversky_keys):
                tversky_state[name] = param.data.cpu()
                
        torch.save({
            'tversky_state_dict': tversky_state,
            'tversky_config': self.tversky_config
        }, path)
        
    def load_tversky_weights(self, path: str) -> Dict[str, List[str]]:
        """Load Tversky-specific weights."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_and_convert first.")
            
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        
        missing, loaded = [], []
        for name, param in checkpoint['tversky_state_dict'].items():
            try:
                self.model.get_parameter(name).data.copy_(param)
                loaded.append(name)
            except Exception:
                missing.append(name)
                
        return {'missing': missing, 'loaded': loaded}