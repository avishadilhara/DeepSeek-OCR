import torch
import torch.nn as nn
from typing import Dict, List, Optional


class SharedFeatureBankManager:
    """
    Manages shared feature banks across multiple Tversky layers.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_features: int,
        num_banks: int = 1,
        bank_names: Optional[List[str]] = None
    ):
        self.input_dim = input_dim
        self.num_features = num_features
        self.num_banks = num_banks
        
        self.bank_names = bank_names or [f'bank_{i}' for i in range(num_banks)]
        self.feature_banks: Dict[str, nn.Parameter] = {}
        
        for name in self.bank_names:
            bank = nn.Parameter(torch.empty(input_dim, num_features))
            nn.init.xavier_uniform_(bank)
            self.feature_banks[name] = bank
            
    def get_bank(self, name: str) -> nn.Parameter:
        return self.feature_banks[name]
    
    def get_default_bank(self) -> nn.Parameter:
        return self.feature_banks[self.bank_names[0]]
    
    def register_to_module(self, module: nn.Module, prefix: str = 'shared_feature_bank'):
        for name, bank in self.feature_banks.items():
            module.register_parameter(f'{prefix}_{name}', bank)
            
    def initialize_from_data(
        self, 
        data_loader,
        model_encoder,
        num_samples: int = 10000,
        device: torch.device = None
    ):
        """
        Initialize feature banks using PCA on encoded representations.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        representations = []
        
        with torch.no_grad():
            sample_count = 0
            for batch in data_loader:
                if sample_count >= num_samples:
                    break
                    
                inputs = batch['input'].to(device) if isinstance(batch, dict) else batch[0].to(device)
                encoded = model_encoder(inputs)
                representations.append(encoded.view(-1, self.input_dim).cpu())
                sample_count += encoded.shape[0]
                
        all_repr = torch.cat(representations, dim=0)[:num_samples]
        
        mean = all_repr.mean(dim=0)
        centered = all_repr - mean
        
        cov = centered.T @ centered / (num_samples - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        top_k_idx = torch.argsort(eigenvalues, descending=True)[:self.num_features]
        init_features = eigenvectors[:, top_k_idx]
        
        for bank in self.feature_banks.values():
            bank.data.copy_(init_features)


class SinhalaFeatureBank(SharedFeatureBankManager):
    """
    Specialized feature bank for Sinhala script OCR.
    
    Sinhala characteristics:
    - 60 basic letters (12 vowels + 48 consonant-vowel combinations)
    - Complex conjunct consonants
    - Vowel signs that attach above, below, or after consonants
    - Unique curves and circular shapes
    """
    
    def __init__(
        self,
        input_dim: int,
        num_features: int = 512
    ):
        super().__init__(
            input_dim=input_dim,
            num_features=num_features,
            num_banks=3,
            bank_names=['stroke_features', 'component_features', 'position_features']
        )
        
        self.stroke_count = num_features // 2
        self.component_count = num_features // 4
        self.position_count = num_features - self.stroke_count - self.component_count
        
    def get_combined_bank(self) -> torch.Tensor:
        """Get concatenated feature bank combining all specialized banks."""
        return torch.cat([
            self.feature_banks['stroke_features'][:, :self.stroke_count],
            self.feature_banks['component_features'][:, :self.component_count],
            self.feature_banks['position_features'][:, :self.position_count]
        ], dim=1)
    
    def get_stroke_bank(self) -> nn.Parameter:
        return self.feature_banks['stroke_features']
    
    def get_component_bank(self) -> nn.Parameter:
        return self.feature_banks['component_features']
    
    def get_position_bank(self) -> nn.Parameter:
        return self.feature_banks['position_features']


class MultiScaleFeatureBank(SharedFeatureBankManager):
    """
    Feature bank with multiple scales for capturing different granularities.
    Useful for OCR where both fine details and overall shapes matter.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_features: int = 512,
        scales: List[str] = None
    ):
        scales = scales or ['fine', 'medium', 'coarse']
        
        super().__init__(
            input_dim=input_dim,
            num_features=num_features,
            num_banks=len(scales),
            bank_names=[f'{s}_scale' for s in scales]
        )
        
        self.scales = scales
        self.features_per_scale = num_features // len(scales)
        
    def get_scale_bank(self, scale: str) -> nn.Parameter:
        return self.feature_banks[f'{scale}_scale']
    
    def get_multi_scale_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute features at each scale."""
        import torch.nn.functional as F
        
        features = {}
        for scale in self.scales:
            bank = self.get_scale_bank(scale)
            features[scale] = F.relu(x @ bank)
            
        return features