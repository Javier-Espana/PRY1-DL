import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardMLP(nn.Module):
    """
    Standard Deep Multi-Layer Perceptron for Tabular Regression.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_blocks: int = 3, dropout: float = 0.2, activation: str = 'silu', **kwargs):
        super(StandardMLP, self).__init__()
        layers = []
        in_dim = input_dim

        # Build hidden_dims based on hidden_dim and num_blocks
        hidden_dims = [max(32, hidden_dim // (2**i)) for i in range(num_blocks)]

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            
            if activation.lower() == 'silu':
                layers.append(nn.SiLU())
            elif activation.lower() == 'gelu':
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())
                
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """
    Residual Block for Tabular Data with skip connection and normalization.
    """
    def __init__(self, dim: int, dropout: float = 0.1, activation: str = 'silu'):
        super(ResidualBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        if activation.lower() == 'silu':
            self.act = nn.SiLU()
        elif activation.lower() == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.act(self.fc1(self.norm1(x)))
        out = self.dropout(out)
        out = self.fc2(self.norm2(out))
        return self.act(out + residual)


class ResNetMLP(nn.Module):
    """
    ResNet-Style Architecture for Tabular Regression.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_blocks: int = 3, dropout: float = 0.15, activation: str = 'silu', **kwargs):
        super(ResNetMLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout, activation=activation) for _ in range(num_blocks)
        ])
        self.norm_final = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_final(x)
        return self.output_layer(x)


class WideAndDeepMLP(nn.Module):
    """
    Wide & Deep Architecture.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_blocks: int = 2, dropout: float = 0.2, activation: str = 'silu', **kwargs):
        super(WideAndDeepMLP, self).__init__()
        self.wide = nn.Linear(input_dim, 1)
        
        deep_layers = []
        in_dim = input_dim
        deep_dims = [max(32, hidden_dim // (2**i)) for i in range(num_blocks)]
        
        for h_dim in deep_dims:
            deep_layers.append(nn.Linear(in_dim, h_dim))
            deep_layers.append(nn.LayerNorm(h_dim))
            if activation.lower() == 'silu':
                deep_layers.append(nn.SiLU())
            elif activation.lower() == 'gelu':
                deep_layers.append(nn.GELU())
            else:
                deep_layers.append(nn.ReLU())
            if dropout > 0:
                deep_layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        deep_layers.append(nn.Linear(in_dim, 1))
        
        self.deep = nn.Sequential(*deep_layers)

    def forward(self, x):
        return self.wide(x) + self.deep(x)


def get_model(architecture: str, input_dim: int, **kwargs):
    """
    Factory method for instantiating model architectures.
    """
    arch = architecture.lower()
    if arch == 'resnet':
        return ResNetMLP(input_dim=input_dim, **kwargs)
    elif arch == 'wide_deep':
        return WideAndDeepMLP(input_dim=input_dim, **kwargs)
    elif arch == 'standard':
        return StandardMLP(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
