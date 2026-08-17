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
    def __init__(self, dim: int, dropout: float = 0.1, activation: str = 'gelu'):
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
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_blocks: int = 2, dropout: float = 0.20, activation: str = 'gelu', **kwargs):
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


class SwiGLUBlock(nn.Module):
    """
    SwiGLU (Swish Gated Linear Unit) Residual Block for Tabular Data.
    """
    def __init__(self, dim: int, dropout: float = 0.15):
        super(SwiGLUBlock, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc_gate = nn.Linear(dim, dim)
        self.fc_val = nn.Linear(dim, dim)
        self.fc_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        normed = self.norm(x)
        gate = F.silu(self.fc_gate(normed))
        val = self.fc_val(normed)
        out = self.fc_out(self.dropout(gate * val))
        return out + residual


class SwiGLUMLP(nn.Module):
    """
    Tabular Multi-Layer Perceptron using SwiGLU gating mechanisms.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_blocks: int = 3, dropout: float = 0.15, **kwargs):
        super(SwiGLUMLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            SwiGLUBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)
        ])
        self.norm_final = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_final(x)
        return self.head(x)


class EmbResNetMLP(nn.Module):
    """
    ResNet-MLP with Trainable Entity Embeddings for Categorical Variables.
    """
    def __init__(self, num_cont_dim: int, emb_dims: list, hidden_dim: int = 512, num_blocks: int = 3, dropout: float = 0.20, **kwargs):
        super(EmbResNetMLP, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(c, e) for c, e in emb_dims
        ])
        self.emb_drop = nn.Dropout(0.25)
        total_emb = sum(e for _, e in emb_dims)
        total_input = num_cont_dim + total_emb

        self.inp_layer = nn.Linear(total_input, hidden_dim)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                'n1': nn.LayerNorm(hidden_dim),
                'f1': nn.Linear(hidden_dim, hidden_dim),
                'n2': nn.LayerNorm(hidden_dim),
                'f2': nn.Linear(hidden_dim, hidden_dim),
                'dr': nn.Dropout(dropout)
            }))
        self.norm_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x_cont, x_cat):
        embs = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x_emb = self.emb_drop(torch.cat(embs, dim=1))
        x = torch.cat([x_cont, x_emb], dim=1)
        x = F.gelu(self.inp_layer(x))
        for blk in self.blocks:
            res = x
            out = F.gelu(blk['f1'](blk['n1'](x)))
            out = blk['dr'](out)
            out = blk['f2'](blk['n2'](out))
            x = F.gelu(out + res)
        return self.head(self.norm_f(x))


class EmbSwiGLUMLP(nn.Module):
    """
    SwiGLU-MLP with Trainable Entity Embeddings for Categorical Variables.
    """
    def __init__(self, num_cont_dim: int, emb_dims: list, hidden_dim: int = 512, num_blocks: int = 3, dropout: float = 0.20, **kwargs):
        super(EmbSwiGLUMLP, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(c, e) for c, e in emb_dims
        ])
        self.emb_drop = nn.Dropout(0.25)
        total_emb = sum(e for _, e in emb_dims)
        total_input = num_cont_dim + total_emb

        self.inp_layer = nn.Linear(total_input, hidden_dim)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                'norm': nn.LayerNorm(hidden_dim),
                'gate': nn.Linear(hidden_dim, hidden_dim),
                'val': nn.Linear(hidden_dim, hidden_dim),
                'out': nn.Linear(hidden_dim, hidden_dim),
                'dr': nn.Dropout(dropout)
            }))
        self.norm_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x_cont, x_cat):
        embs = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x_emb = self.emb_drop(torch.cat(embs, dim=1))
        x = torch.cat([x_cont, x_emb], dim=1)
        x = F.silu(self.inp_layer(x))
        for blk in self.blocks:
            n = blk['norm'](x)
            g = F.silu(blk['gate'](n))
            v = blk['val'](n)
            x = x + blk['out'](blk['dr'](g * v))
        return self.head(self.norm_f(x))


class WideAndDeepMLP(nn.Module):
    """
    Wide & Deep Tabular Architecture.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_blocks: int = 2, dropout: float = 0.2, activation: str = 'gelu', **kwargs):
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


def get_model(architecture: str, input_dim: int = None, **kwargs):
    """
    Factory method for instantiating model architectures.
    """
    arch = architecture.lower()
    if arch == 'resnet':
        return ResNetMLP(input_dim=input_dim, **kwargs)
    elif arch == 'swiglu':
        return SwiGLUMLP(input_dim=input_dim, **kwargs)
    elif arch == 'emb_resnet':
        return EmbResNetMLP(**kwargs)
    elif arch == 'emb_swiglu':
        return EmbSwiGLUMLP(**kwargs)
    elif arch == 'wide_deep':
        return WideAndDeepMLP(input_dim=input_dim, **kwargs)
    elif arch == 'standard':
        return StandardMLP(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
