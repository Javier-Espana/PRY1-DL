import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TabularDataset(Dataset):
    """
    PyTorch Dataset wrapper for processed tabular features and targets.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def create_dataloaders(X_train, y_train, X_val=None, y_val=None, batch_size=32, shuffle=True):
    """
    Helper function to build PyTorch DataLoaders for train and validation splits.
    """
    train_dataset = TabularDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TabularDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
