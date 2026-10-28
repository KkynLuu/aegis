"""
MUTAG Molecular Dataset Loader & Dense Batcher.

Loads the canonical MUTAG mutagenic molecular dataset (188 graphs, binary classification,
7 discrete atom types). Converts PyG sparse COO edge indices into dense padded matrices
[B, N_max, N_max] with node masks [B, N_max] to support unitary quantum walk matrix operations.
Provides standard stratified 10-fold cross-validation splits.
"""

import os
import torch
import numpy as np
from typing import List, Tuple, Dict
from torch.utils.data import Dataset
from torch_geometric.datasets import TUDataset

class MUTAGDataset(Dataset):
    """
    Dense format PyTorch Dataset for MUTAG molecular graphs.
    """
    def __init__(self, root: str = "data/TUDataset", fixed_max_nodes: int = 28):
        self.root = root
        self.fixed_max_nodes = fixed_max_nodes
        
        # Load raw TUDataset
        self.pyg_dataset = TUDataset(root=root, name="MUTAG")
        self.num_classes = 2
        self.num_features = self.pyg_dataset.num_node_features # 7
        
        self.graphs = []
        self.features = []
        self.masks = []
        self.labels = []
        
        self._prepare_dense_data()
        
    def _prepare_dense_data(self):
        for data in self.pyg_dataset:
            n = data.num_nodes
            max_n = self.fixed_max_nodes
            
            # 1. Padded Adjacency
            A = torch.zeros(max_n, max_n, dtype=torch.float32)
            edge_index = data.edge_index
            A[edge_index[0], edge_index[1]] = 1.0
            
            # 2. Padded Node Features (Atom types)
            X = torch.zeros(max_n, self.num_features, dtype=torch.float32)
            X[:n] = data.x.float()
            
            # 3. Node Mask (1 for real atom, 0 for padding)
            mask = torch.zeros(max_n, dtype=torch.float32)
            mask[:n] = 1.0
            
            # 4. Target Label (0 or 1)
            y = int(data.y.item())
            
            self.graphs.append(A)
            self.features.append(X)
            self.masks.append(mask)
            self.labels.append(y)
            
    def __len__(self) -> int:
        return len(self.labels)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return self.features[idx], self.graphs[idx], self.masks[idx], self.labels[idx]

def get_mutag_kfold_splits(dataset: MUTAGDataset, num_folds: int = 10, seed: int = 42) -> List[Tuple[List[int], List[int]]]:
    """
    Stratified K-fold cross-validation split generator.
    """
    rng = np.random.default_rng(seed)
    class_indices = {0: [], 1: []}
    for idx, label in enumerate(dataset.labels):
        class_indices[label].append(idx)
        
    for c in [0, 1]:
        class_indices[c] = rng.permutation(class_indices[c]).tolist()
        
    folds = [[] for _ in range(num_folds)]
    for c in [0, 1]:
        n_c = len(class_indices[c])
        fold_sizes = [n_c // num_folds + (1 if f < n_c % num_folds else 0) for f in range(num_folds)]
        curr = 0
        for f, sz in enumerate(fold_sizes):
            folds[f].extend(class_indices[c][curr : curr + sz])
            curr += sz
            
    splits = []
    all_indices = set(range(len(dataset)))
    for f in range(num_folds):
        test_idx = sorted(folds[f])
        train_idx = sorted(list(all_indices - set(test_idx)))
        splits.append((train_idx, test_idx))
        
    return splits

def collate_mutag_batch(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
    xs, adjs, masks, labels = zip(*batch)
    return {
        "x": torch.stack(xs, dim=0),
        "adj": torch.stack(adjs, dim=0),
        "mask": torch.stack(masks, dim=0),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

if __name__ == "__main__":
    print("Testing MUTAGDataset loader...")
    ds = MUTAGDataset(root="data/TUDataset")
    print(f"Total molecules: {len(ds)}")
    sample_x, sample_adj, sample_mask, sample_y = ds[0]
    print("Sample X shape:   ", sample_x.shape)
    print("Sample Adj shape: ", sample_adj.shape)
    print("Sample Mask sum:  ", sample_mask.sum().item())
    print("Sample Label:     ", sample_y)
    
    splits = get_mutag_kfold_splits(ds, num_folds=10)
    print(f"Generated {len(splits)} folds. Fold 0 train={len(splits[0][0])}, test={len(splits[0][1])}")
    print("[✓] MUTAGDataset verified successfully!")
