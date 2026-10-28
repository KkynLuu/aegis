"""
Circular Skip Links (CSL) Dataset Generator.

Generates the canonical 10-class CSL graph classification benchmark (Murphy et al., ICML 2019;
Dwivedi et al., 2020). Each graph has N=41 nodes and is 4-regular, formed by a cycle (skip 1)
and a skip link R in {2, 3, 4, 5, 6, 9, 11, 12, 13, 16}.
For each class, 15 isomorphic graphs are generated via random node permutations.
Total: 150 graphs. 1-WL is theoretically blind to all 10 classes (accuracy = 10.0%).
"""

import os
import torch
import numpy as np
from typing import List, Tuple, Dict
from torch.utils.data import Dataset

CSL_SKIP_LENGTHS = [2, 3, 4, 5, 6, 9, 11, 12, 13, 16]
NUM_CLASSES = 10
NUM_GRAPHS_PER_CLASS = 15
NUM_NODES = 41

def generate_canonical_csl_adjacency(R: int, N: int = NUM_NODES) -> np.ndarray:
    """Generates the canonical 4-regular adjacency matrix for skip length R."""
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        A[i, (i + 1) % N] = 1.0
        A[i, (i - 1) % N] = 1.0
        A[i, (i + R) % N] = 1.0
        A[i, (i - R) % N] = 1.0
    return A

def permute_graph(A: np.ndarray, seed: int = None) -> np.ndarray:
    """Applies a random node permutation to adjacency matrix A."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(A.shape[0])
    return A[np.ix_(perm, perm)]

class CSLDataset(Dataset):
    """
    PyTorch Dataset for the Circular Skip Links (CSL) benchmark.
    """
    def __init__(self, node_dim: int = 1, seed: int = 42):
        self.node_dim = node_dim
        self.seed = seed
        self.graphs: List[torch.Tensor] = []     # [N, N] adjacency matrices
        self.features: List[torch.Tensor] = []   # [N, node_dim] uniform features
        self.labels: List[int] = []              # class index 0..9
        
        self._generate_dataset()
        
    def _generate_dataset(self):
        rng = np.random.default_rng(self.seed)
        for class_idx, R in enumerate(CSL_SKIP_LENGTHS):
            A_canonical = generate_canonical_csl_adjacency(R)
            for g_idx in range(NUM_GRAPHS_PER_CLASS):
                # Isomorphic permutation
                perm_seed = int(rng.integers(0, 1000000))
                A_perm = permute_graph(A_canonical, seed=perm_seed)
                
                # Uniform uninformative node features (e.g. constant 1s)
                # This ensures the model MUST rely on graph topology, not node features
                x = torch.ones((NUM_NODES, self.node_dim), dtype=torch.float32)
                
                self.graphs.append(torch.from_numpy(A_perm))
                self.features.append(x)
                self.labels.append(class_idx)
                
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            x: Node features [N, node_dim]
            adj: Adjacency matrix [N, N]
            label: Class index (0..9)
        """
        return self.features[idx], self.graphs[idx], self.labels[idx]

def get_csl_kfold_splits(dataset: CSLDataset, num_folds: int = 5, seed: int = 42) -> List[Tuple[List[int], List[int]]]:
    """
    Generates balanced, stratified K-fold cross-validation index splits.
    Each fold contains exactly (NUM_GRAPHS_PER_CLASS // num_folds) graphs per class in test set.
    """
    rng = np.random.default_rng(seed)
    class_indices = {c: [] for c in range(NUM_CLASSES)}
    for idx, label in enumerate(dataset.labels):
        class_indices[label].append(idx)
        
    # Shuffle indices within each class
    for c in range(NUM_CLASSES):
        class_indices[c] = rng.permutation(class_indices[c]).tolist()
        
    folds = [[] for _ in range(num_folds)]
    for c in range(NUM_CLASSES):
        per_fold = len(class_indices[c]) // num_folds
        for f in range(num_folds):
            test_subset = class_indices[c][f * per_fold : (f + 1) * per_fold]
            folds[f].extend(test_subset)
            
    splits = []
    all_indices = set(range(len(dataset)))
    for f in range(num_folds):
        test_idx = sorted(folds[f])
        train_idx = sorted(list(all_indices - set(test_idx)))
        splits.append((train_idx, test_idx))
        
    return splits

def collate_csl_batch(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
    """
    Batches CSL graphs densely.
    Returns:
        x: [B, N, node_dim]
        adj: [B, N, N]
        labels: [B]
    """
    xs, adjs, labels = zip(*batch)
    return {
        "x": torch.stack(xs, dim=0),
        "adj": torch.stack(adjs, dim=0),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

if __name__ == "__main__":
    print("Testing CSL Dataset Generation...")
    ds = CSLDataset(node_dim=4, seed=42)
    print(f"Total graphs: {len(ds)} (Expected: 150)")
    sample_x, sample_adj, sample_y = ds[0]
    print(f"Sample X shape:   {sample_x.shape} (Expected: [41, 4])")
    print(f"Sample Adj shape: {sample_adj.shape} (Expected: [41, 41])")
    print(f"Sample label:     {sample_y}")
    
    # Verify degree regularity
    degrees = sample_adj.sum(dim=-1)
    assert torch.all(degrees == 4.0), "All nodes must have degree 4!"
    print("[✓] 4-regularity confirmed for all nodes.")
    
    # Verify 5-fold split
    splits = get_csl_kfold_splits(ds, num_folds=5)
    print(f"Generated {len(splits)} folds.")
    print(f"Fold 0: Train size={len(splits[0][0])}, Test size={len(splits[0][1])}")
    assert len(splits[0][0]) == 120 and len(splits[0][1]) == 30
    print("[✓] Balanced 5-fold cross-validation splits confirmed!")
