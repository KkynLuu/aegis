"""
AEGIS Models Module: Non-Hermitian Quantum Walk Graph Neural Networks (AEGIS)
and Evaluated Baselines (Unitary CTQW, GCN, GIN, RWPE-GIN, MagNet).
"""

from .aegis_conv import AegisConv, EdgeCentricHoppingEncoder
from .aegis_classifier import AegisClassifier
from .unitary_ctqw_conv import UnitaryCTQWConv
from .unitary_ctqw_classifier import UnitaryCTQWClassifier
from .quasar_conv import QuasarConv
from .quasar_classifier import QuasarClassifier
from .baseline_gnn import (
    GINClassifier,
    GCNClassifier,
    RWPE_GINClassifier,
    MagNetClassifier,
    MagNetConv,
    compute_rwpe,
)

# Backward-compatible aliases
QWConv = UnitaryCTQWConv
QWClassifier = UnitaryCTQWClassifier

__all__ = [
    "AegisConv",
    "AegisClassifier",
    "EdgeCentricHoppingEncoder",
    "UnitaryCTQWConv",
    "UnitaryCTQWClassifier",
    "QuasarConv",
    "QuasarClassifier",
    "QWConv",
    "QWClassifier",
    "GINClassifier",
    "GCNClassifier",
    "RWPE_GINClassifier",
    "MagNetClassifier",
    "MagNetConv",
    "compute_rwpe",
]
