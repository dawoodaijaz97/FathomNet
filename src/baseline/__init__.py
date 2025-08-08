"""
FathomNet Baseline Model Package

This package contains the baseline model implementation for the FathomNet FGVC 2025 competition.
"""

__version__ = "1.0.0"
__author__ = "FathomNet Competition Team"

from .model import BaselineModel
from .dataset import FathomNetDataset
from .evaluate import HierarchicalEvaluator
from .config import BaselineConfig

__all__ = ["BaselineModel", "FathomNetDataset", "HierarchicalEvaluator", "BaselineConfig"]