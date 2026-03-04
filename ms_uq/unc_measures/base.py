from abc import ABC, abstractmethod
from typing import Dict
import torch
import torch.nn as nn

class BaseUncertainty(nn.Module, ABC):
    """Abstract base for uncertainty modules.

    Subclasses must implement forward() and return a dict whose keys contain
    'aleatoric' / 'epistemic' / 'total' substrings for downstream utilities.
    """
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
