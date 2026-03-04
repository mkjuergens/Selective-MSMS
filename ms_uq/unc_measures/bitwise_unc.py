from __future__ import annotations
from typing import Dict, Literal

import torch
from torch import Tensor

from ms_uq.unc_measures.base import BaseUncertainty
from ms_uq.unc_measures.decomposition import decompose_binary


class BitwiseUncertainty(BaseUncertainty):
    """
    Bitwise uncertainty decomposition for ensemble fingerprint predictions.
    
    Supports sparse fingerprints via weighting options.
    """
    
    def __init__(
        self, 
        aggregate: Literal["mean", "sum"] = "mean",
        kind: Literal["entropy", "variance", "both"] = "entropy",
        weighting: Literal["none", "active_bit", "logit"] = "none",
    ):
        """
        Parameters
        ----------
        aggregate : "mean" or "sum"
            How to aggregate across bits.
        kind : "entropy", "variance", or "both"
            Which uncertainty decomposition to compute.
        weighting : "none", "active_bit", or "logit"
            - "none": Standard uncertainty (biased for sparse fingerprints)
            - "active_bit": Weight by mean prediction (ignores inactive bits)
            - "logit": Use log-odds variance (amplifies differences near 0/1)
        """
        super().__init__()
        self.aggregate = aggregate
        self.kind = kind
        self.weighting = weighting
    
    def forward(self, Pbits: Tensor) -> Dict[str, Tensor]:
        if Pbits.dim() == 2:
            Pbits = Pbits.unsqueeze(1)
        
        N, S, K = Pbits.shape
        
        # Standard decomposition
        decomp = decompose_binary(Pbits, sample_dim=1, reduce_dim=None)
        
        # Apply weighting for sparse fingerprints
        if self.weighting == "active_bit":
            w = decomp.mean  # (N, K)
            decomp.total_variance *= w
            decomp.aleatoric_variance *= w
            decomp.epistemic_variance *= w
            decomp.total_entropy *= w
            decomp.aleatoric_entropy *= w
            decomp.epistemic_entropy *= w
            
        elif self.weighting == "logit":
            # Compute logit-space epistemic variance (sparse-aware)
            logit_ep = _logit_epistemic_variance(Pbits)  # (N, K)
            activity = Pbits.max(dim=1).values  # (N, K) - potential activity
            decomp.epistemic_variance = logit_ep * activity
        
        # Aggregate
        agg = lambda x: x.mean(dim=-1) if self.aggregate == "mean" else x.sum(dim=-1)
        
        result = {}
        if self.kind in ["variance", "both"]:
            result["total_var"] = agg(decomp.total_variance)
            result["aleatoric_var"] = agg(decomp.aleatoric_variance)
            result["epistemic_var"] = agg(decomp.epistemic_variance)
            
        if self.kind in ["entropy", "both"]:
            result["total"] = agg(decomp.total_entropy)
            result["aleatoric"] = agg(decomp.aleatoric_entropy)
            result["epistemic"] = agg(decomp.epistemic_entropy)
        
        return result

    def compute(self, Pbits: Tensor) -> Dict[str, Tensor]:
        return self.forward(Pbits)


def _logit_epistemic_variance(Pbits: Tensor, eps: float = 1e-4) -> Tensor:
    """
    Epistemic variance in log-odds space.
    
    For sparse fingerprints, standard variance is biased low because
    Var[p] ≤ p(1-p), and p≈0 for most bits.
    
    Log-odds variance amplifies disagreement near 0 and 1:
    - p=0.02 vs p=0.04 in logit: -3.89 vs -3.18 (large diff)
    - p=0.48 vs p=0.52 in logit: -0.08 vs +0.08 (small diff)
    
    Parameters
    ----------
    Pbits : (N, S, K) ensemble predictions
    eps : float, clipping to avoid log(0)
    
    Returns
    -------
    logit_var : (N, K) variance in log-odds space
    """
    P = Pbits.clamp(eps, 1 - eps)
    logits = torch.log(P / (1 - P))  # (N, S, K)
    return logits.var(dim=1)  # (N, K)


def compute_sparse_aware_epistemic(
    Pbits: Tensor,
    method: Literal["logit", "active_bit", "relative"] = "logit",
) -> Tensor:
    """
    Sparse-aware epistemic uncertainty for fingerprint predictions.
    
    Parameters
    ----------
    Pbits : (N, S, K) or (S, K)
        Ensemble fingerprint predictions.
    method : str
        - "logit": Variance in log-odds space, weighted by activity
        - "active_bit": Standard variance weighted by mean prediction
        - "relative": Coefficient of variation (std/mean)
    
    Returns
    -------
    uncertainty : (N,) or scalar
    """
    squeeze = Pbits.dim() == 2
    if squeeze:
        Pbits = Pbits.unsqueeze(0)
    
    N, S, K = Pbits.shape
    
    if method == "logit":
        logit_var = _logit_epistemic_variance(Pbits)  # (N, K)
        activity = Pbits.max(dim=1).values  # (N, K)
        result = (logit_var * activity).mean(dim=-1)
        
    elif method == "active_bit":
        var = Pbits.var(dim=1)  # (N, K)
        mean = Pbits.mean(dim=1)  # (N, K)
        result = (var * mean).sum(dim=-1)
        
    elif method == "relative":
        mean = Pbits.mean(dim=1)  # (N, K)
        std = Pbits.std(dim=1)  # (N, K)
        cv = std / (mean + 0.01)  # Coefficient of variation
        result = cv.mean(dim=-1)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return result.squeeze(0) if squeeze else result