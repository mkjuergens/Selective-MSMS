import torch
import numpy as np
from typing import Tuple, Dict, List, Optional
import pandas as pd


def rejection_curve(loss: torch.Tensor, u: torch.Tensor, monotone=False):
    """
    Tail-kept mean loss vs % rejected (sort by descending uncertainty).
    Parameters
    ----------
    loss : torch.Tensor
        (N,) loss per instance (e.g. 0/1 error, or continuous
        like Hamming distance or MSE).
    u : torch.Tensor
        (N,) uncertainty per instance (e.g. entropy, variance).
    monotone : bool
        If True, enforce non-increasing kept risk vs rejection.
    Returns
    -------
    rej_pct : torch.Tensor
        (N,) % of instances rejected, from 0% up to ~100% (not inclusive).
    kept : torch.Tensor
        (N,) mean loss of the non-rejected instances.
    """
    loss = loss.float().clone()
    u = u.float().clone()
    N = loss.numel() # total number of instances
    idx = torch.argsort(u, descending=True) # sort by descending uncertainty
    A = loss[idx]
    tail_sum = torch.cumsum(A.flip(0), 0).flip(0)
    kept = tail_sum / torch.arange(N, 0, -1, dtype=torch.float32)
    if monotone:
        kept = torch.minimum(kept, kept.clone().cummin(0).values)  
    rej_pct = torch.arange(0, N, dtype=torch.float32) / N * 100.0 
    return rej_pct, kept



def aurc_from_curve(rej_pct: torch.Tensor, kept_loss: torch.Tensor) -> float:
    """
    Compute Area Under the Risk-Coverage curve (AURC).
    
    The curve plots mean loss (y-axis) vs. coverage (x-axis), where coverage = 1 - rejection%.
    Lower AURC = better (model maintains low loss across coverage levels).
    
    Parameters
    ----------
    rej_pct : torch.Tensor, shape (N,)
        Rejection percentages from 0% to ~100%, as returned by rejection_curve.
    kept_loss : torch.Tensor, shape (N,)
        Mean loss on kept (non-rejected) samples at each rejection level.
    
    Returns
    -------
    aurc : float
        Area under the risk-coverage curve. Lower is better.
    """
    # Convert rejection% to coverage (0→100 becomes 100→0)
    coverage = 100.0 - rej_pct.float()
    
    # Sort by increasing coverage for proper integration
    order = torch.argsort(coverage)
    cov_sorted = coverage[order]
    loss_sorted = kept_loss[order].float()
    
    # Normalize coverage to [0, 1]
    cov_norm = cov_sorted / 100.0
    
    # Trapezoidal integration
    aurc = torch.trapz(loss_sorted, cov_norm).item()
    
    return aurc  # lower is better



def compute_oracle_aurc(hits: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute oracle AURC: perfect rejection based on true outcomes.
    
    The oracle always rejects incorrect predictions first, achieving
    the theoretical maximum AURC for a given hit rate distribution.
    
    Parameters
    ----------
    hits : np.ndarray
        Binary hit indicators (1 = correct, 0 = incorrect).
    
    Returns
    -------
    aurc : float
        Oracle AURC value.
    rej_pct : np.ndarray
        Rejection percentages.
    kept_loss : np.ndarray
        Mean loss on kept samples.
    """
    loss = 1.0 - hits.astype(np.float32)
    # Oracle uncertainty: incorrect predictions get highest uncertainty
    oracle_uncertainty = loss
    
    rej_pct, kept_loss = rejection_curve(
        torch.tensor(loss, dtype=torch.float32),
        torch.tensor(oracle_uncertainty, dtype=torch.float32)
    )
    aurc = aurc_from_curve(rej_pct, kept_loss)
    return aurc, rej_pct.numpy(), kept_loss.numpy()


def compute_random_aurc(hits: np.ndarray, seed: int = 42) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute random baseline AURC: rejection in random order.
    
    Parameters
    ----------
    hits : np.ndarray
        Binary hit indicators (1 = correct, 0 = incorrect).
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    aurc : float
        Random baseline AURC value.
    rej_pct : np.ndarray
        Rejection percentages.
    kept_loss : np.ndarray
        Mean loss on kept samples.
    """
    loss = 1.0 - hits.astype(np.float32)
    rng = np.random.default_rng(seed)
    random_uncertainty = rng.random(len(loss)).astype(np.float32)
    
    rej_pct, kept_loss = rejection_curve(
        torch.tensor(loss, dtype=torch.float32),
        torch.tensor(random_uncertainty, dtype=torch.float32)
    )
    aurc = aurc_from_curve(rej_pct, kept_loss)
    return aurc, rej_pct.numpy(), kept_loss.numpy()


def compute_aurc_with_baselines(
    metrics: Dict[str, np.ndarray],
    hit_rates: Dict[str, np.ndarray],
    include_oracle: bool = True,
    include_random: bool = True,
    confidence_metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute AURC table with optional oracle and random baselines.
    
    Parameters
    ----------
    metrics : dict
        Uncertainty metrics (name -> values).
    hit_rates : dict
        Hit@k results (name -> values).
    include_oracle : bool
        Include oracle (perfect) baseline.
    include_random : bool
        Include random baseline.
    confidence_metrics : list, optional
        Metrics where higher = more confident. If None, auto-detected.
    
    Returns
    -------
    pd.DataFrame
        AURC values with metrics as rows and hit rates as columns.
    """
    from ms_uq.utils import is_confidence_score
    
    if confidence_metrics is None:
        confidence_metrics = [m for m in metrics if is_confidence_score(m)]
    
    results = {}
    
    for hit_name, hits in hit_rates.items():
        loss = 1.0 - hits.astype(np.float32)
        results[hit_name] = {}
        
        # Oracle baseline
        if include_oracle:
            oracle_aurc, _, _ = compute_oracle_aurc(hits)
            results[hit_name]["oracle"] = oracle_aurc
        
        # Random baseline
        if include_random:
            random_aurc, _, _ = compute_random_aurc(hits)
            results[hit_name]["random"] = random_aurc
        
        # All metrics
        for metric_name, values in metrics.items():
            unc = values.copy()
            if metric_name in confidence_metrics:
                unc = -unc  # Negate so higher uncertainty = lower confidence
            
            rej_pct, kept_loss = rejection_curve(
                torch.tensor(loss, dtype=torch.float32),
                torch.tensor(unc, dtype=torch.float32)
            )
            results[hit_name][metric_name] = aurc_from_curve(rej_pct, kept_loss)
    
    df = pd.DataFrame(results)
    return df