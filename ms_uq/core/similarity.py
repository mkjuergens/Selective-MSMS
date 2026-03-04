"""
Core similarity functions 
"""
from typing import Literal
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-8


def normalize(x: Tensor, dim: int = -1) -> Tensor:
    """L2 normalize along given dimension."""
    return F.normalize(x.float(), p=2, dim=dim, eps=EPS)


def cosine_pairwise(a: Tensor, b: Tensor) -> Tensor:
    """
    Row-wise cosine similarity.
    
    Parameters
    ----------
    a, b : (..., K)
    
    Returns
    -------
    (...,) similarities in [-1, 1]
    """
    a, b = a.float(), b.float()
    a_n = normalize(a, dim=-1)
    b_n = normalize(b, dim=-1)
    return (a_n * b_n).sum(dim=-1)


def tanimoto_pairwise(a: Tensor, b: Tensor) -> Tensor:
    """
    Row-wise continuous Tanimoto (soft Jaccard) similarity.
    
    For continuous vectors in [0, 1]:
        T(a, b) = sum(a * b) / (sum(a) + sum(b) - sum(a * b))
    
    This is equivalent to the formula used in fingerprint_mlp.py:
        (total - difference) / (total + difference)
    where total = sum(a + b), difference = sum(|a - b|)
    
    Parameters
    ----------
    a, b : (..., K) tensors in [0, 1]
    
    Returns
    -------
    (...,) similarities in [0, 1]
    """
    a, b = a.float(), b.float()
    intersection = (a * b).sum(dim=-1)
    union = a.sum(dim=-1) + b.sum(dim=-1) - intersection
    return intersection / (union + EPS)


def continuous_iou_pairwise(a: Tensor, b: Tensor) -> Tensor:
    """
    Row-wise continuous IoU (same as tanimoto for soft vectors).
    
    IoU = sum(min(a,b)) / sum(max(a,b))
    
    For vectors in [0,1], this equals:
        (sum(a+b) - sum(|a-b|)) / (sum(a+b) + sum(|a-b|))
    
    Parameters
    ----------
    a, b : (..., K) tensors in [0, 1]
    
    Returns
    -------
    (...,) similarities in [0, 1]
    """
    a, b = a.float(), b.float()
    total = (a + b).sum(dim=-1)
    diff = (a - b).abs().sum(dim=-1)
    return (total - diff) / (total + diff + EPS)


def cosine_matrix(a: Tensor, b: Tensor) -> Tensor:
    """
    All-to-all cosine similarity matrix.
    
    Parameters
    ----------
    a : (..., N, K)
    b : (M, K)
    
    Returns
    -------
    (..., N, M) similarity matrix
    """
    a_n = normalize(a.float(), dim=-1)
    b_n = normalize(b.float(), dim=-1)
    return a_n @ b_n.T


def tanimoto_matrix(a: Tensor, b: Tensor) -> Tensor:
    """
    All-to-all continuous Tanimoto similarity matrix.
    
    Parameters
    ----------
    a : (..., N, K)
    b : (M, K)
    
    Returns
    -------
    (..., N, M) similarity matrix
    """
    a, b = a.float(), b.float()
    
    # a: (..., N, K), b: (M, K)
    # Expand for broadcasting: a -> (..., N, 1, K), b -> (M, K)
    a_exp = a.unsqueeze(-2)  # (..., N, 1, K)
    
    # intersection[..., i, j] = sum_k(a[i,k] * b[j,k])
    intersection = (a_exp * b).sum(dim=-1)  # (..., N, M)
    
    # sum_a[..., i] = sum_k(a[i,k])
    sum_a = a.sum(dim=-1, keepdim=True)  # (..., N, 1)
    # sum_b[j] = sum_k(b[j,k])
    sum_b = b.sum(dim=-1)  # (M,)
    
    union = sum_a + sum_b - intersection  # (..., N, M)
    
    return intersection / (union + EPS)


def continuous_iou_matrix(a: Tensor, b: Tensor) -> Tensor:
    """
    All-to-all continuous IoU similarity matrix.
    
    Parameters
    ----------
    a : (..., N, K)
    b : (M, K)
    
    Returns
    -------
    (..., N, M) similarity matrix
    """
    a, b = a.float(), b.float()
    
    a_exp = a.unsqueeze(-2)  # (..., N, 1, K)
    
    total = (a_exp + b).sum(dim=-1)  # (..., N, M)
    diff = (a_exp - b).abs().sum(dim=-1)  # (..., N, M)
    
    return (total - diff) / (total + diff + EPS)


def hamming_distance(pred: Tensor, target: Tensor, threshold: float = 0.5) -> Tensor:
    """Hamming distance (fraction of differing bits)."""
    pred_binary = (pred > threshold).float()
    return (pred_binary != target.float()).float().mean(-1)



PAIRWISE_FNS = {
    "cosine": cosine_pairwise,
    "cossim": cosine_pairwise,
    "tanimoto": tanimoto_pairwise,
    "tanim": tanimoto_pairwise,
    "iou": continuous_iou_pairwise,
    "contiou": continuous_iou_pairwise,
}

MATRIX_FNS = {
    "cosine": cosine_matrix,
    "cossim": cosine_matrix,
    "tanimoto": tanimoto_matrix,
    "tanim": tanimoto_matrix,
    "iou": continuous_iou_matrix,
    "contiou": continuous_iou_matrix,
}


def similarity_pairwise(
    a: Tensor, 
    b: Tensor, 
    metric: Literal["cosine", "cossim", "tanimoto", "tanim", "iou", "contiou"] = "cosine"
) -> Tensor:
    """
    Compute row-wise similarity between a and b.
    
    Parameters
    ----------
    a, b : (..., K) tensors
    metric : str
        'cosine'/'cossim', 'tanimoto'/'tanim', or 'iou'/'contiou'
    
    Returns
    -------
    (...,) similarity scores
    """
    if metric not in PAIRWISE_FNS:
        raise ValueError(f"Unknown metric: {metric}. Choose from {list(PAIRWISE_FNS.keys())}")
    return PAIRWISE_FNS[metric](a, b)


def similarity_matrix(
    a: Tensor, 
    b: Tensor, 
    metric: Literal["cosine", "cossim", "tanimoto", "tanim", "iou", "contiou"] = "cosine"
) -> Tensor:
    """
    Compute all-to-all similarity matrix.
    
    Parameters
    ----------
    a : (..., N, K) queries
    b : (M, K) candidates
    metric : str
        'cosine'/'cossim', 'tanimoto'/'tanim', or 'iou'/'contiou'
    
    Returns
    -------
    (..., N, M) similarity matrix
    """
    if metric not in MATRIX_FNS:
        raise ValueError(f"Unknown metric: {metric}. Choose from {list(MATRIX_FNS.keys())}")
    return MATRIX_FNS[metric](a, b)


# Keep old names for backward compatibility
cosine_similarity_matrix = cosine_matrix
cosine_similarity_pairwise = cosine_pairwise
tanimoto_similarity = tanimoto_pairwise
continuous_iou = continuous_iou_pairwise