"""
Uncertainty measures for selective prediction.

Core modules:
- BitwiseUncertainty: Fingerprint-level uncertainty decomposition
- RetrievalUncertainty: Retrieval/ranking uncertainty measures
- DistanceUncertainty: Distance-based (k-NN, Mahalanobis) uncertainty
- eval_measures: Unified API for computing uncertainties
- decomposition: Core decomposition primitives
"""

# Core classes
from ms_uq.unc_measures.bitwise_unc import (
    BitwiseUncertainty,
    compute_sparse_aware_epistemic,
)
from ms_uq.unc_measures.retrieval_unc import (
    RetrievalUncertainty,
    RetrievalUncertaintyRagged,
)
from ms_uq.unc_measures.distance_unc import (
    DistanceUncertainty,
    extract_embeddings_from_loader,
    extract_embeddings_from_pbits,
)

# Main evaluation API
from ms_uq.unc_measures.eval_measures import (
    FINGERPRINT_MEASURES,
    RETRIEVAL_MEASURES,
    DISTANCE_MEASURES,
    compute_fingerprint_uncertainties,
    compute_retrieval_uncertainties,
    compute_distance_uncertainties,
    compute_uncertainties,
    fit_distance_model,
    get_embeddings_from_pbits,
)

__all__ = [
    # Core uncertainty classes
    "BitwiseUncertainty",
    "compute_sparse_aware_epistemic",
    "RetrievalUncertainty",
    "RetrievalUncertaintyRagged",
    "DistanceUncertainty",
    "extract_embeddings_from_loader",
    "extract_embeddings_from_pbits",
    # Evaluation API
    "FINGERPRINT_MEASURES",
    "RETRIEVAL_MEASURES",
    "DISTANCE_MEASURES",
    "compute_fingerprint_uncertainties",
    "compute_retrieval_uncertainties",
    "compute_distance_uncertainties",
    "compute_uncertainties",
    "fit_distance_model",
    "get_embeddings_from_pbits",
]