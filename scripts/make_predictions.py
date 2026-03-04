from __future__ import annotations
from pathlib import Path
import argparse
import sys
import gc
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ms_uq.utils import (
    make_test_loader,
    make_train_val_test_loaders,
    discover_ensemble_ckpts,
)
from ms_uq.inference import (
    Predictor,
    MCDropoutSampler,
    EnsembleSampler,
    head_probs_fn,
    save_ranker_from_model,
)
from ms_uq.inference.predictor import save_prestacked_predictions
from ms_uq.inference.retrieve import scores_from_loader
from ms_uq.models.fingerprint_mlp import FingerprintPredicter
from ms_uq.models.laplace_bce import DiagLaplaceBCEHead, PriorTuningConfig

try:
    from torch.serialization import add_safe_globals
    from massspecgym.models.base import Stage
    add_safe_globals([Stage])
except Exception:
    pass


def _cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def save_fp_probs(
    mode: str,
    ckpt: Optional[str],
    ckpts_csv: Optional[str],
    ens_dir: Optional[str],
    ens_metric: Optional[str],
    passes: int,
    device: str,
    dl: DataLoader,
    out_fp_path: Path,
    overwrite: bool,
    save_ranker: bool = True,
) -> Optional[Path]:
    """
    Save fingerprint probabilities and optionally ranker weights.
    
    Parameters
    ----------
    mode : str
        'mcdo', 'ensemble', or 'single'
    ckpt : str, optional
        Single checkpoint path (for mcdo/single)
    ckpts_csv : str, optional
        Comma-separated checkpoint paths
    ens_dir : str, optional
        Ensemble directory
    ens_metric : str, optional
        Metric for ensemble checkpoint discovery
    passes : int
        MC dropout passes
    device : str
        Device to use
    dl : DataLoader
        Test dataloader
    out_fp_path : Path
        Output path for fingerprint probabilities
    overwrite : bool
        Overwrite existing files
    save_ranker : bool
        If True, also save ranker weights if model has one
    
    Returns
    -------
    ranker_path : Path or None
        Path to saved ranker, or None if no ranker
    """
    ranker_path = None
    
    if out_fp_path.exists() and not overwrite:
        print(f"[predict] using cached {out_fp_path}")
        # Check if ranker already exists
        potential_ranker = out_fp_path.parent / "ranker.pt"
        if potential_ranker.exists():
            return potential_ranker
        return None

    if mode == "mcdo":
        if not ckpt:
            sys.exit("ERROR: --ckpt required for --mode mcdo")
        sampler = MCDropoutSampler(Path(ckpt), FingerprintPredicter, passes=passes, device=device)
        # For mcdo, load model once to check for ranker
        if save_ranker:
            model = FingerprintPredicter.load_from_checkpoint(ckpt, map_location=device)
            ranker_out = out_fp_path.parent / "ranker.pt"
            if save_ranker_from_model(model, ranker_out):
                ranker_path = ranker_out
            del model
            _cleanup()
            
    elif mode == "ensemble":
        if ckpts_csv:
            ckpt_list = [Path(p.strip()) for p in ckpts_csv.split(",") if p.strip()]
        elif ens_dir and ens_metric:
            ckpt_list = discover_ensemble_ckpts(ens_dir, ens_metric, prefer="best")
        else:
            sys.exit("ERROR: ensemble requires --ens_dir and --ens_metric (or --ckpts).")
        if not ckpt_list:
            sys.exit("ERROR: no ensemble checkpoints found.")
        print(f"[ensemble] {len(ckpt_list)} members discovered.")
        sampler = EnsembleSampler(ckpt_list, FingerprintPredicter, mc_dropout_eval=False, device=device)
        # For ensemble, use first member's ranker (should be same architecture)
        if save_ranker and ckpt_list:
            model = FingerprintPredicter.load_from_checkpoint(ckpt_list[0], map_location=device)
            ranker_out = out_fp_path.parent / "ranker.pt"
            if save_ranker_from_model(model, ranker_out):
                ranker_path = ranker_out
            del model
            _cleanup()
            
    else:  # single
        if not ckpt:
            sys.exit("ERROR: --ckpt required for --mode single")
        sampler = EnsembleSampler([Path(ckpt)], FingerprintPredicter, mc_dropout_eval=False, device=device)
        if save_ranker:
            model = FingerprintPredicter.load_from_checkpoint(ckpt, map_location=device)
            ranker_out = out_fp_path.parent / "ranker.pt"
            if save_ranker_from_model(model, ranker_out):
                ranker_path = ranker_out
            del model
            _cleanup()

    predictor = Predictor(sampler, head_probs_fn("loss.fp_pred_head", torch.sigmoid))
    predictor.predict_stack(dl, out_fp_path, save_every=100, overwrite=overwrite)

    del sampler, predictor
    _cleanup()
    
    return ranker_path


def save_fp_probs_laplace_bce(
    ckpt_path: str, device: str,
    train_dl: DataLoader, val_dl: Optional[DataLoader], test_dl: DataLoader,
    out_fp_path: Path, n_samples: int = 50,
    tau_w: float = 1.0, tau_b: float = 1.0,
    max_train_batches: int = 200, out_chunk: int = 512,
    prior_opt: str = "gridsearch",
    overwrite: bool = False,
) -> None:
    """Save fingerprint probabilities using Laplace-BCE."""
    if out_fp_path.exists() and not overwrite:
        print(f"[laplace_bce] using cached {out_fp_path}")
        return

    base = FingerprintPredicter.load_from_checkpoint(ckpt_path, strict=False)
    base.eval().to(device)

    la = DiagLaplaceBCEHead(base, device=device, tau_w=tau_w, tau_b=tau_b)

    # Curvature accumulation (cheap last-layer Laplace)
    la.fit(train_dl, max_batches=max_train_batches, scale_to_dataset=True)

    # Optional prior tuning (recommended). Reuse --la_prior_opt.
    if val_dl is not None and prior_opt in {"gridsearch", "CV"}:
        cfg = PriorTuningConfig(
            # use a reasonably wide log grid; users can still pin tau via args
            tau_w_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3),
            tau_b_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3),
            max_batches=50,
            bit_subsample=512,
            seed=0,
        )
        best_tau_w, best_tau_b = la.tune_prior(val_dl, cfg)
        print(f"[laplace_bce] tuned prior: tau_w={best_tau_w:g}, tau_b={best_tau_b:g}")

    # Predict efficiently: per batch we produce (B,S,K) without rerunning backbone S times.
    def _predict_batch(batch: dict) -> torch.Tensor:
        return la.predict_batch(
            batch,
            n_samples=n_samples,
            out_chunk=out_chunk,
            dtype_out=torch.float16,
            method="logit_mc",
        )

    save_prestacked_predictions(
        test_dl,
        out_fp_path,
        _predict_batch,
        dtype_disk=torch.float16,
        overwrite=overwrite,
        chunk_size_batches=50,
        meta_extra={
            "method": "laplace_bce_logit_mc",
            "tau_w": float(la.tau_w),
            "tau_b": float(la.tau_b),
            "max_train_batches": int(max_train_batches),
        },
    )
    print(f"[laplace_bce] wrote {out_fp_path}")

    del base, la
    _cleanup()


def save_scores(
    fp_path: Path,
    out_dir: Path,
    dl: DataLoader,
    metric: str = "cosine",
    aggregation: str = "score",
    temperature: float = 1.0,
    overwrite: bool = False,
    ranker_path: Optional[Path] = None,
    device: str = "cpu",
) -> Path:
    """
    Compute and save retrieval scores with specified aggregation.
    
    Parameters
    ----------
    fp_path : Path
        Path to fingerprint probabilities file.
    out_dir : Path
        Output directory.
    dl : DataLoader
        Test dataloader.
    metric : str
        Similarity metric: 'cosine', 'tanimoto', or 'iou'.
        Ignored if ranker_path is provided.
    aggregation : str
        'score': Average similarity scores across samples
        'fingerprint': Average fingerprints, then compute similarity
        'probability': Average softmax probabilities (uses temperature)
    temperature : float
        Softmax temperature for 'probability' aggregation.
        Higher = softer distribution, lower = sharper.
    overwrite : bool
        Overwrite existing files.
    ranker_path : Path, optional
        Path to saved ranker weights. If provided, uses learned
        similarity instead of metric-based similarity.
    device : str
        Device for ranker computation.
    
    Returns
    -------
    Path to saved scores file.
    """
    from ms_uq.inference import load_ranker
    
    # Check for ranker
    ranker = None
    use_ranker = False
    
    # Auto-detect ranker in same directory as fp_probs
    if ranker_path is None:
        potential_ranker = fp_path.parent / "ranker.pt"
        if potential_ranker.exists():
            ranker_path = potential_ranker
    
    if ranker_path is not None and ranker_path.exists():
        ranker = load_ranker(ranker_path, device=device)
        if ranker is not None:
            use_ranker = True
            print(f"[scores] Using learned ranker from {ranker_path}")
    
    # Build output filename
    if use_ranker:
        base_name = f"scores_ragged_ranker_{aggregation}"
    else:
        base_name = f"scores_ragged_{metric}_{aggregation}"
    
    if aggregation == "probability":
        out_path = out_dir / f"{base_name}_T{temperature}.pt"
    else:
        out_path = out_dir / f"{base_name}.pt"
    
    if out_path.exists() and not overwrite:
        print(f"[scores] using cached {out_path}")
        return out_path
    
    # Load fingerprint probabilities
    data = torch.load(fp_path, map_location="cpu")
    fp_probs = (data["stack"] if isinstance(data, dict) else data).float()
    
    score_method = "ranker" if use_ranker else metric
    print(f"[scores] computing {score_method} scores with '{aggregation}' aggregation" + 
          (f" (T={temperature})" if aggregation == "probability" else "") + "...")
    
    result = scores_from_loader(
        fp_probs=fp_probs,
        loader=dl,
        metric=metric,
        aggregation=aggregation,
        temperature=temperature,
        return_labels=True,
        return_per_sample=True,
        show_progress=True,
        ranker=ranker,
        device=device,
    )
    
    # Store temperature in result for reference
    result["temperature"] = temperature
    if use_ranker:
        result["ranker_path"] = str(ranker_path)
    
    torch.save(result, out_path)
    print(f"[scores] saved {out_path}")
    
    return out_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ... existing arguments ...
    ap.add_argument("--mode", choices=["mcdo", "ensemble", "single", "laplace_bce"], required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--ckpts")
    ap.add_argument("--ens_dir")
    ap.add_argument("--ens_metric")
    ap.add_argument("--passes", type=int, default=50)

    # Laplace BCE options
    ap.add_argument("--laplace_samples", type=int, default=50)
    ap.add_argument("--la_prior_opt", choices=["marglik", "gridsearch", "CV"], default="gridsearch")
    ap.add_argument("--la_tau_w", type=float, default=1.0)
    ap.add_argument("--la_tau_b", type=float, default=1.0)
    ap.add_argument("--la_max_train_batches", type=int, default=200)
    ap.add_argument("--la_out_chunk", type=int, default=512)

    # Data
    ap.add_argument("--dataset_tsv", required=True)
    ap.add_argument("--helper_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    # Runtime
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--bin_width", type=float, default=0.1)
    ap.add_argument("--pin_memory", action="store_true")

    # Scoring
    ap.add_argument("--metric", choices=["cosine", "tanimoto", "iou"], default="cosine")
    ap.add_argument("--aggregation", choices=["score", "fingerprint", "probability"], default="score",
                    help="Ensemble aggregation: 'score' (avg scores), 'fingerprint' (avg FPs), "
                         "'probability' (avg softmax probs, uses --temperature)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Softmax temperature for 'probability' aggregation. "
                         "Lower = sharper (more confident), higher = softer.")
    ap.add_argument("--overwrite", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp_path = out_dir / "fp_probs.pt"

    print(f"\n{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Aggregation: {args.aggregation}" + 
          (f" (T={args.temperature})" if args.aggregation == "probability" else ""))
    print(f"Metric: {args.metric}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")

    # ... existing loader and fp_probs code ...

    # Create loaders and compute fingerprint probabilities
    if args.mode == "laplace_bce":
        train_dl, val_dl, test_dl = make_train_val_test_loaders(
            args.dataset_tsv, args.helper_dir, args.bin_width,
            args.batch_size, args.num_workers, args.pin_memory
        )
        save_fp_probs_laplace_bce(
            args.ckpt, args.device, train_dl, val_dl, test_dl, fp_path,
            n_samples=args.laplace_samples, tau_w=args.la_tau_w,
            tau_b=args.la_tau_b, max_train_batches=args.la_max_train_batches,
            out_chunk=args.la_out_chunk, prior_opt=args.la_prior_opt,
            overwrite=args.overwrite
        )
        dl = test_dl
    else:
        dl = make_test_loader(
            args.dataset_tsv, args.helper_dir, args.bin_width,
            args.batch_size, args.num_workers, args.pin_memory
        )
        save_fp_probs(
            args.mode, args.ckpt, args.ckpts, args.ens_dir, args.ens_metric,
            args.passes, args.device, dl, fp_path, args.overwrite
        )

    # Verify fp_probs
    P = torch.load(fp_path, map_location="cpu")
    Pstack = P["stack"] if isinstance(P, dict) else P
    print(f"[ok] fp_probs: shape {tuple(Pstack.shape)}")
    del P, Pstack
    gc.collect()

    # Compute scores with aggregation
    scores_path = save_scores(
        fp_path, out_dir, dl,
        metric=args.metric,
        aggregation=args.aggregation,
        temperature=args.temperature,
        overwrite=args.overwrite,
    )

    D = torch.load(scores_path, map_location="cpu")
    print(f"[ok] scores: aggregated shape {tuple(D['scores_flat'].shape)}")
    if "scores_stack_flat" in D:
        print(f"     per-sample shape {tuple(D['scores_stack_flat'].shape)}")
    if "temperature" in D:
        print(f"     temperature: {D['temperature']}")

    print(f"\n[done] Artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
