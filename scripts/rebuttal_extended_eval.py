"""Rebuttal Exp-C: evaluate a frozen completed run with extended metrics.

Reuses the strict evaluator's reconstruction and verification unchanged, but
reports R@5/10/20/100, MRR and a tie-aware expected NDCG@10. Core source files
are untouched, so recorded source-hash verification still passes; the NDCG
accumulator is injected at runtime through the tie-metrics helper.

Usage:
  python scripts/rebuttal_extended_eval.py --source-run runs/pilot/... \
      --output runs/pilot-extended --csv-path data/new_york/new_york_checkins.csv --gpu-id -1
"""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import utils.strict_unseen as strict_unseen
from scripts.strict_unseen_train import (  # noqa: E402
    build_model, training_counts, validate_checkpoint_selection, file_sha256,
    verify_source_hashes, split_options, SOURCE_MANIFEST_VERSION, runtime_versions,
)
from utils.strict_unseen import make_split, evaluate, validate_split_options  # noqa: E402
from utils.latlon_dataset import LatLonCoordinateCollate  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

K_VALUES = [5, 10, 20, 100]
NDCG_KEY = "ndcg10"
NDCG_K = 10

_original_tie_metrics = strict_unseen.expected_random_tie_metrics


def tie_metrics_with_ndcg(strictly_above, tie_size, k_values, harmonic=None):
    """Delegate numeric ks to the audited helper; add expected NDCG@10 for ties.

    Target rank is uniform over a+1..a+m; gain(r) = 1/log2(1+r) for r <= 10 else 0,
    so E[gain] = (G[min(a+m,10)] - G[min(a,10)]) / m with G the gain cumsum.
    """
    numeric = [k for k in k_values if k != NDCG_KEY]
    rr, hits = _original_tie_metrics(strictly_above, tie_size, numeric, harmonic)
    if NDCG_KEY in k_values:
        a = strictly_above.long()
        m = tie_size.long()
        gains = torch.zeros(NDCG_K + 1, dtype=torch.float64, device=a.device)
        ranks = torch.arange(1, NDCG_K + 1, dtype=torch.float64, device=a.device)
        gains[1:] = (1.0 / torch.log2(1.0 + ranks)).cumsum(0)
        hits[NDCG_KEY] = (gains[(a + m).clamp(max=NDCG_K)] - gains[a.clamp(max=NDCG_K)]) / m
    return rr, hits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--csv-path", default=None)
    p.add_argument("--gpu-id", type=int, default=None)
    cli = p.parse_args()

    source, output = Path(cli.source_run), Path(cli.output)
    if output.exists():
        raise FileExistsError(f"Extended evaluation requires a fresh output directory: {output}")

    config = json.loads((source / "config.json").read_text())
    verified_sources = verify_source_hashes(config)
    completed = json.loads((source / "training_complete.json").read_text())
    for name in ("config.json", "split.json", "vocabularies.pt", "best.pt", "validation.jsonl"):
        if completed.get("artifact_sha256", {}).get(name) != file_sha256(source / name):
            raise ValueError(f"Training artifact differs from completed run: {name}")

    args = argparse.Namespace(**config["training"])
    opts = argparse.Namespace(**config["strict"])
    validate_split_options(**split_options(opts))

    saved = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
    history = [json.loads(line) for line in (source / "validation.jsonl").read_text().splitlines() if line.strip()]
    validate_checkpoint_selection(saved, history, opts.selection_metric, completed)

    original_manifest = json.loads((source / "split.json").read_text())
    raw_path = Path(cli.csv_path or config["data"]["csv_path"])
    if file_sha256(raw_path) != original_manifest["raw_sha256"]:
        raise ValueError("Raw CSV differs from the dataset used for training.")

    if cli.gpu_id is not None:
        args.gpu_id = cli.gpu_id
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu")

    datasets, vocabs, manifest = make_split(
        raw_path, output, seed=opts.split_seed,
        resolution=args.h3_split_resolution, fraction=args.h3_split_test_ratio,
        **split_options(opts))
    if json.loads(json.dumps(manifest)) != original_manifest:
        raise ValueError("Recreated split differs from the saved training split.")
    original_vocabs = torch.load(source / "vocabularies.pt", map_location="cpu", weights_only=True)
    if not all(torch.equal(v, original_vocabs[n]) for n, v in zip(("train", "validation", "test"), vocabs)):
        raise ValueError("Recreated candidate vocabularies differ from training.")

    encoder, model = build_model(args, opts, vocabs[0], None, device)
    model.load_state_dict(saved["model"], strict=True)
    expected_counts = training_counts(datasets[0], vocabs[0], opts.train_repeats, args.heuristic_popularity)
    if not torch.equal(saved["train_counts"].cpu(), expected_counts):
        raise ValueError("Checkpoint popularity counts differ from the reconstructed training data.")

    # Extended metrics: expected_random ties are required for the injected NDCG.
    if getattr(args, "eval_tie_policy", "worst") != "expected_random":
        raise ValueError("Extended NDCG evaluation supports only the expected_random tie policy.")
    args.k_values = K_VALUES + [NDCG_KEY]
    strict_unseen.expected_random_tie_metrics = tie_metrics_with_ndcg
    try:
        metrics = evaluate(model, DataLoader(datasets[2], batch_size=opts.eval_batch_size,
                                             collate_fn=LatLonCoordinateCollate(args.pad_val_coords, args.pad_val_targets)),
                           vocabs[2], vocabs[0], saved["train_counts"], device, args,
                           reserved_locations=manifest["test_locations"], ablations=True,
                           per_user_path=output / "per_user_extended.json")
    finally:
        strict_unseen.expected_random_tie_metrics = _original_tie_metrics

    report = dict(selected_epoch=saved["epoch"], selection_metric=saved["selection_metric"],
                  validation_selection_score=saved["selection_score"], tie_policy=args.eval_tie_policy,
                  k_values=K_VALUES, ndcg_k=NDCG_K, metrics=metrics,
                  source_run=str(source.resolve()), training_performed=False,
                  checkpoint_sha256=file_sha256(source / "best.pt"), raw_sha256=manifest["raw_sha256"],
                  source_manifest_version=SOURCE_MANIFEST_VERSION, source_sha256=verified_sources,
                  runtime=runtime_versions(), device=str(device))
    (output / "test_extended.json").write_text(json.dumps(report, indent=2) + "\n")
    print("EXTENDED_TEST", json.dumps({g: metrics["model"][g] for g in ("all", "seen", "unseen")}), flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    if code:
        raise SystemExit(code)
