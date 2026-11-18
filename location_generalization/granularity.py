"""One place to run and select spatial granularity for unseen prediction.

Every candidate uses the same declared users, holdouts, seeds and training budget.
Only validation Recall@100 selects a band. Test evaluation is a separate action.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .config import CORE_SOURCE_FILES, ROOT, csv_path, digest, read_json, trials, validate
from .selection import SELECTION_CRITERIA, best_validation_epoch


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def plan(config):
    validate(config)
    declared = list(trials(config))
    return {
        "schema_version": 1, "config_sha256": digest(config), "config": config,
        "objective": "validation Recall@100 on first visits to training-unseen coordinates",
        "selection_metric": config["selection"]["metric"],
        "test_access": "separate evaluate command after selection.json is locked",
        "fits": len(declared), "maximum_epochs_per_fit": config["training"]["epochs"],
        "trials": declared, "source_sha256": source_hashes(),
    }


def source_hashes():
    """Freeze orchestration as well as model, split, training and evaluation code."""
    names = set(CORE_SOURCE_FILES)
    names.update(str(path.relative_to(ROOT)) for path in (ROOT / "location_generalization").glob("*.py"))
    return {name: file_hash(ROOT / name) for name in sorted(names)}


def freeze_experiment(config, output_dir):
    """An output root belongs to exactly one immutable experiment declaration."""
    output = Path(output_dir).resolve()
    expected = plan(config)
    manifest = output / "experiment.json"
    if manifest.exists():
        if read_json(manifest) != expected:
            raise ValueError("This output directory belongs to a different configuration; use a new directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("A new experiment requires an empty output directory")
        write_new(manifest, expected)
    return output


def train_command(config, trial, output_dir, prepare_only=False):
    """Translate the public configuration into explicit, inspectable trainer flags."""
    band, unseen = trial["band"], config["unseen"]
    values = {
        "output_dir": str(output_dir), "city": trial["city"], "seed": trial["seed"],
        "csv_path": str(csv_path(config, trial["city"])),
        "baseline": config["model"]["encoder"],
        "min_scale": band["min_scale"], "max_scale": band["max_scale"],
        "num_bands": config["granularity"]["num_bands"],
        "rff_features": config["granularity"]["rff_features"],
        "h3_split_resolution": unseen["h3_resolution"],
        "h3_split_test_ratio": unseen["holdout_fraction"], "split_seed": unseen["split_seed"],
        "train_user_fraction": unseen["train_user_fraction"],
        "validation_user_fraction": unseen["validation_user_fraction"],
        "heldout_validation_fraction": unseen["heldout_validation_fraction"],
        "max_sequence_length": unseen["max_sequence_length"],
        "selection_metric": config["selection"]["metric"],
        "eval_tie_policy": config["selection"]["tie_policy"],
        "task": "next_event", "multistep_k": 1, "mask_prob": 0, "num_neg_samples": 0,
    }
    values.update({k: v for k, v in config["model"].items() if k != "encoder"})
    values.update({k: v for k, v in config["training"].items() if k != "seeds"})
    values.update({k: v for k, v in config["scoring"].items() if not isinstance(v, bool)})
    command = [sys.executable, str(ROOT / "scripts/strict_unseen_train.py")]
    for key, value in values.items():
        command.extend(["--" + key, str(value)])
    command.extend(["--k_values", "1", "10", "100", "--skip_test"])
    if config["scoring"]["heuristic_loss"]:
        command.append("--heuristic_loss")
    command.append("--heuristic_popularity" if config["scoring"]["heuristic_popularity"]
                   else "--no_heuristic_popularity")
    if prepare_only:
        command.append("--prepare_only")
    return command


def run_logged(command, log_path):
    """Keep a readable log while preserving the trainer's nonzero exit status."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running; log: {log_path}", flush=True)
    with log_path.open("x") as log:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
        if status:
            raise subprocess.CalledProcessError(status, command)


def _request(config, trial, csv_override=None):
    return {"config_sha256": digest(config), "trial": trial,
            "csv_sha256": file_hash(csv_override or csv_path(config, trial["city"]))}


def run_trial(config, trial, output_dir):
    output = freeze_experiment(config, output_dir)
    if (output / "selection.json").exists():
        raise ValueError("Selection is already locked; further training requires a new experiment directory")
    run = output / trial["run_dir"]
    request = _request(config, trial)
    if (run / "trial_complete.json").exists():
        validate_trial(config, trial, output)
        print(f"Already complete and verified: {trial['run_dir']}", flush=True)
        return
    if run.exists():
        raise FileExistsError(f"Incomplete run exists: {run}. Inspect its log; use a new experiment directory to retry.")
    write_new(run / "request.json", request)
    command = train_command(config, trial, run)
    write_new(run / "command.json", command)
    run_logged(command, run / "train.log")
    if (run / "test.json").exists():
        raise RuntimeError("Training unexpectedly evaluated test data")
    artifacts = ["request.json", "config.json", "split.json", "best.pt", "vocabularies.pt",
                 "validation.jsonl", "training_complete.json"]
    write_new(run / "trial_complete.json", {
        "artifact_sha256": {name: file_hash(run / name) for name in artifacts},
        "test_evaluated": False,
    })
    validate_trial(config, trial, output)


def validate_trial(config, trial, output_dir, csv_override=None):
    """Read validation history and provenance only; never inspect test metrics."""
    run = Path(output_dir) / trial["run_dir"]
    complete = read_json(run / "trial_complete.json")
    if complete.get("test_evaluated") is not False or (run / "test.json").exists():
        raise ValueError(f"Test-exposed trial cannot enter this validation search: {run}")
    required = {"request.json", "config.json", "split.json", "best.pt", "vocabularies.pt",
                "validation.jsonl", "training_complete.json"}
    if set(complete.get("artifact_sha256", {})) != required:
        raise ValueError(f"Incomplete artifact manifest: {run}")
    for name, expected in complete["artifact_sha256"].items():
        if file_hash(run / name) != expected:
            raise ValueError(f"Changed trial artifact: {run / name}")
    if read_json(run / "request.json") != _request(config, trial, csv_override):
        raise ValueError(f"Trial configuration or input data differs: {run}")
    saved_config = read_json(run / "config.json")
    if saved_config["strict"]["skip_test"] is not True:
        raise ValueError("Candidate trials must disable test evaluation")
    for name, expected in saved_config["source_sha256"].items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT) or file_hash(source) != expected:
            raise ValueError(f"Training code changed: {name}; use the original code revision")
    history = [json.loads(line) for line in (run / "validation.jsonl").read_text().splitlines() if line.strip()]
    criterion = config["selection"]["metric"]
    best = best_validation_epoch(history, criterion)
    stream, group, _ = SELECTION_CRITERIA[criterion]
    completed = read_json(run / "training_complete.json")
    if (completed.get("selected_epoch") != best["epoch"] or
            completed.get("selection_metric") != criterion or
            completed.get("validation_selection_score") != best["selection_score"]):
        raise ValueError(f"Completed checkpoint does not match earliest-best validation selection: {run}")
    return {"seed": trial["seed"], "run_dir": trial["run_dir"],
            "validation_score": best["selection_score"], "selected_epoch": best["epoch"],
            "validation_events": best["metrics"][stream][group]["n"],
            "checkpoint_sha256": file_hash(run / "best.pt"),
            "trial_complete_sha256": file_hash(run / "trial_complete.json"),
            "split_sha256": file_hash(run / "split.json")}


def validation_comparisons(config, output_dir, csv_template=None):
    """Compare every declared seed on matching splits; ties keep the first band."""
    comparisons, winners = {}, []
    declared = list(trials(config))
    for city in config["data"]["cities"]:
        rows, split_hashes = [], set()
        # A relocated CSV must have the saved hash; it never changes the request.
        relocated = Path(csv_template.format(city=city)).expanduser().resolve() if csv_template else None
        for band in config["granularity"]["candidates"]:
            members = []
            for trial in declared:
                if trial["city"] == city and trial["band"] == band:
                    item = validate_trial(config, trial, output_dir, relocated)
                    members.append(item)
                    split_hashes.add(item["split_sha256"])
            rows.append({"band": band, "mean_validation_score": sum(r["validation_score"] for r in members) / len(members),
                         "runs": members})
        if len(split_hashes) != 1:
            raise ValueError(f"Candidates for {city} used different data splits")
        comparisons[city] = rows
        winners.append({"city": city, **max(rows, key=lambda row: row["mean_validation_score"])})
    return comparisons, winners


def select(config, output_dir):
    """Select one range per city by mean validation score across ALL declared seeds.

    Missing trials fail the comparison. Exact mean-score ties keep the first
    configured range. No city, seed or band is dropped because of its result.
    """
    output = freeze_experiment(config, output_dir)
    if (output / "selection.json").exists():
        raise FileExistsError("selection.json already exists; it is a frozen evaluation decision")
    comparisons, winners = validation_comparisons(config, output)
    result = {"schema_version": 1, "config_sha256": digest(config),
              "experiment_sha256": file_hash(output / "experiment.json"),
              "metric": config["selection"]["metric"], "selection_split": "validation",
              "rule": "highest mean across all configured seeds; exact ties keep first declared band",
              "comparisons": comparisons, "selected": winners}
    write_new(output / "selection.json", result)
    return result


def evaluate_selection(selection_path, output_dir, gpu_id=None, csv_template=None):
    """Evaluate only the locked winning band, retaining every declared seed."""
    selection_path = Path(selection_path).resolve()
    source_root = selection_path.parent
    selected = read_json(selection_path)
    manifest = read_json(source_root / "experiment.json")
    config = validate(manifest["config"])
    if manifest != plan(config):
        raise ValueError("Experiment code or declaration differs; evaluate with its frozen source revision")
    if selected.get("selection_split") != "validation" or selected.get("config_sha256") != digest(config):
        raise ValueError("Selection does not match its experiment")
    if selected.get("experiment_sha256") != file_hash(source_root / "experiment.json"):
        raise ValueError("Experiment declaration changed after selection")
    # Reconstruct the decision from validation artifacts so editing selection.json
    # cannot accidentally evaluate another band or a subset of favorable seeds.
    comparisons, expected_winners = validation_comparisons(config, source_root, csv_template)
    for city, candidates in comparisons.items():
        if selected["comparisons"].get(city) != candidates:
            raise ValueError(f"Validation comparison changed after selection: {city}")
    if selected["selected"] != expected_winners:
        raise ValueError("Selected runs differ from the frozen validation decision")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Final evaluation requires a fresh output directory")
    write_new(output / "selection.json", selected)
    reports = []
    for city_result in expected_winners:
        for run in city_result["runs"]:
            destination = output / run["run_dir"]
            command = [sys.executable, str(ROOT / "scripts/strict_unseen_train.py"),
                       "--evaluate_only", str(source_root / run["run_dir"]),
                       "--evaluation_output_dir", str(destination)]
            if gpu_id is not None:
                command.extend(["--evaluation_gpu_id", str(gpu_id)])
            if csv_template:
                relocated = Path(csv_template.format(city=city_result["city"])).expanduser().resolve()
                command.extend(["--evaluation_csv_path", str(relocated)])
            run_logged(command, output / "logs" / f"{city_result['city']}-seed-{run['seed']}.log")
            report = read_json(destination / "test.json")
            reports.append({"city": city_result["city"], "band": city_result["band"],
                            "seed": run["seed"], "test": report})
    write_new(output / "summary.json", {"selection_sha256": file_hash(selection_path), "results": reports})
    return reports
