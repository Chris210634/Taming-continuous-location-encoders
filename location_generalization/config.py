"""The public experiment contract; reject typos instead of silently using defaults.

The existing trainers retain their historical defaults. This module supplies
explicit settings to the strict unseen trainer for each new experiment.
"""

import hashlib
import json
import math
from pathlib import Path
import re

from .selection import SELECTION_METRICS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/unseen/massive_steps.json"
CITY_NAMES = {
    "new_york": "New-York", "istanbul": "Istanbul", "jakarta": "Jakarta",
    "sao_paulo": "Sao-Paulo", "moscow": "Moscow", "bandung": "Bandung",
    "petaling_jaya": "Petaling-Jaya", "kuwait_city": "Kuwait-City",
}
# Local dependencies used by the strict trainer and frozen-checkpoint evaluator.
# Shared with their source manifest; orchestration adds its own package files.
CORE_SOURCE_FILES = (
    "scripts/strict_unseen_train.py", "utils/strict_unseen.py",
    "utils/strict_encoders.py", "utils/latlon_dataset.py", "utils/evaluation.py",
    "training/setup.py", "training/config.py",
    "models/location_encoder.py", "models/latlon_transformer.py",
    "location_generalization/__init__.py", "location_generalization/config.py",
    "location_generalization/selection.py",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    # JSON permits duplicate keys; our experiment configurations deliberately do not.
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        actual = set(value) if isinstance(value, dict) else set()
        raise ValueError(f"{label}: missing {sorted(set(expected)-actual)}, "
                         f"unknown {sorted(actual-set(expected))}")


def _number(value, label, lower=0, upper=None, integer=False, inclusive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if integer and not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if (value < lower if inclusive else value <= lower) or (upper is not None and value > upper):
        raise ValueError(f"{label} is outside its supported range")


def validate(config):
    _keys(config, ["schema_version", "name", "data", "unseen", "granularity",
                   "model", "training", "scoring", "selection"], "experiment")
    if config["schema_version"] != 1 or isinstance(config["schema_version"], bool):
        raise ValueError("Only schema_version 1 is supported")
    if not isinstance(config["name"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", config["name"]):
        raise ValueError("name must use lowercase letters, numbers, underscores or hyphens")
    data = config["data"]
    _keys(data, ["cities", "csv_template"], "data")
    cities = data["cities"]
    if not isinstance(cities, list) or not cities or any(c not in CITY_NAMES for c in cities) or len(set(cities)) != len(cities):
        raise ValueError(f"cities must be unique members of {list(CITY_NAMES)}")
    if not isinstance(data["csv_template"], str) or "{city}" not in data["csv_template"]:
        raise ValueError("data.csv_template must contain {city}")
    try:
        data["csv_template"].format(city=cities[0])
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError("Only {city} is supported in csv_template") from exc

    unseen = config["unseen"]
    _keys(unseen, ["holdout_fraction", "h3_resolution", "split_seed", "train_user_fraction",
                   "validation_user_fraction", "heldout_validation_fraction", "max_sequence_length"], "unseen")
    for field in ["holdout_fraction", "train_user_fraction", "validation_user_fraction", "heldout_validation_fraction"]:
        _number(unseen[field], f"unseen.{field}", upper=1)
        if unseen[field] == 1:
            raise ValueError(f"unseen.{field} must be less than one")
    if unseen["train_user_fraction"] + unseen["validation_user_fraction"] >= 1:
        raise ValueError("User fractions must leave a positive test fraction")
    _number(unseen["split_seed"], "split_seed", integer=True, inclusive=True)
    _number(unseen["h3_resolution"], "h3_resolution", integer=True, inclusive=True, upper=15)
    _number(unseen["max_sequence_length"], "max_sequence_length", integer=True, lower=1, upper=2048)

    granularity = config["granularity"]
    _keys(granularity, ["num_bands", "rff_features", "candidates"], "granularity")
    for field in ["num_bands", "rff_features"]:
        _number(granularity[field], field, integer=True)
    if granularity["rff_features"] % (2 * granularity["num_bands"]):
        raise ValueError("rff_features must be divisible by 2 * num_bands for paired RFF")
    if not isinstance(granularity["candidates"], list) or not granularity["candidates"]:
        raise ValueError("At least one granularity candidate is required")
    ids, ranges = set(), set()
    for band in granularity["candidates"]:
        _keys(band, ["id", "min_scale", "max_scale"], "granularity candidate")
        if not isinstance(band["id"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", band["id"]):
            raise ValueError("Band ids must be safe lowercase path components")
        _number(band["min_scale"], "min_scale")
        _number(band["max_scale"], "max_scale")
        if band["min_scale"] > band["max_scale"]:
            raise ValueError("min_scale must not exceed max_scale")
        pair = (band["min_scale"], band["max_scale"])
        if band["id"] in ids or pair in ranges:
            raise ValueError("Band ids and frequency ranges must be unique")
        ids.add(band["id"]); ranges.add(pair)

    model = config["model"]
    _keys(model, ["encoder", "d_model", "nhead", "num_layers", "dim_feedforward",
                  "hidden_dim", "num_layers_mlp", "dropout", "backbone_type"], "model")
    if model["encoder"] not in ["rff", "rff_sphere", "rff_advanced", "grid", "hex", "spherec"]:
        raise ValueError("The granularity workflow supports rff, rff_sphere, rff_advanced, grid, hex and spherec")
    if model["backbone_type"] not in ["qwen2", "gpt2", "qwen3"]:
        raise ValueError("Unsupported transformer backbone")
    for field in ["d_model", "nhead", "num_layers", "dim_feedforward", "hidden_dim", "num_layers_mlp"]:
        _number(model[field], f"model.{field}", integer=True)
    if model["d_model"] % model["nhead"] or (model["d_model"] // model["nhead"]) % 2:
        raise ValueError("d_model / nhead must be an even integer")
    if model["backbone_type"] == "qwen3" and model["nhead"] % max(1, model["nhead"] // 4):
        raise ValueError("Qwen3 attention heads must be divisible by its key/value head count")
    if model["encoder"] == "grid" and granularity["rff_features"] % 4:
        raise ValueError("Space2Vec grid requires rff_features divisible by 4")
    if model["encoder"] == "hex" and granularity["rff_features"] % 6:
        raise ValueError("Space2Vec hex requires rff_features divisible by 6")
    if model["encoder"] == "spherec" and granularity["rff_features"] % 3:
        raise ValueError("Sphere2Vec sphereC requires rff_features divisible by 3")
    _number(model["dropout"], "dropout", inclusive=True, upper=1)
    if model["dropout"] == 1:
        raise ValueError("dropout must be less than one")

    training = config["training"]
    _keys(training, ["seeds", "epochs", "batch_size", "eval_batch_size", "train_repeats", "lr",
                     "weight_decay", "temperature", "early_stopping_patience", "clip_grad_norm",
                     "encoder_lr_mult", "head_lr_mult", "lr_step1_epochs", "lr_step2_epochs",
                     "lr_step2_mult", "lr_step3_mult", "gpu_id"], "training")
    seeds = training["seeds"]
    if not isinstance(seeds, list) or not seeds or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("training.seeds must be unique nonnegative integers")
    for field in ["epochs", "batch_size", "eval_batch_size", "train_repeats", "early_stopping_patience", "lr_step1_epochs", "lr_step2_epochs"]:
        _number(training[field], f"training.{field}", integer=True)
    for field in ["lr", "temperature", "clip_grad_norm", "encoder_lr_mult", "head_lr_mult", "lr_step2_mult", "lr_step3_mult"]:
        _number(training[field], f"training.{field}")
    _number(training["weight_decay"], "weight_decay", inclusive=True)
    _number(training["gpu_id"], "gpu_id", integer=True, lower=-1, inclusive=True)
    if training["lr_step1_epochs"] > training["lr_step2_epochs"]:
        raise ValueError("Learning-rate milestones must be ordered")
    scoring = config["scoring"]
    _keys(scoring, ["heuristic_loss", "heuristic_popularity", "heuristic_alpha", "heuristic_beta", "heuristic_gamma"], "scoring")
    for field in ["heuristic_loss", "heuristic_popularity"]:
        if not isinstance(scoring[field], bool):
            raise ValueError(f"scoring.{field} must be boolean")
    for field in ["heuristic_alpha", "heuristic_beta"]:
        _number(scoring[field], field, inclusive=True, upper=1)
    _number(scoring["heuristic_gamma"], "heuristic_gamma")
    _keys(config["selection"], ["metric", "tie_policy"], "selection")
    if config["selection"]["metric"] not in SELECTION_METRICS:
        raise ValueError(f"selection.metric must be one of {sorted(SELECTION_METRICS)}")
    if config["selection"]["tie_policy"] != "expected_random":
        raise ValueError("New experiments use expected_random ties across scorers")
    if config["selection"]["metric"] == "first_visit_unseen_trained_r100" and not scoring["heuristic_loss"]:
        raise ValueError("Learned-prior selection requires heuristic_loss=true")
    return config


def load(path=DEFAULT_CONFIG):
    return validate(read_json(path))


def csv_path(config, city):
    path = Path(config["data"]["csv_template"].format(city=city)).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def trials(config):
    """Declare the entire comparison before reading any validation scores."""
    for city in config["data"]["cities"]:
        for band in config["granularity"]["candidates"]:
            for seed in config["training"]["seeds"]:
                yield {"city": city, "band": band, "seed": seed,
                       "run_dir": f"{city}/{band['id']}/seed-{seed}"}
