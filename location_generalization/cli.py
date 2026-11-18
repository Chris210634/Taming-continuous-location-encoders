"""Human- and agent-readable entry point; imports no ML dependencies for planning."""

import argparse
import json
from pathlib import Path
import subprocess

from . import config as configuration
from . import granularity


def parser():
    result = argparse.ArgumentParser(
        description="Location Generalization Lab | MASSIVE-STEPS unseen-location prediction")
    commands = result.add_subparsers(dest="action", required=True)
    for action, help_text in [
        ("plan", "Validate configuration and print the declared fits; no training or download"),
        ("prepare", "Build and audit a city's unseen split without training"),
        ("train", "Train one declared city/band/seed using validation only"),
        ("tune", "Train the declared granularity candidates; never evaluate test data"),
        ("select", "Lock the best band per city using mean validation recall across seeds"),
    ]:
        command = commands.add_parser(action, help=help_text, description=help_text)
        command.add_argument("--config", type=Path, default=configuration.DEFAULT_CONFIG,
                             help="Public JSON experiment configuration")
        command.add_argument("--output-dir", type=Path, required=action != "plan",
                             help="Experiment output root; use a new root for changed settings")
        if action in ["prepare", "train"]:
            command.add_argument("--city", required=True, choices=list(configuration.CITY_NAMES))
        if action == "train":
            command.add_argument("--band", required=True, help="Candidate id declared in the config")
            command.add_argument("--seed", type=int, required=True, help="Seed declared in the config")
    evaluation = commands.add_parser("evaluate", help="Evaluate locked winning checkpoints once on test")
    evaluation.add_argument("--selection", type=Path, required=True, help="Frozen selection.json from select")
    evaluation.add_argument("--output-dir", type=Path, required=True, help="Fresh directory for test results")
    evaluation.add_argument("--gpu-id", type=int, help="Override device only; -1 for CPU")
    evaluation.add_argument("--csv-template", help="Relocated CSV path containing {city}; bytes must match training")
    data = commands.add_parser("download", help="Fetch a versioned MASSIVE-STEPS city and record hashes")
    data.add_argument("--city", choices=list(configuration.CITY_NAMES), required=True)
    data.add_argument("--revision", required=True, help="Dataset commit or ref; resolved commit is recorded")
    data.add_argument("--data-root", type=Path, default=configuration.ROOT / "data")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        if arguments.action == "download":
            from .data import download
            print(json.dumps(download(arguments.city, arguments.revision, arguments.data_root), indent=2))
            return
        if arguments.action == "evaluate":
            if arguments.csv_template and "{city}" not in arguments.csv_template:
                raise ValueError("--csv-template must contain {city}")
            reports = granularity.evaluate_selection(arguments.selection, arguments.output_dir,
                                                      arguments.gpu_id, arguments.csv_template)
            print(f"Final test complete: {len(reports)} runs; {arguments.output_dir / 'summary.json'}")
            return
        config = configuration.load(arguments.config)
        if arguments.action == "plan":
            print(json.dumps(granularity.plan(config), indent=2))
            return
        if arguments.action == "prepare":
            if arguments.city not in config["data"]["cities"]:
                raise ValueError("City is not declared in the experiment config")
            output = granularity.freeze_experiment(config, arguments.output_dir)
            destination = output / "prepared" / arguments.city
            if destination.exists():
                raise FileExistsError(f"Prepared split exists: {destination}")
            trial = next(t for t in configuration.trials(config) if t["city"] == arguments.city)
            command = granularity.train_command(config, trial, destination, prepare_only=True)
            granularity.run_logged(command, output / "logs" / f"prepare-{arguments.city}.log")
            print(f"Inspect holdouts, user counts and candidate counts: {destination / 'split.json'}")
        elif arguments.action == "train":
            matches = [t for t in configuration.trials(config)
                       if t["city"] == arguments.city and t["band"]["id"] == arguments.band and t["seed"] == arguments.seed]
            if len(matches) != 1:
                raise ValueError("City, band and seed must identify a trial declared in the config")
            granularity.run_trial(config, matches[0], arguments.output_dir)
        elif arguments.action == "tune":
            for trial in configuration.trials(config):
                granularity.run_trial(config, trial, arguments.output_dir)
            print("All validation runs complete. Run select to lock the frequency range before testing.")
        elif arguments.action == "select":
            result = granularity.select(config, arguments.output_dir)
            for winner in result["selected"]:
                print(f"{winner['city']}: {winner['band']['id']}; mean validation Recall@100={winner['mean_validation_score']:.4f}")
            print(f"Frozen decision: {arguments.output_dir / 'selection.json'}")
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"location_generalization: {exc}") from exc
