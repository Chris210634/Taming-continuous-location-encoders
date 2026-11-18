"""Download versioned public MASSIVE-STEPS files; credentials are never stored."""

from pathlib import Path

from .config import CITY_NAMES
from .granularity import file_hash, write_new


def download(city, revision, data_root):
    """Resolve one dataset revision, download its three parts, and record hashes.

    Released parts are combined here because the strict experiment constructs its
    own disjoint user/coordinate splits. Their original names do not define the
    new experiment's training, validation or test membership.
    """
    from huggingface_hub import HfApi, hf_hub_download
    import pandas as pd

    target = Path(data_root).expanduser().resolve() / city
    csv = target / f"{city}_checkins.csv"
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Download requires an empty city directory: {target}")
    repository = f"CRUISEResearchGroup/Massive-STEPS-{CITY_NAMES[city]}"
    resolved = HfApi().dataset_info(repository, revision=revision).sha
    if not resolved:
        raise ValueError("The dataset revision could not be resolved to a commit")
    frames, files = [], []
    for split in ["train", "validation", "test"]:
        filename = f"tabular/{split}-00000-of-00001.parquet"
        path = hf_hub_download(repo_id=repository, repo_type="dataset", filename=filename, revision=resolved)
        frame = pd.read_parquet(path)
        required = {"user_id", "trail_id", "latitude", "longitude", "timestamp"}
        if not required <= set(frame.columns):
            raise ValueError(f"{filename} lacks {sorted(required-set(frame.columns))}")
        frames.append(frame)
        files.append({"filename": filename, "sha256": file_hash(path), "rows": len(frame)})
    target.mkdir(parents=True, exist_ok=True)
    with csv.open("x") as stream:
        pd.concat(frames, ignore_index=True).to_csv(stream, index=False)
    manifest = {"dataset": "MASSIVE-STEPS", "city": city, "repository": repository,
                "requested_revision": revision, "resolved_revision": resolved,
                "source_files": files, "csv_sha256": file_hash(csv),
                "split_note": "Released parts combined; experiment config defines new user and coordinate splits"}
    write_new(target / "download.json", manifest)
    return {"csv": str(csv), **manifest}
