import os
from pathlib import Path
import yaml


def load_yaml(path):
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_root():
    return Path(__file__).resolve().parents[2]


def resolve_path(path, root=None):
    path = Path(path)
    return path if path.is_absolute() else (root or project_root()) / path


def split_dir(config_value="data/splits"):
    """Resolve the split directory, letting FP_SPLIT_DIR override it.

    Unset (the default) -> identical to resolve_path(config_value), so every
    existing run is unaffected. Set (by the CV array job) -> that directory,
    e.g. data/splits_cv/fold3.
    """
    return resolve_path(os.environ.get("FP_SPLIT_DIR") or config_value)

