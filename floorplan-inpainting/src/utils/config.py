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

