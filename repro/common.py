import hashlib
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("single_peak", "dual_peak", "triple_peak")
SEEDS = (42, 123, 456)


def register_vendor_paths(root=PACKAGE_ROOT):
    for relative in ("vendor", "vendor/physics", "vendor/baselines"):
        path = str(Path(root).resolve() / relative)
        if path not in sys.path:
            sys.path.insert(0, path)


def asset(relative, root=PACKAGE_ROOT):
    base = Path(root).resolve() / "assets"
    path = (base / relative).resolve()
    try:
        path.relative_to(base)
    except ValueError:
        raise ValueError("Asset path escapes package")
    return path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def create_output(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    try:
        rel = path.relative_to(root)
    except ValueError:
        raise ValueError("Output must stay inside the package")
    if len(rel.parts) < 2 or rel.parts[0] != "outputs":
        raise ValueError("Choose a new directory under outputs/")
    path.mkdir(parents=True, exist_ok=False)
    return path


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()
