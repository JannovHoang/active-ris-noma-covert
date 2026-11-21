"""UTF-8 records, portable checksums and atomic, resumable stage receipts."""
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import csv
import hashlib
import json
import platform
import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("full_pso", "random_short_pso", "mlp_direct", "gnn_direct", "mlp_short_pso", "gnn_short_pso")
SHORT = ("random_short_pso", "mlp_short_pso", "gnn_short_pso")
DIRECT = ("mlp_direct", "gnn_direct")
PRIMARY = (("gnn_short_pso", "random_short_pso"), ("gnn_short_pso", "mlp_short_pso"), ("gnn_short_pso", "full_pso"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def lines(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def clean(value):
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    return value


def object_hash(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write(path, value, immutable=False):
    path = Path(path)
    value = clean(value)
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen record changed: {path}; use the original plan or a new output")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def commit(path, value):
    write(path, value)
    write(str(path) + ".receipt.json", {"sha256": digest(path)})


def committed(path):
    path = Path(path)
    receipt = Path(str(path) + ".receipt.json")
    if not receipt.exists(): return None
    if not path.is_file() or read(receipt).get("sha256") != digest(path):
        raise ValueError(f"Committed file changed or missing: {path}")
    return read(path)


def safe_path(root, name):
    # Original Windows manifests use both slash conventions. Absolute paths are never accepted.
    name = str(name).replace("\\", "/")
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts or ":" in pure.parts[0]:
        raise ValueError(f"Unsafe manifest path: {name}")
    root = Path(root).resolve()
    result = (root / Path(*pure.parts)).resolve()
    if not result.is_relative_to(root): raise ValueError("Manifest escaped its root")
    return result


def verify(root, record):
    if not isinstance(record.get("sha256"), dict): raise ValueError("Missing SHA-256 manifest")
    for name, expected in record["sha256"].items():
        path = safe_path(root, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"SHA-256 mismatch: {path}")
    return len(record["sha256"])


def freeze(root, name, paths, **metadata):
    root = Path(root)
    record = dict(metadata, sha256={Path(p).relative_to(root).as_posix(): digest(p) for p in paths})
    write(root / name, record, immutable=True)
    return record


def table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(clean(value), ensure_ascii=False) if isinstance(value, (dict, list, tuple))
                             else value for key, value in row.items()})


def code_hashes():
    files = [ROOT / "run_step7h.py", *sorted((ROOT / "step7h").rglob("*.py"))]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in files if "__pycache__" not in p.parts}


def snapshot_code(output):
    from shutil import copy2
    output = Path(output)
    for name, expected in code_hashes().items():
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != expected: raise ValueError("Output source snapshot changed")
        if not target.exists(): copy2(ROOT / name, target)


def versions():
    return {"system": platform.system(), "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "torch": str(torch.__version__), "torch_threads": torch.get_num_threads()}


def now():
    return datetime.now(timezone.utc).isoformat()


def separate(output, *sources):
    output = Path(output).resolve()
    for source in sources:
        source = Path(source).resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output must be separate and non-nested with every source")


def completed_record(root, name, plan_hash):
    root = Path(root)
    if not (root / name).exists(): return None
    record = read(root / name)
    if record["plan_sha256"] != plan_hash: raise ValueError("Completion belongs to another plan")
    verify(root, record)
    return record
