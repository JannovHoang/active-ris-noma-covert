"""Verified reuse of globally frozen 7d SEARCH only, never selections/audits.

Cache reuse requires identical inputs, runtime, factorial settings and actual
search-function ASTs/dependency bytes. Previous and current source snapshots
are retained so an assessment-only numerical correction is auditable.
"""
import ast
import hashlib
from pathlib import Path

from .learning_data import read_json, sha256
from .multik_dataset import immutable_bytes, read_committed

FUNCTIONS = ("table_policy", "lookup_allocation", "load_net", "search_pool", "diagnostic", "search_job")
CONTROL_FILES = {"run_step7d.py", "ris_noma/multik_benchmark.py", "ris_noma/multik_benchmark_analysis.py", "ris_noma/multik_benchmark_core.py", "ris_noma/stable_assessment.py", "ris_noma/multik_benchmark_cache.py"}


def function_hashes(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    return {name: hashlib.sha256(ast.dump(functions[name], include_attributes=False).encode()).hexdigest() for name in FUNCTIONS}


def validate_cache(cache, plan, root):
    cache, root = Path(cache).resolve(), Path(root).resolve()
    old = read_json(cache / "plan.json")
    frozen = read_json(cache / "search_frozen.json")
    for field in ("format", "config", "jobs", "models", "normalization", "lookup_policy", "pso_settings", "streams", "workers", "threads_per_worker", "versions", "dataset_sha256", "training_completion_sha256"):
        if old[field] != plan[field]:
            raise ValueError(f"Search cache has different {field}")
    if frozen["plan_sha256"] != sha256(cache / "plan.json"):
        raise ValueError("Search cache freeze does not match its original plan")
    for name, digest in old["code_sha256"].items():
        snapshot = cache / "source_snapshot" / name
        if not snapshot.is_file() or sha256(snapshot) != digest:
            raise ValueError("Search cache lacks an exact source snapshot")
        if name not in CONTROL_FILES and (not (root / name).is_file() or sha256(root / name) != digest):
            raise ValueError("Search cache numeric dependency changed")
    previous_functions = function_hashes(cache / "source_snapshot/ris_noma/multik_benchmark_core.py")
    if previous_functions != function_hashes(root / "ris_noma/multik_benchmark_core.py"):
        raise ValueError("Search cache algorithm functions changed")
    expected = {"search/" + job["id"] + ".json" for job in plan["jobs"]}
    if set(frozen["sha256"]) != expected:
        raise ValueError("Search cache is incomplete")
    for name, digest in frozen["sha256"].items():
        record = read_committed(cache / name)
        if record is None or sha256(cache / name) != digest or record["plan_sha256"] != frozen["plan_sha256"]:
            raise ValueError("Search cache group is missing or changed")
    return {"original_plan_sha256": frozen["plan_sha256"], "original_search_frozen_sha256": sha256(cache / "search_frozen.json"),
        "search_files_sha256": frozen["sha256"], "source_snapshot_sha256": old["code_sha256"],
        "search_function_sha256": previous_functions,
        "policy": "reuse exact frozen searches only after function/dependency/input checks; recompute every selection and audit"}


def install_cache(cache, output, metadata):
    cache, output = Path(cache), Path(output)
    immutable_bytes(output / "search_origin/plan.json", (cache / "plan.json").read_bytes())
    immutable_bytes(output / "search_origin/search_frozen.json", (cache / "search_frozen.json").read_bytes())
    for name in metadata["source_snapshot_sha256"]:
        immutable_bytes(output / "search_origin/source_snapshot" / name, (cache / "source_snapshot" / name).read_bytes())
    for name in metadata["search_files_sha256"]:
        immutable_bytes(output / name, (cache / name).read_bytes())
        immutable_bytes(output / (name + ".receipt.json"), (cache / (name + ".receipt.json")).read_bytes())


def verify_origin(output, plan, root):
    metadata = plan.get("search_origin")
    if metadata is None: return
    output = Path(output)
    origin = output / "search_origin"
    if sha256(origin / "plan.json") != metadata["original_plan_sha256"] or sha256(origin / "search_frozen.json") != metadata["original_search_frozen_sha256"]:
        raise ValueError("Search-origin plan/freeze changed")
    old = read_json(origin / "plan.json")
    for name, digest in metadata["source_snapshot_sha256"].items():
        if sha256(origin / "source_snapshot" / name) != digest:
            raise ValueError("Search-origin source snapshot changed")
    if function_hashes(Path(root) / "ris_noma/multik_benchmark_core.py") != metadata["search_function_sha256"]:
        raise ValueError("Reused search engine no longer matches")
    if old["config"] != plan["config"] or old["models"] != plan["models"] or old["jobs"] != plan["jobs"]:
        raise ValueError("Reused search context changed")
