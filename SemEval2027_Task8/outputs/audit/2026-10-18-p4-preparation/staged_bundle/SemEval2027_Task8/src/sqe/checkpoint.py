"""Portable inference checkpoints for the project's transformer and TF-IDF models."""
import hashlib
import json
import platform
import uuid
from pathlib import Path
import numpy as np
from .data import TASKS


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def environment():
    import torch, transformers, sklearn
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "transformers": transformers.__version__, "sklearn": sklearn.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def _stage(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists: {destination}")
    stage = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex[:8])
    stage.mkdir(parents=True)
    return stage, destination


def _finish(stage, destination, meta):
    meta = {**meta, "format_version": 1, "complete": True,
            "files_sha256": {str(p.relative_to(stage)).replace("\\", "/"): sha256(p)
                             for p in sorted(stage.rglob("*")) if p.is_file()}}
    write_json(stage / "checkpoint.json", meta)
    stage.rename(destination)
    return meta


def verify_checkpoint(path, expected_identity=None):
    path = Path(path)
    meta = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    if meta.get("format_version") != 1 or not meta.get("complete"):
        raise ValueError(f"Incomplete/unsupported checkpoint: {path}")
    if expected_identity is not None and meta.get("identity") != expected_identity:
        raise ValueError(f"Checkpoint belongs to a different run/config: {path}")
    if meta["labels"] != {t: TASKS[t][1] for t in meta["tasks"]}:
        raise ValueError("Checkpoint label order differs from task definition")
    for rel, expected in meta["files_sha256"].items():
        target = (path / rel).resolve()
        if not target.is_relative_to(path.resolve()) or sha256(target) != expected:
            raise ValueError(f"Checkpoint file failed hash check: {rel}")
    return meta


def save_transformer(model, tokenizer, cfg, tasks, path, identity, history=None):
    import torch
    state = model.state_dict()
    if any(not torch.isfinite(v).all().item() for v in state.values() if v.is_floating_point()):
        raise FloatingPointError("Nonfinite model weights; checkpoint rejected")
    stage, destination = _stage(path)
    model.encoder.config.save_pretrained(stage / "encoder_config")
    tokenizer.save_pretrained(stage / "tokenizer")
    torch.save({k: v.detach().cpu() for k, v in state.items()}, stage / "model.pt")
    return _finish(stage, destination, {"kind": "transformer", "cfg": cfg, "tasks": list(tasks),
           "labels": {t: TASKS[t][1] for t in tasks}, "identity": identity,
           "history": history or [], "environment": environment(),
           "base_model_revision": getattr(model.encoder.config, "_commit_hash", None)})


def load_transformer(path, device="cpu", expected_identity=None):
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from .train import MultiHeadClassifier
    path = Path(path)
    meta = verify_checkpoint(path, expected_identity)
    if meta["kind"] != "transformer":
        raise ValueError("Expected transformer checkpoint")
    cfg = AutoConfig.from_pretrained(path / "encoder_config", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
    model = MultiHeadClassifier(meta["cfg"]["model"], {t: len(meta["labels"][t]) for t in meta["tasks"]},
                                dropout=meta["cfg"]["dropout"], encoder_config=cfg)
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, tok, meta


def save_classical(model, cfg, task, path, identity):
    import joblib
    stage, destination = _stage(path)
    joblib.dump(model, stage / "pipeline.joblib", compress=3)
    return _finish(stage, destination, {"kind": "tfidf", "cfg": cfg, "tasks": [task],
           "labels": {task: TASKS[task][1]}, "identity": identity, "environment": environment()})


def load_classical(path, expected_identity=None):
    import joblib
    meta = verify_checkpoint(path, expected_identity)
    if meta["kind"] != "tfidf":
        raise ValueError("Expected TF-IDF checkpoint")
    model = joblib.load(Path(path) / "pipeline.joblib")
    for t in meta["tasks"]:
        if set(model.classes_) != set(TASKS[t][1]):
            raise ValueError("Classifier classes differ from task labels")
    return model, meta
