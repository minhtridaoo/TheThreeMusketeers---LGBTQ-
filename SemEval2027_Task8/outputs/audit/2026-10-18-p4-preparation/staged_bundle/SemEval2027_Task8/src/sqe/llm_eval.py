"""Reproducible fold-0 Qwen evaluation; no training and no fabricated probabilities."""
import hashlib
import inspect
import json
import os
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import f1_score

from . import llm
from .data import OUT_DIR, TASKS

MODEL = "qwen2.5:7b-instruct-q4_K_M"
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 64}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def prepare_data(df):
    assert len(df) == 2989 and df.StereoQueerEval_id.is_unique
    assert df.groupby("video_id").fold.nunique().eq(1).all()
    evaluation = df[df.fold == 0].reset_index(drop=True)
    assert len(evaluation) == 599
    pool = df[(df.fold != 0) & (df.yt_comment.str.len() < 200)]
    shots = {}
    for task, (column, labels) in TASKS.items():
        examples = pd.concat([pool[pool[column] == label].sample(2, random_state=42)
                              for label in labels]).sample(frac=1, random_state=42)
        assert not set(examples.StereoQueerEval_id) & set(evaluation.StereoQueerEval_id)
        assert not set(examples.video_id) & set(evaluation.video_id)
        shots[task] = [(row, row[column]) for _, row in examples.iterrows()]
    return evaluation, shots


class OllamaClient:
    def __init__(self, model=MODEL, options=None, url="http://127.0.0.1:11434", timeout=300):
        self.model, self.url, self.timeout = model, url.rstrip("/"), timeout
        self.options = dict(OPTIONS if options is None else options)
        self.session = requests.Session()

    def get(self, endpoint):
        response = self.session.get(self.url + endpoint, timeout=15)
        response.raise_for_status()
        return response.json()

    def identity(self):
        models = self.get("/api/tags")["models"]
        found = next((m for m in models if m.get("name") == self.model or m.get("model") == self.model), None)
        if not found or not found.get("digest"):
            raise RuntimeError(f"Model chưa được tải: {self.model}")
        return {"name": self.model, "digest": found["digest"],
                "details": found.get("details", {}), "ollama_version": self.get("/api/version")["version"]}

    def chat(self, messages, labels):
        schema = {"type": "object", "properties": {"label": {"type": "string", "enum": labels}},
                  "required": ["label"], "additionalProperties": False}
        start = time.monotonic()
        response = self.session.post(self.url + "/api/chat", timeout=self.timeout, json={
            "model": self.model, "messages": messages, "format": schema,
            "stream": False, "keep_alive": "30m", "options": self.options})
        response.raise_for_status()
        result = response.json()
        raw = result.get("message", {}).get("content", "")
        metadata = {key: result.get(key) for key in ["done_reason", "prompt_eval_count", "eval_count",
                                                    "total_duration", "load_duration"]}
        error = None
        try:
            label = json.loads(raw).get("label")
        except (ValueError, AttributeError):
            label = None
        if not result.get("done", False) or result.get("done_reason") == "length":
            error = "Phản hồi chưa hoàn chỉnh hoặc đã chạm giới hạn sinh token."
        if label not in labels:
            error = "Phản hồi JSON không có nhãn hợp lệ."
        if (result.get("prompt_eval_count") or 0) >= self.options["num_ctx"] - self.options["num_predict"]:
            error = "Prompt sát giới hạn context; dừng để tránh chấm văn bản bị cắt."
        return {"label": label if not error else None, "raw": raw, "error": error,
                "seconds": time.monotonic() - start, **metadata}

    def warmup_and_check_gpu(self):
        result = self.chat([{"role": "user", "content": 'Return JSON {"label":"no"}.'}], ["no"])
        if result["error"]:
            raise RuntimeError(result["error"])
        running = self.get("/api/ps")["models"]
        model = next((m for m in running if m.get("name") == self.model or m.get("model") == self.model), None)
        if not model or model.get("size_vram", 0) <= 0:
            raise RuntimeError("Qwen chưa dùng GPU. Kiểm tra GPU, driver và ollama_server.log trước khi chạy đủ.")
        return {key: model.get(key) for key in ["name", "size", "size_vram", "context_length"]}


def read_journal(path):
    """Repair only an interrupted final record; reject corruption in the middle."""
    path = Path(path)
    if not path.exists():
        return []
    blob = path.read_bytes()
    records, offset = [], 0
    lines = blob.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1:
                raise ValueError(f"Cache hỏng giữa file: {path}, dòng {index + 1}")
            backup = path.with_name(path.name + f".interrupted-{time.time_ns()}")
            backup.write_bytes(blob[offset:])
            with path.open("r+b") as stream:
                stream.truncate(offset)
            break
        offset += len(line)
    # A valid JSON record without its final newline must not merge with the next append.
    if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as stream:
            stream.write(b"\n")
    return records


class EvaluationRun:
    def __init__(self, evaluation, task, mode, shots, client, identity, output_dir=OUT_DIR):
        if mode not in {"zero_shot", "few_shot"}:
            raise ValueError(mode)
        self.df, self.task, self.mode, self.client = evaluation, task, mode, client
        self.out = Path(output_dir)
        self.labels = TASKS[task][1]
        selected_shots = shots[task] if mode == "few_shot" else []
        self.messages = {row.StereoQueerEval_id: llm.build_messages(row, task, selected_shots)
                         for _, row in evaluation.iterrows()}
        self.hashes = {key: digest(value) for key, value in self.messages.items()}
        self.config = {"schema_version": 1, "task": task, "experiment": mode,
                       "evaluation_scope": "fold0", "model": identity, "options": client.options,
                       "prompt_source_sha256": hashlib.sha256(inspect.getsource(llm).encode()).hexdigest(),
                       "request_hashes": self.hashes,
                       "evaluation_truth": evaluation[["StereoQueerEval_id", TASKS[task][0]]].to_dict("records"),
                       "shot_ids": [row.StereoQueerEval_id for row, _ in selected_shots]}
        self.fingerprint = digest(self.config)
        self.stem = f"qwen7b_{mode}_{task}_{self.fingerprint[:16]}"
        self.path = self.out / "llm_cache" / f"{self.stem}.jsonl"
        atomic_json(self.path.with_suffix(".meta.json"), self.config)
        self.records = self._valid_records()

    def _valid_records(self):
        valid = {}
        for rec in read_journal(self.path):
            if rec.get("fingerprint") != self.fingerprint:
                raise ValueError(f"Cache khác cấu hình: {self.path}")
            sample_id = rec["id"]
            if sample_id not in self.hashes or rec.get("request_hash") != self.hashes[sample_id]:
                raise ValueError(f"Cache khác prompt/dữ liệu: {sample_id}")
            if not rec.get("error") and rec.get("label") in self.labels:
                valid[sample_id] = rec
        return valid

    def check_context(self, tokenizer):
        longest = 0
        for sample_id, messages in self.messages.items():
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            # Transformers 4 returns token IDs; newer versions may return BatchEncoding.
            token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
            if len(token_ids) and isinstance(token_ids[0], (list, tuple)):
                if len(token_ids) != 1:
                    raise ValueError("Tokenizer trả về nhiều prompt cho một request.")
                token_ids = token_ids[0]
            n = len(token_ids)
            longest = max(longest, n)
            if n + self.client.options["num_predict"] + 128 > self.client.options["num_ctx"]:
                raise ValueError(f"{self.mode}/{self.task}: prompt {sample_id} có {n} token. "
                                 "Tăng NUM_CTX rồi chạy lại, không cắt comment âm thầm.")
        return longest

    def progress(self, requested, new_count, started, status, phase):
        elapsed = time.monotonic() - started
        seconds = elapsed / new_count if new_count else None
        done = sum(sample_id in self.records for sample_id in requested)
        atomic_json(self.out / "llm_cache" / "progress.json", {
            "experiment": self.mode, "subtask": self.task, "phase": phase,
            "done": done, "total": len(requested), "status": status,
            "seconds_per_new_request": seconds,
            "eta_minutes": (len(requested) - done) * seconds / 60 if seconds else None,
            "updated_utc": datetime.now(timezone.utc).isoformat()})

    def run(self, sample_ids=None, attempts=3, phase="full", sleep=time.sleep, log=print):
        requested = list(self.messages) if sample_ids is None else list(sample_ids)
        if len(set(requested)) != len(requested) or not set(requested) <= set(self.messages):
            raise ValueError("Danh sách ID cần chạy không hợp lệ")
        started, new_count = time.monotonic(), 0
        self.progress(requested, new_count, started, "running", phase)
        with self.path.open("a", encoding="utf-8") as stream:
            for sample_id in requested:
                if sample_id in self.records:
                    continue
                rec = None
                for attempt in range(1, attempts + 1):
                    try:
                        result = self.client.chat(self.messages[sample_id], self.labels)
                    except (requests.RequestException, ValueError, KeyError) as exc:
                        result = {"label": None, "raw": "", "error": f"{type(exc).__name__}: {exc}", "seconds": None}
                    rec = {"id": sample_id, "fingerprint": self.fingerprint,
                           "request_hash": self.hashes[sample_id], "attempt": attempt,
                           "recorded_utc": datetime.now(timezone.utc).isoformat(), **result}
                    stream.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    if not rec["error"] and rec["label"] in self.labels:
                        break
                    if attempt < attempts:
                        sleep(min(2 ** attempt, 10))
                if rec["error"] or rec["label"] not in self.labels:
                    self.progress(requested, new_count, started, "failed", phase)
                    raise RuntimeError(f"{self.mode}/{self.task}, {sample_id}: thất bại sau {attempts} lần. "
                                       f"Cache đã lưu; không thay lỗi bằng nhãn no. {rec['error']}")
                self.records[sample_id] = rec
                new_count += 1
                self.progress(requested, new_count, started, "running", phase)
                if new_count % 10 == 0:
                    log(f"{self.mode}/{self.task}: {sum(i in self.records for i in requested)}/{len(requested)}", flush=True)
        self.progress(requested, new_count, started, "complete", phase)
        return self.frame(requested)

    def frame(self, requested=None):
        requested = list(self.messages) if requested is None else requested
        out = self.df.set_index("StereoQueerEval_id").loc[requested,
              ["video_id", "fold", TASKS[self.task][0]]].reset_index()
        out = out.rename(columns={TASKS[self.task][0]: "y_true"})
        out["y_pred"] = [self.records[key]["label"] for key in requested]
        out["seconds"] = [self.records[key].get("seconds") for key in requested]
        out["run_fingerprint"] = self.fingerprint
        return out

    def export(self):
        out = self.frame()
        assert len(out) == 599 and out.StereoQueerEval_id.is_unique and out.fold.eq(0).all()
        folder = self.out / "llm_predictions"
        folder.mkdir(parents=True, exist_ok=True)
        out.to_csv(folder / f"{self.stem}.csv", index=False, encoding="utf-8")
        return out


def score_frame(frame, task):
    labels = TASKS[task][1]
    scores = f1_score(frame.y_true, frame.y_pred, labels=labels, average=None, zero_division=0)
    return {"macro_f1": float(scores.mean()), **{f"f1_{label}": float(s) for label, s in zip(labels, scores)}}


def export_results(runs, output_dir=OUT_DIR):
    outdir = Path(output_dir)
    rows = []
    for run in runs:
        frame = run.export()
        records = read_journal(run.path)
        rows.append({"notebook": "06", "model": run.client.model,
                     "input": "title+desc+comment", "experiment": run.mode, "subtask": run.task,
                     "evaluation_scope": "fold0", "n_eval": len(frame),
                     "date": datetime.now(timezone.utc).isoformat(), "macro_f1_std": np.nan,
                     "note": f"{run.mode}; CHỈ fold 0; Q4_K_M; không phải CV 5 fold",
                     "run_fingerprint": run.fingerprint,
                     "failed_attempts": sum(bool(r.get("error")) for r in records),
                     **score_frame(frame, run.task)})
    table = pd.DataFrame(rows)
    assert len(table) == 4 and not table.duplicated(["experiment", "subtask"]).any()
    folder = outdir / "results"
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / "llm_results_fold0.csv", index=False, encoding="utf-8-sig")
    return table


def compare_baselines(evaluation, llm_table, output_dir=OUT_DIR):
    """Use predefined baselines; never pick a winner by its fold-0 score here."""
    outdir = Path(output_dir)
    rows = []
    for task in TASKS:
        target, labels = TASKS[task]
        for name in ["majority_comment", "tfidf_comment", "roberta_comment", "twhate_comment", "mt_twhate_comment"]:
            path = outdir / "oof" / f"{name}_{task}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df = df[df.fold == 0].set_index("StereoQueerEval_id").loc[evaluation.StereoQueerEval_id].reset_index()
            assert len(df) == 599 and np.array_equal(df.y_true, evaluation[target])
            assert set(df.y_pred) <= set(labels)
            rows.append({"system": name, "subtask": task, "evaluation_scope": "fold0", "n_eval": len(df),
                         "input": "comment", **score_frame(df, task)})
    for row in llm_table.to_dict("records"):
        rows.append({"system": f"Qwen {row['experiment']}", **row})
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "results" / "llm_comparison_fold0.csv", index=False, encoding="utf-8-sig")
    return table


def merge_results(output_dir=OUT_DIR):
    """Merge four completed LLM rows into the local master, preserving every other run."""
    folder = Path(output_dir) / "results"
    incoming = pd.read_csv(folder / "llm_results_fold0.csv", dtype={"notebook": str})
    required = {("zero_shot", "A"), ("zero_shot", "B"), ("few_shot", "A"), ("few_shot", "B")}
    assert len(incoming) == 4 and set(zip(incoming.experiment, incoming.subtask)) == required
    assert incoming.n_eval.eq(599).all() and incoming.evaluation_scope.eq("fold0").all()
    assert np.isfinite(incoming.macro_f1).all()
    master = folder / "all_results.csv"
    old = pd.read_csv(master, dtype={"notebook": str}) if master.exists() else pd.DataFrame()
    if len(old):
        if "experiment" not in old:
            old["experiment"] = ""
        old["experiment"] = old.experiment.fillna("")
        remove = pd.Series(False, index=old.index)
        for row in incoming.to_dict("records"):
            remove |= (old.notebook.str.zfill(2).eq("06") & old.model.eq(row["model"])
                       & old.subtask.eq(row["subtask"]) & old.input.eq(row["input"])
                       & old.experiment.eq(row["experiment"]))
        old = old[~remove]
    result = pd.concat([old, incoming], ignore_index=True)
    tmp = master.with_suffix(".tmp")
    result.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(master)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true", required=True)
    parser.parse_args()
    table = merge_results()
    print(f"Đã ghép 4 dòng LLM vào all_results.csv ({len(table)} dòng), giữ các kết quả khác.")
