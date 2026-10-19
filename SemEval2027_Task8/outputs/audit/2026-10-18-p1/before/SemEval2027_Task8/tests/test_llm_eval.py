"""Contract tests using synthetic model replies, never real inference or result files."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import nbformat
import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(BASE / "kaggle/nb06_tools"))
sys.path.insert(0, str(BASE / "kaggle"))
from sqe.data import load_en, TASKS
from sqe.llm_eval import (MODEL, OPTIONS, OllamaClient, EvaluationRun, prepare_data,
                          read_journal, export_results, merge_results, compare_baselines)
from kaggle_support import pack_results, stage_bundle
import merge_nb06_results as importer


@pytest.fixture(scope="module")
def data():
    return prepare_data(load_en())


class FakeClient:
    model = MODEL

    def __init__(self, fail=False):
        self.options = dict(OPTIONS)
        self.calls = 0
        self.fail = fail

    def chat(self, messages, labels):
        self.calls += 1
        return {"label": None if self.fail else "no", "raw": "fake reply",
                "error": "invalid JSON" if self.fail else None, "seconds": 0.1}


def make_run(data, tmp_path, client=None, mode="zero_shot", task="A", identity=None):
    evaluation, shots = data
    return EvaluationRun(evaluation, task, mode, shots, client or FakeClient(),
                         identity or {"name": MODEL, "digest": "test-only"}, tmp_path)


def quiet(*args, **kwargs):
    pass


def test_few_shot_does_not_leak_evaluation(data):
    evaluation, shots = data
    assert len(evaluation) == 599
    for task, count in [("A", 4), ("B", 6)]:
        assert len(shots[task]) == count
        assert all(row.fold != 0 for row, label in shots[task])
        assert not {row.video_id for row, _ in shots[task]} & set(evaluation.video_id)
        assert pd.Series([label for _, label in shots[task]]).value_counts().eq(2).all()


def test_resume_and_configuration_isolation(data, tmp_path):
    client = FakeClient()
    run = make_run(data, tmp_path, client)
    ids = list(run.messages)[:3]
    run.run(ids, log=quiet)
    resumed = make_run(data, tmp_path, client)
    resumed.run(ids, log=quiet)
    assert client.calls == 3
    assert len(resumed.records) == 3
    assert make_run(data, tmp_path, mode="few_shot").path != run.path
    assert make_run(data, tmp_path, identity={"digest": "other"}).path != run.path
    changed = FakeClient()
    changed.options["num_ctx"] = 8192
    assert make_run(data, tmp_path, changed).path != run.path
    changed_df = data[0].copy()
    changed_df.loc[0, "yt_comment"] += " Changed prompt"
    assert make_run((changed_df, data[1]), tmp_path).path != run.path
    record = json.loads(run.path.read_text(encoding="utf-8").splitlines()[0])
    record["fingerprint"] = "wrong"
    run.path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cache khác"):
        make_run(data, tmp_path)


def test_failure_never_becomes_no_and_can_retry(data, tmp_path):
    client = FakeClient(fail=True)
    run = make_run(data, tmp_path, client)
    ids = list(run.messages)[:1]
    with pytest.raises(RuntimeError, match="không thay lỗi"):
        run.run(ids, sleep=quiet, log=quiet)
    assert client.calls == 3 and not run.records
    assert all(r["label"] is None for r in read_journal(run.path))
    good = make_run(data, tmp_path)
    good.run(ids, log=quiet)
    assert good.client.calls == 1 and len(good.records) == 1


def test_interrupted_journal(tmp_path):
    path = tmp_path / "cache.jsonl"
    path.write_bytes(b'{"id":1}\n{"id":')
    assert read_journal(path) == [{"id": 1}]
    assert path.read_bytes() == b'{"id":1}\n'
    assert len(list(tmp_path.glob("*.interrupted-*"))) == 1
    path.write_bytes(b'{"id":1}')
    assert read_journal(path) == [{"id": 1}]
    assert path.read_bytes().endswith(b"\n")
    path.write_bytes(b'broken\n{"id":1}\n')
    with pytest.raises(ValueError, match="hỏng giữa"):
        read_journal(path)


@pytest.mark.parametrize("as_dict", [False, True])
def test_context_counts_tokens_and_rejects_overflow(data, tmp_path, as_dict):
    class Tokenizer:
        length = 100

        def apply_chat_template(self, *args, **kwargs):
            ids = [1] * self.length
            return {"input_ids": ids, "attention_mask": ids} if as_dict else ids

    tokenizer = Tokenizer()
    run = make_run(data, tmp_path)
    assert run.check_context(tokenizer) == 100
    tokenizer.length = 4096
    with pytest.raises(ValueError, match="Tăng NUM_CTX"):
        run.check_context(tokenizer)


@pytest.mark.parametrize("payload,valid", [
    ({"done": True, "message": {"content": '{"label":"no"}'}}, True),
    ({"done": True, "message": {"content": 'not json'}}, False),
    ({"done": True, "message": {"content": '{"label":"unknown"}'}}, False),
    ({"done": True, "done_reason": "length", "message": {"content": '{"label":"no"}'}}, False),
    ({"done": True, "prompt_eval_count": 4090, "message": {"content": '{"label":"no"}'}}, False),
])
def test_chat_contract(monkeypatch, payload, valid):
    client = OllamaClient()

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    def post(url, **kwargs):
        assert kwargs["json"]["format"]["properties"]["label"]["enum"] == ["no", "yes"]
        assert kwargs["json"]["options"]["temperature"] == 0
        return Response()

    monkeypatch.setattr(client.session, "post", post)
    result = client.chat([], ["no", "yes"])
    assert (result["label"] == "no") == valid
    assert bool(result["error"]) != valid


def test_full_exports_merge_and_import(data, tmp_path, monkeypatch):
    project = tmp_path / "download/SemEval2027_Task8"
    output = project / "outputs"
    runs = [make_run(data, output, task=task, mode=mode)
            for mode in ["zero_shot", "few_shot"] for task in "AB"]
    for run in runs:
        run.run(log=quiet)
    table = export_results(runs, output)
    assert len(table) == 4 and table.n_eval.eq(599).all() and table.macro_f1_std.isna().all()
    # Real baselines are read only, then copied into this temporary test sandbox.
    (output / "oof").mkdir()
    for source in (PROJECT / "outputs/oof").glob("*.csv"):
        if source.stem in {f"{name}_{task}" for name in ["majority_comment", "tfidf_comment",
                            "roberta_comment", "twhate_comment", "mt_twhate_comment"] for task in "AB"}:
            (output / "oof" / source.name).write_bytes(source.read_bytes())
    comparison = compare_baselines(data[0], table, output)
    assert len(comparison) == 14 and comparison.n_eval.eq(599).all()
    old = pd.read_csv(PROJECT / "outputs/results/all_results.csv", dtype={"notebook": str})
    old.to_csv(output / "results/all_results.csv", index=False)
    merged = merge_results(output)
    assert len(merged) == len(old) + 4
    assert len(merge_results(output)) == len(merged)
    pd.testing.assert_frame_equal(merged.iloc[:len(old)][old.columns].reset_index(drop=True),
                                  old, check_dtype=False)
    (project / "notebooks").mkdir()
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('test')", execution_count=1)])
    nbformat.write(nb, project / "notebooks/06_llm_zero_few_shot.ipynb")
    (output / "llm_cache/execution_status.json").write_text(
        json.dumps({"status": "complete", "mode": "full"}))
    destination = tmp_path / "destination"
    (destination / "outputs/results").mkdir(parents=True)
    old.to_csv(destination / "outputs/results/all_results.csv", index=False)
    other = destination / "notebooks/08_summary.ipynb"
    other.parent.mkdir()
    other.write_text("preserve this notebook")
    monkeypatch.setattr(importer, "BASE", tmp_path)
    importer.import_results(project, destination)
    importer.import_results(project, destination)
    assert len(pd.read_csv(destination / "outputs/results/all_results.csv")) == len(old) + 4
    assert other.read_text() == "preserve this notebook"
    assert list((tmp_path / "kaggle/nb06_backups").rglob("all_results_before_merge.csv"))
    table.loc[0, "macro_f1"] = -1
    table.to_csv(output / "results/llm_results_fold0.csv", index=False)
    with pytest.raises(AssertionError):
        importer.import_results(project, destination)
    archive = pack_results(project, tmp_path / "results.zip")
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert not any("all_results" in name or "/oof/" in name or "08_summary" in name for name in z.namelist())
        assert sum("llm_predictions/" in name for name in z.namelist()) == 4


def test_bundle_and_notebooks(tmp_path):
    for path in [PROJECT / "notebooks/06_llm_zero_few_shot.ipynb", BASE / "kaggle/kaggle_runner_06.ipynb"]:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
    extracted = tmp_path / "input"
    with zipfile.ZipFile(BASE / "kaggle/sqe_nb06_bundle.zip") as z:
        assert z.testzip() is None
        z.extractall(extracted)
    manifest = json.loads((extracted / "sqe06_manifest.json").read_text())
    for name, checksum in manifest["files"].items():
        assert hashlib.sha256((extracted / name).read_bytes()).hexdigest() == checksum
    staged, _ = stage_bundle(extracted, tmp_path / "work")
    assert (staged / "data/splits/folds_en.csv").is_file()
    assert not (staged / "outputs/llm_cache").exists()
    (extracted / "SemEval2027_Task8/src/sqe/llm_eval.py").write_text("corrupt")
    with pytest.raises(ValueError, match="manifest"):
        stage_bundle(extracted, tmp_path / "work")


@pytest.mark.parametrize("code,expected", [("print('worker verified')", "smoke_complete"),
                                           ("raise ValueError('expected test failure')", "failed")])
def test_worker_saves_notebook_on_success_or_failure(tmp_path, code, expected):
    folder = tmp_path / "notebooks"
    folder.mkdir()
    notebook = folder / "06_llm_zero_few_shot.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)]), notebook)
    result = subprocess.run([sys.executable, "-X", "utf8", str(BASE / "kaggle/nb06_tools/execute_06.py"), str(tmp_path)],
                            env=dict(os.environ, SQE06_MODE="smoke"), capture_output=True, text=True,
                            encoding="utf-8", timeout=90)
    status = json.loads((tmp_path / "outputs/llm_cache/execution_status.json").read_text(encoding="utf-8"))
    assert status["status"] == expected, result.stdout + result.stderr
    assert (result.returncode == 0) == (expected == "smoke_complete")
    saved = nbformat.read(notebook, as_version=4)
    assert saved.cells[0].execution_count == 1 and saved.cells[0].outputs
