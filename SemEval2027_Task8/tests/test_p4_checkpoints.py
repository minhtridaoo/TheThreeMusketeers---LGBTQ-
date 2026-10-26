"""Real small-model round trips, corruption detection, and frozen-recipe contracts."""
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertTokenizerFast
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sqe.checkpoint import save_transformer, load_transformer, save_classical, load_classical, verify_checkpoint
from sqe.train import MultiHeadClassifier, encode, predict_proba, DEFAULT, fit
from sqe.data import TASKS
from sqe.classical import make_model
from sqe.p4 import component_plan, data_for_mode


@pytest.fixture
def tiny(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\nkind\nmean\npeople\ntitle\n", encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
    cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=16, num_hidden_layers=1,
                     num_attention_heads=2, intermediate_size=32, max_position_embeddings=64)
    model = MultiHeadClassifier("not-a-real-remote-model", {"A": 2, "B": 3}, encoder_config=cfg)
    run = {**DEFAULT, "model": "not-a-real-remote-model", "max_length": 32, "epochs": 1, "batch_size": 2}
    df = pd.DataFrame({"yt_comment": ["hello world", "kind people", "mean people", "hello people"],
                       "yt_title": ["title"] * 4, "yt_description": ["world"] * 4,
                       "stereotype": ["no", "yes", "no", "yes"],
                       "hate_speech": ["no", "yes_implicit", "yes_explicit", "no"]})
    return model, tok, run, df


def test_transformer_offline_roundtrip_in_new_process(tiny, tmp_path):
    model, tok, cfg, df = tiny
    cfg["input"] = "pair_title_desc"
    before = predict_proba(model, encode(tok, df, cfg), np.arange(len(df)), tok.pad_token_id, torch.device("cpu"), batch_size=2)
    path = tmp_path / "checkpoint"
    save_transformer(model, tok, cfg, ["A", "B"], path, "test-identity")
    df.to_json(tmp_path / "inputs.json", orient="records")
    worker = tmp_path / "worker.py"
    worker.write_text("""import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from sqe.checkpoint import load_transformer
from sqe.train import encode, predict_proba
root=Path(sys.argv[2]);df=pd.read_json(root/'inputs.json')
m,tok,meta=load_transformer(root/'checkpoint',expected_identity='test-identity')
p=predict_proba(m,encode(tok,df,meta['cfg']),np.arange(len(df)),tok.pad_token_id,torch.device('cpu'),batch_size=2)
np.savez(root/'after.npz',**p)
""",encoding="utf-8")
    env = os.environ.copy(); env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME=str(tmp_path / "empty_cache"))
    result = subprocess.run([sys.executable, str(worker), str(ROOT / "src"), str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    with np.load(tmp_path / "after.npz") as after:
        for task in "AB":
            np.testing.assert_allclose(before[task], after[task], atol=1e-7)
            np.testing.assert_array_equal(before[task].argmax(1), after[task].argmax(1))
    with pytest.raises(ValueError, match="different run"):
        verify_checkpoint(path, "wrong-identity")
    with (path / "model.pt").open("ab") as stream: stream.write(b"corrupt")
    with pytest.raises(ValueError, match="hash check"):
        verify_checkpoint(path)


def test_small_real_training_saves_and_loads_both_heads(tiny, tmp_path):
    model, tok, cfg, df = tiny
    base = tmp_path / "local_base"; model.encoder.save_pretrained(base); tok.save_pretrained(base)
    cfg = {**cfg, "model": str(base), "class_weight": True}
    trained, history = fit(df, np.arange(len(df)), cfg, ["A", "B"], encode(tok, df, cfg), tok, torch.device("cpu"), log=lambda _: None)
    assert len(history) == 1 and np.isfinite(history[0]["train_loss"])
    save_transformer(trained, tok, cfg, ["A", "B"], tmp_path / "trained", "tiny-training", history)
    loaded, _, meta = load_transformer(tmp_path / "trained")
    assert meta["tasks"] == ["A", "B"] and set(loaded.heads) == {"A", "B"}
    for name, value in trained.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name])


def test_tfidf_roundtrip_preserves_feature_pipeline_and_label_order(tmp_path):
    text = ["kind hello world", "mean people word"] * 4
    y = ["no", "yes"] * 4
    model = make_model("tfidf_lr").fit(text, y)
    save_classical(model, {"input": "comment"}, "A", tmp_path / "tfidf", "tfidf-id")
    loaded, meta = load_classical(tmp_path / "tfidf", "tfidf-id")
    np.testing.assert_allclose(model.predict_proba(text), loaded.predict_proba(text), atol=0)
    assert meta["labels"]["A"] == TASKS["A"][1]


def test_recipe_has_seven_unique_components_and_correct_scope():
    # P4 đã chạy với recipe P3 ngày 18/10; recipe hiện hành đổi sau P7/P8.2 (26/10) và dùng sqe.testtime.
    recipe = json.loads((ROOT / "outputs/audit/2026-10-26-p3-rerun/before/outputs/submissions/recipe.json")
                        .read_text(encoding="utf-8"))
    specs = component_plan(recipe)
    assert len(specs) == 7
    assert sum(s["kind"] == "transformer" for s in specs) == 5
    for name in ["mt_twhate_comment", "mtcw_twhate_comment"]:
        assert next(s for s in specs if s["name"] == name)["tasks"] == ["A", "B"]
    for mode, n_train, n_probe in [("smoke",64,16),("dryrun",2390,599),("final",2989,32)]:
        train, probe = data_for_mode(mode)
        assert (len(train),len(probe)) == (n_train,n_probe)
        if mode != "final": assert set(train.video_id).isdisjoint(probe.video_id)


def test_complete_small_suite_resume_and_fresh_offline_verify(tiny, tmp_path, monkeypatch):
    from sqe import p4
    from sqe.checkpoint import environment, sha256
    model, tok, cfg, frame = tiny
    base = tmp_path / "base"; model.encoder.save_pretrained(base); tok.save_pretrained(base)
    cfg = {**cfg, "model": str(base)}
    train = frame.assign(StereoQueerEval_id=[f"train{i}" for i in range(len(frame))])
    probe = frame.assign(StereoQueerEval_id=[f"probe{i}" for i in range(len(frame))])
    recipe = {"A": {"components": {"tiny_mt":1.,"tfidf_comment":1.},"bias":[0.,0.]},
              "B": {"components": {"tiny_mt":1.},"bias":[0.,0.,0.]}}
    plan = {"mode":"smoke", "recipe":recipe, "components":[
        {"name":"tiny_mt","kind":"transformer","tasks":["A","B"],"cfg":cfg},
        {"name":"tfidf_comment","kind":"tfidf","tasks":["A"],"cfg":{"input":"comment"}}],
        "train_ids":train.StereoQueerEval_id.tolist(),"probe_ids":probe.StereoQueerEval_id.tolist(),
        "environment":environment(),"prediction_batch_size":2,"purpose":"local tiny integration test",
        "source_sha256":{"src/sqe/p4.py":sha256(ROOT/'src/sqe/p4.py')}}
    plan['run_fingerprint']=p4.fingerprint(plan)
    monkeypatch.setattr(p4,'create_plan',lambda mode:(plan,train,probe))
    out=tmp_path/'suite'
    p4.train_suite('smoke',out,require_gpu=False)
    before={str(f.relative_to(out)):sha256(f) for f in (out/'checkpoints').rglob('*') if f.is_file()}
    p4.train_suite('smoke',out,require_gpu=False)
    assert before == {str(f.relative_to(out)):sha256(f) for f in (out/'checkpoints').rglob('*') if f.is_file()}
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_p4.py'),'verify','--output-dir',str(out)],
                          capture_output=True,text=True,timeout=240)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    report=json.loads((out/'reload_report.json').read_text(encoding='utf-8'))
    assert report['status']=='complete' and report['offline_reload']
    assert report['training_or_launcher_pid'] != report['reload_pid']
    assert report['checkpoint_count']==2 and report['n_predictions']==4
    assert (out/'predictions/smoke_probe_A.tsv').exists()
