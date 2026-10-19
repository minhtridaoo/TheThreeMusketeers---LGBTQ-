"""Regression checks for nonfinite OOF, row alignment, weights, and result identity."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from sqe import results
from sqe.validation import InvalidOOFError, validate_oof
from sqe.decision import average, greedy_select, nested_greedy, apply_bias

LABELS=["no","yes"]
@pytest.fixture
def frame():
    return pd.DataFrame({"StereoQueerEval_id":["a","b","c","d"],"video_id":["v1","v2","v3","v4"],
        "fold":[0,1,2,3],"y_true":["no","yes","yes","no"],"y_pred":["no","yes","no","yes"],
        "p_no":[.8,.2,.6,.4],"p_yes":[.2,.8,.4,.6]})

def test_zero_weight_excludes_nan_even_in_first_position(frame):
    bad=frame.copy(); bad[["p_no","p_yes"]]=np.nan
    pd.testing.assert_frame_equal(average([bad,frame],LABELS,[0,1]),frame,check_like=True)
    with pytest.raises(InvalidOOFError,match="Nonfinite"):
        average([bad,frame],LABELS,[.1,.9])
    with pytest.raises(InvalidOOFError,match="Nonfinite"):
        greedy_select([frame,bad],LABELS)

@pytest.mark.parametrize("weights",[[0,0],[-1,2],[np.inf,1],[1],[np.nan,1]])
def test_invalid_weights_rejected(frame,weights):
    with pytest.raises(InvalidOOFError,match="Weights"):
        average([frame,frame],LABELS,weights)

def test_positive_weights_and_overflow_safe_normalization(frame):
    other=frame.copy()
    other[["p_no","p_yes"]]=np.array([[.2,.8],[.7,.3],[.8,.2],[.1,.9]])
    other["y_pred"]=np.array(LABELS)[other[["p_no","p_yes"]].to_numpy().argmax(1)]
    actual=average([frame,other],LABELS,[1e308,1e308])
    np.testing.assert_allclose(actual[["p_no","p_yes"]],(frame[["p_no","p_yes"]]+other[["p_no","p_yes"]])/2)

@pytest.mark.parametrize("kind",["reorder","gold","fold","video"])
def test_misaligned_systems_rejected(frame,kind):
    other=frame.copy()
    if kind=="reorder": other=other.iloc[::-1].reset_index(drop=True)
    elif kind=="gold": other.loc[0,"y_true"]="yes"
    elif kind=="fold": other.loc[0,"fold"]=4
    else: other.loc[0,"video_id"]="different"
    with pytest.raises(InvalidOOFError,match="different row order"):
        average([frame,other],LABELS)

@pytest.mark.parametrize("kind",["duplicate","sum","argmax","missing","nonfinite"])
def test_malformed_oof_rejected(frame,kind):
    if kind=="duplicate": frame.loc[1,"StereoQueerEval_id"]="a"
    elif kind=="sum": frame.loc[0,"p_no"]=.5
    elif kind=="argmax": frame.loc[0,"y_pred"]="yes"
    elif kind=="missing": frame=frame.drop(columns="fold")
    else: frame.loc[0,"p_no"]=np.inf
    with pytest.raises(InvalidOOFError): validate_oof(frame,LABELS)

def test_load_oof_checks_canonical_gold(frame,tmp_path,monkeypatch):
    monkeypatch.setattr(results,"OOF_DIR",tmp_path)
    canonical=frame.rename(columns={"y_true":"stereotype"}).copy()
    monkeypatch.setattr(results,"load_en",lambda: canonical)
    path=tmp_path/"sample_A.csv"; frame.to_csv(path,index=False)
    assert len(results.load_oof("sample","A"))==4
    frame.loc[0,"y_true"]="yes"; frame.to_csv(path,index=False)
    with pytest.raises(InvalidOOFError,match="canonical"):
        results.load_oof("sample","A")

def test_registry_keeps_experiments_and_failed_history(tmp_path,monkeypatch):
    path=tmp_path/"all_results.csv"
    monkeypatch.setattr(results,"RESULTS_CSV",path)
    pd.DataFrame([
        {"notebook":"04","model":"bad","input":"comment","subtask":"A","experiment":"","status":"failed_nonfinite"},
        {"notebook":"06","model":"Qwen","input":"context","subtask":"A","experiment":"zero_shot","evaluation_scope":"fold0","macro_f1":.6},
        {"notebook":"06","model":"Qwen","input":"context","subtask":"A","experiment":"few_shot","evaluation_scope":"fold0","macro_f1":.7}
    ]).to_csv(path,index=False)
    scores=pd.DataFrame({"fold":[0,1],"macro_f1":[.7,.8],"f1_no":[.7,.8],"f1_yes":[.7,.8]})
    for name in ["single","MT","MT_cw","MT_bias"]:
        results.log_result("05","encoder","comment","A",scores,experiment=name)
    results.log_result("05","encoder","comment","A",scores,experiment="MT")
    raw=results.load_results(include_invalid=True)
    assert len(raw)==7 and len(results.load_results())==6
    assert set(raw[raw.notebook.eq("05")].experiment)=={"single","MT","MT_cw","MT_bias"}
    assert raw[raw.notebook.eq("06")].macro_f1.tolist()==[.6,.7]

def test_make_oof_does_not_turn_nan_into_no(frame):
    df=frame.rename(columns={"y_true":"stereotype"})
    with pytest.raises(InvalidOOFError,match="Nonfinite"):
        results.make_oof(df,"A",np.full((4,2),np.nan))

def test_real_candidates_reject_deberta_and_greedy_is_finite():
    for task in "AB":
        labels=results.TASKS[task][1]
        with pytest.raises(InvalidOOFError,match="Nonfinite"):
            results.load_oof("deberta_comment",task)
        oofs=[results.load_oof(name,task) for name in ["roberta_comment","twhate_comment","mt_twhate_comment"]]
        combined,picks=nested_greedy(oofs,labels,max_steps=3)
        validate_oof(combined,labels)
        assert len(combined)==2989 and len(picks)==5

