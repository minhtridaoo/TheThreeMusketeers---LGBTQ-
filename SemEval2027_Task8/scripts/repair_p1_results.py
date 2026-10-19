"""Restore notebook-05 experiments and mark invalid/pending results, without training."""
import hashlib, json, sys
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from sqe import results
from sqe.data import TASKS
from sqe.metrics import fold_scores, summary

def repair():
    table=results.load_results(include_invalid=True)
    old_llm=table[table.notebook.eq("06")].copy()
    cfg=json.loads((ROOT/"outputs/runs/mt_twhate_comment_AB.json").read_text(encoding="utf-8"))["cfg"]
    run_date=table.loc[table.notebook.eq("05"),"date"].iloc[0]
    variants=[("single","twhate_comment"),("single + bias","twhate_comment_bias"),
              ("MT","mt_twhate_comment"),("MT + bias","mt_twhate_comment_bias"),
              ("MT + cw","mtcw_twhate_comment"),("MT + cw + bias","mtcw_twhate_comment_bias")]
    restored=[]
    for task in "AB":
        for title,name in variants:
            frame=results.load_oof(name,task)
            restored.append({"date":run_date,"notebook":"05","model":cfg["model"],"input":cfg["input"],
                "subtask":task,"experiment":name,"evaluation_scope":"cv5_video","status":"completed_valid",
                "note":title+("; development OOF cross-fit" if name.endswith("_bias") else ""),
                "source_oof":f"{name}_{task}.csv","source_oof_sha256":hashlib.sha256(results.oof_path(name,task).read_bytes()).hexdigest(),
                "n_eval":len(frame),**summary(fold_scores(frame,TASKS[task][1]))})
    table=table[~table.notebook.eq("05")].copy()
    bad=table.model.eq("microsoft/deberta-v3-base")
    table.loc[bad,"status"]="failed_nonfinite"
    table.loc[bad,"note"]="FAILED: loss/probabilities NaN; excluded; original scores in pre-P1 backup."
    for column in table.columns:
        if column.startswith("f1_") or column.startswith("macro_f1"):
            table.loc[bad,column]=np.nan
    table.loc[table.notebook.eq("08"),"status"]="pending_revalidation"
    table.loc[table.notebook.eq("08"),"evaluation_scope"]="development_oof_crossfit"
    table.loc[table.notebook.eq("08"),"note"]="Historical selection; revalidate after P1 before final report."
    table=pd.concat([table,pd.DataFrame(restored)],ignore_index=True)
    assert not table.duplicated(results.KEY).any()
    assert len(table[table.notebook.eq("05")])==12
    current_llm=table[table.notebook.eq("06")][old_llm.columns]
    pd.testing.assert_frame_equal(old_llm.reset_index(drop=True),current_llm.reset_index(drop=True),check_dtype=False)
    tmp=results.RESULTS_CSV.with_suffix(".tmp")
    table.to_csv(tmp,index=False,encoding="utf-8"); tmp.replace(results.RESULTS_CSV)
    pd.DataFrame(restored).to_csv(ROOT/"outputs/results/nb05_restored.csv",index=False,encoding="utf-8-sig")
    results.load_results().to_csv(ROOT/"outputs/results/validated_results.csv",index=False,encoding="utf-8-sig")
    return {"rows":len(table),"nb05_rows":len(restored),"llm_rows":len(old_llm),
            "status_counts":table.status.value_counts().to_dict(),"repaired_date":date.today().isoformat()}

if __name__=="__main__":
    print(json.dumps(repair(),ensure_ascii=False,indent=2))

