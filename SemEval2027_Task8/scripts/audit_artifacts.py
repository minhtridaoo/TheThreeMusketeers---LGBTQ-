"""Audit saved experiment artifacts without training or modifying source outputs."""
import argparse, hashlib, inspect, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sqe import llm
from sqe.data import load_en, TASKS
from sqe.llm_eval import digest, prepare_data, score_frame
from sqe.metrics import fold_scores, summary, paired_delta
from sqe.decision import average

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def checks_for(frame, truth, task, proba=False):
    column, labels = TASKS[task]
    ids = frame.StereoQueerEval_id
    checks = {"row_count": len(frame) == len(truth), "unique_ids": ids.is_unique,
              "exact_ids": set(ids) == set(truth.index), "valid_labels": set(frame.y_pred) <= set(labels)}
    if checks["exact_ids"] and checks["unique_ids"]:
        aligned = truth.loc[ids]
        checks.update(gold_labels=np.array_equal(frame.y_true, aligned[column]),
                      folds=np.array_equal(frame.fold, aligned.fold),
                      videos=np.array_equal(frame.video_id, aligned.video_id))
    if proba:
        p = frame[[f"p_{label}" for label in labels]].to_numpy()
        checks.update(finite_probabilities=np.isfinite(p).all(),
                      probabilities_in_range=((p >= -1e-7) & (p <= 1+1e-7)).all(),
                      probability_sum_one=np.allclose(p.sum(axis=1), 1, atol=1e-5),
                      argmax_matches_prediction=np.array_equal(np.array(labels)[p.argmax(axis=1)], frame.y_pred))
    return {key: bool(value) for key, value in checks.items()}

def audit():
    df = load_en()
    evaluation, shots = prepare_data(df)
    truth, fold0 = df.set_index("StereoQueerEval_id"), evaluation.set_index("StereoQueerEval_id")
    out = ROOT / "outputs"
    report = {"audited_utc": datetime.now(timezone.utc).isoformat(),
              "dataset": {"n":len(df), "videos":int(df.video_id.nunique()),
              "fold_sizes":{str(k):int(v) for k,v in df.fold.value_counts().sort_index().items()},
              "video_disjoint":bool(df.groupby("video_id").fold.nunique().eq(1).all())},
              "notebooks":[], "llm":[], "oof":[]}
    for path in sorted(p for p in (ROOT / "notebooks").rglob("*.ipynb")  # 01–08 nằm trong notebooks/Baseline/ từ 26/10
                       if not {"_ban_goc", ".ipynb_checkpoints"} & set(p.parts)):
        nb=read_json(path)
        code=[c for c in nb["cells"] if c["cell_type"]=="code"]
        errors=[o.get("evalue") for c in code for o in c.get("outputs",[]) if o.get("output_type")=="error"]
        text="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="markdown")
        report["notebooks"].append({"name":path.name, "code_cells":len(code),
            "executed_cells":sum(c.get("execution_count") is not None for c in code), "errors":errors,
            "conclusion_placeholder":any(s in text for s in ["viết sau khi chạy","viết sau lượt","điền sau lượt"]),
            "sha256":sha(path)})
    status=read_json(out/"llm_cache/execution_status.json")
    env=read_json(out/"llm_cache/environment.json")
    table=pd.read_csv(out/"results/llm_results_fold0.csv", dtype={"notebook":str})
    report["llm_status"], report["llm_model"] = status, env["model"]
    assert len(table)==4 and set(zip(table.experiment,table.subtask))=={(m,t) for m in ["zero_shot","few_shot"] for t in "AB"}
    all_times, llm_frames = [], {}
    for row in table.to_dict("records"):
        task,mode,fp=row["subtask"],row["experiment"],row["run_fingerprint"]
        stem=f"qwen7b_{mode}_{task}_{fp[:16]}"
        path=out/"llm_predictions"/f"{stem}.csv"
        pred=pd.read_csv(path)
        llm_frames[(mode,task)]=pred.set_index("StereoQueerEval_id").loc[fold0.index]
        meta=read_json(out/"llm_cache"/f"{stem}.meta.json")
        records=[json.loads(line) for line in (out/"llm_cache"/f"{stem}.jsonl").read_text(encoding="utf-8").splitlines()]
        checks=checks_for(pred,fold0,task)
        scores=score_frame(pred,task)
        chosen=shots[task] if mode=="few_shot" else []
        hashes={r.StereoQueerEval_id:digest(llm.build_messages(r,task,chosen)) for _,r in evaluation.iterrows()}
        shot_ids=meta["shot_ids"]
        valid={r["id"]:r for r in records if not r.get("error") and r.get("label") in TASKS[task][1]}
        errors=sum(bool(r.get("error")) for r in records)
        checks.update({
          "scores_match_export":all(np.isclose(v,row[k],atol=1e-12) for k,v in scores.items()),
          "std_is_blank":bool(pd.isna(row["macro_f1_std"])),
          "fingerprint_matches_metadata":digest(meta)==fp,
          "fingerprint_matches_prediction":bool(pred.run_fingerprint.eq(fp).all()),
          "environment_matches_model":meta["model"]==env["model"],
          "environment_lists_fingerprint":fp in env["run_fingerprints"],
          "metadata_task_mode":meta["task"]==task and meta["experiment"]==mode,
          "current_prompts_match_cache":hashes==meta["request_hashes"],
          "prompt_source_matches":hashlib.sha256(inspect.getsource(llm).encode()).hexdigest()==meta["prompt_source_sha256"],
          "fewshot_ids_match":shot_ids==[r.StereoQueerEval_id for r,_ in chosen],
          "fewshot_disjoint":not set(shot_ids)&set(fold0.index) and not set(truth.loc[shot_ids].video_id)&set(fold0.video_id),
          "journal_ids_complete":set(valid)==set(fold0.index),
          "journal_matches_prediction":all(r.y_pred==valid.get(r.StereoQueerEval_id,{}).get("label") for r in pred.itertuples()),
          "journal_config_matches":all(r.get("fingerprint")==fp and r.get("request_hash")==hashes.get(r["id"]) for r in records),
          "journal_raw_json_matches":all(json.loads(r["raw"]).get("label")==r["label"] for r in valid.values()),
          "journal_stop_reason":all(r.get("done_reason")=="stop" for r in valid.values()),
          "failure_count_matches":errors==row["failed_attempts"],
          "context_within_limit":all(r["prompt_eval_count"]<meta["options"]["num_ctx"]-meta["options"]["num_predict"] for r in valid.values())})
        cm=confusion_matrix(pred.y_true,pred.y_pred,labels=TASKS[task][1])
        all_times.extend(r["recorded_utc"] for r in records)
        report["llm"].append({"experiment":mode,"subtask":task,"n":len(pred),"valid":all(checks.values()),
            "checks":{k:bool(v) for k,v in checks.items()},"scores":scores,"confusion_labels":TASKS[task][1],
            "confusion_counts":cm.tolist(),"recall":(cm.diagonal()/cm.sum(axis=1)).tolist(),
            "failed_attempts":errors,"journal_records":len(records),"seconds_sum":float(sum(r.get("seconds") or 0 for r in records)),
            "max_prompt_eval_count":max(r["prompt_eval_count"] for r in valid.values()),"fingerprint":fp,"prediction_sha256":sha(path)})
    report["llm_valid"]=all(r["valid"] for r in report["llm"]) and status=={"mode":"full","status":"complete"}
    report["llm_journal_interval_utc"]=[min(all_times),max(all_times)]
    report["llm_changes"]=[]
    for task in "AB":
        zero,few=(llm_frames[(mode,task)] for mode in ["zero_shot","few_shot"])
        z,f=zero.y_pred.eq(zero.y_true),few.y_pred.eq(few.y_true)
        report["llm_changes"].append({"subtask":task,"fixed_by_fewshot":int((~z&f).sum()),
            "broken_by_fewshot":int((z&~f).sum()),"both_wrong":int((~z&~f).sum())})
    frames={}
    for path in sorted((out/"oof").glob("*.csv")):
        task=path.stem[-1]
        if task not in TASKS: continue
        frame=pd.read_csv(path)
        checks=checks_for(frame,truth,task,True)
        entry={"name":path.stem,"subtask":task,"n":len(frame),"valid":all(checks.values()),"checks":checks,"sha256":sha(path)}
        if entry["valid"]:
            frames[path.stem]=frame
            entry["cv_scores"]={k:float(v) for k,v in summary(fold_scores(frame,TASKS[task][1])).items()}
            entry["fold0_scores"]=score_frame(frame[frame.fold==0],task)
        else:
            entry["invalid_probability_cells"]=int((~np.isfinite(frame[[f"p_{l}" for l in TASKS[task][1]]].to_numpy())).sum())
        report["oof"].append(entry)
    report["context_deltas"]=[]
    for task in "AB":
        base=fold_scores(frames[f"roberta_comment_{task}"],TASKS[task][1])
        for variant in ["pair_title","pair_title_desc","concat_tdc"]:
            compared=fold_scores(frames[f"roberta_{variant}_{task}"],TASKS[task][1])
            report["context_deltas"].append({"task":task,"input":variant,**paired_delta(compared,base)})
    report["bad_training_runs"]=[]
    for path in sorted((out/"runs").glob("*.json")):
        data=read_json(path)
        bad=sum(not np.isfinite(epoch["train_loss"]) for history in data.get("history",{}).values() for epoch in history)
        if bad: report["bad_training_runs"].append({"name":path.name,"nonfinite_loss_epochs":bad})
    report["zero_weight_nan_reproduction"]={}
    for task in "AB":
        bad=pd.read_csv(out/"oof"/f"deberta_comment_{task}.csv")
        mixed=average([frames[f"mt_twhate_comment_{task}"],bad],TASKS[task][1],weights=[1,0])
        report["zero_weight_nan_reproduction"][task]=bool(mixed[[f"p_{l}" for l in TASKS[task][1]]].isna().all().all())
    comparison=pd.read_csv(out/"results/llm_comparison_fold0.csv")
    matches=[]
    for row in comparison.to_dict("records"):
        task=row["subtask"]
        frame=llm_frames[(row["system"].removeprefix("Qwen "),task)] if row["system"].startswith("Qwen ") else frames[f"{row['system']}_{task}"].query("fold == 0")
        matches.append(np.isclose(score_frame(frame,task)["macro_f1"],row["macro_f1"]))
    report["fold0_comparison_valid"]=bool(len(comparison)==14 and all(matches))
    report["fold0_comparison"]=comparison[["system","subtask","input","n_eval","macro_f1"]].to_dict("records")
    master=pd.read_csv(out/"results/all_results.csv",dtype={"notebook":str})
    report["master"]={"rows":len(master),"llm_rows":int(master.notebook.str.zfill(2).eq("06").sum()),
        "nb05_rows":int(master.notebook.str.zfill(2).eq("05").sum()),"sha256":sha(out/"results/all_results.csv")}
    report["recipe_saved_not_revalidated"]=read_json(out/"submissions/recipe.json")
    report["dryrun"]=[]
    for task in "AB":
        path=out/"submissions"/f"dryrun_fold0_{task}.tsv"
        pred=pd.read_csv(path,sep="\t",header=None,names=["StereoQueerEval_id","y_pred"])
        assert pred.StereoQueerEval_id.is_unique and set(pred.StereoQueerEval_id)==set(fold0.index)
        pred["y_true"]=fold0.loc[pred.StereoQueerEval_id,TASKS[task][0]].to_numpy()
        assert set(pred.y_pred)<=set(TASKS[task][1])
        report["dryrun"].append({"task":task,"n":len(pred),"scores":score_frame(pred,task),"purpose":"format/pipeline check, not independent evaluation"})
    manual=pd.read_csv(out/"error_analysis/sample_30_B.csv",keep_default_na=False)
    report["manual_error_review"]={"rows":len(manual),"annotated":int(manual.kieu_kho.str.strip().ne("").sum())}
    report["checkpoint_files"]=[str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and p.suffix in {".pt",".pth",".safetensors",".bin"}]
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=audit()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"llm_valid":result["llm_valid"],"fold0_comparison_valid":result["fold0_comparison_valid"],
       "oof_valid":sum(r["valid"] for r in result["oof"]),"oof_total":len(result["oof"]),
       "invalid_oof":[r["name"] for r in result["oof"] if not r["valid"]],"master":result["master"],
       "manual_error_review":result["manual_error_review"],"output":str(args.output)},ensure_ascii=False,indent=2))

