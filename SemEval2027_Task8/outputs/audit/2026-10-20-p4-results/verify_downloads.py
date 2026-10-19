"""Audit downloaded P4 ZIPs without extracting/executing model weights.
Reads every ZIP member to verify CRC and SHA-256; recomputes ensemble output.
Kaggle reload evidence is checked, not rerun in this different CPU environment.
"""
from pathlib import Path, PurePosixPath
import hashlib, io, json, shutil, sys, zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[2]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
from sqe.p4 import data_for_mode, component_plan, fingerprint
from sqe.data import TASKS
from sqe.decision import apply_bias
from sqe.validation import validate_probabilities
from sqe.checkpoint import sha256

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def require(condition, message):
    if not condition:
        raise ValueError(message)

protected = list((ROOT/'outputs/oof').rglob('*.csv'))
protected += list((ROOT/'outputs/results').glob('*.csv'))
protected += list((ROOT/'outputs/selection').rglob('*'))
protected += [ROOT/'outputs/submissions/recipe.json', ROOT/'outputs/error_analysis/sample_30_B.csv', ROOT/'data/splits/folds_en.csv']
protected = sorted(set(p for p in protected if p.is_file()))
before = {p.relative_to(ROOT).as_posix():sha256(p) for p in protected}
save(AUDIT/'protected_hashes.json', before)
for p in [ROOT/'ROADMAP.md', ROOT/'PIPELINE.md', WORKSPACE/'HANDOFF_CS221.md', WORKSPACE/'WORKLOG_CS221.md']:
    dest=AUDIT/'before'/p.relative_to(WORKSPACE)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)

recipe_path=ROOT/'outputs/submissions/recipe.json'
recipe=json.loads(recipe_path.read_text(encoding='utf-8'))
results=[]
for mode in ['smoke','dryrun','final']:
    archive=WORKSPACE/'kaggle/results'/f'sqe_p4_{mode}.zip'
    print('AUDITING',mode, 'archive SHA-256 and all ZIP contents', flush=True)
    archive_hash=sha256(archive)
    prefix=f'SemEval2027_Task8/outputs/p4/{mode}/'
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        require(len(names)==len(set(names)), 'Duplicate archive entry')
        require(not any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n or ':' in n for n in names), 'Unsafe archive path')
        require(not any('failure' in n or '.partial-' in n for n in names), 'Failure/partial artifact found')
        digests={}
        for item in z.infolist():
            if item.is_dir(): continue
            h=hashlib.sha256()
            with z.open(item) as stream:
                for block in iter(lambda:stream.read(4*1024*1024), b''):
                    h.update(block)
            digests[item.filename]=h.hexdigest()
        def get(rel): return json.loads(z.read(prefix+rel))
        plan=get('plan.json'); report=get('reload_report.json'); status=get('status.json')
        require(status==report, 'Status and reload report differ')
        require(report['mode']==mode and report['status']=='complete', 'Incomplete mode')
        require(report['fresh_process'] is True and report['offline_reload'] is True, 'No fresh offline reload')
        require(report['training_or_launcher_pid']!=report['reload_pid'], 'Process IDs equal')
        require(fingerprint({k:v for k,v in plan.items() if k!='run_fingerprint'})==plan['run_fingerprint']==report['run_fingerprint'], 'Run fingerprint mismatch')
        require(plan['recipe']==recipe and plan['recipe_sha256']==sha256(recipe_path), 'Recipe mismatch')
        specs=component_plan(recipe)
        if mode=='smoke':
            specs=[{**s,'cfg':{**s['cfg'],'epochs':1}} if s['kind']=='transformer' else s for s in specs]
        require(specs==plan['components'], 'Component configuration mismatch')
        for rel,digest in plan['source_sha256'].items():
            require(digests['SemEval2027_Task8/'+rel]==digest==sha256(ROOT/rel), 'Code mismatch: '+rel)
        for rel in ['scripts/run_p4.py','data/splits/folds_en.csv','outputs/submissions/recipe.json']:
            require(digests['SemEval2027_Task8/'+rel]==sha256(ROOT/rel), 'Packaged file mismatch: '+rel)
        train,probe=data_for_mode(mode)
        require(plan['train_ids']==train.StereoQueerEval_id.tolist(), 'Train IDs differ')
        require(plan['probe_ids']==probe.StereoQueerEval_id.tolist(), 'Probe IDs differ')
        require(plan['train_data_sha256']==fingerprint(train.to_dict('records')), 'Training data mismatch')
        require(plan['probe_data_sha256']==fingerprint(probe.to_dict('records')), 'Probe data mismatch')
        require(report['n_train']==len(train) and report['n_predictions']==len(probe), 'Sample count mismatch')
        inp=pd.read_csv(io.BytesIO(z.read(prefix+'prediction_input.tsv')),sep='\t',dtype=str,keep_default_na=False)
        truth=pd.read_csv(io.BytesIO(z.read(prefix+'truth.csv')),dtype=str,keep_default_na=False)
        require(inp.equals(probe[['StereoQueerEval_id','yt_title','yt_description','yt_comment']]), 'Probe text mismatch')
        require(truth.equals(probe[['StereoQueerEval_id','stereotype','hate_speech']]), 'Gold mismatch')
        require(not set(TASKS[t][0] for t in 'AB') & set(inp.columns), 'Labels present in inference input')
        overlap=len(set(train.video_id)&set(probe.video_id))
        require(mode=='final' or overlap==0, 'Video leakage into dryrun/smoke')
        refs={}; checkpoint_rows=[]; expected_checks=[]; checked_payloads=0
        metas=[n for n in names if n.startswith(prefix+'checkpoints/') and n.endswith('/checkpoint.json')]
        require(len(metas)==7==report['checkpoint_count']==len(specs), 'Checkpoint count mismatch')
        for spec in specs:
            name=spec['name']; cp=prefix+'checkpoints/'+name+'/'
            meta=json.loads(z.read(cp+'checkpoint.json'))
            require(meta['complete'] is True and meta['format_version']==1, 'Incomplete checkpoint')
            require(meta['kind']==spec['kind'] and meta['cfg']==spec['cfg'] and meta['tasks']==spec['tasks'], 'Metadata/config mismatch')
            require(meta['environment']==plan['environment'], 'Checkpoint environment mismatch')
            require(meta['labels']=={t:TASKS[t][1] for t in spec['tasks']}, 'Class order mismatch')
            require(meta['identity']==fingerprint({'run':plan['run_fingerprint'],'component':spec}), 'Checkpoint identity mismatch')
            actual={n[len(cp):] for n in names if n.startswith(cp) and not n.endswith('/') and n!=cp+'checkpoint.json'}
            require(actual==set(meta['files_sha256']), 'Unhashed or missing checkpoint files')
            for rel,digest in meta['files_sha256'].items():
                require(digests[cp+rel]==digest, 'Checkpoint payload hash mismatch: '+name+'/'+rel)
                checked_payloads+=1
            refpath=prefix+'reference/'+name+'.npz'
            require(digests[refpath]==meta['reference_sha256'], 'Reference hash mismatch')
            with np.load(io.BytesIO(z.read(refpath)),allow_pickle=False) as ref:
                require(set(ref.files)==set(spec['tasks']), 'Reference tasks differ')
                refs[name]={t:ref[t].copy() for t in spec['tasks']}
            for task,P in refs[name].items():
                validate_probabilities(P)
                require(P.shape==(len(probe),len(TASKS[task][1])), 'Probability shape mismatch')
                expected_checks.append((name,task))
            history=meta.get('history',[])
            if spec['kind']=='transformer':
                require([h['epoch'] for h in history]==list(range(1,spec['cfg']['epochs']+1)), 'Missing epochs')
                require(all(np.isfinite(h['train_loss']) and h['train_loss']>=0 for h in history), 'Invalid training loss')
            checkpoint_rows.append({'name':name,'kind':spec['kind'],'tasks':spec['tasks'],'epochs_completed':len(history),'training_minutes':meta['training_minutes'],'train_loss':[h['train_loss'] for h in history]})
        require([(c['component'],c['task']) for c in report['checks']]==expected_checks, 'Missing/duplicate reload checks')
        require(all(c['labels_equal'] is True and c['probabilities_close'] is True and c['max_abs_diff']==0 for c in report['checks']), 'Reload check failed')
        prediction_rows=[]
        for task in 'AB':
            r=recipe[task]
            P=sum(w*refs[n][task] for n,w in r['components'].items())/sum(r['components'].values())
            pred=np.asarray(r['labels'])[apply_bias(P,np.asarray(r['bias'])).argmax(1)]
            filename=f'dryrun_fold0_{task}.tsv' if mode=='dryrun' else f'{mode}_probe_{task}.tsv'
            out=pd.read_csv(io.BytesIO(z.read(prefix+'predictions/'+filename)),sep='\t',header=None,dtype=str,keep_default_na=False)
            require(out.shape==(len(probe),2), 'Prediction TSV shape mismatch')
            require(out[0].is_unique and out[0].tolist()==plan['probe_ids'], 'Prediction ID order mismatch')
            require(out[1].tolist()==pred.tolist(), 'Predictions differ from recomputed frozen recipe')
            row={'task':task,'rows':len(out),'id_order_labels_recipe_match':True,'predicted_label_counts':out[1].value_counts().to_dict()}
            if mode=='dryrun':
                y=truth[TASKS[task][0]].tolist()
                score=f1_score(y,pred,labels=r['labels'],average='macro',zero_division=0)
                recorded=next(x['macro_f1_pipeline_check_only'] for x in report['dryrun_scores'] if x['task']==task)
                require(abs(score-recorded)<1e-12, 'Dryrun score mismatch')
                _,rec,f1,support=precision_recall_fscore_support(y,pred,labels=r['labels'],zero_division=0)
                row.update(macro_f1_pipeline_check_only=score,labels=r['labels'],per_class_f1=f1.tolist(),per_class_recall=rec.tolist(),support=support.tolist(),confusion_matrix=confusion_matrix(y,pred,labels=r['labels']).tolist())
            prediction_rows.append(row)
        require(mode=='dryrun' or report['dryrun_scores']==[], 'Unexpected quality metric on smoke/final')
        # Keep lightweight evidence only. The ZIPs remain the authoritative weight locations.
        for n in names:
            if not n.startswith(prefix): continue
            rel=n[len(prefix):]
            if rel in ['plan.json','status.json','reload_report.json','prediction_input.tsv','truth.csv'] or rel.startswith(('predictions/','reference/')) or rel.endswith('/checkpoint.json'):
                dest=AUDIT/'evidence'/mode/rel
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(z.read(n))
        result={'mode':mode,'archive':archive.relative_to(WORKSPACE).as_posix(),'archive_size_bytes':archive.stat().st_size,'archive_sha256':archive_hash,'member_count':len(names),'uncompressed_bytes':sum(i.file_size for i in z.infolist()),'all_member_crc_read_ok':True,'member_sha256':digests,'checkpoint_payload_hashes_verified':checked_payloads,'run_fingerprint':plan['run_fingerprint'],'environment':plan['environment'],'n_train':len(train),'n_probe':len(probe),'train_probe_video_overlap':overlap,'checkpoint_count':7,'reload_task_checks':len(expected_checks),'reported_max_reload_probability_diff':0.0,'fresh_process_pids':[report['training_or_launcher_pid'],report['reload_pid']],'reference_to_export_recomputed':True,'checkpoints':checkpoint_rows,'component_training_minutes_sum':sum(m['training_minutes'] for m in checkpoint_rows),'predictions':prediction_rows,'status':'verified'}
        save(AUDIT/(mode+'_audit.json'),result)
        results.append(result)
        print(mode,'VERIFIED',len(names),'members;',checked_payloads,'checkpoint hashes;',len(probe),'IDs per task',flush=True)

unchanged=all(sha256(ROOT/rel)==digest for rel,digest in before.items())
require(unchanged,'Protected artifacts changed')
summary={'checked_on':'2026-10-20','status':'passed','p4_complete':True,'final_full_data_checkpoints_available':True,'modes':[{k:v for k,v in r.items() if k!='member_sha256'} for r in results],'protected_artifacts_count':len(before),'protected_artifacts_unchanged':unchanged,'verification_scope':'Local archive CRC/SHA-256, source/config/recipe/data/fold/ID/label checks, training history, stored Kaggle fresh offline reload evidence, independently recomputed predictions and dryrun metrics. No local model forward pass or retraining.','performance_scope':'Dryrun fold 0 is pipeline verification only, not independent test; final 32-row probe is in-sample and unscored.','checkpoint_storage':'Original ZIPs retained in kaggle/results; only lightweight audit evidence copied. No project overwrite.'}
save(AUDIT/'validation.json',summary)
print('ALL VERIFIED; protected files unchanged:',len(before),flush=True)
