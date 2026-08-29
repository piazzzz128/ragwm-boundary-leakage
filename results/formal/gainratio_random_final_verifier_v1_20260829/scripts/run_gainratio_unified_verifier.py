#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import chromadb

COLLECTION='nfcorpus_contriever_cosine'

def db_count(p):
    if not p.is_dir(): return None
    try: return int(chromadb.PersistentClient(path=str(p)).get_collection(COLLECTION).count())
    except Exception: return None

def read_verify(p):
    rows=json.loads(p.read_text()); keys=[tuple(r[0]) for r in rows]; flags=[int(r[2][0]) for r in rows]
    if len(rows)!=30 or len(set(keys))!=30 or any(v not in (0,1) for v in flags): raise RuntimeError(f'Invalid verifier output: {p}')
    return rows,set(keys),flags

def main():
    q=argparse.ArgumentParser(); q.add_argument('--repo',type=Path,required=True); q.add_argument('--old-e2e',type=Path,required=True); q.add_argument('--conditional-root',type=Path,required=True); q.add_argument('--out',type=Path,required=True); a=q.parse_args()
    conditions={
      'watermarked':a.old_e2e/'restored/strict_snapshot/root/autodl-tmp/ragwm_storage/chromadb_db',
      'oracle_restore':a.old_e2e/'vectorstores/e1_oracle_clean',
      'gainratio_fpr05':a.old_e2e/'vectorstores/e2_gainratio_aligned_sensitivity',
    }
    for seed in (101,202,303,404,505):
        conditions[f'random_gainratio_fpr05_seed{seed}']=(
            a.old_e2e/f'vectorstores/e3_random_aligned_v2_seed_{seed}'
        )
    bad={k:{'path':str(v),'count':db_count(v)} for k,v in conditions.items() if db_count(v)!=3633}
    if bad: raise RuntimeError('Database preflight failed: '+json.dumps(bad,indent=2))
    source=a.old_e2e/'conditions/e0_watermarked_before/basepath'
    source_files={
      'wm_prepare/nfcorpus/wmunit.json':source/'wm_prepare/nfcorpus/wmunit.json',
      'wm_generate/nfcorpus/10/wmuint_doc.json':source/'wm_generate/nfcorpus/10/wmuint_doc.json',
      'wm_generate/nfcorpus/10/wmuint_inject.json':source/'wm_generate/nfcorpus/10/wmuint_inject.json',
    }
    for p in source_files.values():
        if not p.is_file(): raise FileNotFoundError(p)
    root=a.out/'verifier_campaign/conditions'; root.mkdir(parents=True,exist_ok=True)
    campaign={'start_utc':datetime.now(timezone.utc).isoformat(),'provider_model':'ephone:gpt-4o-mini (repo alias: gpt4o_mini_e2e)','verify_num':30,'verify_seed':633,'conditions':{k:str(v) for k,v in conditions.items()}}
    (a.out/'verifier_campaign/campaign_manifest_start.json').write_text(json.dumps(campaign,ensure_ascii=False,indent=2)+'\n')
    unit_reference=None
    for name,db in conditions.items():
        condition=root/name; base=condition/'basepath'; output=base/'wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json'
        if output.is_file():
            try:
                _,keys,flags=read_verify(output)
                if unit_reference is not None and keys!=unit_reference: raise RuntimeError('unit mismatch')
                unit_reference=keys if unit_reference is None else unit_reference
                print(f'[SKIP] {name}: WSN={sum(flags)}'); continue
            except Exception:
                quarantine=a.out/'quarantine'/f'verifier_{name}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
                quarantine.parent.mkdir(parents=True,exist_ok=True); shutil.move(str(condition),str(quarantine)); print(f'[QUARANTINE] {condition} -> {quarantine}')
        for rel,src in source_files.items():
            dst=base/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
        log=condition/'verify.log'; log.parent.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'src/main.py','--eval_dataset','nfcorpus','--eval_model_code','contriever','--score_function','cosine','--top_k','5','--mutual_times','10','--model_name_llm','gpt4o_mini_e2e','--model_name_rllm','gpt4o_mini_e2e','--gpu_id','0','--basepath',str(base),'--verify_num','30','--verify_seed','633','--doc','0','--inject','0','--verify','1','--stat','0']
        env=os.environ.copy(); env.update({'PYTHONPATH':str(a.repo),'RAGWM_CHROMA_PATH':str(db),'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8'})
        print(f'[VERIFY] {name}')
        with log.open('w') as h:
            proc=subprocess.Popen(cmd,cwd=a.repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in proc.stdout: print(line,end=''); h.write(line); h.flush()
            code=proc.wait()
        if code!=0: raise RuntimeError(f'Verifier failed: {name}')
        _,keys,flags=read_verify(output)
        if unit_reference is not None and keys!=unit_reference: raise RuntimeError(f'Watermark-unit mismatch: {name}')
        unit_reference=keys if unit_reference is None else unit_reference
        print(f'[PASS] {name}: WSN={sum(flags)}')
    campaign.update({'end_utc':datetime.now(timezone.utc).isoformat(),'status':'complete'})
    (a.out/'verifier_campaign/campaign_manifest_complete.json').write_text(json.dumps(campaign,ensure_ascii=False,indent=2)+'\n')
    print('GAINRATIO RANDOM UNIFIED VERIFIER PASS')

if __name__=='__main__': main()

