"""Frozen offline tagger, full raw-input identity, durable no-replay attempts."""
import gc
import sqlite3
import time
from common import *
from contracts import POLICY, assign_word_labels, tokenize_words, word_map, windows


def cache_key(cfg, text):
    return digest([POLICY, cfg['boundary_model_assets'], text])


def reuse_predictions(cfg,db,unique,cached):
    reused=0
    for source in cfg.get('registered_boundary_authorities',[]):
        authority=connect_ro(source)
        assert not authority.execute("SELECT count(*) FROM attempts WHERE state!='COMPLETED'").fetchone()[0]
        for key,cuts,record in authority.execute('SELECT key,breaks,record FROM prediction'):
            if key not in unique:continue
            doc=unique[key];rec=json.loads(record);breaks=json.loads(cuts)
            assert rec['policy']==POLICY and rec['model_assets']==cfg['boundary_model_assets']
            assert rec['text_sha256']==hashlib.sha256(doc['text'].encode()).hexdigest()
            assert key==cache_key(cfg,doc['text']) and breaks==sorted(set(breaks))
            assert all(0<c<=len(doc['text']) for c in breaks)
            if key in cached:assert cached[key]==breaks,'INCONSISTENT_BOUNDARY_CACHE'
            else:
                with db:
                    db.execute('INSERT INTO prediction VALUES(?,?,?)',(key,cuts,record))
                    db.execute('INSERT INTO reused VALUES(?,?,?)',(key,source,hashlib.sha256(record.encode()).hexdigest()))
                cached[key]=breaks;reused+=1
        authority.close()
    return reused


def predict(cfg, documents, stage):
    import torch
    import transformers
    assert not (R/(stage+'_BOUNDARY_RECEIPT.json')).exists(), 'COMPLETED_DO_NOT_RUN'
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg['boundary_model_path'], local_files_only=True, use_fast=True)
    assert tokenizer.is_fast
    db = sqlite3.connect(R/'boundaries.sqlite3')
    db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=FULL')
    db.executescript('CREATE TABLE IF NOT EXISTS prediction(key TEXT PRIMARY KEY,breaks TEXT,record TEXT);CREATE TABLE IF NOT EXISTS attempts(key TEXT PRIMARY KEY,state TEXT,record TEXT);CREATE TABLE IF NOT EXISTS reused(key TEXT PRIMARY KEY,source TEXT,record_sha256 TEXT);')
    assert not db.execute("SELECT count(*) FROM attempts WHERE state!='COMPLETED'").fetchone()[0], 'UNRESOLVED_TAGGER_NO_REPLAY'
    cached = {k: json.loads(b) for k, b in db.execute('SELECT key,breaks FROM prediction')}
    unique = {cache_key(cfg, d['text']): d for d in documents}
    external_reuse=reuse_predictions(cfg,db,unique,cached)
    missing = [(k, d) for k, d in unique.items() if k not in cached]
    if stage in ['engineering','calibration']:
        assert not missing,'COMPLETED_ENGINEERING_CALIBRATION_TAGGER_MUST_REUSE'
    prepared = []
    for key, doc in missing:
        words, ends = word_map(doc['text'])
        ids, wordids = tokenize_words(tokenizer, words)
        assert set(wordids[1]) == set(range(len(words)))
        prepared.append((key, doc, ends, ids, wordids))
    tokens = sum(b-a+2 for key,d,ends,ids,wi in prepared for a,b,e in windows(len(ids)))
    prior = [json.loads(rec) for rec, in db.execute('SELECT record FROM attempts')]
    caps = cfg['resource_caps']
    assert len(prior)+len(missing) <= caps['new_boundary_documents']
    assert sum(x['forward_input_tokens'] for x in prior)+tokens <= caps['new_boundary_input_tokens']
    save(stage+'_BOUNDARY_PREFORWARD.json', {'utc':utc(), 'protocol_sha256':sha(R/'PROTOCOL.json'),
         'documents':len(documents),'unique':len(unique),'cached':len(unique)-len(missing),
         'new_documents':len(missing),'new_input_tokens_including_overlap':tokens,
         'keys_sha256':digest(sorted(unique)), 'external_registered_reuse':external_reuse,'labels_loaded':False})
    model = None; start = time.monotonic(); chunks = 0
    if any(ids for k,d,e,ids,wi in prepared):
        model = transformers.AutoModelForTokenClassification.from_pretrained(cfg['boundary_model_path'], local_files_only=True, torch_dtype=torch.float32).cuda().eval()
        assert model.dtype == torch.float32
        assert model.config.id2label == {0:'0',1:'.',2:',',3:'?',4:'-',5:':'}
    for index, (key, doc, ends, ids, wordids) in enumerate(prepared, 1):
        rec = {'text_sha256': hashlib.sha256(doc['text'].encode()).hexdigest(),
               'forward_input_tokens':sum(b-a+2 for a,b,e in windows(len(ids))),
               'policy':POLICY,'model_assets':cfg['boundary_model_assets'],'stage':stage}
        with db: db.execute('INSERT INTO attempts VALUES(?,?,?)',(key,'STARTED',canon(rec).decode()))
        logits = []
        for a,b,emit in windows(len(ids)):
            assert time.monotonic()-start < caps['boundary_worker_wall_seconds']
            inpids = tokenizer.build_inputs_with_special_tokens(ids[a:b])
            assert inpids == [tokenizer.cls_token_id]+ids[a:b]+[tokenizer.sep_token_id] and len(inpids)<=512
            inp = torch.tensor([inpids], dtype=torch.long, device='cuda')
            with torch.inference_mode():
                logits.append(model(input_ids=inp, attention_mask=torch.ones_like(inp)).logits[0].argmax(-1).cpu().tolist())
            chunks += 1; del inp
        labels = assign_word_labels(wordids, logits, {0:'0',1:'.',2:',',3:'?',4:'-',5:':'})
        cuts = sorted({ends[w] for w,label in labels.items() if label in ('.','?')})
        assert all(0<c<=len(doc['text']) for c in cuts)
        with db:
            db.execute('INSERT INTO prediction VALUES(?,?,?)',(key,canon(cuts).decode(),canon(rec).decode()))
            db.execute("UPDATE attempts SET state='COMPLETED' WHERE key=?",(key,))
        cached[key] = cuts
        if index%100 == 0: status('BOUNDARY_'+stage,complete=index,total=len(missing),new_API=0)
    model=None; gc.collect(); torch.cuda.empty_cache()
    assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)'); db.close()
    save(stage+'_BOUNDARY_RECEIPT.json', {'status':'PASS_ENGINEERING_NOT_SCIENTIFIC_EFFECT',
         'utc':utc(),'new_documents':len(missing),'new_input_tokens_including_overlap':tokens,
         'forward_chunks':chunks,'external_registered_reuse':external_reuse,'wall_seconds':time.monotonic()-start,'new_API':0})
    return {d['doc_id']:cached[cache_key(cfg,d['text'])] for d in documents}
