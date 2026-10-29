"""Append-only continuation utilities; registered parents are read-only."""
import collections
import datetime
import hashlib
import json
import math
import os
import pathlib
import re
import sqlite3
import threading
import unicodedata
from contracts import canon, digest, positions, request_key

R = pathlib.Path(__file__).resolve().parent


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(pathlib.Path(path).read_text('utf-8-sig'))


def rows(path):
    with pathlib.Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    path = R / name
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canon(value) + b'\n'
    if path.exists():
        assert path.read_bytes() == content, 'REFUSE_OVERWRITE:' + str(path)
        return
    with path.open('xb') as stream:
        stream.write(content)


def freeze_rows(name, data):
    path = R / name
    with path.open('xb') as stream:
        for row in data:
            stream.write(canon(row) + b'\n')


def status(stage, **details):
    value = {'utc': utc(), 'stage': stage, **details}
    tmp = R / ('STATUS.tmp.' + str(os.getpid()) + '.' + str(threading.get_ident()))
    tmp.write_bytes(canon(value) + b'\n')
    os.replace(tmp, R / 'STATUS.json')
    print(canon(value).decode('utf-8'), flush=True)


def connect_ro(path):
    path = pathlib.Path(path)
    assert path.is_file(), 'MISSING_AUTHORITY:' + str(path)
    wal = pathlib.Path(str(path) + '-wal')
    assert not wal.exists() or wal.stat().st_size == 0, 'ACTIVE_WAL:' + str(path)
    return sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)


def contact_risk(text):
    return bool(re.search(r'[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}|(?:https?://)?(?:twitter\.com|x\.com|facebook\.com|instagram\.com)/|(?<!\w)@[A-Za-z][A-Za-z0-9_]{2,}', text, re.I)
                or re.search(r'(?i)(?:telephone|phone|fax|mobile|tel\s*:)\s*[:+]?\s*[\d ()+\-]{7,}', text))


def primitive_pool(cfg):
    """All incompatible losses under the SAME exact key block reuse/forward."""
    observations = collections.defaultdict(list)
    inventory = cfg['legacy_config']['cache_inventory']
    db = connect_ro(inventory)
    valid = set()
    for kind, key, ci, ti in db.execute('SELECT kind,key,context_ids,target_ids FROM requests'):
        assert key == request_key(cfg['request_identity'], kind, json.loads(ci), json.loads(ti))
        valid.add((kind, key))
    for kind, key, loss in db.execute('SELECT kind,key,loss FROM observations'):
        if loss is not None and math.isfinite(loss) and (kind, key) in valid:
            observations[kind, key].append((float(loss), inventory))
    db.close()
    authorities = (cfg['legacy_config']['standard_primitive_caches'] +
                   [cfg['new_closed_clean_cache'], cfg['prior_engineering_cache']] +
                   cfg['extra_primitive_caches'])
    if (R/'nll.sqlite3').exists():
        authorities = authorities + [str(R/'nll.sqlite3')]
    for path in dict.fromkeys(authorities):
        db = connect_ro(path)
        tables = {x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'attempts' in tables:
            cols = {x[1] for x in db.execute('PRAGMA table_info(attempts)')}
            field = 'state' if 'state' in cols else 'status'
            states = dict(db.execute('SELECT ' + field + ',count(*) FROM attempts GROUP BY ' + field))
            assert not (set(states) - {'COMPLETED', 'DONE', 'SUCCESS'}), (path, states)
        for kind, key, loss, record in db.execute('SELECT kind,key,loss,record FROM primitive'):
            rec = json.loads(record)
            ci, ti = rec['context_token_ids'], rec['target_token_ids']
            assert key == request_key(cfg['request_identity'], kind, ci, ti), 'CACHE_INPUT_IDENTITY'
            if 'absolute_label_positions' in rec:
                assert rec['absolute_label_positions'] == positions(ci, ti)[0]
            assert math.isfinite(loss)
            observations[kind, key].append((float(loss), path))
        db.close()
    path = cfg['legacy_config']['fixedT_cache']
    db = connect_ro(path)
    assert not db.execute("SELECT count(*) FROM attempts WHERE state!='COMPLETED'").fetchone()[0]
    for kind, loss, record in db.execute('SELECT kind,loss,record FROM primitives'):
        rec = json.loads(record)
        ci, ti = rec['context_token_ids'], rec['target_token_ids']
        assert rec['absolute_label_positions'] == positions(ci, ti)[0]
        mapped = 'Lu' if kind == 'Lu' else 'Lc'
        observations[mapped, request_key(cfg['request_identity'], mapped, ci, ti)].append((float(loss), path))
    db.close()
    pool, blocked = {}, set()
    for key, values in observations.items():
        if len({v for v, p in values}) > 1:
            blocked.add(key)
        else:
            pool[key] = values[0]
    return pool, blocked


def requests_for(candidates):
    result = {}
    identity = read(R/'PROTOCOL.json')['request_identity']
    for c in candidates:
        for kind, key, ci in [('Lu', c['Lu_key'], []), ('Lc', c['Lc_actual_key'], c['context_token_ids'])]:
            row = {'kind': kind, 'key': key, 'context_token_ids': ci,
                   'target_token_ids': c['target_token_ids']}
            assert key == request_key(identity, kind, ci, row['target_token_ids'])
            if (kind, key) in result:
                assert result[kind, key] == row
            result[kind, key] = row
    return result


def check_implementation(cfg):
    for name, expected in cfg['implementation_sha256'].items():
        assert sha(R / name) == expected, 'IMPLEMENTATION_CHANGED:' + name


def candidate_parser(cfg, tokenizer):
    from contracts import candidates
    return lambda doc: candidates(cfg, tokenizer, doc)


def source_documents(cfg,role):
    documents=rows(pathlib.Path(cfg['prepared_parent'])/(role+'_SOURCE_DOCS.jsonl'))
    metadata={d['doc_id']:d for d in read(R/'SOURCE_ALLOCATION.json')['roles'][role]}
    assert len(documents)==len(metadata)
    output=[]
    for d in documents:
        title,body=d['text'].split('\n',1)
        normalized=' '.join(unicodedata.normalize('NFKC',title).casefold().split())
        expected=metadata[d['doc_id']]
        assert hashlib.sha256(d['text'].encode()).hexdigest()==expected['text_sha256']
        assert hashlib.sha256(normalized.encode()).hexdigest()==expected['normalized_title_sha256']
        assert d['component']==expected['component'] and body.strip()
        output.append(d|{'source_title':title})
    return output
