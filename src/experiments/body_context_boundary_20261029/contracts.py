"""Pure CPU contracts. No source selection, model forward, API, or outcome reads."""
import ast
import bisect
import hashlib
import json
import math
import pathlib
import re
import typing

POLICY = 'FullStop_FP32_joined_offsets_last_subtoken_480_32_raw_word_end_v2'
METHODS = ('RAW', 'BIN', 'VIRTUAL_GR')


def json_scalar(value):
    if type(value).__module__.split('.')[0] == 'numpy' and hasattr(value, 'item'):
        return value.item()
    raise TypeError('unsupported JSON value: ' + type(value).__name__)


def canon(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False, default=json_scalar).encode('utf-8')


def digest(value):
    return hashlib.sha256(canon(value)).hexdigest()


def positions(ci, ti):
    if len(ti) < 2:
        raise ValueError('target too short')
    labels = list(range(len(ci) + 1, len(ci) + len(ti)))
    return labels, [i - 1 for i in labels]


def request_key(identity, kind, ci, ti):
    if kind not in ('Lu', 'Lc') or (kind == 'Lu' and ci):
        raise ValueError('invalid primitive')
    return digest([identity, ti] if kind == 'Lu' else [identity, ci, ti])


def word_map(text):
    """Match the author's preprocessing; map each cleaned word to its RAW word end.

    Consumed terminal punctuation belongs to the original word, rather than a
    second punctuation-only virtual sentence. Never insert punctuation in C/T.
    """
    removed = {m.start() for m in re.finditer(r'(?<!\d)[.,;:!?](?!\d)', text)}
    indices = [i for i in range(len(text)) if i not in removed]
    cleaned = ''.join(text[i] for i in indices)
    words, ends = [], []
    for match in re.finditer(r'\S+', cleaned):
        end = indices[match.end() - 1] + 1
        while end < len(text) and not text[end].isspace() and end in removed:
            end += 1
        words.append(match.group())
        ends.append(end)
    assert words == re.sub(r'(?<!\d)[.,;:!?](?!\d)', '', text).split()
    assert ends == sorted(set(ends))
    return words, ends


def windows(n):
    if n < 0:
        raise ValueError('negative token count')
    result = []
    for a in range(0, n, 448):
        b = min(a + 480, n)
        result.append((a, b, b if b == n else a + 448))
        if b == n:
            break
    return result


def tokenize_words(tokenizer, words):
    """Tokenize the actual joined string, not per-word normalizer invocations.

    Invisible format characters (e.g. U+FEFF) can make the two token streams
    differ. Fast offsets bind the author's actual joined input to raw words.
    An entirely normalized-away word receives label0 and cannot create a cut.
    """
    joined = ' '.join(words)
    encoded = tokenizer(joined, add_special_tokens=False, truncation=False,
                        return_offsets_mapping=True, verbose=False)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    last, start = {}, 0
    for word, text in enumerate(words):
        end = start + len(text)
        overlap = [i for i, (a, b) in enumerate(offsets) if a < end and b > start and b > a]
        last[word] = overlap[-1] if overlap else None
        start = end + 1
    assert len(ids) == len(offsets)
    assert all(p is None or 0 <= p < len(ids) for p in last.values())
    return ids, (len(ids), last)


def assign_word_labels(word_ids, window_logits, id2label):
    """Each word is emitted exactly once, by ownership of its final subtoken."""
    if isinstance(word_ids, tuple):
        count, last = word_ids
    else:
        count = len(word_ids)
        last = {word: i for i, word in enumerate(word_ids) if word is not None}
    assert set(last) == set(range(len(last)))
    predictions = {word:'0' for word,pos in last.items() if pos is None}
    expected = windows(count)
    assert len(window_logits) == len(expected)
    for (a, b, emit), labels in zip(expected, window_logits):
        assert len(labels) == b - a + 2
        for word, pos in last.items():
            if pos is not None and a <= pos < emit:
                assert word not in predictions
                predictions[word] = id2label[labels[pos - a + 1]]
    assert set(predictions) == set(last)
    return predictions


def virtual_spans(text, cuts, original):
    ends = sorted({0, len(text), *(b for a, b in original(text)),
                   *(c for c in cuts if 0 < c < len(text))})
    spans = []
    for a, b in zip(ends, ends[1:]):
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b:
            spans.append((a, b))
    return spans


def text_functions(path):
    env = {'re': re, 'List': typing.List, 'Tuple': typing.Tuple,
           'Optional': typing.Optional, 'MAX_CONTEXT_CHARS': 1200,
           'MIN_TARGET_CHARS': 25}
    names = {'norm_space', 'tail_context', 'sentence_spans', 'last_eligible_sentence_span'}
    nodes = [n for n in ast.parse(pathlib.Path(path).read_text('utf-8')).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'frozen_text_functions', 'exec'), env)
    return env


def candidates(cfg, tokenizer, doc, cuts=None):
    env = text_functions(cfg['scorer_path'])
    if cuts is not None:
        original = env['sentence_spans']
        env['sentence_spans'] = lambda s: virtual_spans(s, cuts, original)
    text, limit, result = doc['text'], len(doc['text']), []
    for rank in range(1, 4):
        span = env['last_eligible_sentence_span'](text[:limit])
        if span is None:
            break
        a, b = span
        context = env['tail_context'](text[:a])
        target = env['norm_space'](text[a:b])
        if not context or len(target) < 25:
            break
        ci = tokenizer(context, add_special_tokens=False, truncation=True, max_length=1024)['input_ids']
        ti = tokenizer(target, add_special_tokens=False, truncation=True, max_length=128)['input_ids']
        if len(ti) < 2:
            break
        result.append({'candidate_id': digest([doc['doc_id'], a, b, ti]),
                       'doc_id': doc['doc_id'], 'component': doc['component'],
                       'a': a, 'b': b, 'rank': rank, 'context_token_ids': ci,
                       'target_token_ids': ti,
                       'Lu_key': request_key(cfg['request_identity'], 'Lu', [], ti),
                       'Lc_actual_key': request_key(cfg['request_identity'], 'Lc', ci, ti)})
        limit = len(text[:a].rstrip())
    assert len(result) <= 3
    if cfg.get('body_context_eligibility',False):
        from body_context_prototype import eligible
        if 'source_title' not in doc:
            raise ValueError('BOUND_TITLE_METADATA_REQUIRED')
        result=[c for c in result if eligible(text,doc['source_title'],c['a'],c['b'])]
    return result


def bin_score(raw, nc, nt, cells):
    key = str((bisect.bisect_right([20, 28, 40], nt), bisect.bisect_right([196, 247, 292], nc)))
    values = cells.get(key, [])
    if len(values) < 200:
        values = sorted(x for group in cells.values() for x in group)
    assert values
    return -(1 + len(values) - bisect.bisect_left(values, raw)) / (len(values) + 1)


def first_action(rows, threshold, field):
    return next((c for c in sorted(rows, key=lambda x: x['rank'])
                 if c[field] >= threshold), None)


def calibrate_joint(documents, rows, field, doc_rate, char_rate):
    """Most permissive clean-only cutoff under common, monotone damage caps.

    Max accepted candidate length per document bounds first-action damage even
    when changing the threshold changes which rank acts first. Ties stay whole.
    """
    ids = {d['doc_id'] for d in documents}
    assert len(ids) == len(documents) and documents
    by_score = {}
    for row in rows:
        assert row['doc_id'] in ids and math.isfinite(row[field])
        by_score.setdefault(row[field], []).append(row)
    if not by_score:
        raise ValueError('NO_FINITE_CALIBRATION_CANDIDATES')
    maxdocs = math.floor(doc_rate * len(documents))
    maxchars = math.floor(char_rate * sum(len(d['text']) for d in documents))
    lengths, threshold = {}, math.nextafter(max(by_score), math.inf)
    for value in sorted(by_score, reverse=True):
        proposed = lengths.copy()
        for c in by_score[value]:
            proposed[c['doc_id']] = max(proposed.get(c['doc_id'], 0), c['b'] - c['a'])
        if len(proposed) > maxdocs or sum(proposed.values()) > maxchars:
            break
        lengths, threshold = proposed, value
    groups = {i: [] for i in ids}
    for row in rows:
        groups[row['doc_id']].append(row)
    actions = [first_action(groups[d['doc_id']], threshold, field) for d in documents]
    realized = sum(a['b'] - a['a'] for a in actions if a)
    assert sum(a is not None for a in actions) <= maxdocs and realized <= maxchars
    return {'threshold': threshold, 'doc_budget': maxdocs, 'char_budget': maxchars,
            'selected_documents': len(lengths), 'conservative_char_upper': sum(lengths.values()),
            'actual_first_action_characters': realized,
            'doc_rate': doc_rate, 'char_rate': char_rate}


def outcome_bounds(flags):
    assert all(f in (0, 1, 2) for f in flags)
    yes, unknown = flags.count(1), flags.count(2)
    return {'YES': yes, 'NO': flags.count(0), 'Unknown': unknown,
            'units': len(flags), 'WSN_Unknown_bounds': [yes, yes + unknown]}
