"""Reviewable next-hypothesis prototype, NOT installed in the frozen experiment.

Uses source title metadata and raw offsets only. Does not score, enumerate,
calibrate, rank, delete, call a model, or inspect watermark labels.
"""


def classify_context(raw_text, source_title, start, end):
    if not isinstance(source_title,str) or not source_title or '\n' in source_title:
        raise ValueError('EXACT_TITLE_METADATA_REQUIRED')
    prefix=source_title+'\n'
    if not raw_text.startswith(prefix):
        raise ValueError('TITLE_METADATA_TEXT_BINDING_MISMATCH')
    if not raw_text[len(prefix):].strip():
        raise ValueError('NONEMPTY_BODY_REQUIRED')
    if not (isinstance(start,int) and isinstance(end,int) and 0<=start<end<=len(raw_text)):
        raise ValueError('INVALID_RAW_OFFSETS')
    body_start=len(prefix)
    if start<body_start:
        return 'WITHIN_TITLE_METADATA'
    if not raw_text[body_start:start].strip():
        return 'NO_BODY_CONTEXT'
    return 'BODY_CONTEXT_PRESENT'


def eligible(raw_text,source_title,start,end):
    return classify_context(raw_text,source_title,start,end)=='BODY_CONTEXT_PRESENT'
