import os
import time
import json
import inspect
import traceback
from pathlib import Path

try:
    from langchain_core.documents import Document
    from langchain_community.graphs.graph_document import GraphDocument
    from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
except Exception as e:
    print("[SAFE_PATCH] import failed:", repr(e))
    LLMGraphTransformer = None


def _is_length_or_parse_error(e: Exception) -> bool:
    msg = (repr(e) + "\n" + str(e)).lower()
    keys = [
        "lengthfinishreasonerror",
        "length limit was reached",
        "finish_reason",
        "could not parse response",
        "outputparserexception",
        "json",
        "parse",
    ]
    return any(k in msg for k in keys)


def _is_transient_api_error(e: Exception) -> bool:
    msg = (repr(e) + "\n" + str(e)).lower()
    keys = [
        "503",
        "502",
        "504",
        "timeout",
        "timed out",
        "temporarily",
        "rate limit",
        "ratelimit",
        "connection",
        "server error",
        "internalservererror",
    ]
    return any(k in msg for k in keys)


def _short_doc(document, max_chars: int):
    content = getattr(document, "page_content", "") or ""
    metadata = dict(getattr(document, "metadata", {}) or {})
    metadata["_safe_truncated"] = True
    metadata["_safe_original_chars"] = len(content)
    metadata["_safe_used_chars"] = max_chars
    return Document(page_content=content[:max_chars], metadata=metadata)


def _log_failed_doc(document, err, stage: str):
    fail_log = os.environ.get(
        "RAGWM_SAFE_FAIL_LOG",
        "/root/autodl-tmp/ragwm_storage/logs/ragwm_safe_failed_docs.jsonl",
    )
    Path(fail_log).parent.mkdir(parents=True, exist_ok=True)

    content = getattr(document, "page_content", "") or ""
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "error_type": type(err).__name__ if err else None,
        "error": repr(err)[:2000] if err else None,
        "content_chars": len(content),
        "metadata": getattr(document, "metadata", {}) or {},
        "preview": content[:500],
    }

    with open(fail_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if LLMGraphTransformer is not None and not getattr(LLMGraphTransformer, "_ragwm_safe_patched", False):
    _ORIG_PROCESS = LLMGraphTransformer.process_response
    _SIG = inspect.signature(_ORIG_PROCESS)
    _HAS_CONFIG = "config" in _SIG.parameters

    def _call_original(self, document, config=None):
        if _HAS_CONFIG:
            return _ORIG_PROCESS(self, document, config=config)
        return _ORIG_PROCESS(self, document)

    def _safe_process_response(self, document, config=None):
        retries = int(os.environ.get("RAGWM_SAFE_RETRIES", "3"))
        sleep_base = int(os.environ.get("RAGWM_SAFE_SLEEP", "20"))

        last_err = None

        # 1. 先按原文档正常跑。API 抖动则重试。
        for attempt in range(1, retries + 1):
            try:
                return _call_original(self, document, config=config)
            except Exception as e:
                last_err = e
                msg = repr(e)[:500]

                if _is_length_or_parse_error(e):
                    print(f"[SAFE_PATCH] length/parse error, will retry with shorter document. err={msg}", flush=True)
                    break

                if _is_transient_api_error(e) and attempt < retries:
                    wait = sleep_base * attempt
                    print(f"[SAFE_PATCH] transient API error, retry {attempt}/{retries}, sleep {wait}s. err={msg}", flush=True)
                    time.sleep(wait)
                    continue

                print(f"[SAFE_PATCH] non-transient error, will try fallback. err={msg}", flush=True)
                break

        # 2. 如果输出截断/解析失败，改用较短文档重试，避免 JSON 被截断。
        content = getattr(document, "page_content", "") or ""
        try_chars = os.environ.get("RAGWM_SAFE_TRY_CHARS", "10000,7000,5000,3000,1500")
        char_list = []
        for x in try_chars.split(","):
            x = x.strip()
            if x.isdigit():
                char_list.append(int(x))

        for max_chars in char_list:
            if len(content) <= max_chars:
                continue

            short_document = _short_doc(document, max_chars)
            print(f"[SAFE_PATCH] retry with truncated doc: original_chars={len(content)}, used_chars={max_chars}", flush=True)

            for attempt in range(1, retries + 1):
                try:
                    return _call_original(self, short_document, config=config)
                except Exception as e:
                    last_err = e
                    msg = repr(e)[:500]

                    if _is_transient_api_error(e) and attempt < retries:
                        wait = sleep_base * attempt
                        print(f"[SAFE_PATCH] fallback transient error, retry {attempt}/{retries}, sleep {wait}s. err={msg}", flush=True)
                        time.sleep(wait)
                        continue

                    if attempt >= retries:
                        print(f"[SAFE_PATCH] fallback failed at used_chars={max_chars}. err={msg}", flush=True)

        # 3. 最后仍失败：记录这条坏文档，返回空图，不让整个程序崩。
        print("[SAFE_PATCH] skip one bad document and continue. This document is logged.", flush=True)
        _log_failed_doc(document, last_err, stage="process_response_failed")

        try:
            return GraphDocument(nodes=[], relationships=[], source=document)
        except Exception:
            raise last_err

    LLMGraphTransformer.process_response = _safe_process_response
    LLMGraphTransformer._ragwm_safe_patched = True
    print("[SAFE_PATCH] LLMGraphTransformer safe patch enabled.", flush=True)
