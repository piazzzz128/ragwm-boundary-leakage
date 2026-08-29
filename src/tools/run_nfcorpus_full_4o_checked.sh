#!/usr/bin/env bash
set -e

cd /root/autodl-tmp/ragwm_storage/repo/ragwm_src
source /root/autodl-tmp/ragwm_storage/.ragwm_api_env
export PYTHONPATH=/root/autodl-tmp/ragwm_storage/repo/ragwm_src:$PYTHONPATH

MODEL="gpt-4o-mini"
BASE="https://api.whatai.cc/v1"
OUT="/root/autodl-tmp/ragwm_storage/output/wm_prepare/nfcorpus"
LOGDIR="/root/autodl-tmp/ragwm_storage/logs"
RUN_ID=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOGDIR"

echo "===== 1. Patch model config to gpt-4o-mini ====="

python - <<'PY'
import json, pathlib, time, os

MODEL = "gpt-4o-mini"
BASE = "https://api.whatai.cc/v1"

cfg_path = pathlib.Path("model_configs/gpt3.5_config.json")
backup = cfg_path.with_suffix(f".json.bak_force_4o_{time.strftime('%Y%m%d_%H%M%S')}")
backup.write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

cfg = json.load(open(cfg_path, "r", encoding="utf-8"))

cfg["model"] = MODEL
cfg["model_name"] = MODEL
cfg["api_base"] = BASE
cfg["base_url"] = BASE
cfg["openai_api_base"] = BASE

cfg.setdefault("params", {})
cfg["params"]["model"] = MODEL
cfg["params"]["model_name"] = MODEL
cfg["params"]["temperature"] = 0.1
cfg["params"]["max_tokens"] = 768
cfg["params"]["seed"] = 2026

json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("patched config:", cfg_path)
print("backup:", backup)
print("model:", MODEL)
PY

echo "===== 2. Patch env model ====="

grep -v '^export RAGWM_MODEL=' /root/autodl-tmp/ragwm_storage/.ragwm_api_env | \
grep -v '^export OPENAI_MODEL=' | \
grep -v '^export MODEL_NAME=' > /root/autodl-tmp/ragwm_storage/.ragwm_api_env.tmp || true

cat >> /root/autodl-tmp/ragwm_storage/.ragwm_api_env.tmp <<'EOF'
export RAGWM_MODEL="gpt-4o-mini"
export OPENAI_MODEL="gpt-4o-mini"
export MODEL_NAME="gpt-4o-mini"
EOF

mv /root/autodl-tmp/ragwm_storage/.ragwm_api_env.tmp /root/autodl-tmp/ragwm_storage/.ragwm_api_env
source /root/autodl-tmp/ragwm_storage/.ragwm_api_env

echo "RAGWM_MODEL=$RAGWM_MODEL"
echo "OPENAI_MODEL=$OPENAI_MODEL"
echo "MODEL_NAME=$MODEL_NAME"

echo "===== 3. Check safe patch / config ====="

python - <<'PY'
import json, pathlib

cfg=json.load(open("model_configs/gpt3.5_config.json","r",encoding="utf-8"))
print("config model:", cfg.get("model"))
print("config model_name:", cfg.get("model_name"))
print("params model:", cfg.get("params",{}).get("model"))
print("base:", cfg.get("base_url") or cfg.get("api_base") or cfg.get("openai_api_base"))

p=pathlib.Path("entity_generate/safe_llm_patch.py")
if p.exists():
    txt=p.read_text(encoding="utf-8", errors="replace")
    print("safe_llm_patch exists:", True)
    print("contains gpt-4o-mini:", "gpt-4o-mini" in txt)
    print("contains qwen-turbo:", "qwen-turbo" in txt)
else:
    print("safe_llm_patch exists:", False)
PY

echo "===== 4. API smoke test ====="

python - <<'PY'
import os, json, urllib.request, urllib.error, time

BASE=os.environ.get("OPENAI_BASE_URL","https://api.whatai.cc/v1").rstrip("/")
KEY=os.environ.get("OPENAI_API_KEY","").strip()
MODEL="gpt-4o-mini"

payload={
    "model": MODEL,
    "messages": [{"role": "user", "content": "Return strict JSON only: {\"ok\": true}"}],
    "temperature": 0,
    "max_tokens": 32
}

for i in range(1, 4):
    req=urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            body=r.read().decode("utf-8", errors="replace")
            print("API_STATUS", r.status)
            print(body[:500])
            raise SystemExit(0)
    except urllib.error.HTTPError as e:
        body=e.read().decode("utf-8", errors="replace")
        print("API_STATUS", e.code)
        print(body[:500])
        if e.code in (429, 500, 502, 503, 504):
            time.sleep(30*i)
            continue
        raise
    except Exception as e:
        print("API_ERR", repr(e))
        time.sleep(30*i)

raise SystemExit("API smoke test failed after retries; do not start full run.")
PY

echo "===== 5. Backup old output if exists ====="

if [ -d "$OUT" ]; then
  tar -czf /root/autodl-tmp/ragwm_storage/backup_nfcorpus_before_full_4o_${RUN_ID}.tar.gz "$OUT" || true
fi

echo "===== 6. Reset and initialize wm_prepare output ====="

rm -rf "$OUT"
mkdir -p "$OUT"

python - <<'PY'
import json, os

base="/root/autodl-tmp/ragwm_storage/output/wm_prepare/nfcorpus"

init = {
    "checkpoint.json": "",
    "entities_dict_llm.json": {},
    "entity_type_llm.json": {},
    "relation_type_llm.json": {},
    "relation_list_llm.json": [],
}

for name, obj in init.items():
    p=os.path.join(base, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print("initialized:", p)
PY

echo "===== 7. Start full NFCorpus wm_prepare ====="
echo "LOG: $LOGDIR/nfcorpus_strict_wm_prepare_4o_full_${RUN_ID}.log"

python entity_generate/generate_entity_llm_check.py \
  --eval_dataset nfcorpus \
  --basepath /root/autodl-tmp/ragwm_storage/output/wm_prepare \
  --dataset_prob 1 \
  2>&1 | tee "$LOGDIR/nfcorpus_strict_wm_prepare_4o_full_${RUN_ID}.log"

echo "===== 8. Final output check ====="

python - <<'PY'
import json, os

base="/root/autodl-tmp/ragwm_storage/output/wm_prepare/nfcorpus"

for f in [
    "checkpoint.json",
    "entities_dict_llm.json",
    "entity_type_llm.json",
    "relation_type_llm.json",
    "relation_list_llm.json",
]:
    p=os.path.join(base,f)
    print("\n", f, "exists=", os.path.exists(p))
    if os.path.exists(p):
        x=json.load(open(p,"r",encoding="utf-8"))
        print("type=",type(x).__name__,"len=",len(x) if hasattr(x,"__len__") else "NA")
        if isinstance(x,dict) and x:
            print("sample keys=",list(x.keys())[:5])
        if isinstance(x,list) and x:
            print("sample item=",x[0])
PY

echo "===== DONE ====="
