import json
import time
import pathlib
import urllib.request
import urllib.error

# 中转站配置
BASE = "https://api.whatai.cc/v1"
MODEL = "gpt-4o-mini"
API_KEY = "<REDACTED_SECRET>"

ROOT = pathlib.Path("/root/autodl-tmp/ragwm_storage/repo/ragwm_src")
LOG_DIR = pathlib.Path("/root/autodl-tmp/ragwm_storage/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def call_api(method, path, payload=None):
    url = BASE.rstrip("/") + path
    data = None
    headers = {
        "Authorization": f"Bearer {API_KEY.strip()}",
        "Content-Type": "application/json",
    }
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            body = r.read().decode("utf-8", errors="replace")
            return r.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, body
    except Exception as e:
        return -1, repr(e)

print("API key check:")
print("len =", len(API_KEY.strip()))
print("prefix =", API_KEY.strip()[:8])
print("suffix =", API_KEY.strip()[-6:])
print("has_space =", any(c.isspace() for c in API_KEY))

models_status, models_body = call_api("GET", "/models")

chat_payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Reply only OK."}],
    "temperature": 0,
    "max_tokens": 8
}
chat_status, chat_body = call_api("POST", "/chat/completions", chat_payload)

result = {
    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    "base": BASE,
    "model": MODEL,
    "key_len": len(API_KEY.strip()),
    "key_prefix": API_KEY.strip()[:8],
    "key_suffix": API_KEY.strip()[-6:],
    "models_status": models_status,
    "models_body_prefix": models_body[:800],
    "chat_status": chat_status,
    "chat_body_prefix": chat_body[:1200],
}

out = LOG_DIR / f"api_preflight_realkey_{time.strftime('%Y%m%d_%H%M%S')}.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps(result, ensure_ascii=False, indent=2))

if chat_status != 200:
    raise SystemExit(
        f"\nAPI_FAIL: /chat/completions status={chat_status}. "
        f"Do not run wm_prepare yet. Log: {out}"
    )

# patch gpt config
cfg_path = ROOT / "model_configs" / "gpt3.5_config.json"
if cfg_path.exists():
    backup = cfg_path.with_suffix(f".json.bak_{time.strftime('%Y%m%d_%H%M%S')}")
    backup.write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
else:
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {}

cfg["model"] = MODEL
cfg["model_name"] = MODEL
cfg["api_key"] = API_KEY.strip()
cfg["base_url"] = BASE
cfg["api_base"] = BASE
cfg["openai_api_base"] = BASE
cfg["openai_api_key"] = API_KEY.strip()
cfg["gpus"] = cfg.get("gpus", [])
cfg["params"] = cfg.get("params", {})
cfg["params"].update({
    "temperature": 0.1,
    "max_tokens": 512,
    "seed": 2026
})

cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

env_file = pathlib.Path("/root/autodl-tmp/ragwm_storage/.ragwm_api_env")
env_file.write_text(
    f"export OPENAI_API_KEY='{API_KEY.strip()}'\n"
    f"export OPENAI_API_BASE='{BASE}'\n"
    f"export OPENAI_BASE_URL='{BASE}'\n"
    f"export RAGWM_API_BASE='{BASE}'\n"
    f"export RAGWM_MODEL='{MODEL}'\n",
    encoding="utf-8"
)

print("\nAPI_PASS: /chat/completions=200")
print(f"patched: {cfg_path}")
print(f"env file: {env_file}")
