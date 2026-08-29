#!/usr/bin/env bash
set -euo pipefail

STORAGE=${STORAGE:-/root/autodl-tmp/ragwm_storage}
REPO=${REPO:-$STORAGE/repo/ragwm_src}
PACKAGE_DIR=${PACKAGE_DIR:-$REPO/tools/RAGWM_multibudget_v1_20260817}
E2E=${E2E:-$STORAGE/output/sanitization_e2e_nf_strict_v1}
OUT_ROOT=${OUT_ROOT:-$STORAGE/output/sanitization_multibudget_nf_aligned_v1}
CONDITION="$OUT_ROOT/conditions/qwen25_7b_fpr01"
DB="$OUT_ROOT/vectorstores/qwen25_7b_fpr01"
GPT_FILE="$REPO/src/models/GPT.py"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

API_BASE=${RAGWM_API_BASE:-https://api.ephone.ai/v1}
VERIFIER_MODEL=${RAGWM_VERIFIER_MODEL:-gpt-4o-mini}

for required in \
  "$GPT_FILE" \
  "$PACKAGE_DIR/scripts/07_run_aligned_verification.sh" \
  "$PACKAGE_DIR/scripts/06_compare_e0_e1_aligned.py" \
  "$CONDITION/sanitization/gainratio_sanitized_corpus.jsonl"; do
  if [[ ! -e "$required" ]]; then
    echo "[ERROR] Missing required path: $required" >&2
    exit 2
  fi
done

if [[ ! -d "$DB" ]]; then
  echo "[ERROR] Missing existing fpr01 vector store: $DB" >&2
  exit 2
fi

echo "API base: $API_BASE"
echo "Verifier model: $VERIFIER_MODEL"
echo "The API key will not be printed or written to a project file."
read -rsp "Paste a newly rotated API key: " RAGWM_API_KEY
echo

if [[ -z "$RAGWM_API_KEY" ]]; then
  echo "[ERROR] Empty API key." >&2
  exit 2
fi

export RAGWM_API_KEY
export RAGWM_API_BASE="$API_BASE"
export RAGWM_VERIFIER_MODEL="$VERIFIER_MODEL"
trap 'unset RAGWM_API_KEY' EXIT

cd "$REPO"

python - <<'PY'
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["RAGWM_API_KEY"],
    base_url=os.environ["RAGWM_API_BASE"],
    timeout=45.0,
)
response = client.chat.completions.create(
    model=os.environ["RAGWM_VERIFIER_MODEL"],
    temperature=0,
    max_tokens=8,
    messages=[
        {"role": "system", "content": "Reply with exactly Yes or No."},
        {"role": "user", "content": "Is one plus one equal to two?"},
    ],
)
answer = response.choices[0].message.content
if not isinstance(answer, str) or answer.strip().lower().rstrip(".") not in {"yes", "no"}:
    raise RuntimeError(f"Unexpected API test output: {answer!r}")
print("EPHONE GPT-4O-MINI API TEST PASS:", answer.strip())
PY

if ! grep -q 'RAGWM_API_KEY' "$GPT_FILE"; then
  cp -a "$GPT_FILE" "$GPT_FILE.before_ephone_env_$STAMP"
  GPT_FILE="$GPT_FILE" python - <<'PY'
from pathlib import Path
import os

path = Path(os.environ["GPT_FILE"])
text = path.read_text(encoding="utf-8")
original = text

if "import os" not in text.split("class GPT", 1)[0]:
    text = text.replace("from openai import OpenAI\n", "from openai import OpenAI\nimport os\n", 1)

old_init = '''        api_keys = config["api_key_info"]["api_keys"]
        api_base = config["api_key_info"]["api_base"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.client = OpenAI(api_key=api_keys[api_pos], base_url=api_base[api_pos])
'''

new_init = '''        api_keys = config["api_key_info"]["api_keys"]
        api_base = config["api_key_info"]["api_base"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        selected_key = os.environ.get("RAGWM_API_KEY", api_keys[api_pos])
        selected_base = os.environ.get("RAGWM_API_BASE", api_base[api_pos])
        self.name = os.environ.get("RAGWM_VERIFIER_MODEL", self.name)
        if not selected_key:
            raise RuntimeError("Missing verifier API key")
        self.client = OpenAI(api_key=selected_key, base_url=selected_base)
'''

if old_init not in text:
    raise SystemExit("[ERROR] GPT.py constructor did not match the audited source; no patch applied")
text = text.replace(old_init, new_init, 1)

old_except = '''        except Exception as e:
            print(e)
            response = ""

        return response
'''

new_except = '''        except Exception as e:
            raise RuntimeError(
                f"Verifier API request failed: {type(e).__name__}: {e}"
            ) from e

        if not isinstance(response, str) or not response.strip():
            raise RuntimeError("Verifier API returned an empty response")
        return response.strip()
'''

if old_except not in text:
    raise SystemExit("[ERROR] GPT.py exception handler did not match the audited source; no patch applied")
text = text.replace(old_except, new_except, 1)

if text == original:
    raise SystemExit("[ERROR] No GPT.py changes were made")
path.write_text(text, encoding="utf-8")
print("GPT ENVIRONMENT HOTFIX APPLIED:", path)
PY
else
  echo "GPT environment hotfix already present; leaving source unchanged."
fi

if [[ -d "$CONDITION/basepath" ]]; then
  cp -a "$CONDITION/basepath" "$CONDITION/basepath_invalid_unknown30_$STAMP"
fi
if [[ -f "$CONDITION/logs/07_verify.log" ]]; then
  cp -a "$CONDITION/logs/07_verify.log" \
    "$CONDITION/logs/07_verify.invalid_unknown30_$STAMP.log"
fi
if [[ -f "$CONDITION/metrics/e0_e1_budget_comparison.json" ]]; then
  cp -a "$CONDITION/metrics/e0_e1_budget_comparison.json" \
    "$CONDITION/metrics/e0_e1_budget_comparison.invalid_unknown30_$STAMP.json"
fi

EA="$CONDITION" \
EA_BASE="$CONDITION/basepath" \
EA_DB="$DB" \
STORAGE="$STORAGE" \
REPO="$REPO" \
E2E="$E2E" \
  bash "$PACKAGE_DIR/scripts/07_run_aligned_verification.sh"

python "$PACKAGE_DIR/scripts/06_compare_e0_e1_aligned.py" \
  --e0 "$E2E/conditions/e0_watermarked_before/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --e1 "$E2E/conditions/e1_oracle_restore/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --aligned "$CONDITION/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --output "$CONDITION/metrics/e0_e1_budget_comparison.json" \
  2>&1 | tee "$CONDITION/logs/06_compare.rerun_$STAMP.log"

CONDITION="$CONDITION" python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

condition = Path(os.environ["CONDITION"])
verify_path = (
    condition
    / "basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json"
)
comparison_path = condition / "metrics/e0_e1_budget_comparison.json"

rows = json.loads(verify_path.read_text(encoding="utf-8"))
counts = Counter(row[2][0] for row in rows)
comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
summary = comparison["summary"]

print("verification_row_count:", len(rows))
print("verification_flag_counts:", dict(counts))
print(json.dumps(summary, ensure_ascii=False, indent=2))

if len(rows) != 30:
    raise SystemExit("[ERROR] Verifier output does not contain exactly 30 rows")
if counts.get(2, 0) != 0:
    raise SystemExit("[ERROR] Verifier still contains unknown outputs")
if not summary.get("run_valid"):
    raise SystemExit("[ERROR] Comparison summary reports run_valid=false")

print("FPR01 VERIFIER RERUN VALIDATION PASS")
PY

sha256sum \
  "$CONDITION/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  "$CONDITION/metrics/e0_e1_budget_comparison.json" \
  > "$CONDITION/metrics/FPR01_VERIFIER_RERUN_SHA256_$STAMP.txt"

echo "FPR01 VERIFIER-ONLY RERUN COMPLETE AND VALID"
echo "Condition preserved at: $CONDITION"
