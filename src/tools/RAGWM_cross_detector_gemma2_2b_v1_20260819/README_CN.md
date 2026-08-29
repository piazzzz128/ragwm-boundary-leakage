# RAG-WM 跨 Detector LM 鲁棒性实验包（Gemma 2 2B，2026-08-19）

## 实验角色

本包运行第三个本地 detector LM：`Gemma-2-2B-base`。它属于
`cross_detector_lm_robustness_aligned_v1` 扩展，不替代 sealed strict-v1
主结果 `E0=26/30, E1=5/30, E2=17/30`，也不覆盖 Qwen 或 DeepSeek 输出。

## 冻结口径

- alignment：`exclude_first_target_token_in_both_conditions`
- no BOS；no separator
- context 最多 1,024 tokens；target 最多 128 tokens
- BF16 模型加载；FP32 logits 交叉熵
- `score = -GainRatio`，越大越可疑
- 5% FPR 阈值只从 NFCorpus clean calibration split 冻结
- 端到端比较使用 2026-08-18 current-provider audit：E0=25/30、E1=4/30

## 固定位置

```text
工具目录：/root/autodl-tmp/ragwm_storage/repo/ragwm_src/tools/RAGWM_cross_detector_gemma2_2b_v1_20260819
模型目录：/root/autodl-tmp/ragwm_storage/local_models/Gemma-2-2B-base-ms
输出目录：/root/autodl-tmp/ragwm_storage/output/cross_detector_lm_robustness_v1/gemma2_2b_base
```

## 1. Preflight

```bash
cd /root/autodl-tmp/ragwm_storage/repo/ragwm_src/tools/RAGWM_cross_detector_gemma2_2b_v1_20260819
chmod +x scripts/*.sh

export STORAGE=/root/autodl-tmp/ragwm_storage
export REPO=$STORAGE/repo/ragwm_src
export E2E=$STORAGE/output/sanitization_e2e_nf_strict_v1
export MULTI_OUT=$STORAGE/output/sanitization_multibudget_nf_aligned_v1
export OUT_ROOT=$STORAGE/output/cross_detector_lm_robustness_v1/gemma2_2b_base
export MODEL_PATH=$STORAGE/local_models/Gemma-2-2B-base-ms

AUDIT_ROOT=$MULTI_OUT/audits/current_provider_verifier_audit_v1_20260818
mkdir -p "$OUT_ROOT/audit" "$OUT_ROOT/scores" "$OUT_ROOT/metrics" "$OUT_ROOT/thresholds" "$OUT_ROOT/logs"

python scripts/00_preflight_gemma.py \
  --model-dir "$MODEL_PATH" \
  --nfcorpus-boundaries "$STORAGE/output/semantic_cliff/aligned_sensitivity/nfcorpus_aligned_scores.json" \
  --trec-boundaries "$STORAGE/output/semantic_cliff/boundary_trec_k5_v3.json" \
  --attack-input "$E2E/conditions/e2_gainratio_delete/corpus/watermarked_attack_input.jsonl" \
  --current-provider-e0 "$AUDIT_ROOT/conditions/e0_current_provider/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --current-provider-e1 "$AUDIT_ROOT/conditions/e1_current_provider/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --output "$OUT_ROOT/audit/preflight.json" \
  2>&1 | tee "$OUT_ROOT/audit/preflight.log"
```

只有出现 `GEMMA CROSS-DETECTOR PREFLIGHT PASS` 才继续。

## 2. NFCorpus 446 条边界评分

```bash
nohup python scripts/01_score_boundaries_aligned.py \
  --input "$STORAGE/output/semantic_cliff/aligned_sensitivity/nfcorpus_aligned_scores.json" \
  --output "$OUT_ROOT/scores/gemma_nfcorpus_aligned_scores.json" \
  --records-dir "$OUT_ROOT/scores/nfcorpus_records" \
  --manifest-output "$OUT_ROOT/scores/gemma_nfcorpus_score_manifest.json" \
  --preflight "$OUT_ROOT/audit/preflight.json" \
  --model-path "$MODEL_PATH" \
  --dataset nfcorpus \
  > "$OUT_ROOT/logs/score_nfcorpus.log" 2>&1 &

tail -f "$OUT_ROOT/logs/score_nfcorpus.log"
```

完成标志：`GEMMA NFCORPUS ALIGNED SCORING PASS`。

## 3. TREC-COVID 458 条边界评分

NFCorpus 完成后再运行，两个 scorer 不得同时占用 GPU。

```bash
nohup python scripts/01_score_boundaries_aligned.py \
  --input "$STORAGE/output/semantic_cliff/boundary_trec_k5_v3.json" \
  --output "$OUT_ROOT/scores/gemma_trec_aligned_scores.json" \
  --records-dir "$OUT_ROOT/scores/trec_records" \
  --manifest-output "$OUT_ROOT/scores/gemma_trec_score_manifest.json" \
  --preflight "$OUT_ROOT/audit/preflight.json" \
  --model-path "$MODEL_PATH" \
  --dataset trec-covid \
  > "$OUT_ROOT/logs/score_trec.log" 2>&1 &

tail -f "$OUT_ROOT/logs/score_trec.log"
```

完成标志：`GEMMA TREC-COVID ALIGNED SCORING PASS`。

## 4. 统计指标与 Gemma 专属 5% FPR 阈值

```bash
python scripts/02_metrics_and_freeze_fpr05.py \
  --nfcorpus-scores "$OUT_ROOT/scores/gemma_nfcorpus_aligned_scores.json" \
  --trec-scores "$OUT_ROOT/scores/gemma_trec_aligned_scores.json" \
  --metrics-output "$OUT_ROOT/metrics/gemma_boundary_metrics.json" \
  --threshold-output "$OUT_ROOT/thresholds/gemma_fpr05.json" \
  2>&1 | tee "$OUT_ROOT/logs/metrics_and_threshold.log"
```

完成标志：`GEMMA METRICS AND MODEL-SPECIFIC FPR05 FREEZE PASS`。

## 5. 正式端到端运行

仅在检查边界指标、阈值、行数和哈希后运行：

```bash
source /root/.config/ragwm/verifier.env
bash scripts/04_run_gemma_fpr05_e2e.sh
```

完成标志：`GEMMA CROSS-DETECTOR FORMAL E2E PASS`。

## 结果边界

- TREC-COVID 仍是 boundary-only generalization，不是第二套端到端攻击。
- 单模型 bootstrap CI 不能证明模型之间显著不同；模型间比较需要 paired cluster-bootstrap。
- 如果最终 WSN 高于 ownership threshold 2，只能写 verification weakening。
- Gemma 的阈值、预算和 WSN 在机器可读输出生成前均为缺失/待核对。
