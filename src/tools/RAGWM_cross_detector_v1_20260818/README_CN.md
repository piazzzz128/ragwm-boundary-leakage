# RAG-WM 跨 detector LM 鲁棒性实验包 v1（DeepSeek，2026-08-18）

## 1. 实验目的

本包补齐老师提出的“跨 detector LM 鲁棒性实验”。新增本地 detector LM：

- `DeepSeek-LLM-7B-base`
- 模型来源：ModelScope snapshot
- 本地目录：`DeepSeek-LLM-7B-base-ms`
- 远端精确 revision：下载时未记录，因此正式标记为“缺失/未记录”；本包改用七个核心文件的 SHA-256 固定模型内容。

实验报告：

1. NFCorpus boundary ROC-AUC、PR-AUC；
2. TREC-COVID boundary ROC-AUC、PR-AUC；
3. DeepSeek 自身 clean-calibration 5% FPR 阈值；
4. NFCorpus 全语料 targeted suffix deletion 后的最终 WSN；
5. 实际修改文档数和删除字符数。

它是 `cross_detector_lm_robustness_aligned_v1` 扩展，不替代：

- sealed strict-v1 主结果 `E0=26/30, E1=5/30, E2=17/30`；
- Qwen2.5-7B aligned 5% sensitivity `21/30`；
- Qwen2.5-7B aligned multi-budget 曲线。

## 2. 固定评分口径

- `aligned_minimal_v1`；
- `L(T)` 与 `L(T|C)` 均排除第一个 target token；
- 不添加 BOS；
- 不添加 separator；
- context 最多 1,024 tokens；
- target 最多 128 tokens；
- 模型以 BF16 加载，交叉熵在 FP32 logits 上计算；
- 分数 `score = -GainRatio`，越大越可疑。

归档旧脚本 `model_scale_ablation_aligned_all_datasets.py` 会给 target-only 前置 EOS 并计入首 token，不符合当前冻结口径，禁止使用。

## 3. verifier 基线

旧 API 服务已经终止，不能把旧 provider 的 `E0=26/E1=5` 与当前 provider 新输出做配对统计。因此本扩展固定使用 2026-08-18 current-provider audit：

- E0 current provider = `25/30`；
- E1 current provider Oracle = `4/30`；
- verifier nominal model = `gpt-4o-mini`；
- raw outputs 由预检哈希锁定。

这不会修改 sealed strict-v1 的 26/5/17 论文主结果。

## 4. 安装

将本目录上传到：

```text
/root/autodl-tmp/ragwm_storage/repo/ragwm_src/tools/RAGWM_cross_detector_v1_20260818
```

然后：

```bash
cd /root/autodl-tmp/ragwm_storage/repo/ragwm_src/tools/RAGWM_cross_detector_v1_20260818
chmod +x scripts/*.sh
```

## 5. 第一步：预检

```bash
export STORAGE=/root/autodl-tmp/ragwm_storage
export REPO=$STORAGE/repo/ragwm_src
export E2E=$STORAGE/output/sanitization_e2e_nf_strict_v1
export MULTI_OUT=$STORAGE/output/sanitization_multibudget_nf_aligned_v1
export OUT_ROOT=$STORAGE/output/cross_detector_lm_robustness_v1/deepseek_llm_7b_base
export MODEL_PATH=$STORAGE/local_models/DeepSeek-LLM-7B-base-ms

AUDIT_ROOT=$MULTI_OUT/audits/current_provider_verifier_audit_v1_20260818
mkdir -p "$OUT_ROOT/audit" "$OUT_ROOT/scores" "$OUT_ROOT/metrics" "$OUT_ROOT/thresholds" "$OUT_ROOT/logs"

python scripts/00_preflight_deepseek.py \
  --model-dir "$MODEL_PATH" \
  --nfcorpus-boundaries "$STORAGE/output/semantic_cliff/aligned_sensitivity/nfcorpus_aligned_scores.json" \
  --trec-boundaries "$STORAGE/output/semantic_cliff/boundary_trec_k5_v3.json" \
  --attack-input "$E2E/conditions/e2_gainratio_delete/corpus/watermarked_attack_input.jsonl" \
  --current-provider-e0 "$AUDIT_ROOT/conditions/e0_current_provider/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --current-provider-e1 "$AUDIT_ROOT/conditions/e1_current_provider/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --output "$OUT_ROOT/audit/preflight.json" \
  2>&1 | tee "$OUT_ROOT/audit/preflight.log"
```

只有出现 `DEEPSEEK CROSS-DETECTOR PREFLIGHT PASS` 才继续。

## 6. 第二步：边界评分

先跑 NFCorpus：

```bash
nohup python scripts/01_score_boundaries_aligned.py \
  --input "$STORAGE/output/semantic_cliff/aligned_sensitivity/nfcorpus_aligned_scores.json" \
  --output "$OUT_ROOT/scores/deepseek_nfcorpus_aligned_scores.json" \
  --records-dir "$OUT_ROOT/scores/nfcorpus_records" \
  --manifest-output "$OUT_ROOT/scores/deepseek_nfcorpus_score_manifest.json" \
  --preflight "$OUT_ROOT/audit/preflight.json" \
  --model-path "$MODEL_PATH" \
  --dataset nfcorpus \
  > "$OUT_ROOT/logs/score_nfcorpus.log" 2>&1 &
```

查看进度：

```bash
tail -f "$OUT_ROOT/logs/score_nfcorpus.log"
```

完成标志：

```text
DEEPSEEK NFCORPUS ALIGNED SCORING PASS
```

然后跑 TREC-COVID：

```bash
nohup python scripts/01_score_boundaries_aligned.py \
  --input "$STORAGE/output/semantic_cliff/boundary_trec_k5_v3.json" \
  --output "$OUT_ROOT/scores/deepseek_trec_aligned_scores.json" \
  --records-dir "$OUT_ROOT/scores/trec_records" \
  --manifest-output "$OUT_ROOT/scores/deepseek_trec_score_manifest.json" \
  --preflight "$OUT_ROOT/audit/preflight.json" \
  --model-path "$MODEL_PATH" \
  --dataset trec-covid \
  > "$OUT_ROOT/logs/score_trec.log" 2>&1 &
```

完成标志：

```text
DEEPSEEK TREC-COVID ALIGNED SCORING PASS
```

不要同时跑两个 7B scorer。

## 7. 第三步：计算指标并冻结 DeepSeek 5% 阈值

```bash
python scripts/02_metrics_and_freeze_fpr05.py \
  --nfcorpus-scores "$OUT_ROOT/scores/deepseek_nfcorpus_aligned_scores.json" \
  --trec-scores "$OUT_ROOT/scores/deepseek_trec_aligned_scores.json" \
  --metrics-output "$OUT_ROOT/metrics/deepseek_boundary_metrics.json" \
  --threshold-output "$OUT_ROOT/thresholds/deepseek_fpr05.json" \
  2>&1 | tee "$OUT_ROOT/logs/metrics_and_threshold.log"
```

完成标志：

```text
DEEPSEEK METRICS AND MODEL-SPECIFIC FPR05 FREEZE PASS
```

阈值必须在查看 WSN 前冻结。DeepSeek 阈值不能复用 Qwen 的 `-0.183218...`。

## 8. 第四步：完整端到端运行

在检查 ROC-AUC、PR-AUC、阈值和输出哈希后运行：

```bash
source /root/.config/ragwm/verifier.env
bash scripts/04_run_deepseek_fpr05_e2e.sh
```

脚本完整执行：

```text
DeepSeek threshold
-> 3,633-document targeted suffix deletion
-> Contriever re-index
-> current-provider gpt-4o-mini verifier
-> paired E0/E1 comparison
-> SHA-256 manifest
```

完成标志：

```text
DEEPSEEK CROSS-DETECTOR FORMAL E2E PASS
```

## 9. 结果边界

- 新结果未生成前，DeepSeek ROC-AUC、PR-AUC、阈值、删除预算和 WSN 均为“缺失/待核对”。
- 不能从加载 loss `7.0721` 推算任何定位或攻击结果。
- 跨 detector 的单模型 CI 不能直接证明模型间差异显著。
- 如果最终 WSN 仍高于 ownership threshold 2，只能写 verification weakening。
- TREC-COVID 在本扩展中仍只做 boundary localization，不是第二个端到端攻击。

