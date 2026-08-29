# RAG-WM canonical multi-budget 实验包 v1（2026-08-17）

## 1. 本包补哪一项老师意见

本包用于补齐真正的 **multi-budget attack–utility curve**。每个预算点都必须完整执行：

`冻结阈值 -> 全语料 targeted suffix deletion -> 重建 Contriever 索引 -> 独立 RAG-WM verifier -> retrieval utility`

它不是 Figure 3 的 TREC-COVID post-hoc 定位敏感性，也不会覆盖 NFCorpus sealed strict-v1 主结果。

## 2. 冻结协议

- 数据：NFCorpus，3,633 documents。
- detector LM：本地 Qwen2.5-7B。
- 评分：canonical token-aligned GainRatio；`L(T)` 与 `L(T|C)` 均排除第一个 target token；不加 BOS，不加额外 separator。
- 边界与攻击策略：与 aligned sensitivity 相同；context tail 1,200 chars，context 1,024 tokens，target 128 tokens，minimum target 25 chars，backward scan `max_steps=8`。
- clean calibration：按 source-document 分组，seed `20260714`，70/30 split；冻结分数哈希为 `e1c7dbc745a83b473883ccf9c0d6a3c43c1bb7a14fa9f0dda1551b08f92dd218`。
- 预算定义：clean-calibration target FPR = 1%、2.5%、5%、10%。横轴最终优先使用 realized deleted-character rate 或 modified-document rate。
- 5% 点复用已经冻结的 aligned sensitivity：threshold `-0.1832180580405571`、E2 WSN `21/30`、174 docs、37,586 chars。不得重新跑后替换。
- 新增点：1%、2.5%、10%。运行前不查看/推断其 WSN。
- verifier：同一 30 watermark units，gpt-4o-mini，temperature 0，verify seed 633。
- retrieval：Contriever + cosine；323 NFCorpus test queries；top-100；5,000 query-paired bootstrap，seed `20260609`。

## 3. 为什么不用旧 model-scale 脚本

归档中的 `model_scale_ablation_aligned_all_datasets.py` 给 target-only 输入添加 EOS，并计入首 target token；这与当前论文的 canonical 公式不一致。本包只复用已审计的 `e2_common_aligned.py` 最小修正实现。

## 4. AutoDL 安装位置

将整个目录上传到：

```text
/root/autodl-tmp/ragwm_storage/repo/ragwm_src/tools/RAGWM_multibudget_v1_20260817
```

然后：

```bash
cd /root/autodl-tmp/ragwm_storage/repo/ragwm_src/tools/RAGWM_multibudget_v1_20260817
chmod +x scripts/*.sh
```

## 5. 第一步：只做预检，不运行正式实验

```bash
export STORAGE=/root/autodl-tmp/ragwm_storage
export REPO=$STORAGE/repo/ragwm_src
export E2E=$STORAGE/output/sanitization_e2e_nf_strict_v1
export OUT_ROOT=$STORAGE/output/sanitization_multibudget_nf_aligned_v1

mkdir -p "$OUT_ROOT/audit" "$OUT_ROOT/thresholds"

python scripts/00_preflight_multibudget.py \
  --scores "$STORAGE/output/semantic_cliff/aligned_sensitivity/nfcorpus_aligned_scores.json" \
  --attack-input "$E2E/conditions/e2_gainratio_delete/corpus/watermarked_attack_input.jsonl" \
  --model-path "$STORAGE/local_models/Qwen2.5-7B" \
  --existing-fpr05 "$E2E/conditions/e2_gainratio_aligned_sensitivity/scores/fixed_threshold_fpr05_aligned.json" \
  --output "$OUT_ROOT/audit/preflight.json" \
  2>&1 | tee "$OUT_ROOT/audit/preflight.log"
```

只有出现 `MULTIBUDGET PREFLIGHT PASS` 才继续。预检应确认原服务器为 RTX 4090 D；若 GPU、输入哈希、446 scores、223+223 labels、3,633 attack rows 或 5% threshold 任一不符，停止。

## 6. 第二步：冻结四个阈值

```bash
python scripts/01_freeze_multibudget_thresholds.py \
  --scores "$STORAGE/output/semantic_cliff/aligned_sensitivity/nfcorpus_aligned_scores.json" \
  --output-dir "$OUT_ROOT/thresholds" \
  2>&1 | tee "$OUT_ROOT/audit/freeze_thresholds.log"
```

应复现：

| budget id | target clean FPR | frozen threshold | calibration empirical FPR | diagnostic injected TPR |
|---|---:|---:|---:|---:|
| fpr01 | 0.010 | -0.13007188598091768 | 0.0118343195 | 0.6098654709 |
| fpr025 | 0.025 | -0.1640843956343344 | 0.0295857988 | 0.6950672646 |
| fpr05 | 0.050 | -0.1832180580405571 | 0.0532544379 | 0.7713004484 |
| fpr10 | 0.100 | -0.20519773948364803 | 0.1005917160 | 0.8295964126 |

这些只是由冻结 boundary scores 计算的阈值和诊断 TPR，不是新的端到端结果。

## 7. 第三步：先跑 1% 单点

先只运行 1% 点，完成后上传/核对六个核心输出，再继续其余预算：

```bash
export PACKAGE_DIR=$REPO/tools/RAGWM_multibudget_v1_20260817
export OUT_ROOT=$STORAGE/output/sanitization_multibudget_nf_aligned_v1
bash scripts/02_run_one_budget.sh fpr01
```

脚本若发现目标 condition 或 vector store 已存在会停止，不会静默覆盖。`fpr05` 被硬性禁止运行，因为它必须复用已有 aligned result。

1% 点完成后先收集：

```text
$OUT_ROOT/thresholds/threshold_fpr01.json
$OUT_ROOT/conditions/qwen25_7b_fpr01/sanitization/gainratio_sanitization_summary.json
$OUT_ROOT/conditions/qwen25_7b_fpr01/metrics/vectorstore_build_summary.json
$OUT_ROOT/conditions/qwen25_7b_fpr01/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json
$OUT_ROOT/conditions/qwen25_7b_fpr01/metrics/e0_e1_budget_comparison.json
$OUT_ROOT/conditions/qwen25_7b_fpr01/metrics/SHA256SUMS.txt
```

这一步的新 WSN、文档数、字符数和显著性在机器输出产生前一律标记“缺失/待核对”。

## 8. 第四步：其余两个新预算

1% 点验收通过后，按顺序：

```bash
bash scripts/02_run_one_budget.sh fpr025
bash scripts/02_run_one_budget.sh fpr10
```

不要并行调用 verifier；每个 condition 固定后独立运行，且 verifier 输出不得用于改阈值。

## 9. 第五步：一次性冻结四个预算的 retrieval utility

```bash
bash scripts/03_evaluate_all_budget_utility.sh
```

它在同一次评估中比较 E0、1%、2.5%、已有 5%、10% 五个索引，保存相同 top-100 rankings，再计算四项 point estimates、paired differences 和 95% CI。

## 10. 第六步：汇总机器可读登记条目

```bash
python scripts/04_collect_multibudget_results.py \
  --out-root "$OUT_ROOT" \
  --existing-aligned-dir "$E2E/conditions/e2_gainratio_aligned_sensitivity" \
  --utility "$OUT_ROOT/metrics/multibudget_retrieval_utility_paired_bootstrap.json" \
  --output "$OUT_ROOT/metrics/multibudget_attack_utility_registry_entry.json" \
  2>&1 | tee "$OUT_ROOT/logs/04_collect_results.log"
```

只有这个命令通过后，才能根据真实数据绘制 attack–utility curve 和修改论文。即使某个点 WSN 较低，只要仍大于 2，就只能写 verification weakening。

## 11. 验收顺序

1. 输入和冻结 scores 哈希正确。
2. 每个 sanitized corpus 为 3,633 documents，`complete=true`。
3. 每个 vector store count 为 3,633。
4. 每个 verifier 正好 30 units，unknown=0。
5. 5% 点与已有 aligned condition完全复用，不重新计算或覆盖。
6. utility 的四项指标全部来自一次 frozen top-100 pass。
7. 为每个核心 JSON 和 verifier raw output保存 SHA-256。
8. 新结果先登记，再进入表格、图和正文。

## 12. 论文证据边界

- 这是 canonical token-aligned NFCorpus implementation-sensitivity extension，不替代 sealed strict-v1 `E0=26/30, E2=17/30` 主结果。
- 曲线上的 5% 点是 aligned `21/30`，不是 strict-v1 `17/30`。
- 不把 calibration target FPR 当成实际删除比例；绘图时同时报告 realized modified docs 和 deleted chars。
- retrieval CI 包含 0 只能写“未检测到统计显著 change”，不能写 equivalence。
- 所有缺失的 WSN、CI、删除预算和显著性禁止推算。

