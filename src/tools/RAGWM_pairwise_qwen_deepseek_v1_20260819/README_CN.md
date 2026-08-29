# RAG-WM Qwen--DeepSeek 配对统计包 v1（2026-08-19）

## 目的

本包只读取已经冻结的 Qwen2.5-7B 与 DeepSeek-LLM-7B-base 输出，不重新评分、不重新删除、不重新调用 verifier。它计算：

1. NFCorpus 上 Qwen minus DeepSeek 的配对 source-document cluster-bootstrap ROC-AUC/PR-AUC 差值与 95% CI；
2. TREC-COVID 上相同的配对差值与 95% CI；
3. 当前 provider 下 Qwen 5% FPR 与 DeepSeek 5% FPR 两个攻击条件的 30 单元配对 McNemar 描述。

边界差值方向固定为 `Qwen - DeepSeek`。端到端 WSN 差值方向固定为 `DeepSeek - Qwen`，正值仅表示 DeepSeek 条件保留了更多阳性。由于两个 detector 的实际删除预算不同，端到端模型比较只能作为描述性 sensitivity，不能解释为相同预算下的模型优劣。

## 安装

将整个目录上传到：

```text
/root/autodl-tmp/ragwm_storage/repo/ragwm_src/tools/RAGWM_pairwise_qwen_deepseek_v1_20260819
```

然后运行：

```bash
cd /root/autodl-tmp/ragwm_storage/repo/ragwm_src/tools/RAGWM_pairwise_qwen_deepseek_v1_20260819
chmod +x scripts/*.py

export STORAGE=/root/autodl-tmp/ragwm_storage
export MULTI_OUT=$STORAGE/output/sanitization_multibudget_nf_aligned_v1
export DEEP_ROOT=$STORAGE/output/cross_detector_lm_robustness_v1/deepseek_llm_7b_base
export PAIR_OUT=$STORAGE/output/cross_detector_lm_robustness_v1/pairwise_qwen_deepseek_v1

export QWEN_NF=$STORAGE/output/semantic_cliff/aligned_sensitivity/nfcorpus_aligned_scores.json
export QWEN_TREC=$STORAGE/output/semantic_cliff/boundary_trec_k5_v3.json
export DEEP_NF=$DEEP_ROOT/scores/deepseek_nfcorpus_aligned_scores.json
export DEEP_TREC=$DEEP_ROOT/scores/deepseek_trec_aligned_scores.json
export DEEP_METRICS=$DEEP_ROOT/metrics/deepseek_boundary_metrics.json

export QWEN_VERIFY=$MULTI_OUT/audits/current_provider_verifier_audit_v1_20260818/conditions/fpr05_current_provider/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json
export DEEP_VERIFY=$DEEP_ROOT/conditions/deepseek_llm_7b_base_fpr05/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json

mkdir -p "$PAIR_OUT"

python scripts/01_pairwise_qwen_deepseek.py \
  --qwen-nf "$QWEN_NF" \
  --deepseek-nf "$DEEP_NF" \
  --qwen-trec "$QWEN_TREC" \
  --deepseek-trec "$DEEP_TREC" \
  --deepseek-metrics "$DEEP_METRICS" \
  --qwen-verify "$QWEN_VERIFY" \
  --deepseek-verify "$DEEP_VERIFY" \
  --output "$PAIR_OUT/qwen_deepseek_pairwise_stats.json" \
  2>&1 | tee "$PAIR_OUT/qwen_deepseek_pairwise_stats.log"
```

正确完成标志：

```text
QWEN-DEEPSEEK PAIRED STATISTICS PASS
```

之后冻结：

```bash
sha256sum \
  "$PAIR_OUT/qwen_deepseek_pairwise_stats.json" \
  "$PAIR_OUT/qwen_deepseek_pairwise_stats.log" \
  | tee "$PAIR_OUT/SHA256SUMS.txt"
```

## 固定保护

- NFCorpus Qwen 输入必须匹配冻结 SHA-256 `e1c7dbc745a83b473883ccf9c0d6a3c43c1bb7a14fa9f0dda1551b08f92dd218`。
- TREC-COVID Qwen 输入必须匹配冻结 SHA-256 `d526aa72f48662521b67339f58849c1ea6c26de4042eb8b3dd842403f8ed293f`。
- NFCorpus DeepSeek 输入必须匹配冻结 SHA-256 `fd29d143ac4c08f23d36b8fac5d4bad7e6d85172f68bb7eb4c1d19275552cc86`。
- DeepSeek TREC 文件由已经冻结并校验通过的 `deepseek_boundary_metrics.json` 反向核对。
- 输出已存在时脚本拒绝覆盖。
- 独立模型 CI 的重叠情况不能代替本包的配对差值 CI。
- 端到端 Qwen/DeepSeek 比较不是预算匹配比较。

