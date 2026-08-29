# RAG-WM Qwen2.5-14B canonical aligned 补充实验包 v1

日期：2026-08-20

## 1. 本包做什么

本包把 `Qwen2.5-14B` base 加入现有 detector LM 扩展，形成：

- Gemma-2-2B-base；
- Qwen2.5-7B；
- DeepSeek-LLM-7B-base；
- Qwen2.5-14B。

14B主要补充同家族7B→14B的尺度敏感性证据。它不是新的主实验，不替代：

- sealed strict-v1：E0/E1/E2 = 26/5/17；
- Qwen2.5-7B aligned 四预算：WSN = 21、20、21、17；
- 已冻结的DeepSeek和Gemma结果。

所有14B结果在生成前均为缺失。本包不会预填ROC-AUC、PR-AUC、CI、阈值、
编辑预算、WSN或显著性。

## 2. 正式模型与硬件

- 官方仓库：`Qwen/Qwen2.5-14B`；
- checkpoint：base/pretrained，不使用Instruct；
- 精度：unquantized BF16；
- 加载：单GPU；
- 最低预检：40 GiB；
- 建议：48 GiB A6000/A40/L40S或更高显存；
- 4090 24GB不能按本正式BF16协议加载完整14B权重；
- 4-bit/8-bit结果不得与现有BF16 7B直接解释为模型规模差异。

模型下载时先把远端revision解析为不可变commit，再对本地权重和tokenizer逐文件
计算SHA-256。

## 3. 固定方法

- canonical aligned；
- 双方均排除第一个target token；
- no BOS / no separator；
- context ≤ 1,024 tokens；
- target ≤ 128 tokens；
- `score = -(L(T)-L(T|C))/L(T)`；
- score越大越可疑；
- threshold只用NFCorpus clean calibration冻结，目标FPR=5%；
- threshold冻结后才能进入3,633文档完整攻击和验证。

更完整的预注册约束见 `EXPERIMENT_SPEC_CN.md`。

## 4. 上传位置

将整个目录上传到：

```text
/root/autodl-tmp/ragwm_storage/repo/ragwm_src/tools/RAGWM_qwen2_5_14b_cross_detector_v1_20260820
```

进入目录：

```bash
export STORAGE=/root/autodl-tmp/ragwm_storage
export REPO=$STORAGE/repo/ragwm_src
export PACKAGE_DIR=$REPO/tools/RAGWM_qwen2_5_14b_cross_detector_v1_20260820
export OUT_ROOT=$STORAGE/output/cross_detector_lm_robustness_v1/qwen2_5_14b_base
export MODEL_PATH=$STORAGE/local_models/Qwen2.5-14B
export E2E=$STORAGE/output/sanitization_e2e_nf_strict_v1
export MULTI_OUT=$STORAGE/output/sanitization_multibudget_nf_aligned_v1
export MULTI_PACKAGE=$REPO/tools/RAGWM_multibudget_v1_20260817

cd "$PACKAGE_DIR"
chmod +x scripts/*.py scripts/*.sh tests/*.py
mkdir -p "$OUT_ROOT"/{audit,scores,metrics,thresholds,logs,conditions,vectorstores}
```

先确认：

```bash
nvidia-smi
python - <<'PY'
import torch, transformers
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda:", torch.cuda.is_available())
print("bf16:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu:", p.name)
    print("memory_GiB:", p.total_memory / 1024**3)
PY
```

## 5. 下载并冻结模型

中国网络环境可使用已验证可访问的镜像端点：

```bash
export HF_ENDPOINT=https://hf-mirror.com

python scripts/00_download_qwen14b.py \
  --repo-id Qwen/Qwen2.5-14B \
  --revision main \
  --local-dir "$MODEL_PATH" \
  --manifest-output "$OUT_ROOT/audit/qwen14b_download_provenance.json" \
  2>&1 | tee "$OUT_ROOT/logs/00_download_qwen14b.log"
```

下载脚本会记录实际commit；后续不能重新把 `main` 当成已冻结版本。

逐文件冻结权重：

```bash
python scripts/01_freeze_model_snapshot.py \
  --model-dir "$MODEL_PATH" \
  --source-manifest "$OUT_ROOT/audit/qwen14b_download_provenance.json" \
  --output "$OUT_ROOT/audit/qwen14b_model_freeze.json" \
  2>&1 | tee "$OUT_ROOT/logs/01_freeze_model.log"
```

必须看到：

```text
QWEN2.5-14B MODEL FREEZE PASS
```

如果模型已经存在但没有下载manifest，可省略 `--source-manifest`。这时revision会正式
记录为 `missing_not_recorded`，但本地内容仍由SHA-256冻结；论文和登记表不能编造revision。

## 6. 实验输入preflight

```bash
REFERENCE_ROOT=$MULTI_OUT/audits/current_provider_verifier_audit_v1_20260818

python scripts/02_preflight_qwen14b.py \
  --model-freeze "$OUT_ROOT/audit/qwen14b_model_freeze.json" \
  --nfcorpus-boundaries "$STORAGE/output/semantic_cliff/aligned_sensitivity/nfcorpus_aligned_scores.json" \
  --trec-boundaries "$STORAGE/output/semantic_cliff/boundary_trec_k5_v3.json" \
  --attack-input "$E2E/conditions/e2_gainratio_delete/corpus/watermarked_attack_input.jsonl" \
  --aligned-e0 "$REFERENCE_ROOT/conditions/e0_current_provider/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --aligned-e1 "$REFERENCE_ROOT/conditions/e1_current_provider/basepath/wm_generate/nfcorpus/gpt4o_mini_e2e/10/wmuint_verify.json" \
  --output "$OUT_ROOT/audit/preflight.json" \
  2>&1 | tee "$OUT_ROOT/logs/02_preflight.log"
```

必须看到：

```text
QWEN2.5-14B EXPERIMENT PREFLIGHT PASS
```

本步强制验证：输入SHA-256、446/458样本、标签平衡、3,633文档和aligned-extension
E0/E1=25/4。任何一项失败都不要继续。

## 7. 先做小样本冒烟测试

```bash
python tests/test_protocol_contract.py

python scripts/03_score_boundaries_aligned.py \
  --input "$STORAGE/output/semantic_cliff/aligned_sensitivity/nfcorpus_aligned_scores.json" \
  --output "$OUT_ROOT/scores/qwen14b_nf_smoke10.json" \
  --records-dir "$OUT_ROOT/scores/nf_smoke10_records" \
  --manifest-output "$OUT_ROOT/scores/qwen14b_nf_smoke10_manifest.json" \
  --preflight "$OUT_ROOT/audit/preflight.json" \
  --model-path "$MODEL_PATH" \
  --dataset nfcorpus \
  --limit 10 \
  2>&1 | tee "$OUT_ROOT/logs/03_nf_smoke10.log"
```

冒烟测试输出不是正式结果，不进入论文。确认无OOM、NaN和模型路径错误后再跑全量。

## 8. 正式boundary scoring

必须顺序运行，不能同时加载两个14B scorer。

### NFCorpus

```bash
nohup python scripts/03_score_boundaries_aligned.py \
  --input "$STORAGE/output/semantic_cliff/aligned_sensitivity/nfcorpus_aligned_scores.json" \
  --output "$OUT_ROOT/scores/qwen14b_nfcorpus_aligned_scores.json" \
  --records-dir "$OUT_ROOT/scores/nfcorpus_records" \
  --manifest-output "$OUT_ROOT/scores/qwen14b_nfcorpus_score_manifest.json" \
  --preflight "$OUT_ROOT/audit/preflight.json" \
  --model-path "$MODEL_PATH" \
  --dataset nfcorpus \
  > "$OUT_ROOT/logs/03_score_nfcorpus.log" 2>&1 &
```

查看：

```bash
tail -f "$OUT_ROOT/logs/03_score_nfcorpus.log"
```

完成标志：

```text
QWEN2.5-14B NFCORPUS ALIGNED SCORING PASS
```

### TREC-COVID

NFCorpus完成后运行：

```bash
nohup python scripts/03_score_boundaries_aligned.py \
  --input "$STORAGE/output/semantic_cliff/boundary_trec_k5_v3.json" \
  --output "$OUT_ROOT/scores/qwen14b_trec_aligned_scores.json" \
  --records-dir "$OUT_ROOT/scores/trec_records" \
  --manifest-output "$OUT_ROOT/scores/qwen14b_trec_score_manifest.json" \
  --preflight "$OUT_ROOT/audit/preflight.json" \
  --model-path "$MODEL_PATH" \
  --dataset trec-covid \
  > "$OUT_ROOT/logs/03_score_trec.log" 2>&1 &
```

完成标志：

```text
QWEN2.5-14B TREC-COVID ALIGNED SCORING PASS
```

每条记录单独落盘，因此中断后可用同一命令继续。不要使用 `--reset`，除非明确决定
丢弃尚未正式冻结的中间记录。

## 9. 计算指标并冻结14B阈值

```bash
python scripts/04_metrics_and_freeze_fpr05.py \
  --nfcorpus-scores "$OUT_ROOT/scores/qwen14b_nfcorpus_aligned_scores.json" \
  --trec-scores "$OUT_ROOT/scores/qwen14b_trec_aligned_scores.json" \
  --metrics-output "$OUT_ROOT/metrics/qwen14b_boundary_metrics.json" \
  --threshold-output "$OUT_ROOT/thresholds/qwen14b_fpr05.json" \
  2>&1 | tee "$OUT_ROOT/logs/04_metrics_and_threshold.log"
```

完成标志：

```text
QWEN2.5-14B METRICS AND MODEL-SPECIFIC FPR05 FREEZE PASS
```

该脚本拒绝覆盖已有metrics和threshold。不要因为结果不好而删除或重算阈值。

## 10. 正式做14B与7B配对定位比较

包内已经附带SHA-256固定的Qwen2.5-7B TREC aligned score文件。NFCorpus 7B
正式分数使用原路径。

```bash
python scripts/05_pairwise_qwen14b_vs_qwen7b.py \
  --qwen7-nf "$STORAGE/output/semantic_cliff/aligned_sensitivity/nfcorpus_aligned_scores.json" \
  --qwen14-nf "$OUT_ROOT/scores/qwen14b_nfcorpus_aligned_scores.json" \
  --qwen7-trec "$PACKAGE_DIR/inputs/trec_aligned_qwen_v1_scores.json" \
  --qwen14-trec "$OUT_ROOT/scores/qwen14b_trec_aligned_scores.json" \
  --qwen14-metrics "$OUT_ROOT/metrics/qwen14b_boundary_metrics.json" \
  --output "$OUT_ROOT/metrics/qwen14b_vs_qwen7b_paired.json" \
  2>&1 | tee "$OUT_ROOT/logs/05_pairwise_qwen14_vs_qwen7.log"
```

该结果的差值方向固定为 `Qwen2.5-14B minus Qwen2.5-7B`。CI包含零不代表等价；
CI排除零也只能说明这两个同家族配置在本实验中的差异。

## 11. 第一检查点：先回传定位结果

此时先不要马上跑端到端。回传以下文件进行核对：

```text
$OUT_ROOT/audit/qwen14b_model_freeze.json
$OUT_ROOT/audit/preflight.json
$OUT_ROOT/scores/qwen14b_nfcorpus_score_manifest.json
$OUT_ROOT/scores/qwen14b_trec_score_manifest.json
$OUT_ROOT/metrics/qwen14b_boundary_metrics.json
$OUT_ROOT/thresholds/qwen14b_fpr05.json
$OUT_ROOT/metrics/qwen14b_vs_qwen7b_paired.json
```

可先生成不含E2E的登记快照：

```bash
python scripts/08_collect_results.py \
  --preflight "$OUT_ROOT/audit/preflight.json" \
  --metrics "$OUT_ROOT/metrics/qwen14b_boundary_metrics.json" \
  --threshold "$OUT_ROOT/thresholds/qwen14b_fpr05.json" \
  --paired-scale "$OUT_ROOT/metrics/qwen14b_vs_qwen7b_paired.json" \
  --output "$OUT_ROOT/metrics/qwen14b_registry_localization.json"
```

## 12. NFCorpus 5%完整端到端

确认定位文件、阈值和哈希均正确后：

```bash
source /root/.config/ragwm/verifier.env
bash scripts/07_run_qwen14b_fpr05_e2e.sh
```

脚本依次执行：

```text
14B model-specific FPR05 threshold
→ 3,633-document backward targeted suffix deletion
→ Contriever vector-store rebuild
→ aligned-extension verification
→ E0/E1=25/4 paired comparison
→ SHA-256 manifest
```

完成标志：

```text
QWEN2.5-14B CROSS-DETECTOR FORMAL E2E PASS
```

此端点使用模型自己的5% FPR阈值，因此与Qwen7B、DeepSeek和Gemma的实际编辑
预算不保证一致。WSN只能作为跨detector敏感性证据。

## 13. 生成最终登记快照

```bash
CONDITION=$OUT_ROOT/conditions/qwen2_5_14b_base_fpr05

python scripts/08_collect_results.py \
  --preflight "$OUT_ROOT/audit/preflight.json" \
  --metrics "$OUT_ROOT/metrics/qwen14b_boundary_metrics.json" \
  --threshold "$OUT_ROOT/thresholds/qwen14b_fpr05.json" \
  --paired-scale "$OUT_ROOT/metrics/qwen14b_vs_qwen7b_paired.json" \
  --endpoint-comparison "$CONDITION/metrics/aligned_reference_comparison.json" \
  --sanitization-summary "$CONDITION/sanitization/gainratio_sanitization_summary.json" \
  --output "$OUT_ROOT/metrics/qwen14b_registry_complete.json"
```

## 14. 论文允许写什么

结果出来后最多新增：

1. cross-detector表中一行Qwen2.5-14B；
2. 一句14B与7B的paired localization差异；
3. 一句14B 5%端点及其实际编辑预算；
4. limitation中说明只有7B/14B两个同家族点，不构成scale curve。

不得新增一张复杂的14B多预算图，不得声称跨所有模型鲁棒，不得把14B WSN与
其他模型做公平排名，不得用14B结果覆盖sealed主结果。
