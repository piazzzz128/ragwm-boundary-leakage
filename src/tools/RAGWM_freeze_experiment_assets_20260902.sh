#!/usr/bin/env bash
set -euo pipefail

RAGWM_STORAGE_ROOT="/root/autodl-tmp/ragwm_storage"
RAGWM_FREEZE_TAG="RAGWM_experiment_freeze_20260902"
RAGWM_FREEZE_PARENT="$RAGWM_STORAGE_ROOT/output/evidence_freezes"
RAGWM_FREEZE_DIR="$RAGWM_FREEZE_PARENT/$RAGWM_FREEZE_TAG"
RAGWM_FREEZE_ARCHIVE="$RAGWM_FREEZE_PARENT/$RAGWM_FREEZE_TAG.tar.gz"

if [[ -e "$RAGWM_FREEZE_DIR" || -e "$RAGWM_FREEZE_ARCHIVE" ]]; then
  echo "STOP: freeze target already exists; nothing was overwritten."
  echo "$RAGWM_FREEZE_DIR"
  echo "$RAGWM_FREEZE_ARCHIVE"
  exit 2
fi

mkdir -p "$RAGWM_FREEZE_DIR/files"

RAGWM_PATH_LIST="$RAGWM_FREEZE_DIR/requested_paths.txt"
RAGWM_FOUND_LIST="$RAGWM_FREEZE_DIR/found_paths.txt"
RAGWM_MISSING_LIST="$RAGWM_FREEZE_DIR/missing_paths.txt"

cat > "$RAGWM_PATH_LIST" <<'PATHS'
/root/autodl-tmp/ragwm_storage/output/formal_likelihood_matrix_v1_20260902
/root/autodl-tmp/ragwm_storage/repo/ragwm_src/tools/RAGWM_formal_likelihood_matrix_v1_20260902
/root/autodl-tmp/ragwm_storage/output/formal_contriever_baseline_v1_20260902
/root/autodl-tmp/ragwm_storage/repo/ragwm_src/local_models/facebook-contriever/config.json
/root/autodl-tmp/ragwm_storage/repo/ragwm_src/local_models/facebook-contriever/tokenizer_config.json
/root/autodl-tmp/ragwm_storage/repo/ragwm_src/local_models/facebook-contriever/special_tokens_map.json
/root/autodl-tmp/ragwm_storage/repo/ragwm_src/local_models/facebook-contriever/tokenizer.json
/root/autodl-tmp/ragwm_storage/repo/ragwm_src/local_models/facebook-contriever/vocab.txt
/root/autodl-tmp/ragwm_storage/repo/ragwm_src/local_models/facebook-contriever/README.md
/root/autodl-tmp/ragwm_storage/output/semantic_cliff/aligned_sensitivity/nfcorpus_aligned_scores.json
/root/autodl-tmp/ragwm_storage/output/sanitization_e2e_trec_covid_aligned_v1/scores/aligned_qwen_v1/trec_aligned_qwen_v1_scores.json
/root/autodl-tmp/ragwm_storage/output/cross_detector_lm_robustness_v1/qwen2_5_14b_base/scores/qwen14b_nfcorpus_aligned_scores.json
/root/autodl-tmp/ragwm_storage/output/cross_detector_lm_robustness_v1/qwen2_5_14b_base/scores/qwen14b_trec_aligned_scores.json
/root/autodl-tmp/ragwm_storage/output/cross_detector_lm_robustness_v1/deepseek_llm_7b_base/scores/deepseek_nfcorpus_aligned_scores.json
/root/autodl-tmp/ragwm_storage/output/cross_detector_lm_robustness_v1/deepseek_llm_7b_base/scores/deepseek_trec_aligned_scores.json
/root/autodl-tmp/ragwm_storage/output/cross_detector_lm_robustness_v1/gemma2_2b_base
/root/autodl-tmp/ragwm_storage/output/semantic_cliff/nfcorpus_strict_contriever_cosine_scores.json
/root/autodl-tmp/ragwm_storage/output/semantic_cliff/nfcorpus_strict_contriever_cosine_metrics.json
/root/autodl-tmp/ragwm_storage/output/semantic_cliff/trec_contriever_cosine_scores.json
/root/autodl-tmp/ragwm_storage/output/semantic_cliff/trec_contriever_cosine_metrics.json
/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_nq_sampled_strict_v2.json
/root/autodl-tmp/ragwm_storage/output/semantic_cliff/boundary_nq_sampled_strict_v2_audit.json
/root/autodl-tmp/ragwm_storage/output/semantic_cliff/nq_sampled_strict_v2_contriever_cosine_scores.json
/root/autodl-tmp/ragwm_storage/output/semantic_cliff/nq_sampled_strict_v2_contriever_cosine_metrics.json
/root/autodl-tmp/ragwm_storage/output/semantic_cliff/main_table_gainratio_nf_trec_nq_aligned_v2.csv
/root/autodl-tmp/ragwm_storage/output/semantic_cliff/main_table_gainratio_nf_trec_nq_aligned_v2.json
/root/autodl-tmp/ragwm_storage/output/semantic_cliff/main_table_gainratio_nf_trec_nq_aligned_v2.md
/root/autodl-tmp/ragwm_storage/output/semantic_cliff/model_scale_ablation_aligned_all_datasets_metrics.json
/root/autodl-tmp/ragwm_storage/output/semantic_cliff/model_scale_ablation_aligned_all_datasets_scores.json
/root/autodl-tmp/ragwm_storage/output/semantic_cliff/model_scale_ablation_aligned_all_datasets_table.csv
/root/autodl-tmp/ragwm_storage/output/semantic_cliff/model_scale_ablation_aligned_all_datasets_table.md
/root/autodl-tmp/ragwm_storage/output/semantic_cliff/context_length_ablation_qwen2_5_7b_aligned_metrics.json
/root/autodl-tmp/ragwm_storage/output/semantic_cliff/context_length_ablation_qwen2_5_7b_aligned_scores.json
/root/autodl-tmp/ragwm_storage/output/semantic_cliff/context_length_ablation_qwen2_5_7b_aligned_table.csv
/root/autodl-tmp/ragwm_storage/output/semantic_cliff/context_length_ablation_qwen2_5_7b_aligned_table.md
/root/autodl-tmp/ragwm_storage/output/adaptive_attack/cleaned
/root/autodl-tmp/ragwm_storage/output/adaptive_attack/cleaned_deepseek_a2
/root/autodl-tmp/ragwm_storage/output/adaptive_attack/adaptive_a2_human_audit_review.md
/root/autodl-tmp/ragwm_storage/output/adaptive_attack/adaptive_a2_human_audit_120_samples_filled.csv
PATHS

: > "$RAGWM_FOUND_LIST"
: > "$RAGWM_MISSING_LIST"

while IFS= read -r RAGWM_SOURCE_PATH; do
  [[ -n "$RAGWM_SOURCE_PATH" ]] || continue
  if [[ -e "$RAGWM_SOURCE_PATH" ]]; then
    printf '%s\n' "$RAGWM_SOURCE_PATH" >> "$RAGWM_FOUND_LIST"
    cp -a --parents "$RAGWM_SOURCE_PATH" "$RAGWM_FREEZE_DIR/files"
  else
    printf '%s\n' "$RAGWM_SOURCE_PATH" >> "$RAGWM_MISSING_LIST"
  fi
done < "$RAGWM_PATH_LIST"

find "$RAGWM_FREEZE_DIR/files" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$RAGWM_FREEZE_DIR/FILE_SHA256SUMS.txt"

{
  echo "freeze_tag=$RAGWM_FREEZE_TAG"
  date -u '+created_utc=%Y-%m-%dT%H:%M:%SZ'
  echo "storage_root=$RAGWM_STORAGE_ROOT"
  echo "found_entries=$(wc -l < "$RAGWM_FOUND_LIST")"
  echo "missing_entries=$(wc -l < "$RAGWM_MISSING_LIST")"
  echo "file_count=$(find "$RAGWM_FREEZE_DIR/files" -type f | wc -l)"
  echo "total_bytes=$(find "$RAGWM_FREEZE_DIR/files" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')"
} > "$RAGWM_FREEZE_DIR/FREEZE_MANIFEST.txt"

tar -C "$RAGWM_FREEZE_PARENT" -czf "$RAGWM_FREEZE_ARCHIVE" "$RAGWM_FREEZE_TAG"
sha256sum "$RAGWM_FREEZE_ARCHIVE" > "$RAGWM_FREEZE_ARCHIVE.sha256"
sha256sum -c "$RAGWM_FREEZE_ARCHIVE.sha256"

echo "PASS: experiment assets frozen"
echo "ARCHIVE=$RAGWM_FREEZE_ARCHIVE"
echo "CHECKSUM=$RAGWM_FREEZE_ARCHIVE.sha256"
echo "MANIFEST=$RAGWM_FREEZE_DIR/FREEZE_MANIFEST.txt"
echo "MISSING=$RAGWM_MISSING_LIST"
