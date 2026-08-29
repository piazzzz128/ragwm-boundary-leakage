# RAG-WM Boundary Leakage

### GainRatio-Guided Targeted Suffix Deletion for Authorized Red-Team Evaluation

本仓库公开 **RAG-WM 边界泄漏（Boundary Leakage）** 研究的代码、实验结果、审计记录与可复现性材料。

本项目研究一个具体的鲁棒性问题：

> 当 RAG 知识水印通过向原始语料追加生成式水印文本时，原始上下文与水印文本之间的拼接边界，是否会产生可被语言模型条件似然捕获的统计信号？

围绕这一问题，本项目研究 **Conditional NLL、DeltaLoss 与 GainRatio**，并进一步评估基于 GainRatio 的目标后缀删除（Targeted Suffix Deletion）在**授权语料编辑威胁模型**下对水印验证结果的影响。

本仓库同时保留：

- 正式最终验证结果；
- localization / conditional-NLL 扩展实验；
- cross-detector 鲁棒性实验；
- multi-budget 实验；
- 历史 sealed strict-v1 实验；
- TREC-COVID 扩展实验；
- 原始机器输出；
- SHA-256 完整性清单；
- 文件大小清单；
- 公开导出脱敏记录；
- 环境快照；
- Git LFS 大文件。

> **重要说明**
>
> 本仓库包含不同时间形成的实验协议。  
> `formal`、`extensions` 和 `historical` 中的结果具有不同实验角色，不能在忽略协议差异的情况下直接混合比较。
>
> 当 README 摘要与机器生成的正式结果文件存在冲突时，应以对应的原始结果文件、正式报告以及 SHA-256 manifest 为准。

---

## 目录

- [1. 研究背景](#1-研究背景)
- [2. 威胁模型](#2-威胁模型)
- [3. Boundary Leakage](#3-boundary-leakage)
- [4. Conditional NLL 与 GainRatio](#4-conditional-nll-与-gainratio)
- [5. 方法概览](#5-方法概览)
- [6. 数据集](#6-数据集)
- [7. 模型与检索器](#7-模型与检索器)
- [8. 结果口径与实验层级](#8-结果口径与实验层级)
- [9. 当前正式最终验证结果](#9-当前正式最终验证结果)
- [10. Boundary Localization 代表性结果](#10-boundary-localization-代表性结果)
- [11. 历史 sealed strict-v1](#11-历史-sealed-strict-v1)
- [12. 扩展实验](#12-扩展实验)
- [13. 仓库结构](#13-仓库结构)
- [14. 快速开始](#14-快速开始)
- [15. 原始 RAG-WM 基础流程](#15-原始-rag-wm-基础流程)
- [16. 如何阅读 formal 结果](#16-如何阅读-formal-结果)
- [17. 完整性与 SHA-256](#17-完整性与-sha-256)
- [18. Git LFS](#18-git-lfs)
- [19. 环境信息](#19-环境信息)
- [20. API 配置与脱敏](#20-api-配置与脱敏)
- [21. strict-v1 与 aligned 的关系](#21-strict-v1-与-aligned-的关系)
- [22. 复现实验注意事项](#22-复现实验注意事项)
- [23. 已知限制](#23-已知限制)
- [24. 负责任使用](#24-负责任使用)
- [25. 结果解释原则](#25-结果解释原则)
- [26. Citation](#26-citation)
- [27. License](#27-license)
- [28. 版本信息](#28-版本信息)

---

# 1. 研究背景

Retrieval-Augmented Generation（RAG）将外部知识库中的检索结果作为上下文输入大语言模型。

知识水印技术尝试通过修改或增强知识库，使语料所有者能够在后续检索与生成过程中验证语料归属。

RAG-WM 的一种核心思路是：

1. 从知识语料中构造 watermark unit；
2. 根据 watermark unit 生成水印文本；
3. 将生成文本注入或追加到相关文档；
4. 建立或更新检索索引；
5. 使用特定查询触发相关水印内容；
6. 通过验证模型判断水印是否存在。

本研究关注这种构造中可能出现的一个附带现象：

> **原始文档与生成水印文本的连接位置可能形成统计不连续性。**

我们将这种现象称为：

**Boundary Leakage（边界泄漏）**

核心问题并不是直接猜测 watermark key，而是研究：

> 在不知道水印 key、也不依赖 verifier feedback 进行候选选择的情况下，能否仅利用文本边界处的条件语言模型统计特征定位可疑拼接位置？

---

# 2. 威胁模型

本项目采用：

**Authorized Corpus-Editing Threat Model**

即攻击方或评估方已经获得对实验语料副本的合法编辑权限。

允许的操作包括：

- 读取实验语料；
- 对语料副本计算语言模型分数；
- 对候选文本执行局部删除；
- 重建检索索引；
- 在完成候选选择后进行端到端验证。

目标不是：

- 获取未经授权的数据；
- 绕过真实生产系统访问控制；
- 窃取 watermark key；
- 攻击第三方系统。

在核心 targeted-selection 阶段，不将 watermark key 或 verifier feedback 作为候选选择信号。

因此，本研究更准确地属于：

> **RAG watermark robustness / authorized red-team evaluation**

---

# 3. Boundary Leakage

考虑一段文档由原始上下文 \(C\) 和后续目标文本 \(T\) 组成：

```text
Original context C | Target text T
                   ↑
             candidate boundary
```

如果 \(T\) 与前文 \(C\) 在生成分布或话语连续性上存在特殊关系，则：

- 单独预测 \(T\)；
- 在给定 \(C\) 的情况下预测 \(T\)；

可能产生不同程度的损失变化。

这种差异可以通过条件语言模型统计量进行量化。

---

# 4. Conditional NLL 与 GainRatio

设：

- \(C\)：目标文本之前的上下文；
- \(T\)：候选目标文本；
- \(L(T)\)：不使用上下文时目标文本的平均负对数似然；
- \(L(T \mid C)\)：给定上下文时目标文本的平均负对数似然。

首先定义：

$$
\Delta L = L(T)-L(T\mid C)
$$

为了降低不同目标文本本身难度差异带来的尺度影响，我们进一步定义：

$$
\mathrm{GainRatio}
=
\frac{L(T)-L(T\mid C)}{L(T)}
$$

直观理解：

- 如果上下文 \(C\) 对目标 \(T\) 几乎没有帮助，GainRatio 较弱；
- 如果给定 \(C\) 后 \(T\) 的预测损失出现明显变化，则该边界可能具有更强统计特征。

本项目利用这一信号进行：

1. candidate boundary scoring；
2. clean-data threshold calibration；
3. backward scanning；
4. targeted suffix deletion；
5. retrieval index rebuilding；
6. end-to-end watermark verification。

---

# 5. 方法概览

整体流程可以概括为：

```text
Original / Watermarked Corpus
            │
            ▼
   Candidate Boundary Extraction
            │
            ▼
       Context C + Target T
            │
            ▼
      Compute L(T)
            │
            ├───────────────┐
            ▼               │
     Compute L(T | C)       │
            │               │
            └───────┬───────┘
                    ▼
               GainRatio
                    │
                    ▼
        Clean-data Calibration
                    │
                    ▼
          Backward Boundary Scan
                    │
                    ▼
       Targeted Suffix Deletion
                    │
                    ▼
         Rebuild Retrieval Index
                    │
                    ▼
       End-to-End Verification
                    │
                    ▼
      Compare with Random Baseline
```

---

# 6. 数据集

本项目不同实验阶段涉及：

### NFCorpus

主要用于：

- boundary localization；
- conditional-NLL；
- sealed strict-v1；
- aligned extension；
- targeted deletion；
- random deletion；
- multi-budget；
- end-to-end verification。

### TREC-COVID

主要用于：

- 独立 boundary evaluation；
- aligned localization；
- detector robustness；
- TREC-COVID end-to-end extension。

### Natural Questions

仓库和历史实验中包含采样子集相关材料，用于部分扩展与稳健性分析。

> 不同实验所使用的数据规模并不完全一致。  
> 复现时必须以对应 experiment manifest / result file 中记录的数据规模为准。

---

# 7. 模型与检索器

## 7.1 检索器

基础 RAG-WM 流程主要使用：

- Contriever
- cosine similarity

源码中同时保留了其他检索配置接口。

---

## 7.2 Detector / Scoring Language Models

不同阶段使用或评估过的模型包括：

- DistilGPT2
- GPT-2 Medium
- Qwen2.5-7B
- Qwen2.5-14B
- DeepSeek-LLM-7B-base
- Gemma-2-2B-base

cross-detector 结果中还包含 Qwen / DeepSeek 的 pairwise 分析。

---

## 7.3 API-based Verifier

当前正式 verifier campaign 记录的 provider/model 为：

```text
ephone:gpt-4o-mini
repository alias: gpt4o_mini_e2e
```

API provider、模型路由和服务端实现可能随时间发生变化，因此：

> API-based verifier 的历史结果应按照其当时保存的 campaign manifest、模型别名、输入单位和机器输出理解，不应被视为跨时间完全不变的确定性函数。

---

# 8. 结果口径与实验层级

这是本仓库最重要的阅读规则之一。

目前 `results/` 分成三层：

```text
results/
├── formal/
├── extensions/
└── historical/
```

## 8.1 formal

`formal/` 保存当前用于公开报告的正式结果。

当前主要目录：

```text
results/formal/
└── gainratio_random_final_verifier_v1_20260829/
```

这是当前 GainRatio 5% clean-FPR 与预算匹配随机删除的正式 verifier campaign。

---

## 8.2 extensions

`extensions/` 保存用于支持主结论的扩展实验：

```text
results/extensions/
├── conditional_nll_e2e_nf_v1_20260828/
├── cross_detector_lm_robustness_v1/
├── sanitization_multibudget_nf_aligned_v1/
└── semantic_cliff/
```

这些实验用于回答：

- Conditional NLL 是否能够支持 boundary localization？
- detector LM 改变后信号是否仍存在？
- 不同模型之间是否存在一致性？
- 不同删除预算下表现如何？
- semantic-cliff / boundary signal 是否具有稳健性？

---

## 8.3 historical

`historical/` 保存已经完成并冻结的历史实验材料：

```text
results/historical/
├── sanitization_e2e_nf_strict_v1/
└── sanitization_e2e_trec_covid_aligned_v1/
```

这些材料用于：

- 保存实验历史；
- 支持审计；
- 追踪协议演进；
- 防止在新实验运行后覆盖历史证据。

**历史结果不应被静默重新解释为当前 formal protocol。**

---

# 9. 当前正式最终验证结果

当前正式报告：

```text
results/formal/
└── gainratio_random_final_verifier_v1_20260829/
    └── gainratio_final_verifier_report.json
```

正式 campaign 使用：

```text
verify_num  = 30
verify_seed = 633
```

并确认：

```text
unit_identity_and_order_match = true
all_unknown_counts_zero       = true
campaign_status               = complete
```

---

## 9.1 Watermarked / Oracle / GainRatio

| Condition | WSN |
|---|---:|
| Watermarked | 26 / 30 |
| Oracle restore | 4 / 30 |
| GainRatio @ 5% clean FPR | 21 / 30 |

其中：

```text
Watermarked → GainRatio
26 → 21
```

GainRatio 条件相对于原 watermarked condition 减少了 5 个 positive verification。

正式 paired comparison 中：

```text
yes_to_no  = 5
no_to_yes  = 0
discordant = 5
McNemar exact two-sided p = 0.0625
```

因此，本结果应被理解为：

> 在当前固定验证 campaign 中，GainRatio-guided deletion 产生了方向一致的 WSN 降低；但该单一 paired test 的双侧精确 McNemar p 值为 0.0625，不应表述为 p < 0.05 的统计显著结果。

---

## 9.2 预算匹配随机删除

五个随机种子的 WSN 为：

| Random seed | WSN |
|---:|---:|
| 101 | 22 / 30 |
| 202 | 25 / 30 |
| 303 | 25 / 30 |
| 404 | 22 / 30 |
| 505 | 24 / 30 |

汇总：

```text
Random WSN = [22, 25, 25, 22, 24]

Mean                 = 23.6
Sample std. dev.     ≈ 1.5166
Minimum              = 22
Maximum              = 25
GainRatio WSN        = 21
```

当前 campaign 中：

```text
all_wsn_above_gainratio = true
```

即五个随机种子的 WSN 均高于 GainRatio 条件的 21。

但正式报告同时明确规定：

> GainRatio 与单个 random seed 的 paired comparison 应以描述性结果为主，除非对应 exact paired test 支持更强的统计结论。

因此，不应将这些结果夸大为已经证明普遍性的 verifier bypass。

---

## 9.3 删除预算

当前正式 GainRatio 5% clean-FPR 条件记录：

```text
modified_documents         = 174
target_deleted_characters  = 37,586
random_character_tolerance = 0.01
random_budget_audit_passed = true
```

也就是说随机 baseline 按照与 GainRatio 相匹配的字符删除预算构造，并经过预算审计。

---

## 9.4 Oracle Gap

正式报告记录：

```text
oracle_gap_closure ≈ 0.22727
```

Oracle restore 作为参考条件提供一个更强的上界式对照，但不代表现实攻击者可获得 oracle information。

---

## 9.5 Formal Claim Guard

正式结果文件中明确保存：

```text
Results do not establish ownership-verification bypass.
```

因此本仓库的结论应限定为：

> 发现并实证评估一个 key-agnostic boundary-localization / corpus-editing 信号，并观察其在固定实验协议下对 watermark verification 的影响。

而不是：

> 已经证明可以普遍绕过所有 RAG watermark ownership verification。

---

# 10. Boundary Localization 代表性结果

不同 boundary evaluation / aligned extension 中，GainRatio 展现出较强区分能力。

代表性记录包括：

| Dataset / Protocol | ROC-AUC | PR-AUC |
|---|---:|---:|
| TREC-COVID independent boundary set | 0.9245 | 0.9279 |
| TREC-COVID aligned, Qwen2.5-7B | 0.9779 | 0.9781 |
| NFCorpus aligned, Qwen2.5-7B | 0.8346 | 0.8034 |

这些结果用于支持：

> boundary-level signal 在多个数据集与协议下可以被条件语言模型统计量检测。

但必须注意：

- independent boundary set；
- aligned extension；
- historical strict-v1；

并不是完全相同的数据构造或计分协议。

因此应分别解释。

---

# 11. 历史 sealed strict-v1

历史 NFCorpus sealed strict-v1 位于：

```text
results/historical/
└── sanitization_e2e_nf_strict_v1/
```

该目录保留了：

```text
audit/
audit_inventory/
conditions/
logs/
restored/
```

其中 `conditions/` 进一步保留：

```text
e0_watermarked_before/
e0_watermarked_smoke/
e1_oracle_restore/
e2_gainratio_delete/
e2_gainratio_smoke/
e2_gainratio_aligned_sensitivity/
e2_gainratio_aligned_smoke/
e3_random_baselines/
e3_random_aligned_v2/
```

这是非常重要的历史审计材料。

---

## 11.1 历史 strict-v1 代表性结果

历史 sealed strict-v1 曾记录：

| Historical condition | WSN |
|---|---:|
| E0 watermarked | 26 / 30 |
| E1 oracle/reference | 5 / 30 |
| E2 GainRatio targeted deletion | 17 / 30 |

历史 budget-matched random deletion：

```text
[23, 24, 24, 23, 23]
```

汇总约为：

```text
mean = 23.4
std  ≈ 0.55
```

这些数字属于：

> **historical sealed strict-v1**

而不是当前 `formal/gainratio_random_final_verifier_v1_20260829` campaign。

因此：

```text
historical strict-v1 E2 = 17/30
```

和：

```text
current formal GainRatio@5%FPR = 21/30
```

不应被视为同一次实验的两个可互换数字。

---

# 12. 扩展实验

# 12.1 Conditional-NLL End-to-End Extension

目录：

```text
results/extensions/
└── conditional_nll_e2e_nf_v1_20260828/
```

包含：

```text
audit/
conditions/
localization/
logs/
metrics/
scan_records/
thresholds/
verifier_refresh/
SHA256SUMS_formal_results.txt
conditional_nll_results_for_review_20260828.tar.gz
```

其中 metrics 包含：

```text
conditional_nll_complete_report.json
retrieval_utility_frozen.json
```

该实验用于把：

```text
boundary localization
        ↓
threshold calibration
        ↓
scan
        ↓
sanitization
        ↓
retrieval utility
        ↓
verification
```

串联为较完整的 end-to-end 分析。

---

# 12.2 Cross-Detector Robustness

目录：

```text
results/extensions/
└── cross_detector_lm_robustness_v1/
```

当前公开子目录包括：

```text
deepseek_llm_7b_base/
gemma2_2b_base/
pairwise_qwen_deepseek_v1/
pairwise_qwen_deepseek_v2/
qwen2_5_14b_base/
```

目的在于检查：

> GainRatio / boundary signal 是否只是某一个 detector LM 的偶然现象。

Qwen2.5-14B 等实验作为 cross-detector extension 保存，不覆盖原有 7B 或 strict-v1 结果。

---

# 12.3 Multi-Budget Extension

目录：

```text
results/extensions/
└── sanitization_multibudget_nf_aligned_v1/
```

用于研究：

- threshold / FPR 改变；
- 删除预算改变；
- targeted deletion 强度变化；

对：

- watermark verification；
- corpus modification；
- retrieval utility；

的影响。

---

# 12.4 Semantic-Cliff Extension

目录：

```text
results/extensions/
└── semantic_cliff/
```

用于保存与 boundary discontinuity / semantic-cliff 分析相关的实验材料。

---

# 13. 仓库结构

当前公开仓库的核心结构如下：

```text
ragwm-boundary-leakage/
│
├── README.md
├── .gitattributes
│
├── src/
│   ├── README.md
│   ├── requirement.txt
│   ├── Statistic of Watermark.xlsx
│   ├── analysis/
│   ├── contriever_src/
│   ├── entity_generate/
│   ├── model_configs/
│   ├── rag/
│   ├── src/
│   └── tools/
│
├── results/
│   │
│   ├── formal/
│   │   └── gainratio_random_final_verifier_v1_20260829/
│   │       ├── gainratio_final_verifier_report.json
│   │       ├── SHA256SUMS_formal_results.txt
│   │       ├── scripts/
│   │       ├── supporting_audits/
│   │       └── verifier_campaign/
│   │
│   ├── extensions/
│   │   ├── conditional_nll_e2e_nf_v1_20260828/
│   │   ├── cross_detector_lm_robustness_v1/
│   │   ├── sanitization_multibudget_nf_aligned_v1/
│   │   └── semantic_cliff/
│   │
│   └── historical/
│       ├── sanitization_e2e_nf_strict_v1/
│       └── sanitization_e2e_trec_covid_aligned_v1/
│
├── environment/
│   ├── python_packages.txt
│   └── system_environment.txt
│
└── manifests/
    ├── EXPORT_SHA256SUMS.txt
    ├── ORIGINAL_OUTPUT_SHA256SUMS.txt
    ├── FILE_SIZES_BYTES.tsv
    ├── FILES_OVER_99MB.tsv
    ├── REDACTED_FILES.txt
    └── SECRET_SCAN.txt
```

---

# 14. 快速开始

## 14.1 安装 Git LFS

仓库包含 Git LFS 对象，因此建议先安装：

```bash
git lfs install
```

---

## 14.2 克隆仓库

HTTPS：

```bash
git clone https://github.com/piazzzz128/ragwm-boundary-leakage.git
```

或者 SSH：

```bash
git clone git@github.com:piazzzz128/ragwm-boundary-leakage.git
```

进入仓库：

```bash
cd ragwm-boundary-leakage
```

获取 LFS 对象：

```bash
git lfs pull
```

---

## 14.3 安装基础依赖

当前基础源码中依赖文件名称为：

```text
src/requirement.txt
```

因此：

```bash
cd src
pip install -r requirement.txt
```

注意：

> `requirement.txt` 来自原始项目环境，不应被理解为经过重新构建验证的最小 lockfile。

完整公开环境快照同时保存在：

```text
environment/python_packages.txt
```

---

# 15. 原始 RAG-WM 基础流程

`src/README.md` 保存了原始项目的基础运行说明。

以下命令应在：

```text
src/
```

目录下运行。

---

## 15.1 数据准备

```bash
python rag/prepare_data.py
```

---

## 15.2 配置 LLM

模型相关配置位于：

```text
model_configs/
```

公开版本已对部分包含敏感 API 配置的文件进行删除或脱敏。

因此复现者需要：

1. 根据自己的 provider 创建配置；
2. 配置自己的 API credential；
3. 不要将 credential 提交至 Git；
4. 根据实际环境修改 base path。

---

## 15.3 创建向量数据库

原始项目使用的 ChromaDB 版本为：

```text
0.5.20
```

以 NFCorpus + Contriever + cosine 为例：

```bash
python rag/vectorstore.py \
  --eval_dataset 'nfcorpus' \
  --eval_model_code 'contriever' \
  --score_function 'cosine'
```

---

## 15.4 实体抽取

```bash
python entity_generate/generate_entity_llm_check.py \
  --eval_dataset 'nfcorpus' \
  --dataset_prob 1
```

该阶段依赖 LLM，运行时间和 API 成本可能较高。

---

## 15.5 Watermark Unit 生成

```bash
python entity_generate/generate_hash_entity.py \
  --eval_dataset 'nfcorpus' \
  -t scratch \
  --entity_num 100 \
  --edge_prob 0.05
```

---

## 15.6 Watermark Text Generation

```bash
python src/main.py \
  --eval_dataset 'nfcorpus' \
  --eval_model_code 'contriever' \
  --score_function 'cosine' \
  --doc 1 \
  --inject 0 \
  --verify 0 \
  --stat 0 \
  --mutual_times 10
```

---

## 15.7 Watermark Injection

```bash
python src/main.py \
  --eval_dataset 'nfcorpus' \
  --eval_model_code 'contriever' \
  --score_function 'cosine' \
  --doc 0 \
  --inject 1 \
  --verify 0 \
  --stat 0 \
  --mutual_times 10
```

---

## 15.8 Watermark Verification

```bash
python src/main.py \
  --eval_dataset 'nfcorpus' \
  --eval_model_code 'contriever' \
  --score_function 'cosine' \
  --doc 0 \
  --inject 0 \
  --verify 1 \
  --stat 1 \
  --mutual_times 10
```

---

# 16. 如何阅读 formal 结果

当前最重要的正式结果入口：

```text
results/formal/gainratio_random_final_verifier_v1_20260829/
```

建议按以下顺序阅读。

---

## 16.1 Final Report

```text
gainratio_final_verifier_report.json
```

包含：

- experiment role；
- verifier model；
- campaign status；
- verify number；
- verify seed；
- condition-level WSN；
- paired comparison；
- random summary；
- deletion budget；
- claim guard。

---

## 16.2 Verifier Campaign

```text
verifier_campaign/
```

保存当前正式验证 campaign 的底层机器输出和 condition 数据。

---

## 16.3 Supporting Audits

```text
supporting_audits/
```

保存用于支持结果完整性和协议检查的审计材料。

---

## 16.4 Scripts

```text
scripts/
```

保存与当前 formal package 对应的实验 / verification 脚本副本。

---

## 16.5 Formal SHA-256

```text
SHA256SUMS_formal_results.txt
```

用于确认 formal package 中关键结果文件未被静默修改。

---

# 17. 完整性与 SHA-256

为了使公开导出具有可审计性，本仓库保留多层 SHA-256 清单。

---

## 17.1 Export SHA-256

```text
manifests/EXPORT_SHA256SUMS.txt
```

保存公开 export 中的文件哈希。

可尝试从仓库根目录验证：

```bash
sha256sum -c manifests/EXPORT_SHA256SUMS.txt
```

---

## 17.2 Original Output SHA-256

```text
manifests/ORIGINAL_OUTPUT_SHA256SUMS.txt
```

用于记录公开 export 对应原始实验输出的校验信息。

---

## 17.3 Formal Results SHA-256

例如：

```text
results/formal/
└── gainratio_random_final_verifier_v1_20260829/
    └── SHA256SUMS_formal_results.txt
```

以及：

```text
results/extensions/
└── conditional_nll_e2e_nf_v1_20260828/
    └── SHA256SUMS_formal_results.txt
```

这些 experiment-local checksum 文件用于固定正式结果。

---

## 17.4 文件大小清单

```text
manifests/FILE_SIZES_BYTES.tsv
```

记录 export 中文件的大小。

---

## 17.5 大文件清单

```text
manifests/FILES_OVER_99MB.tsv
```

用于识别超过普通 GitHub 大文件阈值附近的对象。

---

# 18. Git LFS

当前仓库存在一个历史 ChromaDB binary artifact 通过 Git LFS 管理。

因此克隆后建议执行：

```bash
git lfs pull
```

可以查看：

```bash
git lfs ls-files
```

若仅普通 `git clone` 而未获取 LFS 对象，工作区中的对应文件可能只有 LFS pointer。

---

# 19. 环境信息

公开 export 保存：

```text
environment/
├── python_packages.txt
└── system_environment.txt
```

当前 export 时的系统快照包括：

```text
Python       3.12.3
CUDA         13.2
NVIDIA driver 595.58.03
GPU          NVIDIA RTX PRO 6000 Blackwell
```

需要强调：

> 该文件记录的是公开导出时环境快照，不代表所有历史实验均在完全相同的驱动、CUDA 或硬件环境下执行。

因此复现单个实验时，应同时查看：

- experiment-local logs；
- manifests；
- preflight；
- model metadata；
- frozen result files。

---

# 20. API 配置与脱敏

公开仓库在 export 前进行了敏感信息处理。

脱敏记录：

```text
manifests/REDACTED_FILES.txt
```

其中记录了部分被移除的：

- API model configs；
- historical config backups；
- API-related helper code；
- notebook；
- audit log。

例如包括：

```text
src/model_configs/deepseek_chat_config.json
src/model_configs/gpt3.5_config.json
src/model_configs/gpt4o_mini_e2e_config.json
src/model_configs/qwen_plus_config.json
...
```

这些文件不会作为公开 credential 模板直接发布。

---

## 20.1 Secret Scan

公开导出还保存：

```text
manifests/SECRET_SCAN.txt
```

该文件用于记录 export 阶段的 secret scanning。

即便如此，公开仓库使用者仍应在后续提交前自行检查：

- API keys；
- access tokens；
- passwords；
- private endpoints；
- personal paths；
- private datasets。

---

# 21. strict-v1 与 aligned 的关系

本项目中容易混淆的两个概念是：

```text
strict-v1
```

与：

```text
aligned
```

它们不是 E0 / E2 的同义词，也不是两个攻击方法。

---

## 21.1 strict-v1

`strict-v1` 指一个历史实验协议 / 数据与计分实现状态。

其意义是：

> 保留当时实际运行出的 sealed historical experiment，不事后偷偷改写结果。

它的价值主要是：

- 历史可追溯；
- sealed evidence；
- 支持论文实验演进说明；
- 防止 cherry-picking。

---

## 21.2 aligned

`aligned` 指后续对：

- token scoring range；
- target token definition；
- conditional/unconditional loss；
- BOS / separator；
- first-target-token handling；

等细节进行更统一之后的实验协议。

aligned extension 主要用于：

- cross-detector；
- multi-budget；
- additional dataset；
- model-scale analysis；
- robustness analysis。

---

## 21.3 为什么不直接合并？

因为不同 scoring protocol 下得到的数字未必严格具有同一统计含义。

因此本仓库选择：

```text
保留历史 strict-v1
+
新增 aligned / formal extension
+
明确标注协议
```

而不是为了得到更漂亮的统一数字而重新覆盖历史实验。

---

# 22. 复现实验注意事项

## 22.1 Base Path

历史代码和部分 artifact 内可能包含原始 AutoDL 环境中的绝对路径，例如：

```text
/root/autodl-tmp/ragwm_storage/...
```

复现时不能直接假定这些路径存在。

需要修改为自己的工作目录。

---

## 22.2 Model Weights

大型预训练模型权重没有作为项目依赖整体打包进本仓库。

复现者应从相应模型的官方发布渠道获得模型，并记录：

- model name；
- revision；
- dtype；
- quantization；
- tokenizer revision。

---

## 22.3 API Models

API-based verifier 需要使用复现者自己的合法 API credential。

不要将 API key 写入：

```text
README
Git history
public config
shell script
notebook output
```

建议通过环境变量或本地未跟踪配置传入。

---

## 22.4 Determinism

即使：

- random seed 固定；
- verifier unit 顺序固定；

API-based model 仍可能受到：

- backend revision；
- inference stack；
- provider routing；
- model update；

影响。

因此历史 API result 最可靠的证据是：

> 当时冻结的 machine-generated output，而不是以后重新调用 API 得到的结果。

---

## 22.5 Historical Artifacts

`results/historical/` 的目的之一就是保存历史运行状态。

不建议直接修改其中已有结果。

如果需要重新实验，应创建新的：

```text
results/extensions/<new_experiment_name>/
```

或新的 formal campaign，而不是覆盖原始 artifact。

---

# 23. 已知限制

本研究和当前仓库仍存在以下限制。

### 1. 多协议共存

strict-v1、aligned 和当前 formal campaign 并非完全相同实验协议。

因此不能简单将所有数字放进同一个统计总体。

---

### 2. API verifier 的时间依赖性

API 模型可能发生服务端变化。

虽然本仓库保存了固定 campaign 输出，但未来 rerun 不保证 bitwise-identical。

---

### 3. 尚未形成单一 one-command reproduction

当前仓库首先定位为：

> research artifact + evidence release

而不是已经完全工程化的 Python package。

不同实验仍需按照：

- source code；
- result package；
- manifest；
- experiment-specific scripts；

进行复现。

---

### 4. 部分配置已主动脱敏

为防止公开 API credential 或敏感配置，部分文件没有出现在公开 export 中。

---

### 5. 大模型与数据依赖

部分实验需要：

- 较大 GPU 显存；
- pretrained model weights；
- BEIR datasets；
- API access；

因此完整重跑成本高于读取和审计已有机器结果。

---

### 6. 结果不等于普遍绕过证明

当前 formal report 明确限制结论：

> 本实验结果不建立普遍的 ownership-verification bypass 结论。

它支持的是：

> boundary leakage 是一个值得进一步研究的、可检测且可能影响水印鲁棒性的信号。

---

# 24. 负责任使用

本项目只面向：

- 学术研究；
- 授权安全测试；
- 自有语料；
- watermark robustness analysis；
- reproducibility；
- defense-oriented evaluation。

使用本仓库时应：

1. 只处理有权修改和测试的语料；
2. 遵守模型和数据集许可证；
3. 不将实验方法用于未经授权的数据篡改；
4. 不将研究结果夸大为对所有 watermark 系统的普遍结论；
5. 清楚报告实验协议、模型、数据集和统计限制。

---

# 25. 结果解释原则

为了避免实验演进造成混乱，本项目建议采用以下证据优先级：

```text
1. Frozen machine-generated outputs
2. Experiment-local SHA-256 manifests
3. Formal verifier / metric reports
4. Supporting audits and campaign manifests
5. Extension experiment outputs
6. Historical archived outputs
7. README / prose summaries
```

如果出现：

```text
README 中的数字
        ≠
formal machine report
```

优先采用 formal machine report。

---

## 25.1 不应做的事情

不应为了获得统一或更漂亮的结果而：

- 删除不理想历史实验；
- 静默覆盖 sealed experiment；
- 将 strict-v1 数字改写成 aligned 数字；
- 将不同 token-scoring protocol 混为一个实验；
- 只汇报最好 random seed；
- 忽略不支持强结论的 paired test。

---

## 25.2 推荐做法

应：

- 明确 protocol；
- 明确 dataset；
- 明确 detector；
- 明确 threshold；
- 明确 deletion budget；
- 保存 raw output；
- 保存 SHA-256；
- 将正式结果和历史结果分目录保存。

---

# 26. Citation

正式论文发表后，本节将更新为论文的正式 BibTeX。

当前引用本仓库时，可使用仓库 URL：

```text
https://github.com/piazzzz128/ragwm-boundary-leakage
```

临时项目引用格式可写为：

```bibtex
@misc{ragwm_boundary_leakage_2026,
  title        = {RAG-WM Boundary Leakage: GainRatio-Guided Targeted Suffix Deletion},
  year         = {2026},
  howpublished = {\url{https://github.com/piazzzz128/ragwm-boundary-leakage}},
  note         = {Research code, experimental artifacts, and reproducibility evidence}
}
```

正式论文信息确定后，应补充：

```text
author
paper title
conference / journal
year
DOI / arXiv
```

---

# 27. License

当前仓库尚未附加正式开源许可证。

需要注意：

> GitHub 上“公开可见”并不自动等同于授予任意复制、修改、再发布或商业使用权。

在正式 LICENSE 文件加入仓库之前，请将本仓库视为：

```text
Research artifact made publicly viewable;
reuse terms not yet formally specified.
```

后续可根据论文代码发布策略选择合适许可证，例如：

- MIT
- Apache-2.0
- BSD-3-Clause

具体许可证应结合：

- 原始 RAG-WM 代码许可证；
- 第三方依赖许可证；
- 数据集许可证；
- 学校 / 作者发布要求；

确定。

---

# 28. 版本信息

首次公开 research export：

```text
2026-08-29
```

首次公开 Git commit：

```text
fd5f665d
Initial public release: RAG-WM boundary leakage experiments
```

README 后续更新不会改变初始实验 export 的 provenance。

---

## 当前状态

本仓库目前定位为：

> **Public research artifact / reproducibility repository**

当前已经公开：

- 基础源码；
- 正式 GainRatio vs matched-random verifier campaign；
- conditional-NLL extension；
- cross-detector extension；
- multi-budget extension；
- semantic-cliff artifacts；
- historical strict-v1；
- historical TREC-COVID aligned artifacts；
- environment snapshot；
- SHA-256 manifests；
- redaction inventory；
- secret-scan record；
- Git LFS object。

后续计划进一步整理：

- English README；
- paper-to-artifact result index；
- cleaner reproduction entry points；
- standardized `requirements.txt`；
- one-command reproduction scripts；
- figure/table generation scripts；
- publication citation；
- LICENSE。

---

## Repository

**GitHub**

https://github.com/piazzzz128/ragwm-boundary-leakage

---

**Research focus:**  
RAG Watermarking · Boundary Leakage · Conditional NLL · GainRatio · Targeted Suffix Deletion · Watermark Robustness · Retrieval-Augmented Generation · Authorized Red-Team Evaluation
