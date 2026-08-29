# RAG-WM 边界泄漏研究

本仓库用于公开与整理 **RAG-WM 边界泄漏（Boundary Leakage）** 相关研究代码、实验结果、复现实验材料与完整性校验文件。

本项目重点研究：在 RAG 水印场景中，原始上下文与生成水印文本的拼接位置，是否会暴露可被检测的统计异常；并进一步评估基于 **GainRatio** 的目标后缀删除方法，在授权语料编辑威胁模型下对水印验证鲁棒性的影响。

---

## 1. 研究概述

RAG 水印通常通过向检索语料中插入或附加特定生成文本，使语料所有者能够在后续检索与生成过程中验证语料归属。

本研究关注一个潜在问题：

> 原始文档内容与水印文本之间的拼接边界，是否会形成可检测的“边界泄漏”信号？

围绕这一问题，本项目主要研究：

- 边界定位；
- 条件负对数似然（Conditional NLL）；
- DeltaLoss；
- GainRatio；
- 目标后缀删除；
- 预算匹配随机删除；
- 跨检测器鲁棒性；
- 不同上下文长度影响；
- 不同检测语言模型影响；
- 端到端水印验证变化。

本项目定位为 **授权红队评估（authorized red-team evaluation）**，用于研究 RAG 水印机制的潜在脆弱性及其改进方向。

---

## 2. GainRatio

设目标文本为 \(T\)，目标文本之前的上下文为 \(C\)。

首先计算目标文本在不使用上下文时的损失：

\[
L(T)
\]

以及在给定上下文条件下的目标文本损失：

\[
L(T \mid C)
\]

定义：

\[
\Delta L = L(T) - L(T \mid C)
\]

进一步使用目标文本本身的损失进行归一化，得到 GainRatio：

\[
\mathrm{GainRatio}
=
\frac{L(T)-L(T\mid C)}{L(T)}
\]

GainRatio 用于衡量上下文对目标文本预测带来的相对增益，并据此识别潜在的异常上下文—目标拼接边界。

在后续端到端实验中，该分数进一步用于指导目标后缀删除。

---

## 3. 实验范围

当前仓库包含与以下数据集和实验相关的材料：

### 数据集

- NFCorpus
- TREC-COVID
- Natural Questions（采样子集）

### 实验类型

- RAG-WM 水印生成
- 水印注入
- 水印验证
- Boundary Construction
- Conditional NLL
- DeltaLoss
- GainRatio
- Targeted Suffix Deletion
- Budget-Matched Random Deletion
- Detector LM Robustness
- Cross-Detector Robustness
- Context Length Ablation
- Adaptive Attack
- End-to-End Verification

### 使用过的检测语言模型

不同实验阶段包括：

- DistilGPT2
- GPT-2 Medium
- Qwen2.5-7B
- Qwen2.5-14B

部分检索实验使用 Contriever 进行向量检索。

---

## 4. 主要实验结果

### 4.1 边界定位

GainRatio 在多个边界检测实验中表现出较强的区分能力。

代表性结果包括：

| 数据集 / 设置 | ROC-AUC | PR-AUC |
|---|---:|---:|
| TREC-COVID 独立边界集 | 0.9245 | 0.9279 |
| TREC-COVID aligned，Qwen2.5-7B | 0.9779 | 0.9781 |
| NFCorpus aligned，Qwen2.5-7B | 0.8346 | 0.8034 |

具体实验应以 `results/` 中保存的机器输出和对应 manifest 为准。

---

## 5. NFCorpus strict-v1 端到端结果

在 sealed NFCorpus strict-v1 端到端实验中：

| 条件 | WSN |
|---|---:|
| E0：原始水印语料 | 26 / 30 |
| E1：参考条件 | 5 / 30 |
| E2：GainRatio 引导目标删除 | 17 / 30 |

预算匹配随机删除的五个随机种子结果为：

```text
WSN = [23, 24, 24, 23, 23]

平均值 = 23.4
标准差 ≈ 0.55
