# Qwen2.5-14B 补充实验预注册说明

## 1. 研究问题

本扩展回答两个预先固定的问题：

1. 在相同 Qwen2.5 模型家族中，14B base detector 是否在 NFCorpus 与
   TREC-COVID 上复现 canonical aligned boundary-localization 信号？
2. 在模型自身 clean-calibration 5% FPR 阈值下，14B 定位结果进入完整
   NFCorpus 删除、重建索引与验证流程后，WSN 如何变化？

本实验不是新的 sealed 主实验，不用于覆盖 strict-v1 的
`E0/E1/E2=26/5/17`。

## 2. 模型冻结

- 模型：`Qwen/Qwen2.5-14B`；
- 类型：base / pretrained causal LM；
- 官方参数规模：14.7B；
- 正式精度：BF16，禁止量化；
- 正式运行：单卡；最低检查线 40 GiB，建议 48 GiB；
- 必须记录远端 commit revision 和本地所有模型、tokenizer 文件 SHA-256；
- 禁止替换为 Instruct、GPTQ、AWQ、GGUF 或其他量化版本。

选择该模型的原因是它与已有 Qwen2.5-7B 同家族、同训练阶段，可把
“模型家族差异”控制得比加入另一个14B家族更小。但7B与14B只有两个点，
因此只能称为 same-family scale sensitivity，不能写 scaling law。

## 3. 固定协议

- protocol layer：canonical aligned extension；
- `L(T)` 与 `L(T|C)` 均排除第一个 target token；
- no BOS；
- no separator；
- context 上限 1,024 tokens；
- target 上限 128 tokens；
- `score = -(L(T)-L(T|C))/L(T)`；
- 分数越大越可疑；
- NFCorpus：446 条，clean/injected = 223/223；
- TREC-COVID：458 条，clean/injected = 229/229；
- CI：source-document cluster bootstrap，5,000 replicates，seed 20260609；
- threshold：NFCorpus clean group split，seed 20260714，calibration 169 条，
  holdout 54 条，使用 higher quantile 的95百分位；
- threshold 必须在查看14B端到端 WSN 与 utility 前冻结。

## 4. 预先固定的报告指标

### Localization

- NFCorpus ROC-AUC、PR-AUC及各自95% cluster-bootstrap CI；
- TREC-COVID ROC-AUC、PR-AUC及各自95% cluster-bootstrap CI；
- 14B自身5% FPR threshold；
- calibration empirical FPR、holdout empirical FPR；
- injected TPR仅作为 threshold diagnostic。

### 同家族配对比较

- Qwen2.5-14B minus Qwen2.5-7B 的 ROC-AUC 差值及配对CI；
- Qwen2.5-14B minus Qwen2.5-7B 的 PR-AUC 差值及配对CI；
- CI包含零时只写“未检测到差异”，不能写“等价”。

### NFCorpus 5%端点

- aligned-extension E0/E1 reference = 25/4；
- 14B E2 WSN；
- baseline-positive broken、retained、false gain、unknown；
- exact two-sided McNemar p；
- modified documents、deleted characters、deletion steps；
- 是否仍高于 ownership threshold=2。

## 5. 解释边界

- 14B结果无论正负均按正式方向原样报告，不允许事后取负；
- 不能把14B新结果与 sealed 26/5做混合计算；
- detector-specific 5% FPR会产生不同实际编辑预算，跨模型WSN只能作为
  sensitivity evidence，不能排名；
- 即使14B强于7B，也不能据两个点声称参数规模带来普遍提升；
- 只要自动条件仍高于2，只能写 verification weakening；
- 14B四预算不是本轮必做项，避免增加论文复杂度。

## 6. 停止条件

以下任一条件发生时停止，不进入正式评分：

- 模型不是 Qwen2ForCausalLM / 48 layers / 40 attention heads / 8 KV heads；
- 使用量化权重或非BF16；
- 单卡显存低于40 GiB或不支持BF16；
- 输入SHA-256与20260819 registry不一致；
- NFCorpus/TREC行数或标签数不一致；
- aligned-extension E0/E1不是25/4；
- 旧依赖脚本SHA-256不匹配。
