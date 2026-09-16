# RAGWM 正式六设置全量复现包（2026-09-07）

## 这一步到底做什么

本包逐条重新执行 2026-09-02 formal likelihood matrix 的全部 causal-LM 前向计算：

- 2 个数据集：NF-Corpus、TREC-COVID
- 3 个模型：Qwen2.5-7B、DeepSeek-LLM-7B-base、Qwen2.5-14B
- 6 个正式 settings，共 2,712 条记录
- 每条同时复算 `Lu` 与 `Lc`
- 自动派生 Target-only、Conditional NLL、w/o normalization (`Lc-Lu`) 与 GainRatio (`(Lc-Lu)/Lu`)
- 自动重算 ROC-AUC 与 Average Precision，并与冻结结果逐项比较

它不修改或覆盖任何旧 raw output；新结果只写入：

`/root/autodl-tmp/ragwm_storage/output/formal_full_reproduction_20260907/`

## 为什么现在做

24 条 spot-check 已证明 Qwen2.5-14B 可逐值复现，但当前环境下 Qwen2.5-7B 与 DeepSeek 的逐值结果存在超阈值差异。全量复现用于回答两个问题：

1. 差异是否贯穿全部记录；
2. 差异是否会改变四种分数的正式 ROC-AUC/AP 结论。

只有完成本步，才能决定是否更新论文数字；在结果审计前，旧冻结结果和本次新结果都不能互相覆盖。

## 唯一运行命令

将整个文件夹上传到 AutoDL 后运行：

```bash
bash RAGWM_formal_full_reproduction_20260907/run.sh
```

脚本会先进行不加载模型的硬校验：六个正式输入 SHA、两份 scorer 源码 SHA、记录数和标签数、模型文件完整性，以及所有模型资产 SHA。DeepSeek 还会与历史冻结的模型资产 SHA 逐项核对。任何硬校验失败都会在消耗 GPU 前停止。

随后按 Qwen2.5-7B → DeepSeek-7B → Qwen2.5-14B 的顺序运行。每个模型只加载一次；每完成一条即原子落盘。SSH 断开或进程中断后，再运行同一命令会自动跳过已完成记录。

脚本带有进程锁；不要并行启动两次。同一输出目录若已有任务运行，第二次启动会立即停止，避免双重占用 GPU。

## 完成后上传什么

终端最后会打印：

- `UPLOAD_FILE_1=...formal_full_reproduction_20260907.tar.gz`
- `UPLOAD_FILE_2=...formal_full_reproduction_20260907.tar.gz.sha256`

只上传这两个文件。不要自行挑选或改名其中的 JSON。

## 重要边界

- 这是正式 likelihood matrix 的完整复现，不是重跑项目目录里的所有历史/旧协议实验。
- 不运行 aligned/strict-v1 的旧 NQ、context-length、adaptive 或 GPT-2 数字。
- 不重跑生成、检索或 sanitization 攻击阶段，因为六个正式输入已经由固定 SHA 锁定；本步只复算它们对应的 scorer 与指标。
- `Conditional NLL` 即使论文表格之后不展示，也会保留在审计结果中，不能从实验事实中删除。
- 完成后先审计，再决定 Table 1 使用旧冻结数字还是新统一复现数字。
