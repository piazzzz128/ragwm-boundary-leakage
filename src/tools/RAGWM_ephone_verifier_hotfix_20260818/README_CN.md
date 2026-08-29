# RAG-WM ePhone verifier hotfix（2026-08-18）

用途：修复 `fpr01` 运行中因旧中转 API 停止服务造成的 30/30 unknown。

固定非敏感配置：

- API Base URL：`https://api.ephone.ai/v1`
- verifier model：`gpt-4o-mini`

安全与实验规则：

1. 不在源码、配置、压缩包或结果登记表中存储 API Key。
2. 运行时隐藏输入 Key，仅保留在当前 shell 子进程环境中。
3. 先执行单次 API 测试，通过后才重跑 verifier。
4. 备份原 30/30 unknown 的无效输出。
5. 仅重跑 fpr01 verifier 和 E0/E1 对照，不重跑 3,633 篇删除或索引。
6. unknown 不为 0 或 `run_valid=false` 时强制失败。
7. 不覆盖 sealed strict-v1 主结果。

在 AutoDL 上传并解压后运行：

```bash
chmod +x install_and_rerun_fpr01.sh
bash install_and_rerun_fpr01.sh
```

终端提示时粘贴一枚新生成且未在聊天中公开的 API Key。输入过程不会回显。

若服务商明确要求 Base URL 不包含 `/v1`，可覆盖：

```bash
RAGWM_API_BASE=https://api.ephone.ai bash install_and_rerun_fpr01.sh
```

完成的唯一有效标志：

```text
FPR01 VERIFIER RERUN VALIDATION PASS
FPR01 VERIFIER-ONLY RERUN COMPLETE AND VALID
```
