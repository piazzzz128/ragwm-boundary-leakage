# 正式运行状态登记

## 冻结输入

- [ ] `00_preflight_multibudget.py` 输出 PASS
- [ ] aligned scores SHA-256 = `e1c7dbc745a83b473883ccf9c0d6a3c43c1bb7a14fa9f0dda1551b08f92dd218`
- [ ] attack input SHA-256 = `52c72f64043e4a668b5084b75ac0cfa92f63b77c3493875cf788ee5021da8e38`
- [ ] 5% threshold = `-0.1832180580405571`

## 端到端预算点

| Point | Threshold | Sanitization | Index | Verifier | Utility | WSN | Docs | Chars | Status |
|---|---:|---|---|---|---|---:|---:|---:|---|
| 1% | -0.13007188598091768 | 待运行 | 待运行 | 待运行 | 待运行 | 缺失 | 缺失 | 缺失 | pending |
| 2.5% | -0.1640843956343344 | 待运行 | 待运行 | 待运行 | 待运行 | 缺失 | 缺失 | 缺失 | pending |
| 5% | -0.1832180580405571 | 已完成 | 已完成 | 已完成 | 已完成 | 21/30 | 174 | 37,586 | frozen reuse |
| 10% | -0.20519773948364803 | 待运行 | 待运行 | 待运行 | 待运行 | 缺失 | 缺失 | 缺失 | pending |

## 结果进入论文前

- [ ] `multibudget_attack_utility_registry_entry.json` 生成成功
- [ ] 核心文件 SHA-256 已登记
- [ ] 检查全部 WSN 是否仍大于 ownership threshold=2
- [ ] 检查每项 retrieval paired CI 是否包含 0
- [ ] 图由机器输出 JSON 生成，不手工填写未知结果
- [ ] strict-v1 和 aligned sensitivity 未混写

