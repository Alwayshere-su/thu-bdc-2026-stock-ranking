# 结果说明

仓库不包含行情数据、模型权重与逐日提交文件，原因是体积较大且行情数据受数据源条款约束。这里记录参赛期的成绩与产出位置，便于复现后对照。

## 成绩

| 运行 | `final_score` | 说明 |
| --- | --- | --- |
| 个人最好记录 | **0.1839** | 3 折 OOF + LightGBM Ranker 堆叠版本，参赛期提交 |
| 仓库 notebook 复跑 | 0.0899 | 同一流程的另一次完整运行，输出保留在 notebook 中 |

验证集约为最后一个月（26 个交易日），单日收益波动大，分数对随机种子与早停点敏感。

## 复现后的产出

运行 notebook 后会在 `model/60_158+39/` 下生成：

| 文件 | 内容 |
| --- | --- |
| `best_model.pth` | 验证集 `final_score` 最优的 Transformer 权重 |
| `scaler.pkl` | 特征标准化器 |
| `lgb_stacking_ranker.pkl` | LightGBM LambdaRank 堆叠模型 |
| `config.json` | 本次运行的超参数快照 |
| `final_score.txt` | 最佳 epoch 与分数 |
| `log/` | TensorBoard 日志 |

`model/` 已在 `.gitignore` 中，不会进入版本控制。

## 提交文件格式

取消 notebook 末尾预测单元格的注释后，`save_predictions` 会输出当日 Top-5：

```csv
stock_id,weight
600000,0.2
000001,0.2
...
```
