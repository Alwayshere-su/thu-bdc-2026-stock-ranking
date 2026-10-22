# 方案设计与实验记录

## 1. 问题与指标

每个交易日从沪深300成分股中选出不超过 5 只并等权持有，目标是未来一周收益。模型对当日全部候选股票打分并排序，取分数最高的 5 只。

验证指标 `final_score` 按日计算后取均值：

```text
final_score = (pred_return_sum - random_return_sum) / (max_return_sum - random_return_sum)
```

- `pred_return_sum`：模型 Top-5 的真实收益之和
- `random_return_sum`：随机选 5 只的期望收益（5 × 当日均值）
- `max_return_sum`：当日真实 Top-5 的收益之和（理论上界）

因此 0 表示与随机选择持平，1 表示选中了当日最优 Top-5。

## 2. 基线结构

官方 baseline 是一条 Transformer 排序流水线：

- **输入**：每只股票过去 `sequence_length=60` 个交易日的量价序列；
- **特征**：158 个 Alpha 类特征 + 39 个 TA-Lib 技术指标（合计约 200 维，含去重）；
- **模型**：`StockTransformer`
  - 位置编码 + 2 层 Transformer Encoder 编码单股票时序；
  - `FeatureAttention` 沿时间维聚合；
  - `CrossStockAttention` 在同日股票之间做注意力交互；
  - `score_head` 输出每只股票一个分数。
- **训练**：按日组织排序样本，listwise + pairwise 加权排序损失，验证集 `final_score` 早停；
- **预测**：当日全部分数降序取 Top-5，等权 0.2。

## 3. 本方案的三处改动

### 3.1 横截面特征

同一交易日内，绝对价格量级没有可比性，横截面位置更有信息量。对 13 个基础字段（涨跌幅、换手率、成交额、成交量、return_1/5/10、volume_ratio、volatility_10/20、rsi、macd、atr_14）逐日计算：

- `cs_rank`：当日百分位排名；
- `cs_z`：当日 z-score。

特征维度由约 200 增至 222（含重复列去重后）。

### 3.2 时间序列 OOF

直接用全量训练的 Transformer 分数喂给堆叠模型会泄漏（分数在堆叠训练集上过拟合）。改为按时间顺序切 3 折：

1. 每折只用前面的时间块训练一个 Transformer；
2. 用该模型给后一块打分，得到无偏的 `Transformer_Score`；
3. 把每折的分数与当日截面特征拼成堆叠训练集。

折内只训练 1 个 epoch，目的是提取有泛化性的表示而不是把每折都训练到最优。

### 3.3 LightGBM Ranker 堆叠

- 输入：OOF 阶段的 58 维白名单特征 + `Transformer_Score`（共 59 维）；
- 目标：`lambdarank`，`label_gain=[0,1,3,7,15]`，早停看 `ndcg@5 / ndcg@20`；
- 抗过拟合：`learning_rate=0.01`、`max_depth=5`、`num_leaves=15`、`colsample_bytree=0.6`、`subsample=0.6`、`min_child_samples=50`、`n_estimators=500`；
- 特征白名单包含长周期波动/成交量特征（`WVMA60`、`STD60`、`MIN60`、`BETA60` 等）以及全部截面特征。

堆叠后再按分数取 Top-5，模型文件为 `best_model.pth` + `lgb_stacking_ranker.pkl`。

## 4. 过拟合诊断

参赛期训练日志给出的信号：

```text
Transformer 最佳 epoch: 6
最佳 final_score: 0.1839
LGBM best iteration: 2
valid ndcg@5: 0.178993
valid ndcg@20: 0.206851
```

几个判断：

1. **Transformer 容量大于有效信号**。最佳 epoch 出现在前 10 个 epoch 内，之后验证集连续 10 轮不提升，早停触发。当前 `d_model=256`、2 层、`dim_feedforward=512`、`dropout=0.15` 对约 265 个训练日偏大。
2. **LGBM 在第 2 轮就达到最优**，说明堆叠阶段可用的增量信息有限，主要收益来自 Transformer 分数与少数截面特征。
3. **验证集只有约 26 个交易日**，单日收益波动大，`final_score` 对随机种子和早停点敏感，跨次运行差异明显（见第 5 节）。

据此可以尝试的方向：减小模型宽度/层数、增大 dropout、缩短序列长度、对标签做截面中性化、增加滚动验证的窗口数。

## 5. 实验记录

参赛期保留了多份 notebook，文件名中的数字为该次运行记录的 `final_score`：

| 记录 | 说明 |
| --- | --- |
| 0.0623 | 基线流程首次跑通 |
| 0.0906 | 调整特征与训练参数 |
| 0.1197 | 引入截面特征 |
| 0.1329 | 调整损失权重与序列长度 |
| 0.1525 | 增加特征筛选 |
| **0.1839** | 3 折 OOF + LGBM Ranker 堆叠 |
| 0.0899 | 同一 notebook 的复跑结果（早停点不同） |

0.1839 是参赛期的最好记录；仓库中的 notebook 保留了另一次完整运行的输出（0.0899），用于说明流程可跑通。分数差异主要来自随机种子、早停轮次和验证窗口的选择，不能简单理解为模型能力下降。

## 6. 已知限制

- 验证窗口仅约一个月，不能充分代表全年行情；
- 特征工程依赖 TA-Lib，未做缺失值与停牌日的系统性处理；
- 堆叠特征白名单是在验证集上人工挑选的，存在选择偏差；
- 未考虑交易成本、涨跌停不可成交与流动性约束；
- 结果仅用于竞赛复现，不构成投资建议。
