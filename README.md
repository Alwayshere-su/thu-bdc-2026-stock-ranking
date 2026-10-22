# 沪深300 排序选股：Transformer + LightGBM Ranker

> 2026 中国高校计算机大赛·大数据挑战赛（THU-BDC 2026）参赛方案

[![CI](https://github.com/Alwayshere-su/thu-bdc-2026-stock-ranking/actions/workflows/ci.yml/badge.svg)](https://github.com/Alwayshere-su/thu-bdc-2026-stock-ranking/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue.svg)](https://www.python.org/)

## 简介

赛题要求在每个交易日从沪深300成分股中选出不超过 5 只股票并分配权重，使未来一周的收益尽可能高。本项目把它建模为**同日横截面排序**问题：输入每只股票过去 60 个交易日的量价与技术特征，输出当日分数，按分数取前 5 只等权持有。

方案在赛事官方 Transformer 排序基线的基础上做了三处改动：

1. 加入横截面特征（当日收益、换手、波动等指标的 rank 与 z-score）；
2. 用按时间切分的 3 折 OOF 预测生成 Transformer 的无偏分数；
3. 用 LightGBM LambdaRank 对「截面特征 + Transformer 分数」做二次排序，并小学习率、浅树、行列采样与早停控制过拟合。

## 方法

| 阶段 | 内容 |
| --- | --- |
| 特征 | 158 个 Alpha 类特征 + 39 个 TA-Lib 技术指标；训练/验证分别做截面 rank 与 z-score |
| 标签 | `label = (open_t5 - open_t1) / open_t1`，即次一交易日开盘到第 5 个交易日开盘的收益 |
| 样本 | 按日期组织：一天一个样本，样本内是当日全部可用股票的 60 日特征序列，padding + mask 处理股票数不一致 |
| 主模型 | `StockTransformer`：时序 Transformer 编码 + 时间维特征注意力 + 同日股票间 Cross-Stock Attention，输出每只股票一个排序分数 |
| 损失 | `WeightedRankingLoss`：listwise 加权交叉熵（真实 Top-5 权重 3），pairwise 权重设为 0；AdamW + 线性学习率衰减 |
| 堆叠 | 3 折时间 OOF 分数与 58 维精选特征拼接，训练 `LGBMRanker`（lambdarank，ndcg@5/20 早停） |
| 产出 | 验证集 `final_score` 最优的 Transformer 权重 + LGBM ranker；预测时取 Top-5 等权 |

评估指标 `final_score` 衡量预测 Top-5 相对随机选择的超额收益，并用当日理论最优 Top-5 归一化：

```text
final_score = (pred_return_sum - random_return_sum) / (max_return_sum - random_return_sum)
```

## 结果

| 项 | 数值 |
| --- | --- |
| 个人最好成绩（赛事评分口径） | **0.1839** |
| 项目内记录的一次完整运行 | 0.0899（见 notebook 输出；分数对随机种子与早停较敏感） |
| 模型规模 | 约 152 万参数（`d_model=256`，2 层，4 头） |
| 训练/验证 | 265 / 26 个交易日样本，平均每日约 300 只股票 |

> 0.1839 来自参赛期的历史运行记录；仓库中的 notebook 保留了另一次完整运行的输出（0.0899）。复跑时以实际输出为准。

## 项目结构

```text
.
├── notebooks/
│   └── transformer_lgbm_stacking.ipynb   # 完整流程：特征 → 训练 → OOF → LGBM 堆叠
├── docs/
│   └── approach.md                        # 设计取舍、过拟合诊断与实验记录
├── data/
│   └── get_stock_data.py                  # Baostock 下载沪深300日线
├── results/
│   └── README.md                          # 成绩记录与复现说明
├── requirements.txt
└── .github/workflows/ci.yml               # 校验 notebook JSON 与代码可解析
```

## 快速开始

### 1. 环境

特征工程依赖 TA-Lib 的 C 库，需先安装系统库再装 Python 包：

```bash
# macOS
brew install ta-lib
# Ubuntu
sudo apt-get install build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib && ./configure --prefix=/usr && make && sudo make install && cd ..
# Windows：下载 ta-lib 预编译 whl 后 pip install
```

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 数据

仓库不含行情数据。用 Baostock 下载沪深300成分股日线（后复权）：

```bash
python data/get_stock_data.py --start 2024-10-01 --end 2026-05-29 --output data/stock_data.csv
```

输出列：`股票代码, 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌额, 换手率, 涨跌幅`。

### 3. 训练

打开 `notebooks/transformer_lgbm_stacking.ipynb`，确认 `config['data_path']` 指向 `./data` 后从头运行。主程序会依次完成特征工程、Transformer 训练、OOF 与 LightGBM 堆叠，模型与 `final_score.txt` 写入 `model/60_158+39/`。

如需生成提交文件，取消 notebook 末尾 `06模型预测` 一节的注释，调用 `predict_top_stocks` / `save_predictions` 输出 Top-5。

> 完整训练在单张消费级 GPU 上约十几分钟；CPU 也可运行，时间相应变长。

## 说明

- 赛题与数据版权归赛事主办方所有；本仓库不包含行情数据、训练权重与赛事附件。
- 模型结构参考赛事官方 baseline（[THU-BDC2026](https://github.com/Sherlock1956/THU-BDC2026)），本仓库只收录在此基础上由作者实现的改进与实验代码。
- 本项目仅用于竞赛与技术交流，不构成任何投资建议。

## 许可证

[MIT](LICENSE) © 2026 Alwayshere-su
