#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载沪深300成分股的日线数据（Baostock，后复权）。

输出列与 notebook 中的特征工程一致：
    股票代码, 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌额, 换手率, 涨跌幅

用法：
    python data/get_stock_data.py --start 2024-10-01 --end 2026-05-29 \
        --output data/stock_data.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import baostock as bs
import pandas as pd

FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"
NUMERIC = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]
COLUMNS = ["股票代码", "日期", "开盘", "收盘", "最高", "最低",
           "成交量", "成交额", "振幅", "涨跌额", "换手率", "涨跌幅"]


def login() -> None:
    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(f"Baostock 登录失败: {result.error_msg}")
    print("Baostock 登录成功")


def hs300_codes() -> list[tuple[str, str]]:
    result = bs.query_hs300_stocks()
    if result.error_code != "0":
        raise RuntimeError(f"获取沪深300成分股失败: {result.error_msg}")
    codes = []
    while result.error_code == "0" and result.next():
        row = dict(zip(result.fields, result.get_row_data()))
        codes.append((row["code"], row.get("code_name", "")))
    return codes


def fetch(code: str, start: str, end: str) -> pd.DataFrame | None:
    result = bs.query_history_k_data_plus(
        code,
        FIELDS,
        start_date=start,
        end_date=end,
        frequency="d",
        adjustflag="1",  # 后复权
    )
    if result.error_code != "0":
        return None

    rows = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    if not rows:
        return None

    df = pd.DataFrame(rows, columns=result.fields)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["振幅"] = ((df["high"] - df["low"]) / df["preclose"] * 100).round(2)
    df["涨跌额"] = (df["close"] - df["preclose"]).round(2)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["code"] = df["code"].str.replace("sh.", "", regex=False).str.replace("sz.", "", regex=False).str.zfill(6)

    df = df.rename(columns={
        "code": "股票代码",
        "date": "日期",
        "open": "开盘",
        "close": "收盘",
        "high": "最高",
        "low": "最低",
        "volume": "成交量",
        "amount": "成交额",
        "turn": "换手率",
        "pctChg": "涨跌幅",
    })
    return df[COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="下载沪深300成分股日线数据")
    parser.add_argument("--start", default="2024-10-01", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="2026-05-29", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--output", default="data/stock_data.csv", help="输出 CSV 路径")
    parser.add_argument("--sleep", type=float, default=0.05, help="每只股票之间的间隔秒数")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    login()
    try:
        codes = hs300_codes()
        print(f"沪深300成分股: {len(codes)} 只，区间 {args.start} 至 {args.end}")

        frames, failed = [], []
        for i, (code, name) in enumerate(codes, 1):
            df = fetch(code, args.start, args.end)
            if df is None:
                failed.append(code)
                print(f"[{i}/{len(codes)}] {code} {name} 下载失败")
            else:
                frames.append(df)
                print(f"[{i}/{len(codes)}] {code} {name} {len(df)} 行")
            time.sleep(args.sleep)
    finally:
        bs.logout()

    if not frames:
        raise SystemExit("没有下载到任何数据")

    result = pd.concat(frames, ignore_index=True).sort_values(["日期", "股票代码"])
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"已写出 {output}: {len(result)} 行, {result['股票代码'].nunique()} 只股票")
    if failed:
        print(f"失败 {len(failed)} 只: {', '.join(failed)}")


if __name__ == "__main__":
    main()
