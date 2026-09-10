"""汇总"成对预测迁移"统计（paired prediction transitions）。

本脚本做什么：
    输入是一份逐样本（sample）级别的诊断 CSV：每个样本记录它在干净输入和
    扰动输入（perturbed，例如噪声/攻击后的输入）下的 top-1 预测是否正确，
    以及预测是否发生变化（changed）。本脚本把这些细粒度数据按
    （变体、扰动效应、扫描变量、扫描取值、随机种子）分组成"试验"，
    再按前四维聚合成一个"成对预测迁移"统计表，回答：加了扰动后，
    有多少样本从"对"变成"错"（新错误）、从"错"变成"对"（恢复）、
    净下降多少、预测整体变化率多高。

数据从哪来：
    --input 指定的逐样本 CSV（例如配对扰动诊断实验的输出）。

产出什么：
    --output 指定的汇总 CSV，每行对应一个（变体/效应/变量/取值）组合，
    含均值与 P05/P50/P95 分位数（多个随机种子的分布）、种子列表、
    指标定义与 claim_boundary（结论边界，说明这只是 Imagenette-160 子集
    上的配对诊断，不是全量 ImageNet 或真实设备鲁棒性）。

怎么运行：
    python experiments/scripts/summarize_paired_prediction_transitions.py \
        --input <逐样本CSV> --output <汇总CSV>
    若 --output 落在仓库 tables/ 内，需显式设置 HPAT_WRITE_PROJECT_TABLES=1。
"""

from __future__ import annotations

import argparse
import csv
import pathlib
from collections import defaultdict
from typing import Any

from _common import is_project_table_path, project_writes_enabled, write_csv


# 汇总表固定的列顺序：先放分组维度，再放各统计量（均值 + 分位数），最后是元信息
FIELDS = [
    "variant",
    "effect",
    "sweep_variable",
    "sweep_value",
    "trial_count",
    "sample_count",
    "new_error_rate_mean",
    "new_error_rate_p05",
    "new_error_rate_p50",
    "new_error_rate_p95",
    "recovery_rate_mean",
    "recovery_rate_p05",
    "recovery_rate_p50",
    "recovery_rate_p95",
    "net_top1_drop_mean",
    "prediction_change_rate_mean",
    "seeds",
    "metric_definition",
    "claim_boundary",
]


def quantile(values: list[float], q: float) -> float:
    """计算一份数值列表的分位数（线性插值法）。

    参数：
        values: 原始数值列表（可乱序）。
        q: 分位数位置，0.0（最小）到 1.0（最大）之间。
    返回：
        对应分位数；列表为空时返回 0.0。
    """
    ordered = sorted(values)
    if not ordered:
        return 0.0
    # 目标位置：把数据长度映射到 0..len-1，再在相邻两个样本间线性插值
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    # 加权平均：下界取 (1-fraction) 权重，上界取 fraction 权重
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(input_path: pathlib.Path) -> list[dict[str, Any]]:
    """把逐样本 CSV 汇总为"成对预测迁移"统计表。

    处理步骤：
        1) 按（变体、效应、扫描变量、扫描取值、随机种子）把样本分成"试验"；
        2) 每个试验计算新错误率、恢复率、净下降、变化率（百分比）；
        3) 同一（变体/效应/变量/取值）下的多个种子试验再求均值与 P05/P50/P95
           分位数，得到最终一行汇总。

    参数：
        input_path: 逐样本 CSV 路径，至少需要列 clean_correct_top1、
            perturbed_correct_top1、changed、trial_seed、variant、effect、
           sweep_variable、sweep_value。
    返回：
        汇总行字典列表，可直接交给 write_csv 落盘。
    """
    # 第一层分组：把每个随机种子下的样本聚成一"试验"
    grouped: dict[tuple[str, str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    with input_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["variant"],
                row["effect"],
                row["sweep_variable"],
                row["sweep_value"],
                int(row["trial_seed"]),
            )
            grouped[key].append(row)

    # 第二层：每个试验只算"有效样本"（干净和扰动两列都有标注）的四类统计量
    by_severity: dict[tuple[str, str, str, str], list[dict[str, float | int]]] = defaultdict(list)
    for (variant, effect, variable, value, seed), rows in grouped.items():
        # 只保留两列标注都非空的样本，作为统计分母
        labelled = [row for row in rows if row["clean_correct_top1"] != "" and row["perturbed_correct_top1"] != ""]
        denominator = max(len(labelled), 1)  # 避免除零，空组按 1 处理
        # 新错误：干净时对、扰动后错；恢复：干净时错、扰动后对；变化：预测被翻转
        new_errors = sum(int(row["clean_correct_top1"]) == 1 and int(row["perturbed_correct_top1"]) == 0 for row in labelled)
        recoveries = sum(int(row["clean_correct_top1"]) == 0 and int(row["perturbed_correct_top1"]) == 1 for row in labelled)
        changes = sum(int(row["changed"]) for row in labelled)
        # 按"严重度"维度（不含种子）记录该种子试验的百分比统计
        by_severity[(variant, effect, variable, value)].append(
            {
                "seed": seed,
                "sample_count": len(labelled),
                "new_error_rate": 100.0 * new_errors / denominator,
                "recovery_rate": 100.0 * recoveries / denominator,
                "net_drop": 100.0 * (new_errors - recoveries) / denominator,
                "change_rate": 100.0 * changes / denominator,
            }
        )

    output: list[dict[str, Any]] = []
    # 第三层：把同组合的多个种子试验汇总成一行，按 变体->效应->扫描取值 排序
    for (variant, effect, variable, value), trials in sorted(
        by_severity.items(), key=lambda item: (item[0][0], item[0][1], float(item[0][3]))
    ):
        new_errors = [float(trial["new_error_rate"]) for trial in trials]
        recoveries = [float(trial["recovery_rate"]) for trial in trials]
        net_drops = [float(trial["net_drop"]) for trial in trials]
        changes = [float(trial["change_rate"]) for trial in trials]
        output.append(
            {
                "variant": variant,
                "effect": effect,
                "sweep_variable": variable,
                "sweep_value": value,
                "trial_count": len(trials),
                "sample_count": int(trials[0]["sample_count"]),
                # 均值 = 各种子试验的简单平均；分位数反映跨种子的波动范围
                "new_error_rate_mean": f"{sum(new_errors) / len(new_errors):.4f}",
                "new_error_rate_p05": f"{quantile(new_errors, 0.05):.4f}",
                "new_error_rate_p50": f"{quantile(new_errors, 0.50):.4f}",
                "new_error_rate_p95": f"{quantile(new_errors, 0.95):.4f}",
                "recovery_rate_mean": f"{sum(recoveries) / len(recoveries):.4f}",
                "recovery_rate_p05": f"{quantile(recoveries, 0.05):.4f}",
                "recovery_rate_p50": f"{quantile(recoveries, 0.50):.4f}",
                "recovery_rate_p95": f"{quantile(recoveries, 0.95):.4f}",
                "net_top1_drop_mean": f"{sum(net_drops) / len(net_drops):.4f}",
                "prediction_change_rate_mean": f"{sum(changes) / len(changes):.4f}",
                # 列出参与统计的全部种子，便于复现
                "seeds": ",".join(str(int(trial["seed"])) for trial in trials),
                "metric_definition": "percentage of all labelled samples that are clean-correct and perturbed-wrong; recovery and net-drop columns retained",
                "claim_boundary": "Imagenette-160 subset paired diagnostic; not full ImageNet or measured-device robustness",
            }
        )
    return output


def main() -> None:
    """命令行入口：解析参数、做写仓库保护检查，然后汇总并落盘。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    # 安全护栏：禁止在未显式授权时覆盖仓库正式表，防止误写污染正式数据
    if is_project_table_path(output_path) and not project_writes_enabled():
        parser.error(
            "writing repository tables requires explicit HPAT_WRITE_PROJECT_TABLES=1; "
            "choose an isolated --output path"
        )
    rows = summarize(input_path)
    write_csv(output_path, rows, FIELDS)
    print(output_path)


if __name__ == "__main__":
    # 直接执行本文件时调用命令行入口
    main()
