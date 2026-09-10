"""非理想性敏感性实验（run_nonideality_sensitivity.py）。

实验目的：用"合成向量"（随机生成的 x、w，做点积）模拟三类非理想性——
(1) 输出噪声（高斯噪声，按 LSB 计量）、(2) 相邻通道串扰（crosstalk，
光子阵列通道间信号泄漏）、(3) 均匀量化（低 bit 位宽）——统计相对误差的
均值/中位数/95 分位/最大值，评估每类非理想性对乘法精度的影响程度。
注意：这是数学冒烟测试（sanity check），不是 MobileViT 真实精度。

- 输入：hpat_experiment_config.json（含非理想性参数与随机种子）。
- 产出（--output-dir 下）：nonideality_synthetic_sensitivity.csv、
  nonideality_sensitivity_manifest.json。
- 命令：python run_nonideality_sensitivity.py --output-dir <目录> [--config <配置>]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import statistics
from typing import Any

from _common import REPO_ROOT, base_manifest, ensure_dir, load_json, relative, sha256_file, write_csv, write_json


def dot(a: list[float], b: list[float]) -> float:
    """计算两个等长向量的点积（逐元素相乘后求和）。"""
    return sum(x * y for x, y in zip(a, b))


def roll_one(x: list[float]) -> list[float]:
    """把向量循环右移一位，用于模拟"相邻通道"的信号串扰来源。

    例：roll_one([1,2,3]) == [3,1,2]。
    """
    if not x:
        return x
    return [x[-1]] + x[:-1]


def quantize_unit(x: float, bits: int) -> float:
    """把 [-1,1] 内的浮点数均匀量化到 bits 位。

    步骤：先裁剪到 [-1,1]，映射到 [0,2^bits-1] 的整数刻度取整，
    再还原回 [-1,1]。相当于 ADC/DAC 的有限精度表示。
    """
    levels = (1 << bits) - 1
    clipped = min(1.0, max(-1.0, x))
    q = round((clipped + 1.0) * levels / 2.0)
    return (2.0 * q / levels) - 1.0


def summarize(values: list[float]) -> dict[str, float]:
    """汇总一组误差：均值、中位数、95 分位、最大值。

    参数 values：数值列表。
    返回：{"mean", "median", "p95", "max"} 字典。
    """
    ordered = sorted(values)
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": p95,
        "max": max(values),
    }


def run(output_dir: pathlib.Path, config_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """执行非理想性敏感性主流程。

    参数：
        output_dir：结果输出目录。
        config_path：实验配置路径。
    返回：产出文件路径字典（csv / manifest）。
    """
    ensure_dir(output_dir)
    config = load_json(config_path)
    # 固定随机种子，保证结果可复现
    rng = random.Random(int(config["seed"]))
    cfg = config["nonideality"]
    trials = int(cfg["trials"])
    eps = 1e-9  # 防止除零：分母极小时用 eps 兜底
    rows: list[dict[str, Any]] = []

    # 三重循环：维度 dim × 非理想性种类 effect × 扫描取值 value
    for dim in cfg["dimensions"]:
        for effect_name, sweep_key in [
            ("gaussian_output_noise_lsb", "noise_lsb"),   # 输出噪声：按 LSB 数扫描
            ("adjacent_channel_crosstalk", "crosstalk_alpha"),  # 串扰：按混合系数 alpha 扫描
            ("uniform_quantization", "quant_bits"),      # 量化：按 bit 数扫描
        ]:
            for value in cfg[sweep_key]:
                errors: list[float] = []
                for _ in range(trials):
                    # 每次随机生成输入向量 x 与权重向量 w，计算理想点积 y
                    x = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
                    w = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
                    y = dot(x, w)
                    # 按非理想性种类构造"受污染"的 y_hat：
                    if effect_name == "gaussian_output_noise_lsb":
                        # 噪声幅度以输出 8bit 的 1 LSB 为基准缩放，乘以扫描系数
                        scale = max(abs(y), 1.0) / 255.0
                        y_hat = y + rng.gauss(0.0, float(value) * scale)
                    elif effect_name == "adjacent_channel_crosstalk":
                        # 输入向量与"循环移位后的自己"按 alpha 混合，模拟邻道泄漏
                        alpha = float(value)
                        xr = roll_one(x)
                        mixed = [(1.0 - alpha) * a + alpha * b for a, b in zip(x, xr)]
                        y_hat = dot(mixed, w)
                    else:
                        # 输入与权重都量化到 bits 位后再点积
                        bits = int(value)
                        xq = [quantize_unit(v, bits) for v in x]
                        wq = [quantize_unit(v, bits) for v in w]
                        y_hat = dot(xq, wq)
                    # 相对误差：|y_hat - y| / |y|（分母加 eps 防除零）
                    errors.append(abs(y_hat - y) / max(abs(y), eps))
                stats = summarize(errors)
                rows.append(
                    {
                        "effect": effect_name,
                        "dimension": dim,
                        "sweep_value": value,
                        "trials": trials,
                        "mean_relative_error": f"{stats['mean']:.8f}",
                        "median_relative_error": f"{stats['median']:.8f}",
                        "p95_relative_error": f"{stats['p95']:.8f}",
                        "max_relative_error": f"{stats['max']:.8f}",
                        "evidence_label": "smoke / analytical synthetic vector sensitivity",
                    }
                )

    out_csv = output_dir / "nonideality_synthetic_sensitivity.csv"
    manifest_path = output_dir / "nonideality_sensitivity_manifest.json"
    write_csv(
        out_csv,
        rows,
        [
            "effect",
            "dimension",
            "sweep_value",
            "trials",
            "mean_relative_error",
            "median_relative_error",
            "p95_relative_error",
            "max_relative_error",
            "evidence_label",
        ],
    )
    manifest = base_manifest("nonideality_synthetic_sensitivity", "smoke / analytical synthetic vector sensitivity")
    manifest.update(
        {
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "outputs": [relative(out_csv)],
            "seed": config["seed"],
            # 免责说明：合成向量级冒烟测试，不等于 MobileViT 精度或真实器件鲁棒性
            "promotion_note": "This is a synthetic vector sanity check, not MobileViT accuracy and not fabricated-device robustness.",
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this is a short analytical smoke run.",
        }
    )
    write_json(manifest_path, manifest)
    return {"csv": out_csv, "manifest": manifest_path}


def main() -> None:
    """命令行入口：解析参数并调用 run()。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json"))
    args = parser.parse_args()
    outputs = run(pathlib.Path(args.output_dir), pathlib.Path(args.config))
    print(json.dumps({k: relative(v) for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
