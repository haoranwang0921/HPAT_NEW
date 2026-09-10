"""规模扩展性（scalability）分析：看 HPAT 硬件配置加大后性能怎么变。

背景：HPAT 是光子张量处理器，其"算力"可以通过增加波长数、PDPU
（光点积单元）组数、Tile 数来扩展。本文件用一个解析模型（纯公式，
不跑真实仿真）估算：当这些硬件规模参数变化时，延迟（latency）、
能耗（energy）、利用率（utilization）以及瓶颈环节（bottleneck）如何
变化，从而回答"哪一项硬件资源最卡性能"这类问题。
"""

from __future__ import annotations

from typing import Any

from .mobilevit_loader import primary_token_count, variants_from_config


def scalability_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """按配置生成规模扩展性分析表（每行一种硬件配置组合）。

    模型要点（均为简化解析估计）：
    - 光计算时长 = 总操作数 / (光并行度 × 每周期 25 万次操作)；
    - ADC 时长 = 采样行数 / ADC 并行度 × 每样本 0.25 ns；
    - 内存时长 = 流量比特 / 内存带宽；
    - 热校准时长 = 40 ns × 温度系数 × 组数修正；
    - 总延迟 = 上述四项之和，取最大的一项为"瓶颈"；
    - 能耗倍数 = 1 + 各项硬件规模参数的加权系数（规模越大能耗越高）。

    :param config: 顶层实验配置字典（读 scalability / energy_envelope_mj / qkv）。
    :return: 每行一个 dict 的扩展性分析表。
    """
    # 六个硬件规模参数的待扫取值列表（缺省即扫描默认值）
    scfg = config.get("scalability", {})
    wavelengths = scfg.get("wavelengths", [8, 16, 32])              # 波长数（光并行通道）
    pdpu_banks = scfg.get("pdpu_banks", [1, 2, 4])                  # PDPU 点积单元组数
    tiles = scfg.get("tiles", [1, 2])                               # Tile 数
    adc_parallelism = scfg.get("adc_parallelism", [8, 16, 32])      # ADC 并行度
    memory_bandwidth = scfg.get("memory_bandwidth_gbps", [64, 128]) # 内存带宽（Gbps）
    thermal_multipliers = scfg.get("thermal_calibration_multipliers", [1.0, 1.5])  # 热校准时间系数
    rows: list[dict[str, Any]] = []
    for variant in variants_from_config(config):
        d = int(variant["embedding_dim"])   # 嵌入维度
        n = primary_token_count(variant, config)  # token 数量
        # 基准能耗（从配置的 energy_envelope 里取该变体的值）
        base_energy = float(config.get("energy_envelope_mj", {}).get(variant["variant"], 1.0))
        # 一次前向的总操作数 ≈ Q/K/V 投影 + 注意力矩阵乘 + 输出投影
        base_ops = 3 * n * d * d + 2 * n * n * d + 2 * n * d * (2 * d)
        # 数据搬运比特数 ≈ (4·n·d 个激活 + 3·d·d 个权重) × 位宽
        traffic_bits = (4 * n * d + 3 * d * d) * int(config.get("qkv", {}).get("default_bit_width", 8))
        # 六重循环：穷举全部硬件规模组合
        for w in wavelengths:
            for bank in pdpu_banks:
                for tile in tiles:
                    for adc in adc_parallelism:
                        for bw in memory_bandwidth:
                            for thermal in thermal_multipliers:
                                # 光计算并行度 = 波长 × 组数 × Tile 数（至少为 1，防除零）
                                optical_parallel = max(int(w) * int(bank) * int(tile), 1)
                                # 光计算时长：每秒可做 25 万次（250000.0）× 并行度 次操作
                                compute_ns = base_ops / (optical_parallel * 250000.0)
                                # ADC 采样时长：每 0.25 ns 采一个样本
                                adc_ns = (4 * n * d) / max(int(adc), 1) * 0.25
                                # 内存搬运时长：比特数 / 带宽（×1e9 使秒与纳秒单位对齐）
                                memory_ns = traffic_bits / (float(bw) * 1e9) * 1e9
                                # 热校准时长：40 ns 基线 × 温度系数 ×（bank/2 向上取整）修正
                                thermal_ns = 40.0 * float(thermal) * max(1, bank / 2)
                                latency_ns = compute_ns + adc_ns + memory_ns + thermal_ns
                                # 利用率 = 实际可完成的算力 / 该延迟内硬件能提供的算力（上限 1）
                                utilization = min(1.0, base_ops / (optical_parallel * 250000.0 * max(latency_ns, 1e-9)))
                                # 瓶颈 = 四项时长中最大的一项，报告给读者
                                bottleneck = max(
                                    [
                                        ("optical_compute", compute_ns),
                                        ("adc_dac_parallelism", adc_ns),
                                        ("memory_bandwidth", memory_ns),
                                        ("thermal_calibration", thermal_ns),
                                    ],
                                    key=lambda item: item[1],  # 按时长取最大
                                )[0]
                                # 能耗倍数：硬件规模越大，能耗越高（线性近似）
                                energy_multiplier = 1.0 + 0.015 * int(w) + 0.06 * int(bank) + 0.04 * int(tile)
                                energy_multiplier += 0.08 * (float(thermal) - 1.0)  # 热校准系数的影响
                                rows.append(
                                    {
                                        "variant": variant["variant"],
                                        "n_tokens": n,
                                        "embedding_dim": d,
                                        "wavelengths": int(w),
                                        "pdpu_banks": int(bank),
                                        "tiles": int(tile),
                                        # MRR 阵列规模的"代理指标"：波长 × 维度 × 组数
                                        "mrr_array_size_proxy": int(w) * d * int(bank),
                                        "adc_parallelism": int(adc),
                                        "memory_bandwidth_gbps": float(bw),
                                        "thermal_calibration_multiplier": float(thermal),
                                        "latency_estimate_ns": f"{latency_ns:.6f}",
                                        "energy_estimate_mj": f"{base_energy * energy_multiplier:.6f}",
                                        "utilization_proxy": f"{utilization:.6f}",
                                        "bottleneck": bottleneck,
                                        "evidence_label": "modelled scalability sensitivity",
                                    }
                                )
    return rows


# 扩展性分析表的列名清单（顺序即 CSV 表头顺序），供下游写文件时复用
SCALABILITY_FIELDS = [
    "variant",
    "n_tokens",
    "embedding_dim",
    "wavelengths",
    "pdpu_banks",
    "tiles",
    "mrr_array_size_proxy",
    "adc_parallelism",
    "memory_bandwidth_gbps",
    "thermal_calibration_multiplier",
    "latency_estimate_ns",
    "energy_estimate_mj",
    "utilization_proxy",
    "bottleneck",
    "evidence_label",
]
