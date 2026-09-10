"""QKV 注意力流量（traffic）分析：量化自注意力中的数据搬运开销。

背景：Transformer 自注意力的 Q（查询）、K（键）、V（值）计算都要
读写权重矩阵和中间结果。在光子芯片上，这些数据搬运（含权重编程、
总线传输）的开销往往比光计算本身更值得关注。本文件用解析模型
（纯公式推导，不跑仿真）估算不同配置下的比特级流量，输出一张
"流量敏感度分析表"（QKV_TRAFFIC_FIELDS 定义列名）。
"""

from __future__ import annotations

from typing import Any

from .mobilevit_loader import primary_token_count, variants_from_config


def qkv_traffic_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """按配置生成 QKV 注意力流量分析表（每行一种配置组合）。

    流量构成（都是"比特数"）：
    - input_bits：输入激活的比特数 = token 数 × 嵌入维度 × 输入位宽；
    - qkv_output_bits：Q/K/V 三个投影输出的比特数（≈ 3 份 input_bits）；
    - weight_bits：Q/K/V 三组权重矩阵的总比特数 = 3 × d × d × 位宽；
    - programming_bits：把权重"编程"进 MRR 微环所需的额外比特
      （只有 reprogrammed 模式下才非零，resident 模式权重常驻不重写）；
    - weight_stream_bits：从外存"流式"搬权重进芯片的比特
      （仅 stream_weights=True 时非零）；
    - bus_bits：总线上搬运量 = (输入+输出+权重流) × bus_multiplier。

    :param config: 顶层实验配置字典（读 qkv / bit_width_sweep / mobilevit_variants）。
    :return: 每行一个 dict 的流量分析表。
    """
    rows: list[dict[str, Any]] = []
    qcfg = config.get("qkv", {})
    # 权重模式：resident=权重常驻片上（0 次编程）；reprogrammed=每层重新编程（1 次）
    modes = qcfg.get("weight_modes") or [
        {"name": "resident", "programming_events": 0, "stream_weights": False, "reuse_interval": "layer"},
        {"name": "reprogrammed", "programming_events": 1, "stream_weights": False, "reuse_interval": "layer"},
    ]
    bus_multiplier = float(qcfg.get("bus_multiplier", 1.0))  # 总线流量放大系数（1=理想）
    for variant in variants_from_config(config):
        d = int(variant["embedding_dim"])  # 嵌入维度
        token_sweep = variant.get("token_count_sweep") or [primary_token_count(variant, config)]
        # 三层嵌套循环：token 数 × 位宽 × 权重模式，穷举所有配置组合
        for n_tokens in token_sweep:
            for bit_width in config.get("bit_width_sweep", [8, 10, 12]):
                for mode in modes:
                    b_a = b_o = b_w = int(bit_width)  # 输入/输出/权重位宽取同一值（简化假设）
                    input_bits = int(n_tokens) * d * b_a            # 输入激活流量
                    qkv_output_bits = 3 * int(n_tokens) * d * b_o   # QKV 三路输出流量
                    weight_bits = 3 * d * d * b_w                   # 三组权重矩阵比特数
                    # 编程流量 = 每次编程都要把整个权重写一遍
                    programming_bits = int(mode.get("programming_events", 0)) * weight_bits
                    weight_stream_bits = weight_bits if mode.get("stream_weights", False) else 0
                    bus_bits = int((input_bits + qkv_output_bits + weight_stream_bits) * bus_multiplier)
                    total_bits = input_bits + qkv_output_bits + programming_bits + weight_stream_bits + bus_bits
                    rows.append(
                        {
                            "variant": variant["variant"],
                            "d": d,
                            "n_tokens": int(n_tokens),
                            "bit_width": bit_width,
                            "weight_mode": mode["name"],
                            "reuse_interval": mode.get("reuse_interval", ""),
                            "input_bits": input_bits,
                            "qkv_output_bits": qkv_output_bits,
                            "weight_stream_bits": weight_stream_bits,
                            "programming_bits": programming_bits,
                            "bus_bits": bus_bits,
                            "total_bits": total_bits,
                            "total_kib": f"{total_bits / 8.0 / 1024.0:.4f}",  # 折算成 KiB（1 字节=8 比特）
                            # 编程流量占比：衡量"重写权重"这项开销在总流量中的权重
                            "programming_share_percent": f"{100.0 * programming_bits / total_bits if total_bits else 0.0:.4f}",
                            "evidence_label": "analytical/modelled Q/K/V traffic sensitivity",
                        }
                    )
    return rows


# 流量分析表的列名清单（顺序即 CSV 表头顺序），供下游写文件时复用
QKV_TRAFFIC_FIELDS = [
    "variant",
    "d",
    "n_tokens",
    "bit_width",
    "weight_mode",
    "reuse_interval",
    "input_bits",
    "qkv_output_bits",
    "weight_stream_bits",
    "programming_bits",
    "bus_bits",
    "total_bits",
    "total_kib",
    "programming_share_percent",
    "evidence_label",
]
