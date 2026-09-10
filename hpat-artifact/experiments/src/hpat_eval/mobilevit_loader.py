"""MobileViT 模型规格加载与设备选择工具。

背景：论文实验主要以 MobileViT（一种轻量级视觉 Transformer，常用于
移动端/边缘端图像分类）系列模型作为测试负载。本文件负责：
1) 维护内置的 MobileViT 变体规格表（分辨率、嵌入维度等）；
2) 把用户配置文件（config）里写的变体参数解析成统一规格；
3) 自动选择计算设备（MPS/CUDA/CPU）并给出选择原因；
4) 汇总多次推理的延迟统计（均值/中位数/分位数等）。
"""

from __future__ import annotations

import importlib.util
import statistics
from typing import Any


# ---------------------------------------------------------------------------
# 内置 MobileViT 变体规格表（缺省值）
# ---------------------------------------------------------------------------
# 每个变体记录 4 个关键参数：
# - variant: 变体名称；
# - timm_model: 在 timm 模型库中对应的模型名（用于真正加载网络权重）；
# - input_resolution: 输入图像边长（像素）；
# - embedding_dim: 嵌入向量维度 d（QKV 注意力中每个 token 的特征维数）。
DEFAULT_MODEL_SPECS = [
    {
        "variant": "MobileViT-XXS",
        "timm_model": "mobilevit_xxs",
        "input_resolution": 192,
        "embedding_dim": 128,
    },
    {
        "variant": "MobileViT-XS",
        "timm_model": "mobilevit_xs",
        "input_resolution": 224,
        "embedding_dim": 192,
    },
    {
        "variant": "MobileViT-S",
        "timm_model": "mobilevit_s",
        "input_resolution": 256,
        "embedding_dim": 256,
    },
]


def available(name: str) -> bool:
    """判断某个 Python 包是否已安装（可导入）。

    用 importlib.util.find_spec 探测而不真正导入，避免触发包的初始化副作用。

    :param name: 包名（如 "torch"、"timm"）。
    :return: True 表示可导入。
    """
    return importlib.util.find_spec(name) is not None


def missing_packages(names: list[str]) -> list[str]:
    """从名字列表中筛出"尚未安装"的那些包。

    :param names: 待检查的包名列表。
    :return: 缺失包名列表。
    """
    return [name for name in names if not available(name)]


def variants_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    """从用户配置文件解析出要跑的 MobileViT 变体列表。

    解析逻辑：
    1. 配置里通常写 "mobilevit_variants": [{"name": "MobileViT-XS", ...}]；
    2. 每个条目先以内置规格表为底，再用配置里的值覆盖；
    3. 配置没写任何变体时，退回全部内置规格（DEFAULT_MODEL_SPECS）。

    :param config: 顶层实验配置字典。
    :return: 规范化后的变体规格列表（每项都是 dict）。
    """
    rows: list[dict[str, Any]] = []
    by_name = {row["variant"]: row for row in DEFAULT_MODEL_SPECS}  # 名字 -> 内置规格
    for item in config.get("mobilevit_variants", []):
        base = dict(by_name.get(item["name"], {}))  # 以内置规格为默认值
        base.update(
            {
                "variant": item["name"],
                "timm_model": item.get("timm_model", base.get("timm_model", "")),
                "input_resolution": int(item.get("input_resolution", base.get("input_resolution", 224))),
                # d 是嵌入维度；配置里可简写为 "d"，也可全写 embedding_dim
                "embedding_dim": int(item.get("d", item.get("embedding_dim", base.get("embedding_dim", 128)))),
                # 要扫描的 token 数量列表（如 [196, 784]，对应不同图像 patch 数）
                "token_count_sweep": list(item.get("token_count_sweep", [196])),
                "draft_group": item.get("draft_group", ""),          # 论文草稿分组标签（可选）
                "parameter_count_m": item.get("parameter_count_m", ""),  # 参数量（百万），可选
            }
        )
        rows.append(base)
    return rows or DEFAULT_MODEL_SPECS  # 配置为空则全部用内置规格


def primary_token_count(variant: dict[str, Any], config: dict[str, Any]) -> int:
    """决定该变体"主要的" token 数量（用于默认算例）。

    优先级：配置里显式指定的 primary_token_count > token_count_sweep 里
    恰好有 196（MobileViT 默认 patch 布局的常见值）> 取 sweep 中间值。

    :param variant: 变体规格字典。
    :param config: 顶层配置字典。
    :return: 主 token 数量（整数）。
    """
    configured = config.get("qkv", {}).get("primary_token_count")
    if configured:
        return int(configured)
    sweep = variant.get("token_count_sweep") or [196]
    if 196 in sweep:
        return 196
    return int(sweep[len(sweep) // 2])  # sweep 无 196 时取中间位置的 token 数


def select_device(torch: Any, requested: str) -> tuple[Any, str, str]:
    """按用户要求 + 环境实况选择计算设备（MPS/CUDA/CPU）。

    返回三元组 (设备对象, 设备名, 选择原因说明)。原因说明会写入
    实验清单，让读者知道这次实验到底跑在什么硬件上。

    :param torch: torch 模块（由调用方传入，避免本文件强依赖 torch）。
    :param requested: 请求的设备，取值 "auto" / "mps" / "cuda" / "cpu"。
    :return: (torch.device, 设备类型字符串, 选择原因字符串)。
    :raises RuntimeError: 显式请求了 mps/cuda 但环境不可用时抛出。
    """
    if requested == "auto":
        # auto 策略：优先苹果 MPS，其次 N 卡 CUDA，兜底 CPU
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return torch.device("mps"), "mps", "mps available and selected by auto policy"
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda", "cuda available and selected by auto policy"
        reason = "mps unavailable; selected cpu"
        # 细化 CPU 原因：是根本没编译 MPS，还是编译了但运行时不可用
        if not torch.backends.mps.is_built():
            reason = "torch was not built with mps; selected cpu"
        elif not torch.backends.mps.is_available():
            reason = "torch mps built but unavailable at runtime; selected cpu"
        return torch.device("cpu"), "cpu", reason
    if requested == "mps":
        if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
            raise RuntimeError("Requested mps but torch.backends.mps is unavailable")
        return torch.device("mps"), "mps", "mps explicitly requested"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested cuda but torch.cuda.is_available() is false")
        return torch.device("cuda"), "cuda", "cuda explicitly requested"
    return torch.device("cpu"), "cpu", "cpu explicitly requested"


def sync_device(torch: Any, device: Any) -> None:
    """阻塞等待指定设备上的异步计算完成（计时前必须调用）。

    深度学习框架默认异步执行，直接计时会漏掉排队时间。
    只有调用了这里的同步，后面读到的耗时才是真实执行耗时。

    :param torch: torch 模块。
    :param device: 设备对象（通常来自 select_device 的返回值）。
    """
    device_type = getattr(device, "type", str(device))
    if device_type == "mps":
        torch.mps.synchronize()
    elif device_type == "cuda":
        torch.cuda.synchronize()
    # CPU 设备计算本来就是同步的，无需处理


def latency_stats(samples_ms: list[float]) -> dict[str, float]:
    """汇总一批推理延迟样本（毫秒）的统计量。

    输出 mean/median（均值/中位数）、p05/p95（5%/95% 分位，反映
    尾部延迟）、min/max（最值）。空样本时全部返回 0.0，避免下游除零。

    :param samples_ms: 延迟样本列表（毫秒）。
    :return: {"mean","median","p05","p95","min","max"} 的统计字典。
    """
    ordered = sorted(samples_ms)  # 排序后按下标取分位数
    if not ordered:
        return {key: 0.0 for key in ["mean", "median", "p05", "p95", "min", "max"]}
    p05 = ordered[int(0.05 * (len(ordered) - 1))]  # 5% 分位（线性插值近似）
    p95 = ordered[int(0.95 * (len(ordered) - 1))]  # 95% 分位
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p05": p05,
        "p95": p95,
        "min": min(ordered),
        "max": max(ordered),
    }
