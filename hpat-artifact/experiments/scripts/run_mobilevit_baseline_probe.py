"""MobileViT 模型在本地机器上的基线性能探针（baseline probe）。

做什么：在本地桌面/笔记本上跑一遍 MobileViT（轻量级视觉 Transformer 模型）系列
       三个模型（XXS/XS/S，参数量与输入分辨率递增），实测它们单次推理的延迟
       （latency，毫秒），得到"真实设备能跑多快"的基线数据，供与 HPAT
       （光子张量处理器）论文里的光计算性能做对比参照。
       如果环境里缺少 torch/timm 等依赖，就输出一行行"blocked（被阻止）"状态的
       记录，说明无法测量，而不是悄悄给出假数据。

数据从哪来：模型权重由 timm 从网络拉取（pretrained=False 表示本次只测随机初始化
       模型的结构推理速度，不需要预训练权重，因此不会联网下权重）。
输出到哪：--output-dir 下的：
       - mobilevit_baseline_measured.csv 汇总（各模型统计指标）；
       - raw/mobilevit_baseline_latency_samples.csv 原始逐次延迟采样；
       - tables/ 与项目根目录 tables/ 下各一份副本；
       - mobilevit_baseline_probe_manifest.json 元信息清单。

怎么运行（示例）：
    python run_mobilevit_baseline_probe.py --output-dir run \
        --device auto --warmup 5 --iterations 50 --repeats 3

注意事项：这是本地桌面/MPS（Apple 芯片）的时序，只能当基线参考，
        不能当作边缘设备（edge，如手机）或 HPAT 芯片的实测证据。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import time
from typing import Any

from _common import REPO_ROOT, base_manifest, ensure_dir, relative, write_csv, write_json
from hpat_eval.mobilevit_loader import latency_stats, variants_from_config


# 默认要测的 3 个模型：显示名 / timm 模型名 / 输入分辨率
DEFAULT_MODELS = [
    ("MobileViT-XXS", "mobilevit_xxs", 192),
    ("MobileViT-XS", "mobilevit_xs", 224),
    ("MobileViT-S", "mobilevit_s", 256),
]


# 汇总 CSV 的列：模型信息 + 延迟统计（均值/中位数/5%/95% 分位/最小/最大）
BASELINE_FIELDS = [
    "variant",
    "model_name",
    "input_shape",
    "backend",
    "device",
    "precision",
    "status",
    "blocked_reason",
    "latency_ms_mean",
    "latency_ms_median",
    "latency_ms_p05",
    "latency_ms_p95",
    "latency_ms_min",
    "latency_ms_max",
    "warmup",
    "iterations",
    "synchronized_before_timing",
    "evidence_label",
]

# 原始采样 CSV 的列：每条记录是某一次推理的延迟
RAW_LATENCY_FIELDS = [
    "variant",
    "model_name",
    "input_shape",
    "backend",
    "device",
    "precision",
    "repeat_index",
    "sample_index",
    "latency_ms",
    "warmup",
    "iterations",
    "synchronized_before_timing",
    "evidence_label",
]


def _available(name: str) -> bool:
    """检查某个 Python 包是否已安装（不导入，只查是否存在）。

    参数：name —— 包名（如 'torch'、'timm'）。
    返回：bool —— 已安装返回 True。
    """
    return importlib.util.find_spec(name) is not None


def _select_device(torch, requested: str):
    """根据用户请求选择实际运行的设备（auto/cpu/mps/cuda）。

    参数：torch —— 已导入的 torch 模块；requested —— 请求的设备字符串。
    返回：(device, reason) 二元组：device 为 torch.device，reason 为选择说明。
    """
    # auto：优先用 MPS（Apple 芯片的 GPU 加速），不可用则退回 CPU
    if requested == "auto":
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return torch.device("mps"), "mps available and selected by auto policy"
        return torch.device("cpu"), "mps unavailable; selected cpu"
    # 显式要求 mps 但不可用时直接报错，避免静默降级掩盖问题
    if requested == "mps":
        if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
            raise RuntimeError("Requested mps but torch.backends.mps is unavailable")
        return torch.device("mps"), "mps explicitly requested"
    return torch.device(requested), f"{requested} explicitly requested"


def _sync(torch, device) -> None:
    """让异步执行的 GPU/MPS 任务同步完成，确保测到的延迟是真实计算时间。

    为什么：GPU/MPS 上的计算是异步入队的，若不显式同步，计时器可能
    在计算真正跑完前就返回，得到偏小的假延迟。CPU 不需要同步。
    参数：torch —— torch 模块；device —— 当前设备。
    返回：无。
    """
    if getattr(device, "type", "") == "mps":
        torch.mps.synchronize()
    elif getattr(device, "type", "") == "cuda":
        torch.cuda.synchronize()


def run(
    output_dir: pathlib.Path,
    device_request: str,
    warmup: int | None,
    iterations: int | None,
    repeats: int | None,
    config: dict[str, Any] | None = None,
) -> dict[str, pathlib.Path]:
    """执行基线探针：逐个模型测延迟，输出汇总/原始采样/清单。

    做什么（分步）：
      1) 创建输出目录结构（raw/、tables/）；
      2) 若缺少 torch/timm 依赖，写"blocked"行并提前返回（绝不伪造数据）；
      3) 依配置加载每个 MobileViT 模型（随机权重、eval 模式）；
      4) 先 warmup（预热）若干次排除冷启动影响，再计时 iterations 次，
         整个流程重复 repeats 轮，逐次记录原始延迟；
      5) 用 latency_stats 统计均值/中位数/分位点等，写出汇总 CSV 与 manifest。

    参数：
        output_dir —— 输出目录；device_request —— 设备请求（auto/cpu/mps/cuda）；
        warmup —— 预热次数（None 则用 config 或默认 5）；
        iterations —— 每轮计时次数（None 则用 config 或默认 50）；
        repeats —— 重复轮数（None 则用 config 或默认 3）；
        config —— 可选配置 dict，可覆盖模型清单与以上默认值。
    返回：dict[str, pathlib.Path] —— 输出文件路径映射。
    """
    ensure_dir(output_dir)
    raw_dir = ensure_dir(output_dir / "raw")
    tables_dir = ensure_dir(output_dir / "tables")
    # 从 config 取配置，缺省则用空清单；命令行参数优先于 config 里的值
    config = config or {"mobilevit_variants": []}
    baseline_cfg = config.get("baseline", {})
    warmup = int(warmup if warmup is not None else baseline_cfg.get("warmup", 5))
    iterations = int(iterations if iterations is not None else baseline_cfg.get("iterations", 50))
    repeats = int(repeats if repeats is not None else baseline_cfg.get("repeats", 3))
    manifest_path = output_dir / "mobilevit_baseline_probe_manifest.json"
    manifest = base_manifest("mobilevit_baseline_probe", "blocked or local baseline")
    manifest.update(
        {
            "requested_device": device_request,
            "warmup": warmup,
            "iterations": iterations,
            "repeats": repeats,
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this profiler probe is short by default.",
        }
    )

    # 依赖检查：torch 和 timm 缺一不可，否则无法测量
    missing = [name for name in ["torch", "timm"] if not _available(name)]
    if missing:
        # 被阻止分支：仍为每个配置生成一行"blocked"记录，说明缺什么包
        rows = []
        for spec in variants_from_config(config):
            resolution = int(spec["input_resolution"])
            rows.append(
                {
                    "variant": spec["variant"],
                    "model_name": spec.get("timm_model", ""),
                    "input_shape": f"1x3x{resolution}x{resolution}",
                    "backend": "blocked",
                    "device": "",
                    "precision": "fp32",
                    "status": "blocked",
                    "blocked_reason": f"Missing Python packages: {', '.join(missing)}",
                    "latency_ms_mean": "",
                    "latency_ms_median": "",
                    "latency_ms_p05": "",
                    "latency_ms_p95": "",
                    "latency_ms_min": "",
                    "latency_ms_max": "",
                    "warmup": warmup,
                    "iterations": 0,
                    "synchronized_before_timing": "no",
                    "evidence_label": "blocked",
                }
            )
        out_csv = output_dir / "mobilevit_baseline_measured.csv"
        tables_csv = tables_dir / "mobilevit_baseline_measured.csv"
        write_csv(out_csv, rows, BASELINE_FIELDS)
        write_csv(tables_csv, rows, BASELINE_FIELDS)
        manifest.update(
            {
                "status": "blocked",
                "evidence_label": "blocked",
                "blocked_reason": f"Missing Python packages: {', '.join(missing)}",
                "outputs": [relative(out_csv), relative(tables_csv)],
                "promotion_note": "Install torch and timm, then rerun before citing any author-measured MobileViT baseline.",
            }
        )
        write_json(manifest_path, manifest)
        return {"csv": out_csv, "manifest": manifest_path}

    # 依赖齐全：延迟到此处才真正导入 torch/timm，避免启动开销
    import torch  # type: ignore
    import timm  # type: ignore

    # 选定实际设备（MPS/CPU/CUDA）
    device, device_reason = _select_device(torch, device_request)
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    # 逐个模型实测
    for spec in variants_from_config(config):
        variant = spec["variant"]
        model_name = spec.get("timm_model", "")
        resolution = int(spec["input_resolution"])
        try:
            # 创建模型：pretrained=False 表示随机初始化权重，只测结构推理速度
            model = timm.create_model(model_name, pretrained=False).eval().to(device)
        except Exception as exc:
            # 单个模型加载失败不影响其它模型：记一行 blocked 后跳过
            rows.append(
                {
                    "variant": variant,
                    "model_name": model_name,
                    "input_shape": f"1x3x{resolution}x{resolution}",
                    "backend": getattr(device, "type", str(device)),
                    "device": str(device),
                    "precision": "fp32",
                    "status": "blocked",
                    "blocked_reason": str(exc),
                    "latency_ms_mean": "",
                    "latency_ms_median": "",
                    "latency_ms_p05": "",
                    "latency_ms_p95": "",
                    "latency_ms_min": "",
                    "latency_ms_max": "",
                    "warmup": warmup,
                    "iterations": 0,
                    "synchronized_before_timing": "yes" if getattr(device, "type", "") in {"mps", "cuda"} else "not required",
                    "evidence_label": "blocked",
                }
            )
            continue
        # 造一个随机输入张量（1 张 3 通道分辨率×分辨率 的图）
        x = torch.randn(1, 3, resolution, resolution, device=device)
        with torch.no_grad():
            samples = []
            # repeats 轮，每轮先预热再计时，降低噪声
            for repeat_index in range(repeats):
                for _ in range(warmup):
                    _ = model(x)  # 预热推理：只跑不算时间
                _sync(torch, device)
                for sample_index in range(iterations):
                    _sync(torch, device)  # 计时前先同步，清空队列
                    t0 = time.perf_counter()  # 高精度计时起点
                    _ = model(x)
                    _sync(torch, device)  # 计时后再同步，确保计算完成
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    samples.append(latency_ms)
                    raw_rows.append(
                        {
                            "variant": variant,
                            "model_name": model_name,
                            "input_shape": f"1x3x{resolution}x{resolution}",
                            "backend": getattr(device, "type", str(device)),
                            "device": str(device),
                            "precision": "fp32",
                            "repeat_index": repeat_index,
                            "sample_index": sample_index,
                            "latency_ms": f"{latency_ms:.6f}",
                            "warmup": warmup,
                            "iterations": iterations,
                            "synchronized_before_timing": "yes" if getattr(device, "type", "") in {"mps", "cuda"} else "not required",
                            "evidence_label": "raw local desktop baseline sample; not edge evidence",
                        }
                    )
        # 汇总行：用 latency_stats 统计所有采样的均值/中位数/分位点等
        rows.append(
            {
                "variant": variant,
                "model_name": model_name,
                "input_shape": f"1x3x{resolution}x{resolution}",
                "backend": getattr(device, "type", str(device)),
                "device": str(device),
                "precision": "fp32",
                "status": "ok",
                "blocked_reason": "",
                "latency_ms_mean": f"{latency_stats(samples)['mean']:.6f}",
                "latency_ms_median": f"{latency_stats(samples)['median']:.6f}",
                "latency_ms_p05": f"{latency_stats(samples)['p05']:.6f}",
                "latency_ms_p95": f"{latency_stats(samples)['p95']:.6f}",
                "latency_ms_min": f"{latency_stats(samples)['min']:.6f}",
                "latency_ms_max": f"{latency_stats(samples)['max']:.6f}",
                "warmup": warmup,
                "iterations": iterations,
                "synchronized_before_timing": "yes" if getattr(device, "type", "") in {"mps", "cuda"} else "not required",
                "evidence_label": "local desktop baseline; not edge evidence",
            }
        )

    # 写出全部输出：运行目录汇总/原始/表格 + 项目根目录 tables 汇总
    out_csv = output_dir / "mobilevit_baseline_measured.csv"
    raw_csv = raw_dir / "mobilevit_baseline_latency_samples.csv"
    tables_csv = tables_dir / "mobilevit_baseline_measured.csv"
    project_csv = REPO_ROOT / "tables" / "mobilevit_baseline_measured.csv"
    write_csv(out_csv, rows, BASELINE_FIELDS)
    write_csv(raw_csv, raw_rows, RAW_LATENCY_FIELDS)
    write_csv(tables_csv, rows, BASELINE_FIELDS)
    write_csv(project_csv, rows, BASELINE_FIELDS)
    manifest.update(
        {
            "status": "ok",
            "evidence_label": "local desktop baseline; not edge evidence",
            "device": str(device),
            "device_reason": device_reason,
            "torch_version": getattr(torch, "__version__", "unknown"),
            "timm_version": getattr(timm, "__version__", "unknown"),
            "raw_sample_count": len(raw_rows),
            "outputs": [relative(out_csv), relative(raw_csv), relative(tables_csv), relative(project_csv)],
            "promotion_note": "Local desktop/MPS timing only. Do not use as edge evidence.",
        }
    )
    write_json(manifest_path, manifest)
    return {"csv": out_csv, "raw_csv": raw_csv, "project_csv": project_csv, "manifest": manifest_path}


def main() -> None:
    """命令行入口：解析参数、可选加载 config、调用 run() 并打印输出路径。

    参数：全部来自命令行（--output-dir / --device / --warmup / --iterations /
        --repeats / --config）。
    返回：无。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--config", default=None, help="Accepted for run_all compatibility; currently unused.")
    args = parser.parse_args()
    cfg = None
    # 若给了 --config，就加载 JSON 配置（目前主要为兼容 run_all 的调用方式）
    if args.config:
        from _common import load_json

        cfg = load_json(pathlib.Path(args.config))
    outputs = run(pathlib.Path(args.output_dir), args.device, args.warmup, args.iterations, args.repeats, cfg)
    # 用相对路径打印结果，方便上层脚本接住
    print(json.dumps({k: relative(v) for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
