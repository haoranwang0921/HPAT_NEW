"""苹果 MPS（Metal Performance Shaders，苹果 GPU 上的计算框架）功耗探针（power probe）。

【做什么】在苹果设备（目标机型为 Apple M5 Pro MPS 桌面机）上实测 MobileViT 各
变体的功耗与延迟数据。原理是：让模型在 MPS 上连续跑推理，同时用系统自带的
powermetrics 命令采样 CPU/GPU/ANE（Apple Neural Engine 神经网络引擎）功耗；
再测量一段"空闲"功耗做基线，用"活跃功耗 - 空闲功耗"得到模型净功耗（能耗），
从而为 HPAT 的能耗模型校准提供本地实测证据。

【数据从哪来】
- 配置：experiments/config/hpat_experiment_config.json（要测的 MobileViT 变体清单）
- 实测：本机 powermetrics（需 root/管理员，失败时脚本输出 blocked 行而非硬报错）

【输出到哪】（--output-dir 下）
- tables/mobilevit_mps_power_measured.csv 汇总表（功耗均值、延迟、能量等）
- raw/mobilevit_mps_powermetrics_samples.csv 逐采样功耗原始数据
- raw/mobilevit_mps_power_window_latency_samples.csv 功耗窗口内延迟原始数据
- mps_power_probe_manifest.json 运行台账
- 还会尽力把汇总表提升（promote）到仓库根的 tables/ 下（幂等，不覆盖已有 ok 行）

【怎么运行】（需在苹果设备上，且 powermetrics 可用）
    python run_mps_power_probe.py --output-dir <输出目录> [--device mps] [--warmup 12] \
        [--interval-ms 500] [--samples 12] [--config <配置路径>]

本文件只负责测量与落盘；它不会改动实验代码，也不修改配置。
"""

from __future__ import annotations

import argparse  # 解析命令行参数
import json  # 读写 manifest JSON
import math  # 判断数值是否有限（isfinite）
import os  # 读取环境变量（caffeinate 标记）
import pathlib  # 路径操作
import re  # 解析 powermetrics 文本
import shutil  # 查找可执行文件路径
import statistics  # 均值/中位数统计
import subprocess  # 启动 powermetrics 子进程
import tempfile  # 用临时文件承接 powermetrics 输出
import time  # 计时（推理延迟、超时判定）
from typing import Any

# 复用同目录 _common.py 的工具与 hpat_eval 里的设备选择/变体加载。
from _common import REPO_ROOT, base_manifest, ensure_dir, project_write_skipped, read_csv, relative, write_csv, write_json
from hpat_eval.mobilevit_loader import select_device, sync_device, variants_from_config


SUMMARY_FIELDS = [
    # 汇总 CSV（summary）的列名：一个变体一行，记录功耗/延迟/能量的均值与采样信息。
    "variant",
    "model_name",
    "input_shape",
    "backend",
    "device",
    "precision",
    "status",
    "blocked_reason",
    "power_method",
    "power_scope",
    "idle_power_w_mean",
    "active_power_w_mean",
    "idle_subtracted_power_w_mean",
    "active_gpu_power_w_mean",
    "latency_ms_mean",
    "latency_ms_median",
    "inference_count",
    "energy_mj_per_inference_mean",
    "idle_sample_count",
    "active_sample_count",
    "powermetrics_interval_ms",
    "powermetrics_samples",
    "raw_power_source",
    "raw_latency_source",
    "evidence_label",
    "claim_boundary",
]

RAW_POWER_FIELDS = [
    # 逐采样功耗原始 CSV 的列名：每个 powermetrics 采样点一行。
    "variant",
    "phase",
    "sample_index",
    "cpu_power_w",
    "gpu_power_w",
    "ane_power_w",
    "combined_power_w",
    "selected_power_w",
    "raw_excerpt",
]

RAW_LATENCY_FIELDS = [
    # 功耗窗口内逐次推理延迟的原始 CSV 列名。
    "variant",
    "model_name",
    "input_shape",
    "backend",
    "device",
    "precision",
    "sample_index",
    "latency_ms",
    "measurement_phase",
    "evidence_label",
]

CLAIM_BOUNDARY = (
    # 声明边界（claim boundary）：这份数据只算"本机 powermetrics 实测证据"，
    # 不能外推成移动端/边缘部署、HPAT 加速比或 HPAT 硅片功耗——写进每一行，
    # 防止结果被滥用为过强声明。
    "Apple M5 Pro MPS desktop power is local powermetrics evidence only; "
    "not mobile/edge deployment, not HPAT speedup, and not HPAT silicon power."
)


def _available(name: str) -> bool:
    """检查某个 Python 包能否被 import（用于探测 torch/timm 是否已安装）。

    参数 name：包名（如 "torch"）。
    返回：能 import 返回 True，任何异常返回 False。
    """
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _powermetrics_cmd(interval_ms: int, samples: int) -> list[str]:
    """构造 powermetrics 命令（含必要参数与 sudo 前缀）。

    参数 interval_ms：采样间隔（毫秒）；samples：采样次数。
    返回：命令参数列表。非 root 用户会自动加 "sudo -n"（非交互式 sudo，
    不会弹密码框）；root 用户则直接调用。
    """
    executable = shutil.which("powermetrics") or "/usr/bin/powermetrics"
    cmd = [executable, "--samplers", "cpu_power,gpu_power", "-i", str(interval_ms), "-n", str(samples)]
    if os.geteuid() == 0:
        return cmd
    return ["sudo", "-n", *cmd]


def _run_powermetrics(interval_ms: int, samples: int, timeout_s: float | None = None) -> subprocess.CompletedProcess[str]:
    """同步运行 powermetrics 并等待其结束，返回完成结果。

    参数 interval_ms：采样间隔（毫秒）；samples：采样次数；
          timeout_s：超时秒数，不传则按采样时长自动 +10s 兜底。
    返回：subprocess.CompletedProcess[str]（含 stdout/stderr）。
    """
    timeout_s = timeout_s or max(10.0, samples * interval_ms / 1000.0 + 10.0)
    return subprocess.run(
        _powermetrics_cmd(interval_ms, samples),
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )


def _preflight_powermetrics(interval_ms: int) -> tuple[bool, str]:
    """预检（preflight）：确认 powermetrics 可执行且能用 sudo 跑通一次采样。

    参数 interval_ms：采样间隔（毫秒）。
    返回：(是否通过, 说明文本)。未通过时文本会写进 blocked 行的 blocked_reason。
    这是"软失败"设计：探针跑不起来时不要中断，而是产出 blocked 状态行。
    """
    executable = shutil.which("powermetrics")
    if not executable:
        return False, "powermetrics executable not found"
    try:
        proc = _run_powermetrics(interval_ms, 1)
    except Exception as exc:
        return False, f"powermetrics preflight failed: {exc}"
    if proc.returncode != 0:
        text = (proc.stderr or proc.stdout or "").strip()
        return False, text or f"powermetrics returned {proc.returncode}"
    return True, "powermetrics preflight ok"


def _split_samples(text: str) -> list[str]:
    """把 powermetrics 的整段输出按 "*** Sampled ..." 标记切成多个采样块。

    参数 text：powermetrics 原始输出。
    返回：各采样块文本的列表。用 lookahead 正则在每个 "*** Sampled" 前切一刀；
    若没有标记则把整段当一块。
    """
    chunks = re.split(r"(?=^\*\*\* Sampled)", text, flags=re.MULTILINE)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return chunks or ([text.strip()] if text.strip() else [])


def _parse_power_line(line: str) -> tuple[str, float] | None:
    """解析 powermetrics 输出里的一行功耗数据，返回 (字段名, 瓦数)。

    参数 line：powermetrics 输出的一行文本。
    返回：形如 ("gpu_power_w", 12.3) 的元组；行里没有功耗数据则返回 None。
    支持 CPU/GPU/ANE/Combined 四类功耗行，单位 mW 会换算成 W（瓦）。
    """
    lower = line.lower()
    key = ""
    if "cpu power" in lower:
        key = "cpu_power_w"
    elif "gpu power" in lower:
        key = "gpu_power_w"
    elif "ane power" in lower:
        key = "ane_power_w"
    elif "combined power" in lower:
        key = "combined_power_w"
    if not key:
        return None
    # 匹配数值 + 单位（mW 或 W）；mW 除以 1000 转成 W。
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(mW|W)\b", line)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "mW":
        value /= 1000.0
    return key, value


def _parse_powermetrics(text: str, variant: str, phase: str) -> list[dict[str, Any]]:
    """把 powermetrics 整段输出解析成"逐采样"的功耗行列表。

    参数 text：powermetrics 原始输出；variant：当前模型变体名；
          phase：阶段标记，仅用于 idle/active 两种取值。
    返回：行字典列表，每行含 variant/phase/sample_index、四类功耗、selected_power_w
    以及 raw_excerpt（原始文本摘录，最多 8 行 700 字符，方便追溯）。只保留
    至少解析出一项功耗的采样块。
    """
    rows: list[dict[str, Any]] = []
    for sample_index, chunk in enumerate(_split_samples(text)):
        row: dict[str, Any] = {
            "variant": variant,
            "phase": phase,
            "sample_index": sample_index,
            "cpu_power_w": "",
            "gpu_power_w": "",
            "ane_power_w": "",
            "combined_power_w": "",
            "selected_power_w": "",
            "raw_excerpt": " | ".join(chunk.splitlines()[:8])[:700],
        }
        for line in chunk.splitlines():
            parsed = _parse_power_line(line)
            if parsed:
                key, value = parsed
                row[key] = f"{value:.6f}"
        # selected_power_w：优先用 combined；否则 cpu+gpu+ane 求和。
        selected = _selected_power_w(row)
        if selected is not None:
            row["selected_power_w"] = f"{selected:.6f}"
        # 采样块里完全没有功耗数据（例如开头几行说明文字）则丢弃。
        if any(row[key] != "" for key in ["cpu_power_w", "gpu_power_w", "ane_power_w", "combined_power_w"]):
            rows.append(row)
    return rows


def _mean(values: list[float]) -> float:
    """求有限数值列表的算术平均值；全是空/非法值时返回 0.0。

    参数 values：数值列表。
    返回：均值；过滤掉非有限值（NaN/inf）后没有值则返回 0.0。
    """
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else 0.0


def _float_or_none(value: Any) -> float | None:
    """把任意值安全转成有限浮点数；空串/None/非法值返回 None。

    参数 value：待转换值（CSV 里读出的常是字符串）。
    返回：有限的 float，否则 None。避免后面计算时被脏数据带偏。
    """
    if value in ("", None):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _selected_power_w(row: dict[str, Any]) -> float | None:
    """从一行功耗数据中选出"本行代表功耗"（瓦），用于后续均值计算。

    参数 row：一行功耗 dict（含 combined/cpu/gpu/ane 字段）。
    返回：有 combined 用 combined；否则 cpu+gpu+ane 求和；总和 <=0 返回 None。
    """
    combined = _float_or_none(row.get("combined_power_w"))
    if combined is not None:
        return combined
    cpu = _float_or_none(row.get("cpu_power_w")) or 0.0
    gpu = _float_or_none(row.get("gpu_power_w")) or 0.0
    ane = _float_or_none(row.get("ane_power_w")) or 0.0
    total = cpu + gpu + ane
    return total if total > 0 else None


def _phase_mean(rows: list[dict[str, Any]], phase: str, field: str) -> float:
    """对指定阶段（idle/active）的行、指定功耗字段求均值。

    参数 rows：功耗行列表；phase：阶段名；field：字段名（如 selected_power_w）。
    返回：该阶段该字段的均值；没有可用数值则返回 0.0。
    """
    return _mean([value for value in (_float_or_none(row.get(field)) for row in rows if row.get("phase") == phase) if value is not None])


def _existing_project_power_ok(path: pathlib.Path) -> bool:
    """检查仓库根下已有的汇总表里是否存在 status=ok 的行。

    参数 path：项目汇总 CSV 路径。
    返回：存在 ok 行返回 True；文件不存在或读取失败返回 False。
    用于决定是否允许覆盖仓库根的表（保护已有有效测量结果）。
    """
    if not path.exists():
        return False
    try:
        return any(row.get("status") == "ok" for row in read_csv(path))
    except Exception:
        return False


def _write_project_summary(path: pathlib.Path, rows: list[dict[str, Any]]) -> bool:
    """把汇总行写入仓库根的项目表，但保护已有 ok 数据不被新 blocked 行覆盖。

    参数 path：项目汇总 CSV 路径；rows：本次测量的汇总行。
    返回：是否真的发生了写入（False 表示被跳过/未写入）。
    跳过规则：文件被标记跳过（project_write_skipped），或本次没有 ok 行
    而旧文件里已有 ok 行——避免 blocked 行把历史有效数据冲掉。
    """
    if project_write_skipped(path):
        return False
    has_new_ok = any(row.get("status") == "ok" for row in rows)
    if not has_new_ok and _existing_project_power_ok(path):
        return False
    write_csv(path, rows, SUMMARY_FIELDS)
    return True


def _measure_idle(variant: str, interval_ms: int, samples: int) -> list[dict[str, Any]]:
    """测量"空闲"功耗基线：机器空转、不跑模型时的功耗。

    参数 variant：变体名（仅作标记）；interval_ms/samples：采样参数。
    返回：解析后的 idle 阶段功耗行列表。
    测量失败（powermetrics 退出码非 0）直接抛 RuntimeError，让上层按异常处理。
    """
    proc = _run_powermetrics(interval_ms, samples)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "").strip() or f"powermetrics returned {proc.returncode}")
    return _parse_powermetrics(proc.stdout, variant, "idle")


def _measure_active(
    variant: str,
    model_name: str,
    input_shape: str,
    torch: Any,
    model: Any,
    x: Any,
    device: Any,
    interval_ms: int,
    samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """测量"活跃"功耗：让模型连续推理，同时后台采功耗并逐次记延迟。

    参数 variant：变体名；model_name：模型名；input_shape：输入形状字符串；
          torch/torch 模块；model：已加载模型；x：输入张量；device：运行设备；
          interval_ms/samples：powermetrics 采样参数。
    返回：(active 阶段功耗行列表, 延迟行列表)。

    流程拆解：
    1) 用临时文件承接 powermetrics 的 stdout/stderr（不占管道，避免阻塞）；
    2) 后台启动 powermetrics（Popen，非阻塞）；
    3) 主循环：在 powermetrics 没退出前，不断执行 model(x) 推理并计时
       （perf_counter 前后各加 sync_device 保证 MPS 异步执行被真正落盘）；
    4) 到超时或 powermetrics 结束就收尾；超时会 terminate 进程并报错；
    5) 读回 stdout 解析成 active 功耗行，连同延迟行一起返回。
    """
    latency_rows: list[dict[str, Any]] = []
    sample_index = 0
    # 超时 = 采样时长 + 15s 缓冲，防模型初始化卡住时无限等待。
    timeout_s = max(20.0, samples * interval_ms / 1000.0 + 15.0)
    deadline = time.monotonic() + timeout_s
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8") as stdout_file, tempfile.NamedTemporaryFile(
        "w+", encoding="utf-8"
    ) as stderr_file:
        proc = subprocess.Popen(
            _powermetrics_cmd(interval_ms, samples),
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
        )
        timed_out = False
        with torch.no_grad():
            # 在采样窗口内持续推理：模型越跑，越能反映"负载下的真实功耗"。
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    timed_out = True
                    proc.terminate()
                    break
                sync_device(torch, device)
                t0 = time.perf_counter()
                _ = model(x)
                sync_device(torch, device)  # 等 MPS 异步计算完成，再读钟
                latency_ms = (time.perf_counter() - t0) * 1000.0
                latency_rows.append(
                    {
                        "variant": variant,
                        "model_name": model_name,
                        "input_shape": input_shape,
                        "backend": getattr(device, "type", str(device)),
                        "device": str(device),
                        "precision": "fp32",
                        "sample_index": sample_index,
                        "latency_ms": f"{latency_ms:.6f}",
                        "measurement_phase": "active_power_window",
                        "evidence_label": "raw Apple M5 Pro MPS desktop latency sample during powermetrics window",
                    }
                )
                sample_index += 1
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        # 从临时文件读回子进程输出（Popen 的 stdout/stderr 都重定向到了文件）。
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()
    if timed_out:
        raise RuntimeError(f"powermetrics active window exceeded {timeout_s:.1f}s timeout")
    if proc.returncode != 0:
        raise RuntimeError((stderr or stdout or "").strip() or f"powermetrics returned {proc.returncode}")
    return _parse_powermetrics(stdout, variant, "active"), latency_rows


def _blocked_rows(config: dict[str, Any], reason: str, interval_ms: int, samples: int) -> list[dict[str, Any]]:
    """为配置里的每个变体生成一行"blocked（被阻断/不可测）"汇总行。

    参数 config：实验配置；reason：阻断原因（如 powermetrics 不可用）；
          interval_ms/samples：采样参数（写进行里便于追溯）。
    返回：汇总行列表，每行 status="blocked"、blocked_reason=reason，
    其余功耗/延迟字段留空。用于"软失败"：探针不可用时仍能落盘可解释的结果。
    """
    rows: list[dict[str, Any]] = []
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
                "blocked_reason": reason,
                "power_method": "powermetrics cpu_power,gpu_power",
                "power_scope": "Apple M5 Pro MPS desktop local",
                "idle_power_w_mean": "",
                "active_power_w_mean": "",
                "idle_subtracted_power_w_mean": "",
                "active_gpu_power_w_mean": "",
                "latency_ms_mean": "",
                "latency_ms_median": "",
                "inference_count": "",
                "energy_mj_per_inference_mean": "",
                "idle_sample_count": 0,
                "active_sample_count": 0,
                "powermetrics_interval_ms": interval_ms,
                "powermetrics_samples": samples,
                "raw_power_source": "",
                "raw_latency_source": "",
                "evidence_label": "blocked Apple M5 Pro MPS power measurement",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return rows


def run(
    output_dir: pathlib.Path,
    config: dict[str, Any],
    device_request: str,
    warmup: int,
    interval_ms: int,
    samples: int,
) -> dict[str, pathlib.Path]:
    """探针主流程：预检 → 逐变体测功耗/延迟 → 落盘 CSV 与 manifest。

    参数 output_dir：输出目录；config：实验配置（含变体清单）；
          device_request：请求的设备（auto/mps/cpu/cuda）；warmup：预热推理次数；
          interval_ms：采样间隔；samples：采样次数。
    返回：{raw_power_csv, raw_latency_csv, summary_csv, project_summary_csv, manifest}
    各输出文件路径。

    流程拆解：
    1) 建目录、初始化 manifest（含 claim_boundary 等元数据）；
    2) powermetrics 预检：失败则整批输出 blocked 行并结束（软失败）；
    3) 检查 torch/timm：缺失同样走 blocked；
    4) 逐个变体：加载模型 → 预热 → 测 idle 功耗 → 测 active 功耗+延迟；
       计算净功耗（active−idle）与每推理能量（净功耗×延迟均值）；
       单个变体测量异常只把该行标为 blocked，不中断整体；
    5) 写三份 CSV（raw 功耗、raw 延迟、汇总），尽力提升项目汇总到仓库根；
    6) 更新 manifest 并落盘，返回各输出路径。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    raw_dir = ensure_dir(output_dir / "raw")
    manifest_path = output_dir / "mps_power_probe_manifest.json"
    summary_csv = tables_dir / "mobilevit_mps_power_measured.csv"
    project_summary_csv = REPO_ROOT / "tables" / "mobilevit_mps_power_measured.csv"
    raw_power_csv = raw_dir / "mobilevit_mps_powermetrics_samples.csv"
    raw_latency_csv = raw_dir / "mobilevit_mps_power_window_latency_samples.csv"
    manifest = base_manifest("mps_power_probe", "Apple M5 Pro MPS desktop power probe")
    manifest.update(
        {
            "requested_device": device_request,
            "warmup": warmup,
            "powermetrics_interval_ms": interval_ms,
            "powermetrics_samples": samples,
            "power_method": "powermetrics cpu_power,gpu_power",
            "claim_boundary": CLAIM_BOUNDARY,
            # 记录是否通过 caffeinate 防止系统睡眠，保证采样窗口不被中断。
            "caffeinate_used": os.environ.get("HPAT_CAFFEINATE_USED", "").strip() == "1",
            "caffeinate_reason": "Use caffeinate for power probes so the active window is not interrupted by sleep.",
        }
    )

    # ---------- 阶段2：powermetrics 预检，失败则软失败 ----------
    ok, reason = _preflight_powermetrics(interval_ms)
    if not ok:
        rows = _blocked_rows(config, reason, interval_ms, samples)
        write_csv(summary_csv, rows, SUMMARY_FIELDS)
        project_written = _write_project_summary(project_summary_csv, rows)
        manifest.update(
            {
                "status": "blocked",
                "blocked_reason": reason,
                "outputs": [relative(summary_csv)] + ([relative(project_summary_csv)] if project_written else []),
                "project_write_performed": project_written,
                # 说明：blocked 行不会覆盖项目表里已有的 ok 行。
                "project_write_note": "Blocked probe rows do not overwrite an existing project table with status=ok.",
                "promotion_note": "Run `sudo -v` in a local terminal, then rerun the mps-power-probe target to collect powermetrics samples.",
            }
        )
        write_json(manifest_path, manifest)
        return {"summary_csv": summary_csv, "project_summary_csv": project_summary_csv, "manifest": manifest_path}

    # ---------- 阶段3：检查 torch/timm 是否可 import ----------
    missing = [name for name in ["torch", "timm"] if not _available(name)]
    if missing:
        reason = f"Missing Python packages: {', '.join(missing)}"
        rows = _blocked_rows(config, reason, interval_ms, samples)
        write_csv(summary_csv, rows, SUMMARY_FIELDS)
        project_written = _write_project_summary(project_summary_csv, rows)
        manifest.update(
            {
                "status": "blocked",
                "blocked_reason": reason,
                "outputs": [relative(summary_csv)] + ([relative(project_summary_csv)] if project_written else []),
                "project_write_performed": project_written,
                "project_write_note": "Blocked probe rows do not overwrite an existing project table with status=ok.",
            }
        )
        write_json(manifest_path, manifest)
        return {"summary_csv": summary_csv, "project_summary_csv": project_summary_csv, "manifest": manifest_path}

    import torch  # type: ignore
    import timm  # type: ignore

    # ---------- 阶段4：选设备，逐变体测量 ----------
    device, device_type, device_reason = select_device(torch, device_request)
    summary_rows: list[dict[str, Any]] = []
    raw_power_rows: list[dict[str, Any]] = []
    raw_latency_rows: list[dict[str, Any]] = []
    for spec in variants_from_config(config):
        variant = spec["variant"]
        model_name = spec.get("timm_model", "")
        resolution = int(spec["input_resolution"])
        input_shape = f"1x3x{resolution}x{resolution}"
        try:
            # 加载无预训练权重的模型并搬到目标设备（纯推理，不训）。
            model = timm.create_model(model_name, pretrained=False).eval().to(device)
            x = torch.randn(1, 3, resolution, resolution, device=device)
            # 预热：先跑若干次让 MPS 完成内核编译/分配，避免把首推的抖动算进去。
            with torch.no_grad():
                for _ in range(warmup):
                    _ = model(x)
                sync_device(torch, device)
            idle_rows = _measure_idle(variant, interval_ms, samples)
            active_rows, latency_rows = _measure_active(variant, model_name, input_shape, torch, model, x, device, interval_ms, samples)
            raw_power_rows.extend(idle_rows)
            raw_power_rows.extend(active_rows)
            raw_latency_rows.extend(latency_rows)
            # 计算指标：空闲/活跃均值功耗，净功耗=活跃−空闲，GPU 功耗单独记，
            # 每推理能量=净功耗(瓦)×延迟均值(毫秒)/1000 → 毫焦。
            idle_power = _phase_mean(idle_rows, "idle", "selected_power_w")
            active_power = _phase_mean(active_rows, "active", "selected_power_w")
            active_gpu_power = _phase_mean(active_rows, "active", "gpu_power_w")
            net_power = max(0.0, active_power - idle_power)
            latency_values = [
                value
                for value in (_float_or_none(row.get("latency_ms")) for row in latency_rows)
                if value is not None
            ]
            latency_mean = _mean(latency_values)
            latency_median = statistics.median(latency_values) if latency_values else 0.0
            summary_rows.append(
                {
                    "variant": variant,
                    "model_name": model_name,
                    "input_shape": input_shape,
                    "backend": device_type,
                    "device": str(device),
                    "precision": "fp32",
                    "status": "ok",
                    "blocked_reason": "",
                    "power_method": "powermetrics cpu_power,gpu_power",
                    "power_scope": "idle-subtracted CPU+GPU+ANE when combined power is unavailable",
                    "idle_power_w_mean": f"{idle_power:.6f}",
                    "active_power_w_mean": f"{active_power:.6f}",
                    "idle_subtracted_power_w_mean": f"{net_power:.6f}",
                    "active_gpu_power_w_mean": f"{active_gpu_power:.6f}",
                    "latency_ms_mean": f"{latency_mean:.6f}",
                    "latency_ms_median": f"{latency_median:.6f}",
                    "inference_count": len(latency_values),
                    "energy_mj_per_inference_mean": f"{net_power * latency_mean:.6f}",
                    "idle_sample_count": len(idle_rows),
                    "active_sample_count": len(active_rows),
                    "powermetrics_interval_ms": interval_ms,
                    "powermetrics_samples": samples,
                    "raw_power_source": relative(raw_power_csv),
                    "raw_latency_source": relative(raw_latency_csv),
                    "evidence_label": "measured Apple M5 Pro MPS desktop power with idle baseline; not edge evidence",
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
        except Exception as exc:
            # 单个变体失败只标 blocked，继续测下一个；不抛断整体。
            summary_rows.append(_blocked_rows({"mobilevit_variants": [{"name": variant, "timm_model": model_name, "input_resolution": resolution}]}, str(exc), interval_ms, samples)[0])

    # ---------- 阶段5：写三份 CSV，尽力提升项目汇总到仓库根 ----------
    write_csv(raw_power_csv, raw_power_rows, RAW_POWER_FIELDS)
    write_csv(raw_latency_csv, raw_latency_rows, RAW_LATENCY_FIELDS)
    write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)
    project_written = _write_project_summary(project_summary_csv, summary_rows)
    # ---------- 阶段6：更新 manifest 并落盘 ----------
    manifest.update(
        {
            # 有 ok 行且无任何非 ok 行才叫 ok；否则叫 partial（部分成功）。
            "status": "ok" if summary_rows and all(row["status"] == "ok" for row in summary_rows) else "partial",
            "device": str(device),
            "device_type": device_type,
            "device_reason": device_reason,
            "torch_version": getattr(torch, "__version__", "unknown"),
            "timm_version": getattr(timm, "__version__", "unknown"),
            "outputs": [relative(raw_power_csv), relative(raw_latency_csv), relative(summary_csv)]
            + ([relative(project_summary_csv)] if project_written else []),
            "project_write_performed": project_written,
            "project_write_note": "Project table is updated only for new ok rows or when no prior ok project power rows exist.",
            "promotion_note": "Use only as local Apple M5 Pro MPS desktop power/energy reference. Do not use as mobile/edge or HPAT silicon evidence.",
        }
    )
    write_json(manifest_path, manifest)
    return {
        "raw_power_csv": raw_power_csv,
        "raw_latency_csv": raw_latency_csv,
        "summary_csv": summary_csv,
        "project_summary_csv": project_summary_csv,
        "manifest": manifest_path,
    }


def main() -> None:
    """命令行入口：解析参数 → 加载配置 → 调用 run() 并打印输出文件清单。

    命令行参数：
        --output-dir  输出目录（必填）
        --config      实验配置 JSON（默认 experiments/config/hpat_experiment_config.json）
        --device      设备请求：auto/mps/cpu/cuda（默认 mps）
        --warmup      预热推理次数（默认 12）
        --interval-ms powermetrics 采样间隔毫秒（默认 500）
        --samples     powermetrics 采样次数（默认 12）
    返回：None。运行结果以 JSON（各输出文件的相对路径）打印到 stdout。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json"))
    parser.add_argument("--device", default="mps", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--interval-ms", type=int, default=500)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    from _common import load_json  # 延迟 import，避免脚本头部就要求 _common 可导入

    outputs = run(
        pathlib.Path(args.output_dir),
        load_json(pathlib.Path(args.config)),
        args.device,
        args.warmup,
        args.interval_ms,
        args.samples,
    )
    print(json.dumps({key: relative(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()  # 作为脚本直接运行时进入 main
