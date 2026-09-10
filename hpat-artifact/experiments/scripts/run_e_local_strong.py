"""E-local 强化全流程驱动（run_e_local_strong.py）。

实验目的：按固定顺序串行调用一批 e-local（边缘本地推理）相关脚本，
完成一条"强化证据链"：环境采集 → MobileViT 活动跟踪 → 基线溯源 →
trace-driven 活动 → 盈亏平衡 → 证据强度汇总 → 算子映射闭合 → 刷新
外部边缘上下文 → e-local 就绪度 → 渲染图表。可选使用固定图片子集
（public 或 imagenette）跑真实精度实验（run_e_local_fixed_subset.py）。

本脚本是"编排器"，自己不计算；可用环境变量 HPAT_ELOCAL_* 定制子集、
设备、精度等参数。

- 输入：仓库配置与上游数据；可选环境变量指定固定子集。
- 产出：results/<run_时间戳>/ 下完整结果集 + e_local_run_summary.json。
- 命令：python run_e_local_strong.py [--output-root <目录>] [--config <配置>]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

from _common import REPO_ROOT, ensure_dir, relative, run_id, write_json


def _use_public_subset() -> bool:
    """是否启用"公共固定 ImageNet 子集"（通过环境变量开关）。"""
    return os.environ.get("HPAT_ELOCAL_USE_PUBLIC_SUBSET") == "1"


def _use_imagenette_subset() -> bool:
    """是否启用 imagenette 固定子集（通过环境变量开关）。"""
    return os.environ.get("HPAT_ELOCAL_USE_IMAGENETTE") == "1"


def _public_subset_dir() -> pathlib.Path:
    """返回公共固定子集的目录（可用环境变量覆盖默认位置）。"""
    return pathlib.Path(
        os.environ.get(
            "HPAT_ELOCAL_PUBLIC_SUBSET_DIR",
            str(REPO_ROOT / "experiments" / "data" / "public_fixed_imagenet_subset"),
        )
    )


def _imagenette_subset_dir() -> pathlib.Path:
    """返回 imagenette 固定子集目录（可用环境变量覆盖默认位置）。"""
    return pathlib.Path(
        os.environ.get(
            "HPAT_ELOCAL_IMAGENETTE_DIR",
            str(REPO_ROOT / "experiments" / "data" / "imagenette160_fixed_subset"),
        )
    )


def _fixed_subset_args() -> list[str]:
    """把 HPAT_ELOCAL_* 环境变量翻译成 run_e_local_fixed_subset.py 的命令行参数。

    若启用了 imagenette/public 子集且未显式指定 dataset/subset，则自动补默认路径；
    其它环境变量（设备、模型变体、批大小、重复种子、阈值等）一一透传。
    """
    args: list[str] = []
    dataset_root = os.environ.get("HPAT_ELOCAL_DATASET_ROOT", "")
    subset_file = os.environ.get("HPAT_ELOCAL_SUBSET_FILE", "")
    if _use_imagenette_subset() and not dataset_root and not subset_file:
        imagenette_dir = _imagenette_subset_dir()
        dataset_root = str(imagenette_dir)
        subset_file = str(imagenette_dir / "fixed_subset.csv")
    if _use_public_subset() and not dataset_root and not subset_file:
        public_dir = _public_subset_dir()
        dataset_root = str(public_dir)
        subset_file = str(public_dir / "fixed_subset.csv")
    if dataset_root:
        args.extend(["--dataset-root", dataset_root])
    if subset_file:
        args.extend(["--subset-file", subset_file])
    # 以下均为可选透传参数，只有设置了对应环境变量才会出现在命令行里
    if os.environ.get("HPAT_ELOCAL_MAX_SAMPLES"):
        args.extend(["--max-samples", os.environ["HPAT_ELOCAL_MAX_SAMPLES"]])
    if os.environ.get("HPAT_ELOCAL_DEVICE"):
        args.extend(["--device", os.environ["HPAT_ELOCAL_DEVICE"]])
    if os.environ.get("HPAT_ELOCAL_MODEL_VARIANT"):
        args.extend(["--model-variant", os.environ["HPAT_ELOCAL_MODEL_VARIANT"]])
    if os.environ.get("HPAT_ELOCAL_PRETRAINED") == "1":
        args.append("--pretrained")
    if os.environ.get("HPAT_ELOCAL_MIN_BOUNDARY_COVERAGE"):
        args.extend(["--min-boundary-coverage", os.environ["HPAT_ELOCAL_MIN_BOUNDARY_COVERAGE"]])
    if os.environ.get("HPAT_ELOCAL_BATCH_SIZE"):
        args.extend(["--batch-size", os.environ["HPAT_ELOCAL_BATCH_SIZE"]])
    if os.environ.get("HPAT_ELOCAL_SAVE_LOGITS"):
        args.extend(["--save-logits", os.environ["HPAT_ELOCAL_SAVE_LOGITS"]])
    if os.environ.get("HPAT_ELOCAL_SAVE_PREDICTION_DETAIL"):
        args.extend(["--save-prediction-detail", os.environ["HPAT_ELOCAL_SAVE_PREDICTION_DETAIL"]])
    if os.environ.get("HPAT_ELOCAL_REPEAT_SEEDS"):
        args.extend(["--repeat-seeds", os.environ["HPAT_ELOCAL_REPEAT_SEEDS"]])
    if os.environ.get("HPAT_ELOCAL_SAFE_TOP1_DROP"):
        args.extend(["--safe-top1-drop-threshold", os.environ["HPAT_ELOCAL_SAFE_TOP1_DROP"]])
    if os.environ.get("HPAT_ELOCAL_SAFE_TOP5_DROP"):
        args.extend(["--safe-top5-drop-threshold", os.environ["HPAT_ELOCAL_SAFE_TOP5_DROP"]])
    if os.environ.get("HPAT_ELOCAL_SAFE_CHANGE_RATE"):
        args.extend(["--safe-prediction-change-threshold", os.environ["HPAT_ELOCAL_SAFE_CHANGE_RATE"]])
    return args


def main() -> None:
    """主入口：组织步骤序列，逐脚本执行并汇总结果。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(REPO_ROOT / "experiments" / "results"))
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json"))
    args = parser.parse_args()

    # 每次运行一个独立时间戳目录，避免覆盖历史结果
    output_dir = ensure_dir(pathlib.Path(args.output_root) / run_id())
    ensure_dir(output_dir / "raw")
    ensure_dir(output_dir / "tables")
    ensure_dir(output_dir / "figures")
    logs_dir = ensure_dir(output_dir / "logs")
    shutil.copy2(args.config, output_dir / "config_snapshot.json")

    script_dir = pathlib.Path(__file__).resolve().parent
    # 按依赖顺序定义步骤序列：(脚本名, 额外参数列表)
    steps: list[tuple[str, list[str]]] = [
        ("collect_environment.py", []),
        ("run_mobilevit_activity_trace.py", []),
        ("run_baseline_provenance_check.py", []),
        ("run_e_local_trace_driven_activity.py", []),
        ("run_energy_break_even.py", []),
        ("run_evidence_strength_summary.py", []),
        ("run_operator_mapping_closure.py", []),
        ("refresh_external_edge_context.py", []),
        ("run_e_local_readiness_summary.py", []),
        ("render_matplotlib_evidence_figures.py", ["--output-dir", str(REPO_ROOT / "experiments" / "results" / "matplotlib_figures_latest")]),
    ]
    # 若启用 imagenette 子集，先准备固定子集数据
    if _use_imagenette_subset():
        prepare_args = [
            "--dataset-dir",
            str(_imagenette_subset_dir()),
            "--source",
            "imagenette160-valid",
        ]
        if os.environ.get("HPAT_ELOCAL_IMAGENETTE_ARCHIVE_URL"):
            prepare_args.extend(["--archive-url", os.environ["HPAT_ELOCAL_IMAGENETTE_ARCHIVE_URL"]])
        steps.insert(5, ("prepare_public_fixed_subset.py", prepare_args))
    # 若启用 public 子集，同样先准备数据
    if _use_public_subset():
        steps.insert(
            5,
            (
                "prepare_public_fixed_subset.py",
                ["--dataset-dir", str(_public_subset_dir())],
            ),
        )
    # 在步骤 5/6 位置插入固定子集精度实验（跑真实 MobileViT）
    fixed_insert = 6 if (_use_public_subset() or _use_imagenette_subset()) else 5
    steps.insert(fixed_insert, ("run_e_local_fixed_subset.py", _fixed_subset_args()))
    results = []
    # 顺序执行每个步骤：日志落盘、记录退出码，失败即停
    for script, extra in steps:
        cmd = [sys.executable, str(script_dir / script)]
        # 除显式传 --output-dir 的脚本外，统一把结果写到本次运行目录
        if "--output-dir" not in extra:
            cmd.extend(["--output-dir", str(output_dir)])
        cmd.extend(extra)
        # 环境采集与子集准备脚本不接收 --config，其余脚本统一传入配置
        if script not in {"collect_environment.py", "prepare_public_fixed_subset.py"} and "--config" not in extra:
            cmd.extend(["--config", args.config])
        proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
        stdout_log = logs_dir / f"{pathlib.Path(script).stem}.stdout.log"
        stderr_log = logs_dir / f"{pathlib.Path(script).stem}.stderr.log"
        stdout_log.write_text(proc.stdout, encoding="utf-8")
        stderr_log.write_text(proc.stderr, encoding="utf-8")
        results.append(
            {
                "script": script,
                "returncode": proc.returncode,
                "stdout_log": relative(stdout_log),
                "stderr_log": relative(stderr_log),
            }
        )
        if proc.returncode != 0:
            break

    # 汇总整条链的运行信息与固定子集配置
    summary = {
        "output_dir": relative(output_dir),
        "python": sys.executable,
        "config_snapshot": relative(output_dir / "config_snapshot.json"),
        "results": results,
        "all_passed": all(row["returncode"] == 0 for row in results),
        "fixed_subset_dataset_root": os.environ.get("HPAT_ELOCAL_DATASET_ROOT", ""),
        "fixed_subset_file": os.environ.get("HPAT_ELOCAL_SUBSET_FILE", ""),
        "public_fixed_subset_enabled": _use_public_subset(),
        "public_fixed_subset_dir": relative(_public_subset_dir()) if _use_public_subset() else "",
        "imagenette_fixed_subset_enabled": _use_imagenette_subset(),
        "imagenette_fixed_subset_dir": relative(_imagenette_subset_dir()) if _use_imagenette_subset() else "",
        "batch_size": os.environ.get("HPAT_ELOCAL_BATCH_SIZE", ""),
        "save_logits": os.environ.get("HPAT_ELOCAL_SAVE_LOGITS", ""),
        "save_prediction_detail": os.environ.get("HPAT_ELOCAL_SAVE_PREDICTION_DETAIL", ""),
        "repeat_seeds": os.environ.get("HPAT_ELOCAL_REPEAT_SEEDS", ""),
        "caffeinate_used": os.environ.get("HPAT_CAFFEINATE_USED") == "1",
        "caffeinate_reason": (
            "Caller set HPAT_CAFFEINATE_USED=1; command was expected to be wrapped in caffeinate."
            if os.environ.get("HPAT_CAFFEINATE_USED") == "1"
            else "Not recorded; wrap long E-local runs in caffeinate -dimsu."
        ),
    }
    write_json(output_dir / "e_local_run_summary.json", summary)
    print(json.dumps({"output_dir": relative(output_dir), "all_passed": summary["all_passed"]}, indent=2))
    # 有失败则以非零码退出，便于上层感知整体成败
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
