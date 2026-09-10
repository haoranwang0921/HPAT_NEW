"""一键跑完所有核心实验的驱动脚本（run_all.py）。

本脚本不自己计算，而是按固定顺序依次调用 scripts 目录下的各个 run_*.py，
把完整实验管线串起来，方便一键复现：
- 输入：读取 hpat_experiment_config.json 实验总配置。
- 产出：在 experiments/results/<run_时间戳>/ 下新建目录（raw/tables/figures/logs），
  拷贝一份配置快照 config_snapshot.json，保存每个子脚本的 stdout/stderr 日志，
  最后汇总成 run_all_summary.json（含各脚本退出码与是否全部通过）。
- 命令：python run_all.py [--output-root 输出根目录] [--config 配置文件路径]

注意：会真实执行实验，跑完约 30 分钟级，属"全量复现"入口。
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


# 按依赖顺序列出要依次执行的脚本：先采集环境、再跑模型基线/能耗/精度/
# 非理想性/规模扩展等核心实验，最后渲染图表与汇总证据强度
SCRIPTS = [
    "collect_environment.py",
    "run_mobilevit_baseline_probe.py",
    "run_edge_baseline_import.py",
    "run_mobilevit_activity_trace.py",
    "run_hpat_activity_export.py",
    "run_energy_accounting.py",
    "run_energy_unit_cost_calibration.py",
    "run_qkv_traffic_sensitivity.py",
    "run_qkv_traffic_calibration.py",
    "run_precision_sweep.py",
    "run_nonideality_sensitivity.py",
    "run_nonideality_accuracy_sweep.py",
    "run_scalability_sweep.py",
    "run_scalability_physical_proxy.py",
    "run_architecture_ablation_sweep.py",
    "run_p1_readiness_summary.py",
    "run_layout_area_feasibility_proxy.py",
    "run_thermal_tuning_stress.py",
    "run_public_edge_context_refresh.py",
    "run_additional_model_family_check.py",
    "run_p2_readiness_summary.py",
    "run_baseline_provenance_check.py",
    "run_evidence_strength_summary.py",
    "render_experiment_figures.py",
]


def main() -> None:
    """主入口：解析命令行参数，依次执行所有核心实验脚本。

    流程：
    1) 解析 --output-root（结果根目录）与 --config（实验配置）。
    2) 创建带时间戳的独立输出目录，避免与历史结果冲突。
    3) 逐个运行 SCRIPTS 中的脚本，把每个脚本的输出/错误日志落盘。
    4) 任一脚本失败立即中断（break），最后写汇总 JSON 并据此决定退出码。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(REPO_ROOT / "experiments" / "results"))
    parser.add_argument("--config", default=str(REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json"))
    args = parser.parse_args()

    # 每个 run 独立目录：results/<run_时间戳>/，内部再分 raw/tables/figures/logs
    output_dir = ensure_dir(pathlib.Path(args.output_root) / run_id())
    ensure_dir(output_dir / "raw")
    ensure_dir(output_dir / "tables")
    ensure_dir(output_dir / "figures")
    logs_dir = ensure_dir(output_dir / "logs")
    # 把本次使用的配置原样复制一份，保证之后能核对"当时用了什么配置"
    shutil.copy2(args.config, output_dir / "config_snapshot.json")
    script_dir = pathlib.Path(__file__).resolve().parent
    results = []
    for script in SCRIPTS:
        cmd = [
            sys.executable,
            str(script_dir / script),
            "--output-dir",
            str(output_dir),
        ]
        # 除环境采集脚本外，其余脚本都统一传入配置路径
        if script != "collect_environment.py":
            cmd.extend(["--config", args.config])
        proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
        # 每个子脚本的 stdout/stderr 分别存日志，便于排查
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
        # 关键：某一步失败就停止后续，避免在错误基础上继续产生误导结果
        if proc.returncode != 0:
            break

    summary = {
        "output_dir": relative(output_dir),
        "python": sys.executable,
        "config_snapshot": relative(output_dir / "config_snapshot.json"),
        "results": results,
        "all_passed": all(r["returncode"] == 0 for r in results),
        # 记录是否用 caffeinate 防止 mac 休眠打断长实验（仅作元信息记录）
        "caffeinate_used": os.environ.get("HPAT_CAFFEINATE_USED") == "1",
        "caffeinate_reason": (
            "Caller set HPAT_CAFFEINATE_USED=1; command was expected to be wrapped in caffeinate."
            if os.environ.get("HPAT_CAFFEINATE_USED") == "1"
            else "Not used or not recorded; run_all default scripts are short unless real torch/timm profiling is enabled."
        ),
    }
    write_json(output_dir / "run_all_summary.json", summary)
    print(json.dumps({"output_dir": relative(output_dir), "all_passed": summary["all_passed"]}, indent=2))
    # 有失败则以非零码退出，便于脚本/CI 感知整体成功与否
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
