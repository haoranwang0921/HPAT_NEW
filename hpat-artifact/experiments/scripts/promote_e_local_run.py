"""E-local 结果提升脚本（promote_e_local_run.py）。

实验目的：把某次 e-local 运行目录 tables/ 下的一组"规范表"（CANONICAL_TABLES）
显式复制提升到仓库公共 tables/ 目录，作为论文可引用的正式产物。
提升是"显式、可控"的：只复制清单里列出的文件，不涉及实验计算本身；
同时保留提升产物里的 claim boundary（证据边界）说明。

- 输入：--run-dir 指向某次 e-local 运行目录（其 tables/ 下需有规范表）。
- 产出：仓库 tables/ 下覆盖对应文件 +
  <输出目录>/promote_e_local_run_manifest.json（记录每个提升文件的来源/哈希）。
- 命令：python promote_e_local_run.py --run-dir <运行目录>
  [--output-dir <清单输出目录>]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

from _common import REPO_ROOT, base_manifest, ensure_dir, relative, sha256_file, write_json


# 允许提升到仓库公共目录的"规范表"清单（白名单，防止误提升其它文件）
CANONICAL_TABLES = [
    "nonideality_accuracy_sweep.csv",
    "nonideality_accuracy_measured.csv",
    "precision_accuracy_sweep.csv",
    "nonideality_accuracy_trials.csv",
    "nonideality_accuracy_by_severity.csv",
    "nonideality_accuracy_safe_region.csv",
    "nonideality_accuracy_failure_thresholds.csv",
    "nonideality_accuracy_fixed_subset.csv",
    "nonideality_accuracy_summary.csv",
    "imagenette_clean_accuracy_by_model.csv",
    "nonideality_accuracy_classwise.csv",
    "nonideality_accuracy_bootstrap_ci.csv",
    "nonideality_accuracy_worst_class.csv",
    "nonideality_margin_drift_by_severity.csv",
    "nonideality_boundary_ablation_summary.csv",
    "nonideality_boundary_ablation_delta.csv",
    "operator_fx_profiler_closure.csv",
    "operator_fx_profiler_closure.md",
    "external_edge_context_online_verification.csv",
    "e_local_readiness_summary.csv",
    "e_local_readiness_summary.json",
    "e_local_readiness_summary.md",
]


def run(run_dir: pathlib.Path, output_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """执行提升主流程：把运行目录下的规范表复制到仓库 tables/。

    参数：
        run_dir：源 e-local 运行目录。
        output_dir：清单（manifest）输出目录。
    返回：{"manifest": 清单文件路径}。
    """
    ensure_dir(output_dir)
    project_tables = ensure_dir(REPO_ROOT / "tables")
    promoted: list[dict[str, str]] = []
    missing: list[str] = []
    for name in CANONICAL_TABLES:
        source = run_dir / "tables" / name
        if not source.exists():
            missing.append(name)
            continue
        destination = project_tables / name
        shutil.copy2(source, destination)
        promoted.append(
            {
                "source": relative(source),
                "destination": relative(destination),
                "sha256": sha256_file(destination) or "",
            }
        )
    manifest_path = output_dir / "promote_e_local_run_manifest.json"
    manifest = base_manifest("promote_e_local_run", "explicit promotion of selected E-local run artifacts")
    manifest.update(
        {
            "status": "ok" if promoted else "no_artifacts_promoted",
            "run_dir": relative(run_dir),
            "promoted": promoted,
            "missing": missing,
            # 免责说明：提升是显式的，且保留各产物中的证据边界说明；
            # 不会提升被阻塞的作者实测边缘/移动或硅片证据线。
            "promotion_note": (
                "Promotion is explicit and preserves claim boundaries in promoted artifacts. "
                "It does not promote blocked author-measured edge/mobile or silicon lanes."
            ),
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; promotion is file copying.",
        }
    )
    write_json(manifest_path, manifest)
    return {"manifest": manifest_path}


def main() -> None:
    """命令行入口：解析参数并调用 run()。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "experiments" / "results" / "e_local_promotion_latest"))
    args = parser.parse_args()
    outputs = run(pathlib.Path(args.run_dir), pathlib.Path(args.output_dir))
    print(json.dumps({key: relative(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
