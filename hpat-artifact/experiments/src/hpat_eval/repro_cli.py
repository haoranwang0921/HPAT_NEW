"""Command-line interface for the HPAT reproducibility package."""
# （英文原 docstring）HPAT 可复现性包的命令行接口。
# 中文说明：本文件是命令行入口（hpat-repro），用法示例：
#     python -m hpat_eval.repro_cli prepare-data --profile paper
#     python -m hpat_eval.repro_cli run --profile paper --output-dir runs/canonical
#     python -m hpat_eval.repro_cli verify --run-dir runs/canonical
#     python -m hpat_eval.repro_cli compare --run-dir runs/canonical --reference release.tar
#     python -m hpat_eval.repro_cli promote --run-dir runs/canonical
#     python -m hpat_eval.repro_cli freeze --run-dir runs/canonical --output-dir runs/freeze
# 它只做"解析参数 + 调用 reproducibility.py 的函数 + 输出 JSON"，
# 真正的打包/哈希/环境锁逻辑都在 reproducibility.py 里。

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from typing import Sequence

from .reproducibility import (
    DEFAULT_PREPARED_DIR,
    PAPER_CONFIG,
    ReproducibilityError,
    prepare_data,
    run_pipeline,
    sha256_file,
    verify_reference_archive,
    write_reference_comparison_report,
    write_verification_report,
)


# 仓库根目录：本文件位于 experiments/src/hpat_eval/ 下，向上 3 级即仓库根
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# 需要"提升"到仓库 tables/ 目录的规范表格清单（promote 命令使用）
PROMOTED_TABLES = (
    "mapping_scenario_speedup_bound.csv",
    "mobilevit_mapping_scenarios.csv",
    "mobilevit_operator_activity.csv",
    "hpat_energy_by_component_reasonable_max_mapping.csv",
    "energy_uncertainty_summary.csv",
    "nonideality_accuracy_by_severity.csv",
    "nonideality_paired_transition_summary.csv",
    "p0_readiness_summary.json",
    "no_silicon_evidence_gate.md",
)


def _release_version() -> str:
    """读取发布规范文件里的 release_version（版本锁定）。

    :return: 发布版本号字符串。
    """
    spec = json.loads(
        (REPO_ROOT / "experiments" / "config" / "release_spec_v1.json").read_text(
            encoding="utf-8"
        )
    )
    return str(spec["release_version"])


def _path(value: str) -> pathlib.Path:
    """argparse 类型转换：字符串 → pathlib.Path。

    :param value: 命令行传入的路径字符串。
    :return: Path 对象。
    """
    return pathlib.Path(value)


def _freeze_release(run_dir: pathlib.Path, output_dir: pathlib.Path, version: str) -> dict:
    """freeze 命令的实现：把已验证的运行结果冻结成发布产物。

    流程：
    1) 校验版本号与发布规范锁定的一致；
    2) 生成验证报告，确认 release_ready 为真；
    3) 调用 scripts/freeze_repro_release.py 子进程做实际冻结；
    4) 解析其 JSON 输出，要求状态为 "Green"。

    :param run_dir: 运行目录。
    :param output_dir: 冻结产物输出目录。
    :param version: 请求的发布版本。
    :return: 冻结子进程返回的 payload 字典。
    :raises ReproducibilityError: 版本不匹配、未就绪或子进程失败时抛出。
    """
    expected_version = _release_version()
    if version != expected_version:
        raise ReproducibilityError(
            f"The release specification is locked to {expected_version}, not {version}"
        )
    verification = write_verification_report(run_dir)
    if not verification.get("release_ready"):
        raise ReproducibilityError(
            "Run is not release-ready: " + "; ".join(verification.get("errors", []))
        )
    command = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "scripts" / "freeze_repro_release.py"),
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(output_dir),
        "--spec",
        str(REPO_ROOT / "experiments" / "config" / "release_spec_v1.json"),
    ]
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip() or "release freeze failed"
        raise ReproducibilityError(detail)
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise ReproducibilityError("Release freezer returned invalid JSON") from exc
    if payload.get("status") != "Green":
        raise ReproducibilityError("Release freeze gate did not reach Green")
    return payload


def _promote_reference_tables(run_dir: pathlib.Path) -> dict:
    """Explicitly update repository canonical experiment tables."""
    # （英文原注释）显式地把仓库的"规范实验表"更新为本次运行的结果。
    # 通俗说：论文附录引用的表格统一从 runs/…/tables/ 复制到仓库根 tables/，
    # 并记录每个文件的 SHA-256。promote 是"提交参考结果"的唯一受控入口。
    verification = write_verification_report(run_dir)
    if not verification.get("release_ready"):
        raise ReproducibilityError(
            "Run is not release-ready: " + "; ".join(verification.get("errors", []))
        )
    source_tables = run_dir / "tables"
    missing = [name for name in PROMOTED_TABLES if not (source_tables / name).is_file()]
    subset = run_dir / "inputs" / "subset.csv"
    if not subset.is_file():
        missing.append("inputs/subset.csv")
    if missing:
        raise ReproducibilityError("Promotion inputs are missing: " + ", ".join(missing))

    table_dir = REPO_ROOT / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    promoted: list[pathlib.Path] = []
    # 逐个复制规范表格
    for name in PROMOTED_TABLES:
        destination = table_dir / name
        shutil.copyfile(source_tables / name, destination)
        promoted.append(destination)
    # 复制标注子集样例
    sample_destination = table_dir / "canonical_v1_samples.csv"
    shutil.copyfile(subset, sample_destination)
    promoted.append(sample_destination)

    return {
        "schema_version": "hpat-reference-table-promotion-v1",
        "status": "Green",
        "source_run": run_dir.as_posix(),
        "verification_manifest_sha256": verification["run_manifest_sha256"],  # 关联验证清单哈希
        "artifacts": [
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(promoted)
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器（定义 6 个子命令及其选项）。

    子命令：
    - prepare-data：准备固定标注子集与固定预训练权重缓存；
    - run：跑一次隔离的 smoke 或 canonical 实验链；
    - verify：校验 run 的 schema、哈希与发布门槛；
    - verify-reference：校验下载的 Release 归档（比对前先验明正身）；
    - compare：把完整论文 run 与不可变参考结果比对；
    - promote：把验证过的 run 的表格提升为仓库规范表；
    - freeze：把验证过的 canonical run 冻结为发布产物。

    :return: 配置好的 ArgumentParser。
    """
    parser = argparse.ArgumentParser(
        prog="hpat-repro",
        description="Prepare, run, verify, and freeze the HPAT experiment artifact.",
    )
    # 全局选项：配置文件（默认用论文 canonical 配置）
    parser.add_argument(
        "--config",
        type=_path,
        default=PAPER_CONFIG,
        help="canonical JSON config (compatibility filename: experiments/config/paper_v1.json)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare-data 子命令
    prepare_parser = subparsers.add_parser(
        "prepare-data", help="prepare the fixed subset and pinned pretrained-weight cache"
    )
    prepare_parser.add_argument("--config", type=_path, default=argparse.SUPPRESS)
    prepare_parser.add_argument("--profile", choices=["paper", "smoke"], default="paper")
    prepare_parser.add_argument(
        "--output-dir",
        type=_path,
        default=None,
        help="isolated prepared-data directory (paper default: runs/prepared/paper_v1)",
    )
    prepare_parser.add_argument(
        "--dataset-root",
        type=_path,
        default=None,
        help="optional existing Imagenette root; selected images are copied into the prepared directory",
    )
    prepare_parser.add_argument(
        "--archive-url",
        default=None,
        help="override the download URL; the canonical profile still enforces the pinned archive SHA-256",
    )
    prepare_parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="skip weight preparation (useful for data-only diagnostics; canonical release remains blocked)",
    )

    # run 子命令
    run_parser = subparsers.add_parser("run", help="run an isolated smoke or canonical experiment chain")
    run_parser.add_argument("--config", type=_path, default=argparse.SUPPRESS)
    run_parser.add_argument("--profile", choices=["paper", "smoke"], default="paper")
    run_parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    run_parser.add_argument("--output-dir", type=_path, required=True)
    run_parser.add_argument(
        "--prepared-dir",
        type=_path,
        default=DEFAULT_PREPARED_DIR,
        help="prepared canonical inputs (ignored by smoke)",
    )
    run_parser.add_argument(
        "--no-caffeinate",
        action="store_true",
        help="do not wrap long canonical subprocesses in caffeinate (the manifest records this)",
    )

    # verify 子命令
    verify_parser = subparsers.add_parser("verify", help="verify run schema, hashes, and release gates")
    verify_parser.add_argument("--run-dir", type=_path, required=True)

    # verify-reference 子命令
    verify_reference_parser = subparsers.add_parser(
        "verify-reference", help="verify a downloaded Release archive before comparison"
    )
    verify_reference_parser.add_argument("--reference", type=_path, required=True)

    # compare 子命令
    compare_parser = subparsers.add_parser(
        "compare", help="compare a complete paper run with the immutable reference results"
    )
    compare_parser.add_argument("--run-dir", type=_path, required=True)
    compare_parser.add_argument("--reference", type=_path, required=True)
    compare_parser.add_argument(
        "--output",
        type=_path,
        default=None,
        help="comparison report path (default: RUN_DIR/reference_comparison_report.json)",
    )

    # promote 子命令
    promote_parser = subparsers.add_parser(
        "promote", help="explicitly update canonical reference tables from a verified run"
    )
    promote_parser.add_argument("--run-dir", type=_path, required=True)

    # freeze 子命令
    freeze_parser = subparsers.add_parser(
        "freeze", help="freeze a verified canonical run into experiment release artifacts"
    )
    freeze_parser.add_argument("--run-dir", type=_path, required=True)
    freeze_parser.add_argument("--output-dir", type=_path, required=True)
    freeze_parser.add_argument("--version", default=_release_version())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主入口：解析参数、分发到对应实现函数、输出 JSON 结果。

    返回码约定：0=成功；1=可复现性错误或校验不通过；2=未知命令。

    :param argv: 命令行参数（None 时用 sys.argv）。
    :return: 进程退出码。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        # 按子命令分发到 reproducibility.py 的对应函数
        if args.command == "prepare-data":
            output_dir = args.output_dir
            if output_dir is None:
                output_dir = DEFAULT_PREPARED_DIR if args.profile == "paper" else pathlib.Path("runs/prepared/smoke")
            payload = prepare_data(
                profile=args.profile,
                output_dir=output_dir,
                config_path=args.config,
                dataset_root=args.dataset_root,
                archive_url=args.archive_url,
                download_weights=False if args.skip_weights else None,
            )
        elif args.command == "run":
            payload = run_pipeline(
                profile=args.profile,
                device_request=args.device,
                output_dir=args.output_dir,
                config_path=args.config,
                prepared_dir=args.prepared_dir,
                use_caffeinate=False if args.no_caffeinate else None,
            )
        elif args.command == "verify":
            payload = write_verification_report(args.run_dir)
        elif args.command == "verify-reference":
            payload = verify_reference_archive(args.reference)
        elif args.command == "compare":
            payload = write_reference_comparison_report(
                args.run_dir, args.reference, args.output
            )
        elif args.command == "promote":
            payload = _promote_reference_tables(args.run_dir)
        elif args.command == "freeze":
            payload = _freeze_release(args.run_dir, args.output_dir, args.version)
        else:  # pragma: no cover - argparse enforces the command set
            parser.error(f"Unsupported command: {args.command}")
            return 2
    except ReproducibilityError as exc:
        # 业务错误统一输出 JSON 到 stderr 并返回 1
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    # 命令特有失败判断：非 Green / 不匹配 / 校验无效都返回非零
    if args.command == "verify-reference" and payload.get("status") != "Green":
        return 1
    if args.command == "compare" and not payload.get("match"):
        return 1
    if args.command == "verify" and not payload.get("valid"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
