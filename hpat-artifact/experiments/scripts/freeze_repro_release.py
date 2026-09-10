"""冻结（frozen）HPAT 可复现发布包的主脚本。

【做什么】把一次完整论文实验的产物"打包固定"成可复现发布包（reproducible
release package）：代码、数据表格、manifest（清单/台账）、SHA256 哈希校验和、
发布规格全部打包在一起，并跑一遍多项审计（表格结构、论文声明、匿名性、Git
提交绑定等），全部通过才判定为 Green（绿灯/合格）。保证任何人拿到同一份包，
都能在相同环境下复现论文里的结果。

【数据从哪来】
- run_dir（--run-dir）：一次完整规范实验运行的输出目录，里面应有
  repro_manifest.json / manifest.json 等台账文件、tables/ 表格、raw/ 原始数据。
- spec（--spec）：发布规格文件 experiments/config/release_spec_v1.json，
  说明版本号、必须包含哪些表格、canonical（标准/金标准）参数、声明边界等。

【输出到哪】output_dir（--output-dir）：冻结后的发布包目录，包含
release_manifest.json（发布清单）、SHA256SUMS（校验和文件）、gate_summary.json
（门禁总表）、table_schema_audit.json 等审计报告，以及 release_assets/ 下
以版本号命名的 tar.gz 发布压缩包。

【怎么运行】
    python freeze_repro_release.py --run-dir <运行目录> --output-dir <输出目录> [--spec <规格文件>]

本文件只负责打包与审计；它不修改任何实验数据，也不运行实验。
"""

from __future__ import annotations

import argparse  # 解析命令行参数（--run-dir / --output-dir / --spec）
import csv  # 读写 CSV 表格，用于表格结构审计
import gzip  # 打包时压缩（gzip 层，mtime 置 0 保证可复现）
import hashlib  # 计算 SHA256 校验和
import io  # 用内存字节流（BytesIO）往 tar 里写覆盖版 manifest
import json  # 读写 JSON 清单/报告
import pathlib  # 跨平台路径操作
import re  # 正则：匹配哈希格式、限定词等
import shutil  # 复制文件
import subprocess  # 调 git 拿 commit 哈希
import sys  # 退出码与 stderr
import tarfile  # 打包/读取 tar.gz
from collections import Counter  # 统计类别分布
from typing import Any, Iterable, Mapping  # 类型标注

# 复用同目录审计脚本的默认单文件体积上限与匿名化审计函数。
from audit_public_release import DEFAULT_MAX_FILE_BYTES, audit_public_release
# 复用 hpat_eval 里的可复现性工具：canonical JSON 序列化、canonical 输出记录、参照包校验。
from hpat_eval.reproducibility import (
    REFERENCE_COMPARISON_SCHEMA_VERSION,
    canonical_json_bytes,
    canonical_output_record,
    verify_reference_archive,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
# 仓库根目录：本文件位于 <repo>/experiments/scripts/ 下，向上退两级即仓库根。
DEFAULT_SPEC = REPO_ROOT / "experiments" / "config" / "release_spec_v1.json"
# 默认的发布规格文件：规定发布版本、必须包含的表格、canonical 参数等。

REPORT_NAMES = {
    # 各类审计/清单报告的文件名，统一在此登记，避免散落各处。
    "manifest": "release_manifest.json",  # 发布清单：总览整个发布包
    "checksums": "SHA256SUMS",  # 校验和文件：每个文件的 SHA256 哈希
    "tables": "table_schema_audit.json",  # 表格结构审计报告
    "claims": "claim_lint.json",  # 论文声明措辞检查（lint）报告
    "anonymity": "anonymity_report.json",  # 匿名化审计报告
    "reference": "reference_package_audit.json",  # 参照发布包审计报告
    "gate": "gate_summary.json",  # 门禁汇总：各检查项是否全部通过
}

FORBIDDEN_CLAIM_PHRASES = (
    # 论文文本/元数据中禁止出现的"过强声明"短语。项目是仿真/校准结果，
    # 这些短语（如"已在硅片上验证"）暗示已流片验证，不允许出现在正式发布物里。
    "validated on silicon",
    "silicon validation",
    "deployed on edge",
    "edge deployment",
    "measured hpat speedup",
    "measured hpat energy",
    "state-of-the-art superiority",
    "full imagenet robustness",
    "full imagenet validation",
    "first silicon",
)

CLAIM_QUALIFIERS = (
    # 声明"限定词"：如果上述禁用短语附近出现了这些词，说明该句已被限定/否认
    # （例如 "not validated on silicon"），就不算违规声明。
    "not ",
    "no ",
    "cannot",
    "does not",
    "do not",
    "must not",
    "outside",
    "blocked",
    "future",
    "context only",
)

TEXT_SUFFIXES = {".bib", ".cff", ".csv", ".json", ".md", ".tex", ".toml", ".txt", ".yaml", ".yml"}
# 需要做声明检查的纯文本类文件后缀。
EXCLUDED_LARGE_DIAGNOSTIC_PATTERNS = (
    # 这些"大型诊断产物"（体积大、含逐样本 logits 等）默认不打进公开发布包，
    # 避免包过大或泄露中间诊断数据；匹配规则用 glob 模式。
    "raw/*.npz",
    "raw/*logits*",
    "raw/*topk_margin*",
    "**/*.pt",
    "**/*.pth",
    "**/*.bin",
    "**/*.mlmodel",
    "**/*.mlpackage/**",
)

P0_REQUIRED_STATUSES = {
    # P0（最高优先级）证据通道（evidence lane）的期望状态。只有部分通道
    # 允许"paper_eligible_with_limitations"（带限制可入论文），其余必须 blocked（封堵），
    # 因为项目并未实际流片验证。
    "P0-E1": "blocked",
    "P0-E2": "blocked",
    "P0-E3": "paper_eligible_with_limitations",
    "P0-E4": "blocked",
}


def _sha256(path: pathlib.Path) -> str:
    """计算文件的 SHA256 哈希值（校验和），用于后续比对文件是否原样未改。

    参数 path：要计算的文件路径。
    返回：64 位十六进制字符串形式的 SHA256 摘要。
    做法：按 1MB 一块分块读取（iter 配哨兵值 b""），避免大文件一次性读进内存。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    """读取一个 JSON 文件并保证其顶层是 JSON 对象（dict）。

    参数 path：JSON 文件路径。
    返回：解析后的 dict。
    若顶层不是对象则抛 ValueError，避免后续代码对非 dict 结构做错误假设。
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    """把 JSON 对象写到磁盘，保证输出是"确定性"的（可复现）。

    参数 path：输出文件路径；payload：要写入的 JSON 对象。
    返回：None。
    细节：indent=2 缩进 + sort_keys=True 键排序 + 末尾换行 + UTF-8 编码，
    这样每次生成内容逐字节一致，方便用哈希直接比对。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _safe_label(path: pathlib.Path, root: pathlib.Path) -> str:
    """把文件路径转成相对仓库根的安全标签，用于报告里的路径展示。

    参数 path：目标文件；root：仓库根目录。
    返回：相对路径字符串；若无法相对化（如文件在仓库外），退回 "external/<文件名>"。
    """
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return f"external/{path.name}"


def _copy_file(source: pathlib.Path, destination: pathlib.Path) -> None:
    """复制单个普通文件到目标位置，并拒绝复制符号链接。

    参数 source：源文件；destination：目标路径（含文件名）。
    返回：None。
    为什么要拒绝符号链接：发布包必须"自包含"，不能偷偷带上指向外部文件的链接，
    否则别人解包后可能读到本机上的其他文件，破坏可复现性。
    """
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Release input is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _release_asset_members(run_dir: pathlib.Path) -> list[str]:
    """Return the sanitized diagnostics allowed in the GitHub Release archive."""

    # 上面英文注释保留：返回"净化后"允许进 GitHub Release 压缩包的成员列表。
    members: list[str] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(run_dir).as_posix()
        pure = pathlib.PurePosixPath(relative)
        if relative.startswith("logs/"):
            continue
        if any(pure.match(pattern) for pattern in EXCLUDED_LARGE_DIAGNOSTIC_PATTERNS):
            continue
        keep = (
            # 允许打包的"白名单"：台账、输入子集、tables/ 表格、
            # 运行目录根下的 *_manifest.json、raw/ 下的 csv/json 小文件。
            relative in {"config_snapshot.json", "repro_manifest.json", "inputs/subset.csv"}
            or relative.startswith("tables/")
            or ("/" not in relative and relative.endswith("_manifest.json"))
            or (relative.startswith("raw/") and pure.suffix.lower() in {".csv", ".json"})
        )
        if keep:
            members.append(relative)
    return members


def _write_deterministic_release_asset(
    run_dir: pathlib.Path,
    output_dir: pathlib.Path,
    version: str,
) -> tuple[pathlib.Path, list[str]]:
    """生成"确定性"的发布压缩包（tar.gz），同一输入永远得到逐字节相同的包。

    参数 run_dir：规范运行的输出目录；output_dir：发布包输出目录；
          version：发布版本号（用作压缩包内目录名）。
    返回：(压缩包路径, 打包成员文件相对路径列表)。
    为什么强调确定性：把 tar 头里的用户 ID、时间戳全部清零、gzip 的 mtime 也
    置 0，这样不同机器/不同时间打出的包哈希一致，别人可以核对校验和。
    """
    members = _release_asset_members(run_dir)
    if not members:
        raise ValueError("No sanitized canonical-run members were selected for the release asset")
    destination = output_dir / "release_assets" / f"hpat-artifact-{version}.tar.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 用"净化版"manifest 覆盖包里原有的 repro_manifest.json，让包里清单只描述
    # 实际打包的文件（大诊断文件被排除后，原 manifest 会指向不存在的文件）。
    manifest_payload = _sanitized_release_manifest(run_dir, members)
    overrides = {"repro_manifest.json": canonical_json_bytes(manifest_payload)}
    with destination.open("wb") as raw_handle:
        # gzip mtime=0 与 tar 头清零（uid/gid/mtime/mode 固定），保证打包结果可复现。
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle:
            with tarfile.open(fileobj=gzip_handle, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for relative in members:
                    source = run_dir / relative
                    override = overrides.get(relative)
                    if override is None:
                        # 普通文件：从磁盘读入 tar 归档，并指定包内路径。
                        info = archive.gettarinfo(
                            str(source), arcname=f"hpat-artifact-{version}/{relative}"
                        )
                    else:
                        # 被覆盖的 manifest：内容来自内存（canonical_json_bytes 的输出）。
                        info = tarfile.TarInfo(f"hpat-artifact-{version}/{relative}")
                        info.size = len(override)
                    # 以下字段全部固定，确保 tar 头不随打包环境变化。
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    if override is None:
                        with source.open("rb") as handle:
                            archive.addfile(info, handle)
                    else:
                        archive.addfile(info, io.BytesIO(override))
    return destination, members


def _sanitized_release_manifest(
    run_dir: pathlib.Path, members: Iterable[str]
) -> dict[str, Any]:
    """Create a self-verifying manifest while preserving the full run digest.
    （构造一个"可自校验"的 manifest，同时保留完整运行的摘要哈希。）

    Large logits and per-sample diagnostics are intentionally absent from the
    public Release archive. The packaged manifest therefore describes only the
    included files, while ``reference_comparison`` retains the immutable digest
    of the complete canonical run for reviewer comparisons.
    （大型 logits 与逐样本诊断数据刻意不进公开压缩包，因此包内 manifest 只描述
    实际打包的文件；而 reference_comparison 字段仍保留完整 canonical 运行的
    不可变哈希，供审稿人把完整复现结果与之比对。）

    参数 run_dir：规范运行输出目录；members：实际打包进压缩包的文件列表。
    返回：净化后的 manifest（dict），其中 outputs 只含打包文件、reference_comparison
    保留完整运行的 digest。
    """
    source_path = run_dir / "repro_manifest.json"
    source = _read_json(source_path)
    # 打包清单 = 全部成员去掉 repro_manifest.json 本身（它会被覆盖重写）。
    packaged_paths = sorted(set(members) - {"repro_manifest.json"})
    # 为每个打包文件记录 path、sha256、bytes，构成可校验的输出台账。
    packaged_outputs = [
        {
            "path": relative,
            "sha256": _sha256(run_dir / relative),
            "bytes": (run_dir / relative).stat().st_size,
        }
        for relative in packaged_paths
    ]
    source_outputs = source.get("outputs", [])
    source_output_paths = {
        str(record.get("path"))
        for record in source_outputs
        if isinstance(record, dict) and record.get("path")
    }
    # 原 manifest 里"没被打包"的文件，单独记录为 excluded_outputs（排除清单）。
    excluded_outputs = [
        dict(record)
        for record in source_outputs
        if isinstance(record, dict) and str(record.get("path")) not in packaged_paths
    ]
    # 安全检查：如果打包文件不在原台账里，说明台账与磁盘不一致，直接报错。
    missing_from_source = sorted(set(packaged_paths) - source_output_paths)
    if missing_from_source:
        raise ValueError(
            "Release members are absent from the canonical output ledger: "
            + ", ".join(missing_from_source)
        )
    full_canonical = source.get("canonical_outputs")
    if not isinstance(full_canonical, dict) or not full_canonical.get("sha256"):
        raise ValueError("Canonical run manifest is missing canonical_outputs")

    # 深拷贝原始 manifest，然后替换 outputs 为打包版、补充 reference_comparison。
    sanitized = json.loads(json.dumps(source))
    sanitized["outputs"] = packaged_outputs
    sanitized["canonical_outputs"] = canonical_output_record(packaged_outputs)
    sanitized["reference_comparison"] = {
        "schema_version": REFERENCE_COMPARISON_SCHEMA_VERSION,
        "source_manifest_sha256": _sha256(source_path),  # 原完整 manifest 的哈希
        "canonical_outputs": full_canonical,  # 完整运行的标准输出哈希，原样保留
        "excluded_outputs": excluded_outputs,
        "note": (
            "Compare a complete reproduced paper run against canonical_outputs. "
            "Excluded files are not redistributed in this sanitized Release archive."
        ),
    }
    return sanitized


def _resolve_named_file(name: str, run_dir: pathlib.Path, repo_root: pathlib.Path) -> pathlib.Path | None:
    """按名称在若干候选位置中查找表格/输入文件。

    参数 name：文件名；run_dir：运行输出目录；repo_root：仓库根。
    返回：找到的第一个存在的文件路径，全都没找到则返回 None。
    查找顺序：运行目录的 tables/ → inputs/ → 根下 → 仓库根的 tables/，
    这样既允许文件出自运行产物，也允许直接引用仓库自带的表格。
    """
    candidates = [
        run_dir / "tables" / name,
        run_dir / "inputs" / name,
        run_dir / name,
        repo_root / "tables" / name,
    ]
    return next((path for path in candidates if path.is_file()), None)


def _resolve_run_manifest(run_dir: pathlib.Path, spec: Mapping[str, Any]) -> pathlib.Path | None:
    """在运行目录中定位"规范运行台账"（manifest）文件。

    参数 run_dir：运行输出目录；spec：发布规格。
    返回：找到的 manifest 路径，找不到返回 None。
    查找策略：先按规格里列出的固定文件名找；找不到则扫目录下 *_manifest.json，
    取第一个带 dataset.sample_count（样本数）字段的，据此判断它属于某次真实实验。
    """
    names = spec.get("canonical_manifest_names", ["repro_manifest.json", "manifest.json"])
    for name in names:
        path = run_dir / str(name)
        if path.is_file():
            return path
    manifests = sorted(run_dir.glob("*_manifest.json"))
    for path in manifests:
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        dataset = payload.get("dataset", {})
        if isinstance(dataset, dict) and dataset.get("sample_count"):
            return path
    return None


def _extract_device(manifest: Mapping[str, Any]) -> tuple[str, bool | None]:
    """从 manifest 中提取运行设备名与是否回退（fallback）到其他设备的标记。

    参数 manifest：运行台账 dict。
    返回：(设备名, fallback_used 或 None)。不同运行版本把设备信息放在不同
    字段（device / dataset / backend / selected_device），这里统一兼容提取。
    """
    device = manifest.get("device")
    if isinstance(device, dict):
        return str(device.get("selected") or device.get("device") or ""), device.get("fallback_used")
    dataset = manifest.get("dataset")
    if isinstance(dataset, dict) and dataset.get("device"):
        return str(dataset.get("device")), manifest.get("fallback_used")
    backend = manifest.get("backend")
    if isinstance(backend, dict):
        return str(backend.get("device") or backend.get("selected") or ""), backend.get("fallback_used")
    return str(manifest.get("selected_device") or device or ""), manifest.get("fallback_used")


def _canonical_run_audit(manifest_path: pathlib.Path | None, spec: Mapping[str, Any]) -> dict[str, Any]:
    """审计"规范运行"（canonical run）是否满足发布规格要求。

    参数 manifest_path：运行台账路径（可能为 None）；spec：发布规格。
    返回：审计结果 dict，含 status（Green/Red）、manifest 文件名、逐项 checks。
    检查项包括：设备、是否回退、样本数、随机种子、实验效应、profile、证据等级，
    并与规格里 canonical 字段的期望值逐项比对。
    """
    canonical = spec.get("canonical", {})
    checks: list[dict[str, Any]] = []
    if manifest_path is None:
        # 连台账都找不到，直接判红。
        return {
            "status": "Red",
            "manifest": "",
            "checks": [{"check": "canonical_manifest_exists", "passed": False, "observed": "missing"}],
        }
    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        # 台账存在但不是合法 JSON，也判红。
        return {
            "status": "Red",
            "manifest": manifest_path.name,
            "checks": [{"check": "canonical_manifest_valid_json", "passed": False, "observed": str(exc)}],
        }

    # 从台账里抽取各项可观察指标（有的在 dataset，有的在 data，做兼容处理）。
    device, fallback = _extract_device(manifest)
    dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
    if not dataset and isinstance(manifest.get("data"), dict):
        dataset = manifest["data"]
    sample_count = dataset.get("sample_count", manifest.get("sample_count"))
    randomness = manifest.get("randomness") if isinstance(manifest.get("randomness"), dict) else {}
    seeds = manifest.get("repeat_seeds") or manifest.get("seeds") or randomness.get("repeat_seeds")
    effects = manifest.get("selected_effects") or manifest.get("effects")
    if not effects:
        # 台账里没直接写效应列表时，退而去读旁边的 config_snapshot.json。
        config_snapshot = manifest_path.parent / "config_snapshot.json"
        if config_snapshot.is_file():
            try:
                config = _read_json(config_snapshot)
                repro = config.get("reproducibility", {})
                nonideality = {}
                if isinstance(repro, dict):
                    nonideality = repro.get("nonideality_run") or repro.get("nonideality") or {}
                effects = nonideality.get("effects") if isinstance(nonideality, dict) else None
            except (OSError, ValueError, json.JSONDecodeError):
                effects = None
    evidence = manifest.get("evidence") if isinstance(manifest.get("evidence"), dict) else {}
    # 证据等级字段在多个地方可能出现过，逐个尝试取。
    evidence_tier = (
        manifest.get("artifact_status")
        or manifest.get("evidence_tier")
        or evidence.get("tier")
        or manifest.get("status")
    )
    profile = manifest.get("profile") or manifest.get("paper_profile")
    # 组装"观测 vs 期望"对照表，期望值为 None 或空列表的项跳过检查。
    observations = {
        "device": (device, canonical.get("device")),
        "fallback_used": (fallback, False if canonical.get("allow_device_fallback") is False else fallback),
        "sample_count": (sample_count, canonical.get("sample_count")),
        "repeat_seeds": (seeds, canonical.get("repeat_seeds")),
        "effects": (sorted(effects or []), sorted(canonical.get("effects", []))),
    }
    for key, (observed, expected) in observations.items():
        if expected is None or expected == []:
            continue
        checks.append({"check": key, "passed": observed == expected, "observed": observed, "expected": expected})
    # profile 必须属于 {期望 profile, "paper"} 之一。
    expected_profile = str(spec.get("canonical_profile") or "")
    allowed_profiles = sorted({value for value in [expected_profile, "paper"] if value})
    if allowed_profiles:
        checks.append(
            {
                "check": "profile",
                "passed": profile in allowed_profiles,
                "observed": profile,
                "expected_one_of": allowed_profiles,
            }
        )
    # 证据等级必须落在规格允许的集合内。
    allowed_evidence = {canonical.get("evidence_tier"), canonical.get("evidence_class")}
    allowed_evidence.discard(None)
    if allowed_evidence and evidence_tier:
        checks.append(
            {
                "check": "evidence_tier",
                "passed": evidence_tier in allowed_evidence,
                "observed": evidence_tier,
                "expected_one_of": sorted(allowed_evidence),
            }
        )
    # 有检查项且全部通过才判 Green。
    return {
        "status": "Green" if checks and all(row["passed"] for row in checks) else "Red",
        "manifest": manifest_path.name,
        "checks": checks,
    }


def _collect_required_tables(
    run_dir: pathlib.Path,
    repo_root: pathlib.Path,
    spec: Mapping[str, Any],
) -> tuple[list[tuple[str, pathlib.Path]], list[dict[str, Any]]]:
    """按发布规格收集"必须包含"的表格文件，并记录每个分组的命中情况。

    参数 run_dir：运行输出目录；repo_root：仓库根；spec：发布规格。
    返回：(去重后的 [(表名, 路径)] 列表, 分组解析结果列表)。
    规格里 required_artifact_groups 定义了若干分组，每组有模式 all（全部必到）
    或 any（至少一个）；结果会记录每组 passed / missing 供审计用。
    """
    selected: list[tuple[str, pathlib.Path]] = []
    resolution: list[dict[str, Any]] = []
    modes = spec.get("required_group_modes", {})
    for group, names in spec.get("required_artifact_groups", {}).items():
        mode = str(modes.get(group, "all"))
        found: list[tuple[str, pathlib.Path]] = []
        missing: list[str] = []
        for raw_name in names:
            name = str(raw_name)
            path = _resolve_named_file(name, run_dir, repo_root)
            if path is None:
                missing.append(name)
            else:
                found.append((name, path))
                if mode == "any":
                    break  # any 模式下找到第一个就够，不用继续找
        passed = bool(found) if mode == "any" else not missing
        resolution.append(
            {
                "group": group,
                "mode": mode,
                "passed": passed,
                "selected": [name for name, _ in found],
                "missing": [] if mode == "any" and found else missing,
            }
        )
        selected.extend(found)
    # 同名表格只保留第一个命中的路径，避免重复打包同一文件。
    deduplicated: dict[str, pathlib.Path] = {}
    for name, path in selected:
        deduplicated.setdefault(name, path)
    return sorted(deduplicated.items()), resolution


def _read_table(path: pathlib.Path) -> tuple[list[str], list[dict[str, str]]]:
    """读取 CSV 表格，返回 (列名列表, 数据行列表，每行是 dict)。

    参数 path：CSV 文件路径。
    返回：(fieldnames, rows)。用 csv.DictReader 把每行按列名映射成字典。
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _table_schema_audit(
    tables: Iterable[tuple[str, pathlib.Path]],
    group_resolution: list[dict[str, Any]],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """对发布包要带的表格做"结构审计"：列名、行数、哈希字段、样本分布等。

    参数 tables：[(表名, 路径)]；group_resolution：分组命中情况；
          spec：发布规格（内含每张表的期望列 table_schemas 与 canonical 参数）。
    返回：审计结果 dict，含 schema_version、status（Green/Red）、逐表 rows、
          以及 issue_count（问题数）。
    特殊处理：
    - p0_readiness_summary.json 按 JSON 结构检查（必须含全部 P0 通道且状态匹配）；
    - CSV 表格检查：必需列是否齐全、image_sha256 是否 64 位十六进制、
      样本数/类别分布是否符合 canonical、claim_boundary 列是否为空。
    """
    rows: list[dict[str, Any]] = []
    expected_schemas = spec.get("table_schemas", {})
    canonical = spec.get("canonical", {})
    # 需要按样本数校验的表格组：spec 的 sample_manifest 分组。
    sample_names = set(spec.get("required_artifact_groups", {}).get("sample_manifest", []))
    for name, path in tables:
        if name == "p0_readiness_summary.json":
            # ---------- 分支一：P0 就绪汇总 JSON，走专门检查 ----------
            try:
                payload = _read_json(path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                rows.append(
                    {
                        "table": name,
                        "format": "json",
                        "status": "Red",
                        "issues": ["invalid_p0_readiness_json"],
                        "error": str(exc),
                    }
                )
                continue
            experiments = payload.get("experiments")
            observed: dict[str, str] = {}
            duplicate_ids: list[str] = []
            if isinstance(experiments, list):
                for record in experiments:
                    if not isinstance(record, dict):
                        continue
                    experiment = str(record.get("experiment", ""))
                    if not experiment:
                        continue
                    if experiment in observed:
                        duplicate_ids.append(experiment)
                    observed[experiment] = str(record.get("status", ""))
            # 缺了哪些必填 P0 通道 / 哪些通道状态与期望不符 / 有无重复实验 ID。
            missing = sorted(set(P0_REQUIRED_STATUSES) - set(observed))
            unexpected = {
                experiment: observed.get(experiment, "missing")
                for experiment, expected in P0_REQUIRED_STATUSES.items()
                if observed.get(experiment) != expected
            }
            issues: list[str] = []
            if not isinstance(experiments, list):
                issues.append("missing_experiments_list")
            if missing:
                issues.append("missing_required_p0_lanes")
            if unexpected:
                issues.append("p0_status_mismatch")
            if duplicate_ids:
                issues.append("duplicate_p0_lane")
            rows.append(
                {
                    "table": name,
                    "format": "json",
                    "status": "Green" if not issues else "Red",
                    "required_statuses": P0_REQUIRED_STATUSES,
                    "observed_statuses": {
                        experiment: observed.get(experiment, "missing")
                        for experiment in P0_REQUIRED_STATUSES
                    },
                    "missing_lanes": missing,
                    "duplicate_lanes": sorted(set(duplicate_ids)),
                    "issues": issues,
                }
            )
            continue
        # ---------- 分支二：普通 CSV 表格 ----------
        required = [str(column) for column in expected_schemas.get(name, [])]
        try:
            columns, data = _read_table(path)
        except (OSError, csv.Error, UnicodeError) as exc:
            rows.append({"table": name, "status": "Red", "error": str(exc)})
            continue
        missing = [column for column in required if column not in columns]
        # 统计 image_sha256 列里不符合 64 位十六进制格式的行数。
        invalid_hash_count = 0
        if "image_sha256" in columns:
            invalid_hash_count = sum(
                1 for row in data if not re.fullmatch(r"[0-9a-fA-F]{64}", row.get("image_sha256", ""))
            )
        # 对样本清单表：行数必须等于 canonical 样本数；若 10 类 1024 张，还要求
        # 类别分布精确为 6 个类各 102 张 + 4 个类各 103 张（ImageNet 采样）。
        sample_count_ok = True
        class_distribution: dict[str, int] = {}
        if name in sample_names:
            sample_count_ok = len(data) == canonical.get("sample_count")
            if "synset" in columns:
                class_distribution = dict(sorted(Counter(row.get("synset", "") for row in data).items()))
                expected_count = int(canonical.get("sample_count", 0))
                if expected_count == 1024 and len(class_distribution) == 10:
                    sample_count_ok = sample_count_ok and sorted(class_distribution.values()) == [102] * 6 + [103] * 4
        # 声称边界列（claim_boundary）不允许留空行，声明必须逐行写清楚。
        empty_boundary_count = 0
        if "claim_boundary" in required and "claim_boundary" in columns:
            empty_boundary_count = sum(1 for row in data if not row.get("claim_boundary", "").strip())
        passed = bool(data) and not missing and not invalid_hash_count and sample_count_ok and not empty_boundary_count
        rows.append(
            {
                "table": name,
                "status": "Green" if passed else "Red",
                "row_count": len(data),
                "required_columns": required,
                "missing_columns": missing,
                "invalid_image_sha256_count": invalid_hash_count,
                "empty_claim_boundary_count": empty_boundary_count,
                "sample_count_ok": sample_count_ok,
                "class_distribution": class_distribution,
            }
        )
    # 汇总：分组失败 + 表格失败都算问题，一个 Green 都不给。
    group_failures = [row for row in group_resolution if not row["passed"]]
    failures = [row for row in rows if row.get("status") != "Green"]
    return {
        "schema_version": "hpat-table-schema-audit-v1",
        "status": "Green" if not group_failures and not failures else "Red",
        "group_resolution": group_resolution,
        "tables": rows,
        "issue_count": len(group_failures) + len(failures),
    }


def _qualified_claim(text: str, phrase: str) -> bool:
    """判断某行文本中的禁用短语是否已被"限定词"限定（从而不算违规）。

    参数 text：整行文本；phrase：FORBIDDEN_CLAIM_PHRASES 中的一个禁用短语。
    返回：True 表示该短语没有裸奔（出现处附近有否定/限定词，可接受）；
          False 表示是"裸声明"（违规）。若短语根本不在行里，返回 True。
    做法：定位短语，截取前后各 72 字符的窗口，看窗口里是否有 CLAIM_QUALIFIERS。
    """
    lowered = text.lower()
    index = lowered.find(phrase)
    if index < 0:
        return True
    window = lowered[max(0, index - 72) : index + len(phrase) + 72]
    return any(qualifier in window for qualifier in CLAIM_QUALIFIERS)


def _claim_lint(paths: Iterable[pathlib.Path], root: pathlib.Path) -> dict[str, Any]:
    """对文本类文件做"声明 lint"：找未完成标记和未加限定的禁用声明。

    参数 paths：要扫描的文件路径；root：仓库根（用于生成相对路径标签）。
    返回：lint 报告 dict，含 schema_version、status、violation_count、violations。
    检查两类问题：
    1. 行内含 TODO / FIXME → 未完成标记（unfinished_marker）；
    2. 行内含 FORBIDDEN_CLAIM_PHRASES 且未被限定 → 不支持的声明（unsupported_claim）。
    """
    violations: list[dict[str, Any]] = []
    for path in sorted(set(paths)):
        # 只看文本类后缀，且跳过超过 8MB 的大文件（文本大文件没必要逐行扫）。
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 8 * 1024 * 1024:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            if "todo" in lowered or "fixme" in lowered:
                violations.append({"path": _safe_label(path, root), "line": lineno, "code": "unfinished_marker"})
            for phrase in FORBIDDEN_CLAIM_PHRASES:
                if phrase in lowered and not _qualified_claim(line, phrase):
                    violations.append(
                        {"path": _safe_label(path, root), "line": lineno, "code": "unsupported_claim", "phrase": phrase}
                    )
    return {
        "schema_version": "hpat-claim-lint-v1",
        "status": "Green" if not violations else "Red",
        "violation_count": len(violations),
        "violations": violations,
    }


def _git_commit(repo_root: pathlib.Path) -> str:
    """取仓库当前 HEAD 的完整 commit 哈希，用于把发布包"绑定"到具体代码版本。

    参数 repo_root：仓库根目录。
    返回：40~64 位十六进制 commit 哈希；拿不到（无 git、非仓库、输出不合法）
    则返回 "unavailable"。
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"
    value = result.stdout.strip()
    return value if re.fullmatch(r"[0-9a-f]{40,64}", value) else "unavailable"


def _artifact_rows(output_dir: pathlib.Path, *, exclude_reports: bool = False) -> list[dict[str, Any]]:
    """枚举输出目录里的所有文件并计算其 SHA256，作为发布包"工件清单"。

    参数 output_dir：发布包输出目录；exclude_reports：为 True 时跳过各类审计报告
          （用于生成清单时避免把报告本身也算进"待校验工件"）。
    返回：[{path, sha256, size_bytes}] 列表。SHA256SUMS 自身不参与统计，
    避免"校验和文件算自己"形成闭环。
    """
    rows: list[dict[str, Any]] = []
    report_set = set(REPORT_NAMES.values())
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == REPORT_NAMES["checksums"]:
            continue
        relative = path.relative_to(output_dir).as_posix()
        if exclude_reports and relative in report_set:
            continue
        rows.append({"path": relative, "sha256": _sha256(path), "size_bytes": path.stat().st_size})
    return rows


def freeze_release(
    run_dir: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    spec_path: pathlib.Path = DEFAULT_SPEC,
    repo_root: pathlib.Path = REPO_ROOT,
) -> dict[str, Any]:
    """主流程：把一次实验运行冻结成可复现发布包并跑完整套审计。

    参数 run_dir：规范实验运行的输出目录；output_dir：发布包输出目录（必须不存在或为空）；
          spec_path：发布规格文件；repo_root：仓库根目录。
    返回：{"status": overall, "output_dir": 绝对路径, "gate_summary": 门禁总表}。
    status 为 "Green" 表示全部门禁通过，否则 "Red"。

    整体步骤：
    1) 生成确定性的 tar.gz 发布压缩包（含净化后的 manifest）；
    2) 对压缩包做参照审计（verify_reference_archive）；
    3) 收集并审计必须的表格，复制到发布包 tables/；
    4) 复制 canonical 配置与运行台账；
    5) 对复制进包的文件做声明 lint；
    6) 两遍匿名化审计：先写"pending"门禁，再审计整个最终目录，
       随后重写门禁与发布清单；若匿名报告状态变化，再收敛一次；
    7) 最后生成 SHA256SUMS 校验和文件（包含全部工件，覆盖门禁与清单）。
    """
    repo_root = repo_root.resolve()
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    spec = _read_json(spec_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Freeze output must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    release_version = str(spec.get("release_version", "v1.0.1"))
    # 步骤1：写发布压缩包（确定性 tar.gz）。
    release_asset, release_asset_members = _write_deterministic_release_asset(
        run_dir, output_dir, release_version
    )
    # 步骤2：校验参照发布包（重新解包验证哈希/结构是否自洽）。
    reference_package_audit = verify_reference_archive(release_asset)
    _write_json(output_dir / REPORT_NAMES["reference"], reference_package_audit)

    # 步骤3：收集规格要求的表格，做结构审计，并把命中的表复制进发布包。
    tables, group_resolution = _collect_required_tables(run_dir, repo_root, spec)
    table_audit = _table_schema_audit(tables, group_resolution, spec)
    copied_sources: list[pathlib.Path] = []
    for name, source in tables:
        _copy_file(source, output_dir / "tables" / name)
        copied_sources.append(source)

    # 步骤4：复制 canonical profile 的配置快照与运行台账。
    config_source = repo_root / "experiments" / "config" / f"{spec.get('canonical_profile', 'paper_v1')}.json"
    if config_source.is_file():
        _copy_file(config_source, output_dir / "config" / config_source.name)
        copied_sources.append(config_source)

    run_manifest = _resolve_run_manifest(run_dir, spec)
    canonical_audit = _canonical_run_audit(run_manifest, spec)
    if run_manifest is not None:
        _copy_file(run_manifest, output_dir / "canonical_run" / run_manifest.name)
        copied_sources.append(run_manifest)

    # 步骤5：对复制进包的文件做声明 lint，并落盘两份审计报告。
    claim_report = _claim_lint(copied_sources, repo_root)
    _write_json(output_dir / REPORT_NAMES["tables"], table_audit)
    _write_json(output_dir / REPORT_NAMES["claims"], claim_report)
    # 收集工件清单（不含报告本身）与 Git commit，用于绑定代码版本。
    payload_rows = _artifact_rows(output_dir, exclude_reports=True)
    repository_commit = _git_commit(repo_root)
    require_git_commit = bool(spec.get("require_git_commit", False))
    # 若规格要求绑定 git commit，则拿不到 commit 时仓库绑定门禁判红。
    repository_binding_status = (
        "Green" if not require_git_commit or repository_commit != "unavailable" else "Red"
    )

    def write_gate_and_manifest(anonymity_status: str) -> tuple[str, dict[str, Any]]:
        """（内嵌函数）写门禁总表与发布清单，返回 (总状态, 门禁总表)。

        参数 anonymity_status：当前匿名化审计状态（pending / Green / Red）。
        返回：(overall_status, summary)。总状态 = 六个门禁全部 Green 才 Green。
        """
        gates = {
            "canonical_run": canonical_audit["status"],  # 规范运行审计
            "reference_package": reference_package_audit["status"],  # 参照包审计
            "table_schema": table_audit["status"],  # 表格结构审计
            "claim_lint": claim_report["status"],  # 声明 lint
            "anonymity_and_size": anonymity_status,  # 匿名化与体积
            "repository_binding": repository_binding_status,  # 仓库 commit 绑定
        }
        overall_status = "Green" if all(status == "Green" for status in gates.values()) else "Red"
        summary = {
            "schema_version": "hpat-release-gate-v1",
            "status": overall_status,
            "release_version": spec.get("release_version"),
            "gates": gates,
            "canonical_run_audit": canonical_audit,
            "repository_binding": {
                "required": require_git_commit,
                "commit": repository_commit,
                "policy": (
                    "The frozen release is generated after the single public commit; the release manifest and "
                    "Git tag bind that immutable commit without creating a self-referential tracked manifest."
                ),
            },
            "blocked_evidence_lanes": spec.get("blocked_evidence_lanes", []),
            "claim_boundary": spec.get("claim_boundary", ""),
        }
        _write_json(output_dir / REPORT_NAMES["gate"], summary)
        manifest = {
            "schema_version": "hpat-repro-release-manifest-v1",
            "status": overall_status,
            "release_version": spec.get("release_version"),
            "canonical_profile": spec.get("canonical_profile"),
            "commit": repository_commit,
            "source_run": _safe_label(run_dir, repo_root),
            "canonical": spec.get("canonical", {}),
            "release_asset": {
                "path": release_asset.relative_to(output_dir).as_posix(),
                "sha256": _sha256(release_asset),
                "size_bytes": release_asset.stat().st_size,
                "members": release_asset_members,
                "excludes_dataset_weights_and_logits": True,
            },
            "artifacts": payload_rows,
            "artifact_count": len(payload_rows),
            "reports": {key: value for key, value in REPORT_NAMES.items() if key != "checksums"},
            "excluded_large_diagnostics": list(EXCLUDED_LARGE_DIAGNOSTIC_PATTERNS),
            "blocked_evidence_lanes": spec.get("blocked_evidence_lanes", []),
            "claim_boundary": spec.get("claim_boundary", ""),
        }
        _write_json(output_dir / REPORT_NAMES["manifest"], manifest)
        return overall_status, summary

    # Write provisional reports, then scan the complete final-format tree.  The
    # anonymity report excludes only itself, so release_manifest and gate_summary
    # are covered rather than being trusted as unscanned generated metadata.
    # 步骤6：先写一份"pending（待定）"门禁，再对完整最终目录做匿名化审计；
    # 匿名报告只排除自己，因此 release_manifest / gate_summary 也要被它覆盖，
    # 而不是当作"免检"的生成元数据直接信任。
    write_gate_and_manifest("pending")
    anonymity_path = output_dir / REPORT_NAMES["anonymity"]
    anonymity_report = audit_public_release(
        output_dir,
        output=anonymity_path,
        max_file_bytes=DEFAULT_MAX_FILE_BYTES,
    )
    # 用真实匿名结果重写门禁与清单。
    overall, gate_summary = write_gate_and_manifest(anonymity_report["status"])
    # 重写后目录又变了，再审计一次；若状态变化则再收敛一次，保证报告自洽。
    final_anonymity_report = audit_public_release(
        output_dir,
        output=anonymity_path,
        max_file_bytes=DEFAULT_MAX_FILE_BYTES,
    )
    if final_anonymity_report["status"] != anonymity_report["status"]:
        overall, gate_summary = write_gate_and_manifest(final_anonymity_report["status"])

    # 步骤7：最后生成 SHA256SUMS（含所有工件），作为最终的完整性校验依据。
    checksum_rows = _artifact_rows(output_dir)
    checksum_text = "".join(f"{row['sha256']}  {row['path']}\n" for row in checksum_rows)
    (output_dir / REPORT_NAMES["checksums"]).write_text(checksum_text, encoding="utf-8", newline="\n")
    return {"status": overall, "output_dir": output_dir.as_posix(), "gate_summary": gate_summary}


def main() -> int:
    """命令行入口：解析参数并调用 freeze_release，输出 JSON 结果。

    命令行用法：
        --run-dir   规范运行输出目录（必填）
        --output-dir 发布包输出目录（必填）
        --spec      发布规格文件（可选，默认 experiments/config/release_spec_v1.json）
    返回：进程退出码。0 表示门禁通过（Green）；1 表示门禁未通过；2 表示运行出错。
    """
    parser = argparse.ArgumentParser(description="Freeze the bounded HPAT experiment release.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    args = parser.parse_args()
    try:
        result = freeze_release(
            pathlib.Path(args.run_dir),
            pathlib.Path(args.output_dir),
            spec_path=pathlib.Path(args.spec),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        # 出错信息输出到 stderr，并以退出码 2 表示"脚本自身失败"。
        print(json.dumps({"status": "Red", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "Green" else 1


if __name__ == "__main__":
    sys.exit(main())  # 作为脚本直接运行时进入 main，返回值作为进程退出码
