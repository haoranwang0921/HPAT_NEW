# =============================================================================
# 可复现性（reproducibility）打包核心库 —— 模块导读
# =============================================================================
# 这是 hpat-artifact 里"把实验变成可复现包"的核心模块，也是本库最大的文件。
# 它的目标：让任何人拿到仓库 + 一条命令，就能原样重跑论文实验并逐字节
# 对得上（哈希一致）。为此它做了四件大事：
#
# 1) 数据锁（prepare_data / validate_prepared_data）：
#    下载/选样 Imagenette 固定 1024 样本子集 + 钉死（pin）的 MobileViT 预训练
#    权重，所有文件记录 SHA-256，任何字节不一致直接报错，拒绝"悄悄换了数据"。
#
# 2) 环境锁（run_pipeline）：
#    锁定 Python 版本、torch/timm/numpy/Pillow 版本、Git commit、随机种子、
#    设备（auto/cpu/mps/cuda）、caffeinate 防睡眠等；canonical（论文）模式
#    明确禁止把显式请求的加速器悄悄退回 CPU。
#
# 3) 哈希审计（verify_run / canonical_output_record）：
#    跑完后给 run 目录里每个文件算哈希并写 repro_manifest.json；再重新算一遍
#    比对，确认无人动过。输出还分"canonical 科学输出"（CSV/NPZ）与
#    "非 canonical 诊断"（每样本置信度细节，MPS 下浮点可能轻微不同，故不
#    参与跨机比对）。
#
# 4) 发布冻结（freeze_run / verify_reference_archive / compare_run_to_reference）：
#    把验证过的 run 挑出"引用文件"（tables、manifest、subset）打包成确定性
#    tar.gz（所有时间戳/用户/权限归零，保证同内容同哈希）；读者下载后先
#    自校验，再与自己的 run 比对 canonical 摘要。
#
# 设计原则（原英文 docstring）：本模块只用 Python 标准库；真正耗时的科学
# 实验都在 experiments/scripts/ 里，通过子进程调用（run_pipeline 里一条条
# _run_experiment_script），编排边界放在这里，让 CLI 有唯一确定的数据契约，
# 而不重复科学内核。
# =============================================================================

"""Reproducibility package primitives for the HPAT experiment artifact.

This module deliberately uses only the Python standard library.  The expensive
experiment implementations remain in ``experiments/scripts`` and are invoked
as subprocesses by :func:`run_pipeline`.  Keeping the orchestration boundary
here gives the public CLI one deterministic data contract without duplicating
the scientific kernels.
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
SCRIPTS_DIR = EXPERIMENTS_DIR / "scripts"
TEST_FIXTURES_DIR = EXPERIMENTS_DIR / "tests" / "fixtures"
PAPER_CONFIG = EXPERIMENTS_DIR / "config" / "paper_v1.json"
DEFAULT_PREPARED_DIR = REPO_ROOT / "runs" / "prepared" / "paper_v1"

PREPARED_SCHEMA_VERSION = "hpat-prepared-data-v1"
REPRO_MANIFEST_SCHEMA_VERSION = "hpat-repro-manifest-v1"
VERIFICATION_SCHEMA_VERSION = "hpat-repro-verification-v1"
REFERENCE_VERIFICATION_SCHEMA_VERSION = "hpat-reference-verification-v1"
REFERENCE_COMPARISON_SCHEMA_VERSION = "hpat-reference-comparison-v1"
FREEZE_SCHEMA_VERSION = "hpat-repro-freeze-v1"

PAPER_SAMPLE_COUNT = 1024
PAPER_CLASS_COUNT = 10
PAPER_SAMPLING_SEED = 20260706
PAPER_REPEAT_SEEDS = [20260706, 20260707, 20260708, 20260709, 20260710]
PAPER_ARTIFACT_STATUS = "paper_eligible_with_limitations"
PAPER_EVIDENCE_TIER = "local-modelled diagnostic"

SUBSET_FIELDS = ["path", "label", "synset", "class_name", "image_sha256"]
CLAIM_BOUNDARY = (
    "Fixed-subset local/modelled diagnostic evidence only; not full ImageNet "
    "validation, not fabricated silicon, not calibrated silicon energy, not "
    "author-measured HPAT edge/mobile deployment, and not cross-platform superiority."
)


class ReproducibilityError(RuntimeError):
    """Raised when an artifact contract or a pipeline prerequisite is invalid."""
    # （英文原注释）当产物契约或管线前置条件不合法时抛出。
    # 通俗说：一切"可复现性被破坏"的情况（哈希不匹配、目录非空、schema 不对、
    # 数据缺失）都抛这个异常，由 CLI 统一转成 JSON 错误输出。


class PipelineExecutionError(ReproducibilityError):
    """Raised after a wrapped experiment script exits unsuccessfully."""
    # （英文原注释）被包装的实验脚本非零退出时抛出。
    # 与 ReproducibilityError 的区别：这个专门表示"科学脚本本身跑挂了"。


@dataclass(frozen=True)
class DeviceSelection:
    """设备选择结果：记录请求/实际选择/原因/是否回退/是否 canonical 后端。
    frozen=True 表示不可变，保证记录写入 manifest 后不会被意外篡改。
    """
    requested: str
    selected: str
    reason: str
    fallback_used: bool
    canonical_backend: bool

    def as_dict(self) -> dict[str, Any]:
        """把设备选择记录转成普通字典（供写入 manifest JSON）。"""
        return {
            "requested": self.requested,
            "selected": self.selected,
            "reason": self.reason,
            "fallback_used": self.fallback_used,
            "canonical_backend": self.canonical_backend,
        }


def utc_now() -> str:
    """返回当前 UTC 时间的 ISO 字符串（精确到秒），用于清单打时间戳。"""
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def sha256_file(path: pathlib.Path) -> str:
    """计算文件 SHA-256 哈希（分 1MiB 块读取，避免大文件占满内存）。

    :param path: 文件路径。
    :return: 十六进制摘要字符串。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """计算字节串的 SHA-256 哈希（用于内存里的清单/摘要）。"""
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    """把对象序列化成"canonical JSON 字节"：缩进 2、键排序、UTF-8、末尾换行。

    同一份数据无论谁在哪个机器上调用，输出字节都完全一致——这是跨机
    比对哈希的前提。

    :param payload: 可 JSON 序列化的对象。
    :return: 规范化后的 UTF-8 字节串。
    """
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path: pathlib.Path, payload: Any) -> None:
    """以 canonical JSON 格式写文件（自动建父目录）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def load_json(path: pathlib.Path) -> dict[str, Any]:
    """读取 JSON 文件并强制要求顶层是字典（manifest 结构约定）。

    :param path: JSON 文件路径。
    :return: 顶层字典。
    :raises ReproducibilityError: 顶层不是 JSON 对象时抛出。
    """
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ReproducibilityError(f"Expected a JSON object: {path}")
    return payload


def write_subset_csv(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """把子集行写入 CSV（按 SUBSET_FIELDS 列序，含表头）。

    :param path: CSV 输出路径。
    :param rows: 子集行列表（每行含 path/label/synset/class_name/image_sha256）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUBSET_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUBSET_FIELDS})


def read_csv_rows(path: pathlib.Path) -> list[dict[str, str]]:
    """读取 CSV 文件为行字典列表（首行为表头）。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _is_relative_to(path: pathlib.Path, root: pathlib.Path) -> bool:
    """判断 path 是否位于 root 目录之下（路径逃逸安全检测用）。

    :param path: 待判断路径。
    :param root: 基准根目录。
    :return: True 表示在 root 之内。
    """
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_join(root: pathlib.Path, relative_path: str) -> pathlib.Path:
    """把相对路径拼到根目录下，并阻止任何"越狱"路径（如 ../ 或绝对路径）。

    这是防路径穿越（zip-slip）攻击的安全函数：解压归档、读数据子集时
    都经过它，保证解析后的路径绝不逃出 root。

    :param root: 根目录。
    :param relative_path: 相对路径字符串。
    :return: 拼接并解析后的安全路径。
    :raises ReproducibilityError: 路径逃逸根目录时抛出。
    """
    candidate = (root / pathlib.PurePosixPath(relative_path)).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and not _is_relative_to(candidate, root_resolved):
        raise ReproducibilityError(f"Path escapes the dataset root: {relative_path}")
    return candidate


def _command_path(path: pathlib.Path) -> str:
    """Return a usable cwd-relative path without embedding a local account path."""
    # （英文原注释）返回可用的相对路径，避免把本机账号路径嵌进命令记录。
    # 通俗说：manifest 里记录的命令必须"可移植"——不能带 /Users/whr 这种
    # 私人绝对路径，否则换台机器就复现不了。这里统一折成相对仓库根的路径。
    return pathlib.Path(os.path.relpath(path.resolve(), REPO_ROOT.resolve())).as_posix()


def _run_relative(path: pathlib.Path, run_dir: pathlib.Path) -> str:
    """把 run 目录内的文件路径折成相对 run 目录的字符串；越界则报错。

    :param path: 文件路径。
    :param run_dir: run 根目录。
    :return: 相对 run 目录的 POSIX 路径。
    :raises ReproducibilityError: 输出逃出 run 目录时抛出。
    """
    resolved = path.resolve()
    root = run_dir.resolve()
    if not _is_relative_to(resolved, root):
        raise ReproducibilityError(f"Output escaped the run directory: {path}")
    return resolved.relative_to(root).as_posix()


def _require_empty_output(path: pathlib.Path) -> None:
    """要求输出目录"不存在或为空"，防止把两次 run 的结果混在一起。

    :param path: 输出目录。
    :raises ReproducibilityError: 目录已存在且有内容时抛出。
    """
    if path.exists() and any(path.iterdir()):
        raise ReproducibilityError(
            f"Output directory must be absent or empty to avoid mixing runs: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def load_paper_config(config_path: pathlib.Path = PAPER_CONFIG) -> dict[str, Any]:
    """加载并校验论文 canonical 配置文件。

    校验两点：schema_version 必须匹配；必须带 reproducibility 配置段。

    :param config_path: 配置文件路径。
    :return: 配置字典。
    :raises ReproducibilityError: 版本不支持或缺 reproducibility 段时抛出。
    """
    config = load_json(config_path)
    if config.get("schema_version") != "hpat-paper-config-v1":
        raise ReproducibilityError("paper_v1 config has an unsupported schema_version")
    repro = config.get("reproducibility")
    if not isinstance(repro, dict):
        raise ReproducibilityError("paper_v1 config is missing reproducibility settings")
    return config


def _canonical_source_row(
    row: Mapping[str, Any], dataset_root: pathlib.Path, *, verify_declared_hash: bool = True
) -> dict[str, Any]:
    """把一行"源数据行"转成"canonical 子集行"，并逐项校验。

    校验内容：路径必须相对且文件存在；若行里声明了 image_sha256 则必须
    与磁盘文件哈希一致（防篡改）；必须带 label。输出行固定五个字段，
    保证下游（选样、写 CSV、算摘要）看到的字段完全统一。

    :param row: 源行（可能来自 CSV 或目录扫描）。
    :param dataset_root: 数据集根目录。
    :param verify_declared_hash: 是否校验声明哈希。
    :return: canonical 子集行。
    :raises ReproducibilityError: 任一项校验失败时抛出。
    """
    relative_path = str(row.get("path") or row.get("image_path") or row.get("filename") or "")
    if not relative_path or pathlib.PurePath(relative_path).is_absolute():
        raise ReproducibilityError("Each dataset row must contain a relative image path")
    image_path = _safe_join(dataset_root, relative_path)
    if not image_path.is_file():
        raise ReproducibilityError(f"Dataset image is missing: {relative_path}")
    actual_hash = sha256_file(image_path)
    declared_hash = str(row.get("image_sha256") or row.get("sha256") or "")
    if verify_declared_hash and declared_hash and declared_hash.lower() != actual_hash:
        raise ReproducibilityError(f"Image SHA-256 mismatch: {relative_path}")
    synset = str(row.get("synset") or image_path.parent.name)  # 无 synset 时用父目录名
    label = row.get("label", "")
    if label in (None, ""):
        raise ReproducibilityError(f"Dataset row has no label: {relative_path}")
    return {
        "path": pathlib.PurePosixPath(relative_path).as_posix(),
        "label": int(label),
        "synset": synset,
        "class_name": str(row.get("class_name") or row.get("class") or ""),
        "image_sha256": actual_hash,
        "archive_sha256": str(row.get("archive_sha256") or ""),
    }


def source_rows_from_csv(dataset_root: pathlib.Path, source_csv: pathlib.Path) -> list[dict[str, Any]]:
    """从"数据集台账 CSV"读取并规范化源行。

    :param dataset_root: 数据集根目录。
    :param source_csv: 台账 CSV 路径。
    :return: canonical 源行列表。
    :raises ReproducibilityError: CSV 为空时抛出。
    """
    rows = read_csv_rows(source_csv)
    if not rows:
        raise ReproducibilityError(f"Dataset ledger is empty: {source_csv}")
    return [_canonical_source_row(row, dataset_root) for row in rows]


def _prepare_synthetic_smoke_rows(data_root: pathlib.Path) -> list[dict[str, Any]]:
    """Create tiny deterministic images without redistributing third-party data."""
    # （英文原注释）生成极小的确定性图片，避免重新分发第三方数据。
    # 通俗说：smoke（冒烟）模式不需要真的 Imagenette，这里用标准库手工
    # 生成 3 张纯色 PPM 图（8×8 像素），每张对应一个"类"。
    rows: list[dict[str, Any]] = []
    colors = [(220, 40, 40), (40, 180, 80), (40, 80, 220)]  # 红/绿/蓝三色
    for index, color in enumerate(colors):
        relative_path = pathlib.PurePosixPath(f"synthetic/class_{index}/sample_{index}.ppm")
        image_path = _safe_join(data_root, relative_path.as_posix())
        image_path.parent.mkdir(parents=True, exist_ok=True)
        # Portable pixmap keeps the smoke fixture standard-library-only while
        # remaining readable by Pillow/timm when a caller elects to load it.
        # （英文原注释）PPM 格式让冒烟夹具只用标准库即可生成，同时 Pillow/timm
        # 也能读它。下面用纯色填充 8×8 像素。
        pixels = bytes(color) * (8 * 8)
        image_path.write_bytes(b"P6\n8 8\n255\n" + pixels)
        rows.append(
            {
                "path": relative_path.as_posix(),
                "label": index,
                "synset": f"smoke{index}",
                "class_name": f"synthetic-smoke-{index}",
                "image_sha256": sha256_file(image_path),
                "archive_sha256": "",
            }
        )
    return rows


def scan_imagenette_rows(dataset_root: pathlib.Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """扫描 Imagenette-160 验证集的 10 个类别目录，生成源行列表。

    在几个候选目录里找包含全部类别子目录的那一个（imagenette2-160/val、
    val、或数据集根），然后逐类别枚举图像文件（jpg/jpeg/png），
    每张图记录路径/标签/synset/类名/哈希。

    :param dataset_root: 数据集根目录。
    :param config: 论文配置（读 reproducibility.dataset.classes）。
    :return: 源行列表。
    :raises ReproducibilityError: 找不到全部 10 个类目录时抛出。
    """
    class_rows = config["reproducibility"]["dataset"]["classes"]
    class_by_synset = {str(row["synset"]): row for row in class_rows}
    candidates = [
        dataset_root / "imagenette2-160" / "val",
        dataset_root / "val",
        dataset_root,
    ]
    # 找到同时包含全部类目录的那个路径
    val_dir = next((path for path in candidates if all((path / synset).is_dir() for synset in class_by_synset)), None)
    if val_dir is None:
        raise ReproducibilityError("Could not locate all ten Imagenette validation class directories")
    rows: list[dict[str, Any]] = []
    for synset in sorted(class_by_synset):
        info = class_by_synset[synset]
        for image_path in sorted((val_dir / synset).iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue  # 只收常见图像格式
            relative_path = image_path.relative_to(dataset_root).as_posix()
            rows.append(
                {
                    "path": relative_path,
                    "label": int(info["label"]),
                    "synset": synset,
                    "class_name": str(info["class_name"]),
                    "image_sha256": sha256_file(image_path),
                    "archive_sha256": "",
                }
            )
    return rows


def stratified_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    total: int = PAPER_SAMPLE_COUNT,
    seed: int = PAPER_SAMPLING_SEED,
    expected_synsets: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Select the canonical fixed subset with one independently seeded draw per class.

    For ten classes and 1024 samples this yields 102 samples per class plus one
    additional sample for the first four lexicographically sorted synsets.
    Sampling is without replacement; output rows are sorted for byte stability.
    """
    # （英文原注释）用"每个类独立播种抽取"选择 canonical 固定子集。
    # 10 类 1024 样本 = 每类 102 张 + 按字典序前 4 个 synset 各多 1 张。
    # 无放回抽样；输出按路径排序保证字节级稳定。
    #
    # 通俗理解分层抽样：先按类别分组，每类内部用同一个种子（随机数起点）
    # 抽 required 张。这样任何机器重跑，抽出的集合完全一样。
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        synset = str(row.get("synset") or "")
        if not synset:
            raise ReproducibilityError("Every sampling row must include a synset")
        grouped[synset].append(row)
    synsets = sorted(expected_synsets if expected_synsets is not None else grouped)
    if not synsets:
        raise ReproducibilityError("Cannot sample from an empty dataset")
    # 类集合必须与预期完全一致（不多不少）
    if set(grouped) != set(synsets):
        missing = sorted(set(synsets) - set(grouped))
        extra = sorted(set(grouped) - set(synsets))
        raise ReproducibilityError(f"Dataset class mismatch; missing={missing}, extra={extra}")
    base, remainder = divmod(total, len(synsets))  # 每类基数 + 余数分摊
    selected: list[dict[str, Any]] = []
    for class_index, synset in enumerate(synsets):
        required = base + (1 if class_index < remainder else 0)  # 前 remainder 类多抽 1 张
        candidates = sorted(grouped[synset], key=lambda row: str(row.get("path", "")))
        unique_paths = {str(row.get("path", "")) for row in candidates}
        if len(unique_paths) != len(candidates):
            raise ReproducibilityError(f"Duplicate image paths in class {synset}")
        if len(candidates) < required:
            raise ReproducibilityError(
                f"Class {synset} has {len(candidates)} samples but {required} are required"
            )
        # The protocol intentionally resets the same seed for every class.
        # （英文原注释）协议故意对每个类重置同一个种子（保证确定性）。
        draw = random.Random(seed).sample(candidates, required)
        selected.extend(sorted(draw, key=lambda row: str(row.get("path", ""))))
    selected.sort(key=lambda row: (str(row.get("synset", "")), str(row.get("path", ""))))
    return selected


def subset_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """统计子集的类别分布（synset → 样本数），按键排序后返回。"""
    return dict(sorted(Counter(str(row["synset"]) for row in rows).items()))


def subset_content_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """计算子集"内容摘要"：把每行的 path/label/synset/哈希拼串再哈希。

    这个摘要锁住的是"选了哪些图片"，即使 CSV 里换了顺序或改了表头，
    只要选出的图片集合一致，摘要就不变——用于跨 run 比对子集是否一致。

    :param rows: 子集行。
    :return: SHA-256 摘要。
    """
    payload = "".join(
        f"{row['path']}\0{row['label']}\0{row['synset']}\0{row['image_sha256']}\n"
        for row in rows
    ).encode("utf-8")
    return sha256_bytes(payload)


def _download(url: str, destination: pathlib.Path) -> pathlib.Path:
    """从 URL 下载文件到目标路径；也支持本地文件路径/file:// 协议。

    网络下载时先写 .part 临时文件再原子改名，避免下载中途被杀留下半截文件。
    超时 180 秒，防止网络挂起卡死整个管线。

    :param url: 下载地址（http/https/file/本地路径）。
    :param destination: 目标文件路径。
    :return: 目标路径。
    :raises ReproducibilityError: 本地源不存在时抛出。
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in {"", "file"}:
        # 本地复制路径：file:// 需先还原成普通路径
        source = pathlib.Path(urllib.request.url2pathname(parsed.path) if parsed.scheme == "file" else url)
        if not source.is_file():
            raise ReproducibilityError(f"Download source does not exist: {url}")
        shutil.copyfile(source, destination)
        return destination
    request = urllib.request.Request(url, headers={"User-Agent": "hpat-artifact/1.0"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        temporary.replace(destination)  # 原子替换
    finally:
        if temporary.exists():
            temporary.unlink()  # 清理残留的 .part 文件
    return destination


def _safe_extract_tar(archive_path: pathlib.Path, destination: pathlib.Path) -> None:
    """安全解压 tar.gz：解压前逐成员检查，拒绝路径逃逸与链接文件。

    这是防 zip-slip 的第二个闸口：先遍历归档成员，凡目标路径逃出解压目录、
    或是符号链接/硬链接的一律报错，之后才真正 extractall。

    :param archive_path: 归档文件路径。
    :param destination: 解压目标目录。
    :raises ReproducibilityError: 存在不安全成员时抛出。
    """
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if target != destination_resolved and not _is_relative_to(target, destination_resolved):
                raise ReproducibilityError(f"Unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise ReproducibilityError(f"Links are not accepted in the dataset archive: {member.name}")
        archive.extractall(destination)


def _materialize_selected_images(
    rows: Sequence[Mapping[str, Any]], source_root: pathlib.Path, target_root: pathlib.Path
) -> list[dict[str, Any]]:
    """把选中的图片从源数据集复制到隔离的 prepared 目录，并复核哈希。

    :param rows: 选中的子集行。
    :param source_root: 源数据集根目录。
    :param target_root: 目标（prepared）目录。
    :return: 复制后的行（内容不变）。
    :raises ReproducibilityError: 复制后哈希与声明不一致时抛出。
    """
    materialized: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        relative_path = str(row["path"])
        source = _safe_join(source_root, relative_path)
        target = _safe_join(target_root, relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        # 复制完立即复核哈希，确保搬运过程没有破坏内容
        if sha256_file(target) != row["image_sha256"]:
            raise ReproducibilityError(f"Copied image hash changed: {relative_path}")
        materialized.append(row)
    return materialized


def _weight_cache_repo_name(repo_id: str) -> str:
    """把 HuggingFace repo id 转成 HF hub 缓存目录名（models-- 前缀 + -- 替换 /）。

    例：repo_id="timm/mobilevit_xxs" → "models--timm--mobilevit_xxs"。
    """
    return "models--" + repo_id.replace("/", "--")


def _weight_download_url(record: Mapping[str, Any]) -> str:
    """拼出某个权重的 HF 下载 URL（repo_id/revision/filename 各做 URL 编码）。

    :param record: 权重记录（repo_id/revision/filename）。
    :return: https://huggingface.co/<repo>/resolve/<revision>/<filename>。
    """
    repo_id = urllib.parse.quote(str(record["repo_id"]), safe="/")
    revision = urllib.parse.quote(str(record["revision"]), safe="")
    filename = urllib.parse.quote(str(record["filename"]), safe="/")
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"


def _prepare_weight_cache(output_dir: pathlib.Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """把单个"钉死"的模型权重放进隔离的 HF 缓存目录，并校验哈希。

    布局（与 huggingface_hub 兼容，便于 timm 离线加载）：
      hf_home/hub/models--<repo>/blobs/<sha256>   # 权重本体
      hf_home/hub/models--<repo>/snapshots/<revision>/<filename>  # 软链/拷贝
      hf_home/hub/models--<repo>/refs/main        # revision 指针
    优先复用本机全局缓存（~/.cache/huggingface/hub）里已验证的 blob，
    否则下载；下载后必须与 record["sha256"] 完全一致。

    :param output_dir: prepared 输出目录。
    :param record: 权重记录（model/repo_id/revision/filename/sha256）。
    :return: 权重缓存记录（含 cache_path/verified=True）。
    :raises ReproducibilityError: 哈希不匹配时抛出。
    """
    expected_hash = str(record["sha256"])
    repo_name = _weight_cache_repo_name(str(record["repo_id"]))
    cache_repo = output_dir / "hf_home" / "hub" / repo_name
    blob = cache_repo / "blobs" / expected_hash  # HF 约定：blob 以内容哈希命名
    global_blob = pathlib.Path.home() / ".cache" / "huggingface" / "hub" / repo_name / "blobs" / expected_hash
    if not blob.exists():
        blob.parent.mkdir(parents=True, exist_ok=True)
        if global_blob.is_file() and sha256_file(global_blob) == expected_hash:
            shutil.copyfile(global_blob, blob)  # 复用本机已验证缓存
        else:
            _download(_weight_download_url(record), blob)  # 否则下载
    actual_hash = sha256_file(blob)
    if actual_hash != expected_hash:
        raise ReproducibilityError(f"Weight SHA-256 mismatch for {record['model']}")
    # snapshots 视图：让 timm 通过 <snapshot>/<filename> 找到权重
    snapshot_file = cache_repo / "snapshots" / str(record["revision"]) / str(record["filename"])
    snapshot_file.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot_file.exists():
        relative_blob = os.path.relpath(blob, snapshot_file.parent)
        try:
            snapshot_file.symlink_to(relative_blob)  # 优先软链省空间
        except OSError:
            shutil.copyfile(blob, snapshot_file)  # 平台不支持软链则拷贝
    refs_main = cache_repo / "refs" / "main"
    refs_main.parent.mkdir(parents=True, exist_ok=True)
    # huggingface_hub treats the ref file as an exact commit identifier; a
    # trailing newline is interpreted as part of the revision and corrupts
    # offline cache lookup.
    # （英文原注释）HF hub 把 ref 文件当精确 commit 标识；末尾换行会被当成
    # revision 的一部分，破坏离线缓存查找。故这里用 newline="" 不追加换行。
    refs_main.write_text(str(record["revision"]), encoding="utf-8", newline="\n")
    return {
        "model": record["model"],
        "repo_id": record["repo_id"],
        "revision": record["revision"],
        "filename": record["filename"],
        "sha256": expected_hash,
        "license": record.get("license", ""),
        "cache_path": snapshot_file.relative_to(output_dir).as_posix(),
        "verified": True,
    }


def _archive_hash_from_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    """从源行里收集 archive_sha256；多行不一致则报错。

    :param rows: 源行列表。
    :return: 唯一的归档哈希（可能为空串）。
    :raises ReproducibilityError: 各行归档哈希互相矛盾时抛出。
    """
    hashes = {str(row.get("archive_sha256") or "") for row in rows}
    hashes.discard("")
    if len(hashes) > 1:
        raise ReproducibilityError("Dataset rows disagree on the source archive SHA-256")
    return next(iter(hashes), "")


def prepare_data(
    *,
    profile: str,
    output_dir: pathlib.Path,
    config_path: pathlib.Path = PAPER_CONFIG,
    dataset_root: pathlib.Path | None = None,
    archive_url: str | None = None,
    download_weights: bool | None = None,
) -> dict[str, Any]:
    """Prepare a hash-locked dataset subset and pinned timm weight cache."""
    # （英文原注释）准备"哈希锁定的数据集子集"和"钉死的 timm 权重缓存"。
    # 这是可复现性的第一道锁。产物是一个隔离目录：
    #   data/                  # 选中的图片（canonical 时）
    #   subset.csv             # 固定子集清单（含每图哈希）
    #   hf_home/               # 隔离的 HF 权重缓存
    #   prepared_manifest.json # 数据清单（含全部哈希与分布）
    #
    # profile 两种模式：
    # - paper：真 Imagenette 固定 1024 样本 + 钉死权重（canonical）；
    # - smoke：3 张合成 PPM 图，纯标准库生成，用于快速冒烟自检。
    if profile not in {"paper", "smoke"}:
        raise ReproducibilityError(f"Unsupported profile: {profile}")
    output_dir = output_dir.resolve()
    _require_empty_output(output_dir)  # 输出目录必须干净，防止混入旧数据
    config = load_paper_config(config_path)
    config_hash = sha256_file(config_path)  # 记录配置文件本身的哈希
    data_root = output_dir / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    if profile == "smoke":
        selected = _prepare_synthetic_smoke_rows(data_root)
        archive_hash = ""
    else:
        # ---- canonical（paper）分支：准备真实数据集 ----
        dataset_cfg = config["reproducibility"]["dataset"]
        expected_archive_hash = str(dataset_cfg["archive_sha256"])
        if dataset_root is None:
            # 没有现成数据集：下载钉死的 Imagenette 归档并校验哈希
            archive_url = archive_url or str(dataset_cfg["archive_url"])
            archive_path = output_dir / "downloads" / "imagenette2-160.tgz"
            _download(archive_url, archive_path)
            actual_archive_hash = sha256_file(archive_path)
            if actual_archive_hash != expected_archive_hash:
                raise ReproducibilityError(
                    f"Imagenette archive SHA-256 mismatch: expected {expected_archive_hash}, got {actual_archive_hash}"
                )
            _safe_extract_tar(archive_path, data_root)  # 安全解压
            source_root = data_root
            source_rows = scan_imagenette_rows(source_root, config)
            archive_hash = actual_archive_hash
        else:
            # 提供现成数据集根：优先读台账 CSV，否则扫描目录
            source_root = dataset_root.resolve()
            source_csv = source_root / "fixed_subset.csv"
            source_rows = (
                source_rows_from_csv(source_root, source_csv)
                if source_csv.is_file()
                else scan_imagenette_rows(source_root, config)
            )
            archive_hash = _archive_hash_from_rows(source_rows)
        expected_synsets = [str(row["synset"]) for row in dataset_cfg["classes"]]
        # 分层抽样：选固定 1024 样本子集（每个类独立种子，完全确定）
        selected = stratified_sample(
            source_rows,
            total=int(dataset_cfg["sample_count"]),
            seed=int(dataset_cfg["sampling_seed"]),
            expected_synsets=expected_synsets,
        )
        # 若源不在 prepared 目录里，把选中图片复制进来（隔离依赖）
        if source_root.resolve() != data_root.resolve():
            selected = _materialize_selected_images(selected, source_root, data_root)
        if archive_hash and archive_hash != expected_archive_hash:
            raise ReproducibilityError(
                f"Dataset ledger archive SHA-256 mismatch: expected {expected_archive_hash}, got {archive_hash}"
            )

    subset_path = output_dir / "subset.csv"
    write_subset_csv(subset_path, selected)
    selected_rows = read_csv_rows(subset_path)  # 回读，确保写出的与内存一致
    weights: list[dict[str, Any]] = []
    should_download_weights = profile == "paper" if download_weights is None else download_weights
    if should_download_weights:
        for record in config["reproducibility"]["weights"]:
            weights.append(_prepare_weight_cache(output_dir, record))

    distribution = subset_distribution(selected_rows)
    # 汇总清单：记录数据来源、哈希、分布、权重信息与声明边界
    manifest = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "created_utc": utc_now(),
        "profile": profile,
        "status": "ready" if profile == "paper" else "smoke_ready",
        "config": {"path": "experiments/config/paper_v1.json", "sha256": config_hash},
        "dataset": {
            "name": (
                config["reproducibility"]["dataset"]["name"]
                if profile == "paper"
                else "three generated synthetic smoke fixtures"
            ),
            "dataset_root": "data",
            "subset_file": "subset.csv",
            "subset_sha256": sha256_file(subset_path),
            "content_sha256": subset_content_sha256(selected_rows),  # 内容摘要（选图集合）
            "sample_count": len(selected_rows),
            "class_count": len(distribution),
            "class_distribution": distribution,
            "sampling_seed": PAPER_SAMPLING_SEED if profile == "paper" else None,
            "archive_url": (
                config["reproducibility"]["dataset"]["archive_url"] if profile == "paper" else ""
            ),
            "archive_sha256": archive_hash,
            "archive_sha256_verified": bool(
                profile == "paper"
                and archive_hash == config["reproducibility"]["dataset"]["archive_sha256"]
            ),
            "images_redistributable": profile == "smoke",
        },
        "weights": weights,
        "weights_redistributable": False,  # 权重不随仓库分发（许可证原因）
        "claim_boundary": CLAIM_BOUNDARY,
    }
    write_json(output_dir / "prepared_manifest.json", manifest)
    return manifest


def validate_prepared_data(
    prepared_dir: pathlib.Path,
    *,
    expected_profile: str = "paper",
    config_path: pathlib.Path = PAPER_CONFIG,
) -> dict[str, Any]:
    """Validate prepared inputs and return resolved, non-serialized runtime paths."""
    # （英文原注释）校验 prepared 输入，并返回解析好的"非序列化"运行时路径。
    # 通俗说：run 之前先全量复核一次 prepared 目录——manifest 存在且 schema
    # 匹配、profile 匹配、配置文件哈希一致、子集 CSV 哈希一致、逐图哈希一致、
    # 类分布一致、canonical 时 10 类 1024 样本协议 + 全部钉死权重在位且哈希正确。
    # 全部通过后，返回带私有键（_prepared_dir 等）的运行时会话字典。
    prepared_dir = prepared_dir.resolve()
    manifest_path = prepared_dir / "prepared_manifest.json"
    if not manifest_path.is_file():
        raise ReproducibilityError(f"Prepared-data manifest is missing: {manifest_path}")
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != PREPARED_SCHEMA_VERSION:
        raise ReproducibilityError("Prepared-data manifest has an unsupported schema")
    if manifest.get("profile") != expected_profile:
        raise ReproducibilityError(
            f"Prepared-data profile is {manifest.get('profile')!r}, expected {expected_profile!r}"
        )
    config = load_paper_config(config_path)
    if manifest.get("config", {}).get("sha256") != sha256_file(config_path):
        raise ReproducibilityError("Prepared data was generated from a different paper_v1 config")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise ReproducibilityError("Prepared-data manifest is missing dataset metadata")
    dataset_root = _safe_join(prepared_dir, str(dataset.get("dataset_root", "")))
    subset_path = _safe_join(prepared_dir, str(dataset.get("subset_file", "")))
    if not subset_path.is_file() or sha256_file(subset_path) != dataset.get("subset_sha256"):
        raise ReproducibilityError("Prepared subset CSV is missing or has a SHA-256 mismatch")
    rows = read_csv_rows(subset_path)
    for row in rows:
        _canonical_source_row(row, dataset_root)  # 逐行复核：路径/文件/哈希/label
    distribution = subset_distribution(rows)
    if len(rows) != int(dataset.get("sample_count", -1)):
        raise ReproducibilityError("Prepared subset sample count does not match its manifest")
    if distribution != {str(k): int(v) for k, v in dataset.get("class_distribution", {}).items()}:
        raise ReproducibilityError("Prepared subset class distribution does not match its manifest")
    if subset_content_sha256(rows) != dataset.get("content_sha256"):
        raise ReproducibilityError("Prepared subset content digest does not match its manifest")

    if expected_profile == "paper":
        # canonical 额外强校验：10 类 / 1024 样本 / 固定种子 / 归档哈希 / 全部权重
        dataset_cfg = config["reproducibility"]["dataset"]
        synsets = sorted(str(row["synset"]) for row in dataset_cfg["classes"])
        expected_distribution = {
            synset: 102 + (1 if index < 4 else 0) for index, synset in enumerate(synsets)
        }
        if len(rows) != PAPER_SAMPLE_COUNT or distribution != expected_distribution:
            raise ReproducibilityError("Prepared paper subset is not the canonical 10-class/1024 protocol")
        if dataset.get("sampling_seed") != PAPER_SAMPLING_SEED:
            raise ReproducibilityError("Prepared paper subset uses the wrong sampling seed")
        if dataset.get("archive_sha256") != dataset_cfg["archive_sha256"]:
            raise ReproducibilityError("Prepared paper subset does not record the pinned archive hash")
        expected_weights = {record["model"]: record for record in config["reproducibility"]["weights"]}
        actual_weights = {record.get("model"): record for record in manifest.get("weights", [])}
        if set(actual_weights) != set(expected_weights):
            raise ReproducibilityError("Prepared paper inputs do not contain every pinned model weight")
        for model, expected in expected_weights.items():
            actual = actual_weights[model]
            if any(actual.get(key) != expected.get(key) for key in ["repo_id", "revision", "filename", "sha256"]):
                raise ReproducibilityError(f"Pinned weight metadata mismatch for {model}")
            weight_path = _safe_join(prepared_dir, str(actual.get("cache_path", "")))
            if not weight_path.is_file() or sha256_file(weight_path) != expected["sha256"]:
                raise ReproducibilityError(f"Pinned weight file is missing or invalid for {model}")

    # 附上解析好的运行时路径（下划线前缀表示"非序列化"内部字段）
    runtime = dict(manifest)
    runtime["_prepared_dir"] = prepared_dir
    runtime["_dataset_root"] = dataset_root
    runtime["_subset_path"] = subset_path
    return runtime


def _mps_available(torch_module: Any) -> tuple[bool, str]:
    """探测苹果 MPS 是否可用（编译了 + 运行时可用），返回 (可用, 原因)。"""
    try:
        built = bool(torch_module.backends.mps.is_built())
        available = bool(torch_module.backends.mps.is_available())
    except (AttributeError, RuntimeError) as exc:
        return False, f"MPS probe failed: {exc}"
    if not built:
        return False, "torch was not built with MPS support"
    if not available:
        return False, "torch MPS is built but unavailable at runtime"
    return True, "MPS is built and available"


def _cuda_available(torch_module: Any) -> tuple[bool, str]:
    """探测 CUDA 是否可用，返回 (可用, 原因)。"""
    try:
        available = bool(torch_module.cuda.is_available())
    except (AttributeError, RuntimeError) as exc:
        return False, f"CUDA probe failed: {exc}"
    return (True, "CUDA is available") if available else (False, "CUDA is unavailable")


def resolve_device(
    requested: str,
    *,
    profile: str,
    torch_module: Any | None = None,
) -> DeviceSelection:
    """Resolve a backend without ever converting an explicit accelerator request to CPU."""
    # （英文原注释）解析计算后端，但绝不把"显式请求的加速器"悄悄退回 CPU。
    # 这是可复现性的关键约束：如果用户显式请求 mps/cuda 而环境不可用，直接
    # 报错（不静默降级），否则两次 run 可能跑在不同硬件上，结果没可比性。
    # auto 模式下则：MPS 可用优先 MPS，其次 CUDA；两者都没有就报错，
    # 并要求用户显式 --device cpu 跑"仅对比"模式。
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ReproducibilityError(f"Unsupported device: {requested}")
    if profile not in {"paper", "smoke"}:
        raise ReproducibilityError(f"Unsupported profile: {profile}")
    if profile == "smoke":
        # 冒烟模式刻意只允许 CPU（保证任何机器都能快速自检）
        if requested not in {"auto", "cpu"}:
            raise ReproducibilityError("The smoke profile is intentionally CPU-only")
        return DeviceSelection(
            requested=requested,
            selected="cpu",
            reason="smoke profile requires explicit CPU execution",
            fallback_used=False,
            canonical_backend=False,
        )
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise ReproducibilityError("Paper runs require the torch ML dependency") from exc
    mps_ok, mps_reason = _mps_available(torch_module)
    cuda_ok, cuda_reason = _cuda_available(torch_module)
    if requested == "mps":
        if not mps_ok:
            raise ReproducibilityError(f"Requested MPS but it is unavailable: {mps_reason}")
        return DeviceSelection("mps", "mps", "MPS explicitly requested and verified", False, True)
    if requested == "cuda":
        if not cuda_ok:
            raise ReproducibilityError(f"Requested CUDA but it is unavailable: {cuda_reason}")
        return DeviceSelection("cuda", "cuda", "CUDA explicitly requested and verified", False, False)
    if requested == "cpu":
        return DeviceSelection(
            "cpu",
            "cpu",
            "CPU explicitly requested; paper output is comparison-only",
            False,
            False,
        )
    # auto 分支
    if mps_ok:
        return DeviceSelection("auto", "mps", "auto selected verified MPS", False, True)
    if cuda_ok:
        return DeviceSelection(
            "auto",
            "cuda",
            f"auto selected CUDA because MPS was unavailable ({mps_reason})",
            False,
            False,
        )
    raise ReproducibilityError(
        "Paper auto-device found no accelerator and will not silently fall back to CPU; "
        f"MPS: {mps_reason}; CUDA: {cuda_reason}. Use --device cpu explicitly for a comparison-only run."
    )


def _git_commit() -> str:
    """读取仓库当前 HEAD 的完整 commit（40 位 hex）；失败则 "unavailable"。

    Git commit 是"代码锁"的一部分：读者可以用它精确还原生成结果的源码版本。
    """
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    commit = process.stdout.strip()
    return commit if process.returncode == 0 and re.fullmatch(r"[0-9a-fA-F]{40}", commit) else "unavailable"


def _package_versions(names: Iterable[str]) -> dict[str, str]:
    """查询一批包的已安装版本（环境锁的一部分）。

    :param names: 包名列表。
    :return: {包名: 版本号}；未安装的记 "not-installed"。
    """
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _output_records(run_dir: pathlib.Path) -> list[dict[str, Any]]:
    """扫描 run 目录里所有"产物文件"，返回 path/sha256/bytes 记录。

    排除清单文件本身（repro_manifest / verification_report /
    reference_comparison_report）与 logs/ 目录——它们要么是元信息，
    要么因含时间戳而不可复现。

    :param run_dir: run 根目录。
    :return: 排序后的产物记录列表。
    """
    excluded_names = {
        "repro_manifest.json",
        "verification_report.json",
        "reference_comparison_report.json",
    }
    records: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in excluded_names:
            continue
        relative = path.relative_to(run_dir).as_posix()
        if relative.startswith("logs/"):
            continue
        records.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return records


def canonical_output_record(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the stable digest used to compare scientific run outputs.

    Only canonical CSV and NPZ outputs participate. Per-sample floating-point
    confidence details remain integrity-checked, but are intentionally excluded
    from the cross-run digest because sub-decision MPS values may vary slightly.
    """
    # （英文原注释）构建用于跨 run 比对的稳定摘要。
    # 只有"canonical"的 CSV/NPZ 参与比对；每样本的浮点置信度细节仍做完整性
    # 校验（在 outputs[] 里全量哈希），但故意排除在跨机比对摘要之外——
    # 因为 MPS 上低于"判决粒度"的浮点尾数可能轻微不同，纳入会影响可比性。
    noncanonical_paths = {"raw/nonideality_prediction_topk_margin.csv"}
    excluded = [
        {
            **dict(record),
            "reason": (
                "Non-paper-facing per-sample confidence/margin float detail; MPS perturbed-forward values "
                "can vary below the categorical decision granularity. Integrity remains covered by outputs[]."
            ),
        }
        for record in records
        if str(record.get("path")) in noncanonical_paths
    ]
    scientific = [
        dict(record)
        for record in records
        if pathlib.PurePosixPath(str(record["path"])).suffix.lower() in {".csv", ".npz"}  # 只收 CSV/NPZ
        and str(record["path"]) not in noncanonical_paths
    ]
    payload = "".join(
        f"{record['path']}\0{record['sha256']}\0{record['bytes']}\n" for record in scientific
    ).encode("utf-8")
    return {
        "algorithm": (
            "sha256(path\\0sha256\\0bytes\\n for sorted canonical CSV/NPZ outputs; explicitly excludes "
            "non-paper-facing MPS confidence/margin float detail while full output integrity remains hashed)"
        ),
        "sha256": sha256_bytes(payload),
        "files": scientific,
        "excluded_diagnostics": excluded,
    }


def _canonical_output_record(
    _run_dir: pathlib.Path, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Backward-compatible wrapper for the original private helper."""
    # （英文原注释）旧私有辅助函数的向后兼容包装。
    # _run_dir 参数已不再使用（原实现用它定位目录），保留签名以兼容调用方。
    return canonical_output_record(records)


def _script_command(
    script_name: str,
    arguments: Sequence[str],
    *,
    use_caffeinate: bool,
) -> tuple[list[str], list[str]]:
    """构造"实际执行命令"与"记录进 manifest 的命令"两份列表。

    两者区别：实际命令用 sys.executable 的绝对路径（本机可执行），
    而记录命令只用解释器文件名 + 相对脚本路径（可移植、无本机路径）。
    caffeinate（macOS 防睡眠）同样只加在需要时。

    :param script_name: 脚本文件名（experiments/scripts/ 下）。
    :param arguments: 传给脚本的参数列表。
    :param use_caffeinate: 是否包 caffeinate 前缀。
    :return: (实际命令, 记录命令)。
    """
    executable = pathlib.Path(sys.executable)
    actual = [str(executable), str(SCRIPTS_DIR / script_name), *arguments]
    recorded = [executable.name, f"experiments/scripts/{script_name}", *arguments]
    if use_caffeinate:
        actual = ["caffeinate", "-dimsu", *actual]
        recorded = ["caffeinate", "-dimsu", *recorded]
    return actual, recorded


def _run_experiment_script(
    *,
    script_name: str,
    arguments: Sequence[str],
    run_dir: pathlib.Path,
    environment: Mapping[str, str],
    use_caffeinate: bool,
    caffeinate_recorded: bool | None = None,
    caffeinate_mode: str = "none",
) -> dict[str, Any]:
    """以子进程方式运行一个实验脚本，记录 stdout/stderr 日志与命令元信息。

    成功时返回一条"命令记录"（写进 manifest）；脚本非零退出时抛出
    PipelineExecutionError，run_pipeline 捕获后把 run 标记为 failed。

    :param script_name: 脚本文件名。
    :param arguments: 参数列表。
    :param run_dir: run 目录（日志写到这里）。
    :param environment: 子进程环境变量。
    :param use_caffeinate: 是否实际加 caffeinate。
    :param caffeinate_recorded: 记录里是否标记用了 caffeinate（None 时取 use_caffeinate）。
    :param caffeinate_mode: caffeinate 模式标签（outer-wrapper/per-command/none）。
    :return: 命令记录字典。
    :raises PipelineExecutionError: 脚本非零退出时抛出。
    """
    actual, recorded = _script_command(script_name, arguments, use_caffeinate=use_caffeinate)
    process = subprocess.run(
        actual,
        cwd=REPO_ROOT,
        env=dict(environment),
        text=True,
        capture_output=True,
        check=False,
    )
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stem = pathlib.Path(script_name).stem
    stdout_log = logs_dir / f"{stem}.stdout.log"
    stderr_log = logs_dir / f"{stem}.stderr.log"
    stdout_log.write_text(process.stdout, encoding="utf-8", newline="\n")
    stderr_log.write_text(process.stderr, encoding="utf-8", newline="\n")
    record = {
        "script": f"experiments/scripts/{script_name}",
        "command": recorded,
        "returncode": process.returncode,
        "stdout_log": stdout_log.relative_to(run_dir).as_posix(),
        "stderr_log": stderr_log.relative_to(run_dir).as_posix(),
        "caffeinate_used": use_caffeinate if caffeinate_recorded is None else caffeinate_recorded,
        "caffeinate_mode": caffeinate_mode,
    }
    if process.returncode != 0:
        tail = process.stderr.strip().splitlines()[-1] if process.stderr.strip() else "no stderr"
        raise PipelineExecutionError(f"{script_name} failed with exit code {process.returncode}: {tail}")
    return record


def _paper_promotion_checks(manifest: Mapping[str, Any], selected_device: str) -> dict[str, bool]:
    """对非理想性 manifest 做"升格入论文"的资格检查，返回逐项布尔。

    升格（promotion）不是随便能做的：必须同时满足样本数 1024、五个
    重复种子、四个必测非理想性效果、reasonable-max 映射边界、预训练
    权重、设备记录一致，缺一不可。

    :param manifest: 非理想性实验的 manifest。
    :param selected_device: 实际选中的设备名。
    :return: {检查名: 是否通过} 字典。
    """
    dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
    return {
        "sample_count_1024": dataset.get("sample_count") == PAPER_SAMPLE_COUNT,
        "label_count_1024": dataset.get("label_count") == PAPER_SAMPLE_COUNT,
        "five_repeat_seeds": manifest.get("repeat_seeds") == PAPER_REPEAT_SEEDS,
        "four_required_effects": set(manifest.get("selected_effects", []))
        == {
            "gaussian_pd_tia_noise",   # PD/TIA 高斯噪声
            "wdm_adjacent_crosstalk",  # WDM 相邻波长串扰
            "mrr_variation",           # MRR 制造偏差
            "thermal_drift",           # 热漂移
        },
        "reasonable_max_boundary": manifest.get("injection_boundary") == "reasonable-max",
        "pretrained_weights": manifest.get("pretrained") is True,
        "selected_device_recorded": dataset.get("device") == selected_device,
    }


def _promote_nonideality_manifest(path: pathlib.Path, selected_device: str) -> dict[str, Any]:
    """把非理想性实验 manifest 从"运行结果"升格为"论文可用（有局限）"。

    升格步骤：跑完资格检查，全通过才把 status 改成
    paper_eligible_with_limitations、evidence_tier 改成
    local-modelled diagnostic，并写入声明边界与检查明细；不通过则抛错。

    :param path: 非理想性 manifest 文件路径。
    :param selected_device: 实际选中的设备名。
    :return: 修改后的 manifest 字典。
    :raises ReproducibilityError: 检查未全部通过时抛出。
    """
    manifest = load_json(path)
    checks = _paper_promotion_checks(manifest, selected_device)
    if not all(checks.values()):
        failed = ", ".join(key for key, passed in checks.items() if not passed)
        raise ReproducibilityError(f"Paper non-ideality promotion checks failed: {failed}")
    manifest["runner_status_before_promotion"] = manifest.get("status")  # 备份原状态
    manifest["status"] = PAPER_ARTIFACT_STATUS
    manifest["evidence_tier"] = PAPER_EVIDENCE_TIER
    manifest["claim_eligible_for_local_modelled_diagnostic"] = True
    manifest["claim_eligible_for_silicon_or_edge_claims"] = False  # 明确：不得用来支撑硅片/端侧声明
    manifest["paper_promotion_checks"] = checks
    manifest["claim_boundary"] = CLAIM_BOUNDARY
    manifest["promotion_note"] = (
        "Eligible with limitations for the fixed 1024-sample, five-seed local/modelled diagnostic only. "
        "The reasonable-max boundary remains a mapping scenario, not measured HPAT execution, silicon "
        "validation, calibrated energy, or edge/mobile deployment evidence."
    )
    write_json(path, manifest)  # 就地写回
    return manifest


def _base_repro_manifest(
    *,
    profile: str,
    run_dir: pathlib.Path,
    config_path: pathlib.Path,
    device: DeviceSelection,
    commands: Sequence[Mapping[str, Any]],
    status: str,
    error: str = "",
    prepared: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """组装"可复现 manifest"：记录数据、环境、命令、输出与证据声明。

    这是 run 的"出生证明"，包含：
    - 数据锁：数据集/子集/权重哈希；
    - 环境锁：Git commit、Python 版本、包版本、OS、随机种子、设备；
    - 命令链：逐条子进程命令及退出码、日志路径、caffeinate 情况；
    - 产物：outputs（全量哈希）与 canonical_outputs（跨机比对摘要）；
    - 证据声明：evidence.tier / claim_boundary / 明确 blocked 的外部通道。

    :param profile: 运行模式（paper/smoke）。
    :param run_dir: run 目录。
    :param config_path: 配置文件路径。
    :param device: 设备选择结果。
    :param commands: 已执行命令记录列表。
    :param status: 整体状态（completed/failed）。
    :param error: 错误信息（失败时）。
    :param prepared: prepared 数据字典（canonical 时提供）。
    :return: manifest 字典。
    """
    config = load_paper_config(config_path)
    repro = config["reproducibility"]
    # artifact 状态判定：paper + canonical 后端 + 完成 → 论文可用（有局限）；
    # paper + 完成但非 canonical 后端 → 仅对比；smoke → smoke
    artifact_status = (
        PAPER_ARTIFACT_STATUS
        if profile == "paper" and device.canonical_backend and status == "completed"
        else ("comparison_only" if profile == "paper" and status == "completed" else "smoke")
    )
    data_record: dict[str, Any]
    weight_records: list[dict[str, Any]]
    if prepared is not None:
        # canonical：从 prepared 清单摘取数据/权重信息
        dataset = prepared["dataset"]
        data_record = {
            "name": dataset["name"],
            "subset_file": "inputs/subset.csv",
            "subset_sha256": dataset["subset_sha256"],
            "content_sha256": dataset["content_sha256"],
            "archive_url": dataset.get("archive_url", ""),
            "archive_sha256": dataset.get("archive_sha256", ""),
            "sample_count": dataset["sample_count"],
            "class_count": dataset["class_count"],
            "class_distribution": dataset["class_distribution"],
            "sampling_seed": dataset.get("sampling_seed"),
            "data_and_weights_not_redistributed": True,
        }
        weight_records = [
            {
                key: record.get(key)
                for key in ["model", "repo_id", "revision", "filename", "sha256", "license", "verified"]
            }
            for record in prepared.get("weights", [])
        ]
    else:
        # smoke：用仓库自带的 fixture（固定算子活动 CSV）
        fixture = TEST_FIXTURES_DIR / "trace_operator_activity.csv"
        data_record = {
            "name": "repository smoke fixtures",
            "sample_count": 0,
            "fixture": "experiments/tests/fixtures/trace_operator_activity.csv",
            "fixture_sha256": sha256_file(fixture),
        }
        weight_records = []

    output_records = _output_records(run_dir)
    return {
        "schema_version": REPRO_MANIFEST_SCHEMA_VERSION,
        "created_utc": utc_now(),
        "profile": profile,
        "run_status": status,
        "error": error,
        "artifact_status": artifact_status,
        "git": {"commit": _git_commit()},  # 代码锁：commit
        "config": {
            "path": "config_snapshot.json",
            "sha256": sha256_file(run_dir / "config_snapshot.json"),
            "source_path": "experiments/config/paper_v1.json",
            "source_sha256": sha256_file(config_path),
            "schema_version": config["schema_version"],
        },
        "data": data_record,
        "weights": weight_records,
        "backend": device.as_dict(),
        "runtime": {  # 环境锁：解释器/OS/机器/包版本
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "packages": _package_versions(["torch", "timm", "numpy", "Pillow"]),
        },
        "randomness": {  # 随机种子锁
            "sampling_seed": PAPER_SAMPLING_SEED if profile == "paper" else None,
            "experiment_seed": int(config["seed"]),
            "repeat_seeds": list(config["repeat_seeds"]) if profile == "paper" else [int(config["seed"])],
        },
        "numeric": {  # 数值/同步设置
            "precision": config["precision"],
            "dtype": repro["dtype"],
            "warmup": repro["warmup"],
            "synchronization": repro["synchronization"].get(device.selected, "not required"),
            "broad_cpu_fallback_enabled": False,
        },
        "power_lease": {  # caffeinate 防睡眠记录
            "caffeinate_used": any(bool(command.get("caffeinate_used")) for command in commands),
            "modes": sorted(
                {
                    str(command.get("caffeinate_mode", "none"))
                    for command in commands
                    if command.get("caffeinate_used")
                }
            ),
        },
        "mapping": repro["mapping"],
        "effects": list(repro["nonideality_run"]["effects"]) if profile == "paper" else [],
        "commands": list(commands),
        "outputs": output_records,
        "canonical_outputs": _canonical_output_record(run_dir, output_records),
        "evidence": {  # 证据声明
            "tier": PAPER_EVIDENCE_TIER if profile == "paper" else "smoke",
            "claim_boundary": CLAIM_BOUNDARY,
            "local_apple_silicon_only": profile == "paper" and device.selected == "mps",
            "blocked_external_lanes": [  # 明确声明哪些证据通道被阻塞
                {
                    "lane": "P0-E1",
                    "status": "blocked",
                    "reason": "No fabricated-silicon validation or calibrated silicon activity/energy package.",
                },
                {
                    "lane": "P0-E2",
                    "status": "blocked",
                    "reason": "No comparable author-measured HPAT edge/mobile deployment evidence.",
                },
            ],
        },
    }


def run_pipeline(
    *,
    profile: str,
    device_request: str,
    output_dir: pathlib.Path,
    config_path: pathlib.Path = PAPER_CONFIG,
    prepared_dir: pathlib.Path | None = None,
    use_caffeinate: bool | None = None,
) -> dict[str, Any]:
    """Run the smoke or canonical paper chain through existing experiment scripts."""
    # （英文原注释）通过既有的实验脚本运行 smoke 或 canonical 论文链条。
    # 这是 run 命令的总指挥，编排若干个子进程脚本：
    #   smoke： 3 个脚本（映射闭包 / E-local 能耗 / 非理想性精度，全部 CPU）；
    #   paper： 5 个脚本（算子轨迹 / 映射闭包 / E-local 能耗 / 非理想性精度
    #           + 配对转换汇总 / 证据强度汇总）。
    # 同时完成环境锁：Python 版本校验、PYTHONHASHSEED、PYTHONDONTWRITEBYTECODE、
    # PYTHONPATH、HF 离线权重缓存、caffeinate 防睡眠。任何一步失败都会把
    # run_status 置为 failed 并抛出 PipelineExecutionError。
    if profile not in {"paper", "smoke"}:
        raise ReproducibilityError(f"Unsupported profile: {profile}")
    output_dir = output_dir.resolve()
    _require_empty_output(output_dir)
    config = load_paper_config(config_path)
    # 环境锁 1：canonical 运行必须使用配置锁定的 Python 版本
    if profile == "paper" and platform.python_version() != config["reproducibility"]["python"]:
        raise ReproducibilityError(
            "Canonical runs require Python "
            f"{config['reproducibility']['python']}; current runtime is {platform.python_version()}"
        )
    shutil.copyfile(config_path, output_dir / "config_snapshot.json")  # 快照配置（自包含）
    (output_dir / "inputs").mkdir(parents=True, exist_ok=True)

    prepared: dict[str, Any] | None = None
    if profile == "paper":
        # 环境锁 2：canonical 必须使用经验证的 prepared 数据（哈希全对才行）
        prepared = validate_prepared_data(
            (prepared_dir or DEFAULT_PREPARED_DIR), expected_profile="paper", config_path=config_path
        )
        shutil.copyfile(prepared["_subset_path"], output_dir / "inputs" / "subset.csv")
    device = resolve_device(device_request, profile=profile)  # 设备锁

    # caffeinate（macOS 防睡眠）三层策略：外层环境变量 / 每条命令加 / 不加
    outer_caffeinate = os.environ.get("HPAT_CAFFEINATE_USED", "").strip().lower() in {"1", "true", "yes"}
    if use_caffeinate is None:
        use_caffeinate = (
            profile == "paper"
            and sys.platform == "darwin"
            and shutil.which("caffeinate") is not None
            and not outer_caffeinate
        )
    if use_caffeinate and shutil.which("caffeinate") is None:
        raise ReproducibilityError("caffeinate was requested but is unavailable")
    caffeinate_protected = outer_caffeinate or bool(use_caffeinate)
    caffeinate_mode = "outer-wrapper" if outer_caffeinate else ("per-command" if use_caffeinate else "none")

    # 子进程环境：锁随机种子、关字节码缓存、注入库路径、强制离线 HF 权重
    environment = dict(os.environ)
    environment["HPAT_WRITE_PROJECT_TABLES"] = "0"  # 禁止脚本直接写仓库 tables/
    environment["PYTHONHASHSEED"] = str(config["seed"])  # 字典顺序等也锁死
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(EXPERIMENTS_DIR / "src"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)  # 不允许 MPS 隐式回退 CPU
    if caffeinate_protected:
        environment["HPAT_CAFFEINATE_USED"] = "1"
    else:
        environment.pop("HPAT_CAFFEINATE_USED", None)
    if prepared is not None:
        # 让 timm 走隔离的权重缓存，且强制离线（不许联网换权重）
        environment["HF_HOME"] = str(prepared["_prepared_dir"] / "hf_home")
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"

    # 命令路径统一折成相对仓库根的"可移植"形式（记录用）
    run_arg = _command_path(output_dir)
    config_arg = _command_path(output_dir / "config_snapshot.json")
    commands: list[dict[str, Any]] = []  # 累计命令记录，写进 manifest
    caught_error = ""
    try:
        if profile == "smoke":
            # ---- smoke 分支：3 个脚本，全部 CPU + 仓库自带 fixture ----
            fixture_activity = _command_path(TEST_FIXTURES_DIR / "trace_operator_activity.csv")
            fixture_costs = _command_path(TEST_FIXTURES_DIR / "unit_costs_complete.json")
            commands.append(
                _run_experiment_script(
                    script_name="run_operator_mapping_closure.py",
                    arguments=["--output-dir", run_arg, "--config", config_arg, "--operator-activity-csv", fixture_activity],
                    run_dir=output_dir,
                    environment=environment,
                    use_caffeinate=False,
                    caffeinate_recorded=outer_caffeinate,
                    caffeinate_mode=caffeinate_mode,
                )
            )
            commands.append(
                _run_experiment_script(
                    script_name="run_e_local_trace_driven_activity.py",
                    arguments=[
                        "--output-dir",
                        run_arg,
                        "--config",
                        config_arg,
                        "--operator-activity-csv",
                        fixture_activity,
                        "--unit-costs",
                        fixture_costs,
                        "--mapping-scenario",
                        "reasonable_max",
                    ],
                    run_dir=output_dir,
                    environment=environment,
                    use_caffeinate=False,
                    caffeinate_recorded=outer_caffeinate,
                    caffeinate_mode=caffeinate_mode,
                )
            )
            commands.append(
                _run_experiment_script(
                    script_name="run_nonideality_accuracy_sweep.py",
                    arguments=["--output-dir", run_arg, "--config", config_arg, "--device", "cpu"],
                    run_dir=output_dir,
                    environment=environment,
                    use_caffeinate=False,
                    caffeinate_recorded=outer_caffeinate,
                    caffeinate_mode=caffeinate_mode,
                )
            )
        else:
            # ---- paper（canonical）分支：5 个脚本，真实数据 + 选定设备 ----
            assert prepared is not None
            activity_csv = output_dir / "tables" / "mobilevit_operator_activity.csv"
            activity_arg = _command_path(activity_csv)
            # 1) 抓 MobileViT 算子活动轨迹（真实加载 timm 模型跑一次前向）
            commands.append(
                _run_experiment_script(
                    script_name="run_mobilevit_activity_trace.py",
                    arguments=[
                        "--output-dir",
                        run_arg,
                        "--config",
                        config_arg,
                        "--device",
                        device.selected,
                        "--precision",
                        str(config["precision"]),
                        "--project-write-policy",
                        "never",
                    ],
                    run_dir=output_dir,
                    environment=environment,
                    use_caffeinate=bool(use_caffeinate),
                    caffeinate_recorded=caffeinate_protected,
                    caffeinate_mode=caffeinate_mode,
                )
            )
            # 检查轨迹确实在选定后端完成（拒绝代理/回退输出）
            activity_manifest = load_json(output_dir / "mobilevit_activity_manifest.json")
            trace_meta = activity_manifest.get("trace_meta", {})
            if activity_manifest.get("status") != "ok" or trace_meta.get("backend") != device.selected:
                raise ReproducibilityError(
                    "Operator trace did not complete on the selected backend; proxy/fallback output is not accepted"
                )
            # 2) 算子映射闭包（哪些层能上光子端）
            commands.append(
                _run_experiment_script(
                    script_name="run_operator_mapping_closure.py",
                    arguments=["--output-dir", run_arg, "--config", config_arg, "--operator-activity-csv", activity_arg],
                    run_dir=output_dir,
                    environment=environment,
                    use_caffeinate=False,
                    caffeinate_recorded=outer_caffeinate,
                    caffeinate_mode=caffeinate_mode,
                )
            )
            # 3) 轨迹驱动的 E-local 能耗建模
            commands.append(
                _run_experiment_script(
                    script_name="run_e_local_trace_driven_activity.py",
                    arguments=[
                        "--output-dir",
                        run_arg,
                        "--config",
                        config_arg,
                        "--operator-activity-csv",
                        activity_arg,
                        "--unit-costs",
                        _command_path(EXPERIMENTS_DIR / "config" / "energy_unit_costs.yaml"),
                        "--mapping-scenario",
                        str(config["reproducibility"]["mapping"]["scenario"]),
                    ],
                    run_dir=output_dir,
                    environment=environment,
                    use_caffeinate=False,
                    caffeinate_recorded=outer_caffeinate,
                    caffeinate_mode=caffeinate_mode,
                )
            )
            nonideality = config["reproducibility"]["nonideality_run"]
            mapping = config["reproducibility"]["mapping"]
            # 4) 非理想性精度扫描（固定子集 + 预训练权重 + 误差注入）
            commands.append(
                _run_experiment_script(
                    script_name="run_nonideality_accuracy_sweep.py",
                    arguments=[
                        "--output-dir",
                        run_arg,
                        "--config",
                        config_arg,
                        "--dataset-root",
                        _command_path(prepared["_dataset_root"]),
                        "--subset-file",
                        _command_path(prepared["_subset_path"]),
                        "--seed",
                        str(config["seed"]),
                        "--device",
                        device.selected,
                        "--pretrained",
                        "--model-variant",
                        str(nonideality["model_variant"]),
                        "--operator-activity-csv",
                        activity_arg,
                        "--injection-boundary",
                        str(mapping["injection_boundary"]),
                        "--min-boundary-coverage",
                        str(mapping["min_boundary_coverage"]),
                        "--batch-size",
                        str(nonideality["batch_size"]),
                        "--save-logits",
                        str(nonideality["save_logits"]),
                        "--save-prediction-detail",
                        str(nonideality["save_prediction_detail"]),
                        "--repeat-seeds",
                        ",".join(str(seed) for seed in config["repeat_seeds"]),
                        "--effects",
                        ",".join(str(effect) for effect in nonideality["effects"]),
                    ],
                    run_dir=output_dir,
                    environment=environment,
                    use_caffeinate=bool(use_caffeinate),
                    caffeinate_recorded=caffeinate_protected,
                    caffeinate_mode=caffeinate_mode,
                )
            )
            # 升格非理想性 manifest（论文可用有局限），不过关这里直接抛错
            _promote_nonideality_manifest(
                output_dir / "nonideality_accuracy_sweep_manifest.json", device.selected
            )
            # 4b) 配对预测转换汇总（干净 vs 扰动预测的变化统计）
            commands.append(
                _run_experiment_script(
                    script_name="summarize_paired_prediction_transitions.py",
                    arguments=[
                        "--input",
                        _command_path(output_dir / "raw" / "nonideality_prediction_topk_margin.csv"),
                        "--output",
                        _command_path(output_dir / "tables" / "nonideality_paired_transition_summary.csv"),
                    ],
                    run_dir=output_dir,
                    environment=environment,
                    use_caffeinate=False,
                    caffeinate_recorded=outer_caffeinate,
                    caffeinate_mode=caffeinate_mode,
                )
            )
            # 5) 证据强度汇总（Amdahl 上界 / 能耗不确定性等）
            commands.append(
                _run_experiment_script(
                    script_name="run_evidence_strength_summary.py",
                    arguments=[
                        "--output-dir",
                        run_arg,
                        "--config",
                        config_arg,
                        "--energy-csv",
                        _command_path(
                            output_dir
                            / "tables"
                            / "hpat_energy_by_component_reasonable_max_mapping.csv"
                        ),
                    ],
                    run_dir=output_dir,
                    environment=environment,
                    use_caffeinate=False,
                    caffeinate_recorded=outer_caffeinate,
                    caffeinate_mode=caffeinate_mode,
                )
            )
    except (ReproducibilityError, OSError) as exc:
        caught_error = str(exc)  # 记录首个失败原因，run 整体标记 failed

    status = "failed" if caught_error else "completed"
    manifest = _base_repro_manifest(
        profile=profile,
        run_dir=output_dir,
        config_path=config_path,
        device=device,
        commands=commands,
        status=status,
        error=caught_error,
        prepared=prepared,
    )
    write_json(output_dir / "repro_manifest.json", manifest)  # 写"出生证明"
    if caught_error:
        raise PipelineExecutionError(caught_error)
    return manifest


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    """构造一行"检查记录"（检查名 / 是否通过 / 详细说明）。"""
    return {"name": name, "passed": bool(passed), "detail": detail}


# 匿名扫描用的模式：本机绝对路径（macOS /home /Windows）与禁止引用的树
_ABSOLUTE_PATH_PATTERNS = [
    re.compile(b"/" + b"Users" + rb"/[A-Za-z0-9._-]+/"),   # macOS: /Users/<name>/
    re.compile(b"/" + b"home" + rb"/[A-Za-z0-9._-]+/"),    # Linux: /home/<name>/
    re.compile(rb"[A-Za-z]:\\" + b"Users" + rb"\\[A-Za-z0-9._-]+\\"),  # Windows: C:\Users\<name>\
]
_FORBIDDEN_RELEASE_TOKENS = [
    b".research-resource.json",
    b"review_packets/",
    b"experiments/results/",
]


def _anonymous_scan(run_dir: pathlib.Path) -> list[str]:
    """扫描 run 产物，找出可能泄露本机身份/内部树结构的内容。

    检查两类内容：
    - 绝对本机账号路径（/Users/xxx、/home/xxx、C:\\Users\\xxx）；
    - 禁止引用的内部 token（research-resource、review_packets、results）。
    发布前必须清空这些，否则匿名性不达标（release gate 会拦）。

    :param run_dir: run 目录。
    :return: 发现问题列表（无则空列表）。
    """
    findings: list[str] = []
    for record in _output_records(run_dir):
        path = run_dir / record["path"]
        if path.suffix.lower() not in {".json", ".csv", ".md", ".txt", ".yaml", ".yml"}:
            continue  # 只扫文本类文件
        payload = path.read_bytes()
        for pattern in _ABSOLUTE_PATH_PATTERNS:
            if pattern.search(payload):
                findings.append(f"{record['path']}: absolute local account path")
        for token in _FORBIDDEN_RELEASE_TOKENS:
            if token in payload:
                findings.append(f"{record['path']}: forbidden release reference {token.decode('ascii')}")
    return sorted(set(findings))


def _validate_repro_manifest_shape(manifest: Mapping[str, Any]) -> list[str]:
    """校验 manifest 的"外形"：必需顶层键、schema 版本、关键子记录结构。

    :param manifest: repro_manifest 字典。
    :return: 结构错误列表（无则空）。
    """
    errors: list[str] = []
    required_top_level = {
        "schema_version",
        "profile",
        "run_status",
        "artifact_status",
        "git",
        "config",
        "data",
        "weights",
        "backend",
        "runtime",
        "randomness",
        "numeric",
        "mapping",
        "commands",
        "outputs",
        "canonical_outputs",
        "evidence",
    }
    missing = sorted(required_top_level - set(manifest))
    if missing:
        errors.append("missing manifest keys: " + ", ".join(missing))
    if manifest.get("schema_version") != REPRO_MANIFEST_SCHEMA_VERSION:
        errors.append("unsupported repro manifest schema_version")
    backend = manifest.get("backend")
    if not isinstance(backend, dict) or not {"requested", "selected", "reason", "fallback_used"}.issubset(backend):
        errors.append("backend record is incomplete")
    output_records = manifest.get("outputs")
    if not isinstance(output_records, list) or any(
        not isinstance(record, dict) or not {"path", "sha256", "bytes"}.issubset(record)
        for record in output_records
    ):
        errors.append("outputs must be a list of path/SHA-256/size records")
    return errors


def verify_run(run_dir: pathlib.Path) -> dict[str, Any]:
    """Verify hashes, isolation, schema, and canonical paper eligibility."""
    # （英文原注释）校验哈希、隔离性、schema 与 canonical 论文资格。
    # 这是 verify 命令的核心：对 run 目录做全量审计——
    # 1) manifest schema 是否合法；
    # 2) config 快照哈希是否一致；
    # 3) 每个记录产物的 size/sha256 是否与磁盘一致（多出来的"未记录"文件也算错）；
    # 4) canonical 摘要是否一致；
    # 5) 子进程是否全部成功退出；
    # 6) 匿名扫描是否通过；
    # 7) paper 模式额外检查 canonical 资格（MPS 无回退、1024 样本、10 类、
    #    5 种子、分层分布、映射边界、Python 版本锁定、manifest 已升格）。
    # 输出 valid（基础校验）与 release_ready（canonical 全通过）。
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "repro_manifest.json"
    if not manifest_path.is_file():
        raise ReproducibilityError(f"Run manifest is missing: {manifest_path}")
    manifest = load_json(manifest_path)
    checks: list[dict[str, Any]] = []
    shape_errors = _validate_repro_manifest_shape(manifest)
    checks.append(_check("manifest_schema", not shape_errors, "; ".join(shape_errors) or "schema complete"))

    config_path = run_dir / str(manifest.get("config", {}).get("path", ""))
    config_ok = config_path.is_file() and sha256_file(config_path) == manifest.get("config", {}).get("sha256")
    checks.append(_check("config_snapshot_sha256", config_ok, "config snapshot hash matches" if config_ok else "config snapshot missing or changed"))

    # 逐产物核对：存在性 / 字节数 / 哈希；并发现"未记录"的文件
    output_errors: list[str] = []
    recorded_paths: set[str] = set()
    for record in manifest.get("outputs", []):
        if not isinstance(record, dict):
            continue
        relative = str(record.get("path", ""))
        try:
            path = _safe_join(run_dir, relative)
        except ReproducibilityError as exc:
            output_errors.append(str(exc))
            continue
        recorded_paths.add(relative)
        if not path.is_file():
            output_errors.append(f"missing output: {relative}")
            continue
        if path.stat().st_size != record.get("bytes"):
            output_errors.append(f"size mismatch: {relative}")
        if sha256_file(path) != record.get("sha256"):
            output_errors.append(f"SHA-256 mismatch: {relative}")
    actual_paths = {record["path"] for record in _output_records(run_dir)}
    unrecorded = sorted(actual_paths - recorded_paths)
    if unrecorded:
        output_errors.append("unrecorded outputs: " + ", ".join(unrecorded))
    checks.append(_check("output_hashes", not output_errors, "; ".join(output_errors) or "all output hashes match"))
    # 重算 canonical 摘要并逐字段比对（含排除项与算法描述）
    expected_canonical = _canonical_output_record(run_dir, manifest.get("outputs", []))
    canonical_digest_ok = (
        manifest.get("canonical_outputs", {}).get("sha256") == expected_canonical["sha256"]
        and manifest.get("canonical_outputs", {}).get("files") == expected_canonical["files"]
        and manifest.get("canonical_outputs", {}).get("excluded_diagnostics")
        == expected_canonical["excluded_diagnostics"]
        and manifest.get("canonical_outputs", {}).get("algorithm") == expected_canonical["algorithm"]
    )
    checks.append(
        _check(
            "canonical_output_digest",
            canonical_digest_ok,
            "canonical scientific-output digest matches"
            if canonical_digest_ok
            else "canonical scientific-output digest is missing or changed",
        )
    )

    commands = manifest.get("commands", [])
    command_ok = bool(commands) and all(
        isinstance(command, dict) and command.get("returncode") == 0 for command in commands
    )
    checks.append(_check("subprocesses_succeeded", command_ok, "all wrapped scripts returned zero" if command_ok else "one or more wrapped scripts are missing or failed"))
    run_status_ok = manifest.get("run_status") == "completed" and not manifest.get("error")
    checks.append(_check("run_completed", run_status_ok, "run completed" if run_status_ok else str(manifest.get("error") or "run incomplete")))

    anonymity_findings = _anonymous_scan(run_dir)
    checks.append(_check("anonymous_paths", not anonymity_findings, "; ".join(anonymity_findings) or "no absolute identity paths or excluded-tree references"))

    profile = manifest.get("profile")
    canonical_checks: list[dict[str, Any]] = []
    if profile == "paper":
        # ---- canonical 专属资格检查 ----
        backend = manifest.get("backend", {})
        locked_python = (
            load_json(config_path).get("reproducibility", {}).get("python") if config_ok else None
        )
        canonical_checks.extend(
            [
                _check(
                    "canonical_mps_backend",
                    backend.get("selected") == "mps" and backend.get("canonical_backend") is True,
                    f"selected backend: {backend.get('selected')}",
                ),
                _check(
                    "no_device_fallback",
                    backend.get("fallback_used") is False,
                    "no fallback recorded" if backend.get("fallback_used") is False else "fallback was recorded",
                ),
                _check(
                    "fixed_1024_samples",
                    manifest.get("data", {}).get("sample_count") == PAPER_SAMPLE_COUNT,
                    f"sample count: {manifest.get('data', {}).get('sample_count')}",
                ),
                _check(
                    "ten_classes",
                    manifest.get("data", {}).get("class_count") == PAPER_CLASS_COUNT,
                    f"class count: {manifest.get('data', {}).get('class_count')}",
                ),
                _check(
                    "five_repeat_seeds",
                    manifest.get("randomness", {}).get("repeat_seeds") == PAPER_REPEAT_SEEDS,
                    f"repeat seeds: {manifest.get('randomness', {}).get('repeat_seeds')}",
                ),
                _check(
                    "paper_artifact_status",
                    manifest.get("artifact_status") == PAPER_ARTIFACT_STATUS,
                    f"artifact status: {manifest.get('artifact_status')}",
                ),
                _check(
                    "locked_python_version",
                    manifest.get("runtime", {}).get("python") == locked_python,
                    f"runtime Python: {manifest.get('runtime', {}).get('python')}",
                ),
                _check(
                    "reasonable_max_mapping",
                    manifest.get("mapping", {}).get("scenario") == "reasonable_max"
                    and manifest.get("mapping", {}).get("injection_boundary") == "reasonable-max",
                    f"mapping: {manifest.get('mapping')}",
                ),
            ]
        )
        # 分层分布必须精确是 [103,103,103,103,102,102,102,102,102,102]
        distribution = manifest.get("data", {}).get("class_distribution", {})
        ordered_counts = [int(distribution[key]) for key in sorted(distribution)] if isinstance(distribution, dict) else []
        canonical_checks.append(
            _check(
                "stratified_distribution",
                ordered_counts == [103, 103, 103, 103, 102, 102, 102, 102, 102, 102],
                f"class counts: {ordered_counts}",
            )
        )
        # 非理想性 manifest 必须已升格（有局限的论文可用状态）
        source_nonideality = run_dir / "nonideality_accuracy_sweep_manifest.json"
        promoted = load_json(source_nonideality) if source_nonideality.is_file() else {}
        canonical_checks.append(
            _check(
                "nonideality_promoted_with_limitations",
                promoted.get("status") == PAPER_ARTIFACT_STATUS
                and promoted.get("evidence_tier") == PAPER_EVIDENCE_TIER,
                f"nonideality status/tier: {promoted.get('status')}/{promoted.get('evidence_tier')}",
            )
        )
    checks.extend(canonical_checks)
    base_valid = all(check["passed"] for check in checks if check not in canonical_checks)
    release_ready = profile == "paper" and base_valid and all(check["passed"] for check in canonical_checks)
    report = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "run_manifest_sha256": sha256_file(manifest_path),
        "profile": profile,
        "valid": base_valid,
        "release_ready": release_ready,
        "checks": checks,
        "errors": [check["detail"] for check in checks if not check["passed"]],
    }
    return report


def write_verification_report(run_dir: pathlib.Path) -> dict[str, Any]:
    """执行 verify_run 并把报告写回 run 目录（verification_report.json）。"""
    report = verify_run(run_dir)
    write_json(run_dir / "verification_report.json", report)
    return report


def _safe_archive_name(name: str) -> pathlib.PurePosixPath:
    """Validate an archive member name before using it as a logical path."""
    # （英文原注释）在把归档成员名当作逻辑路径前先做安全校验。
    # 拒绝绝对路径、空路径与含 ".." 的路径（防解压逃逸）。
    path = pathlib.PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ReproducibilityError(f"Unsafe reference archive member: {name}")
    return path


def _reference_archive_payloads(
    reference: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Read a sanitized reference archive without extracting it to disk."""
    # （英文原注释）读取"净化过的"参考归档，但不解压到磁盘。
    # 直接在内部分析 tar 成员：要求恰好一个 repro_manifest.json 作根，
    # 其余文件必须是该根之下的普通文件（拒绝链接、重复、越界成员），
    # 全部读到内存字典里。
    payloads: dict[str, bytes] = {}
    manifest_names: list[str] = []
    try:
        archive = tarfile.open(reference, mode="r:*")
    except (tarfile.TarError, OSError) as exc:
        raise ReproducibilityError(f"Could not read reference archive: {reference.name}") from exc
    with archive:
        regular_members = [member for member in archive.getmembers() if member.isfile()]
        unsafe = [member.name for member in archive.getmembers() if member.issym() or member.islnk()]
        if unsafe:
            raise ReproducibilityError("Reference archive contains links: " + ", ".join(unsafe))
        for member in regular_members:
            logical = _safe_archive_name(member.name)
            if logical.name == "repro_manifest.json":
                manifest_names.append(logical.as_posix())
        if len(manifest_names) != 1:
            raise ReproducibilityError(
                "Reference archive must contain exactly one repro_manifest.json"
            )
        manifest_name = manifest_names[0]
        root = pathlib.PurePosixPath(manifest_name).parent  # manifest 所在目录 = 包根
        for member in regular_members:
            logical = _safe_archive_name(member.name)
            try:
                relative = logical.relative_to(root).as_posix()
            except ValueError as exc:
                raise ReproducibilityError(
                    f"Reference archive member is outside the package root: {member.name}"
                ) from exc
            if relative in payloads:
                raise ReproducibilityError(f"Duplicate reference archive member: {relative}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ReproducibilityError(f"Could not read reference archive member: {relative}")
            payloads[relative] = extracted.read()  # 全量读入内存
    try:
        manifest = json.loads(payloads["repro_manifest.json"].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReproducibilityError("Reference archive manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ReproducibilityError("Reference archive manifest must be a JSON object")
    return manifest, payloads


def _reference_directory_payloads(
    reference: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Read an extracted reference package using its manifest as the root."""
    # （英文原注释）以 manifest 位置为根，读取"已解压"的参考包。
    # 用法：读者手动解压 tar.gz 后，直接把这个目录交给 compare/verify。
    candidates = sorted(reference.rglob("repro_manifest.json"))
    if len(candidates) != 1:
        raise ReproducibilityError(
            "Reference directory must contain exactly one repro_manifest.json"
        )
    root = candidates[0].parent
    payloads = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }
    manifest = load_json(candidates[0])
    return manifest, payloads


def _load_reference_manifest(reference: pathlib.Path) -> dict[str, Any]:
    """Load a reference manifest from JSON, a tar archive, or an extracted package."""
    # （英文原注释）从 JSON 文件 / tar 归档 / 已解压目录加载参考 manifest。
    # 三种输入形态统一入口，方便 compare 命令灵活接收参考物。
    reference = reference.resolve()
    if reference.is_dir():
        manifest, _ = _reference_directory_payloads(reference)
        return manifest
    if reference.suffix.lower() == ".json":
        return load_json(reference)
    manifest, _ = _reference_archive_payloads(reference)
    return manifest


def verify_reference_archive(reference: pathlib.Path) -> dict[str, Any]:
    """Verify that a sanitized Release archive is complete and self-consistent."""
    # （英文原注释）校验"净化过的"Release 归档是否完整且自洽。
    # 读者侧的第一道闸：解压前后先验证——manifest schema 合法、每个记录的
    # 产物都在包里且大小/哈希一致、没有多余或缺失文件、canonical 摘要一致、
    # 且含"完整参考摘要"（供 compare 用）。全过 = Green，否则 Red。
    reference = reference.resolve()
    if reference.is_dir():
        manifest, payloads = _reference_directory_payloads(reference)
    elif reference.is_file() and reference.suffix.lower() != ".json":
        manifest, payloads = _reference_archive_payloads(reference)
    else:
        raise ReproducibilityError(
            "Reference verification requires a Release archive or its extracted directory"
        )

    checks: list[dict[str, Any]] = []
    shape_errors = _validate_repro_manifest_shape(manifest)
    checks.append(
        _check("manifest_schema", not shape_errors, "; ".join(shape_errors) or "schema complete")
    )

    # 逐产物核对包内文件
    output_errors: list[str] = []
    recorded_paths: set[str] = set()
    for record in manifest.get("outputs", []):
        if not isinstance(record, dict):
            continue
        relative = str(record.get("path", ""))
        try:
            logical = _safe_archive_name(relative).as_posix()
        except ReproducibilityError as exc:
            output_errors.append(str(exc))
            continue
        recorded_paths.add(logical)
        payload = payloads.get(logical)
        if payload is None:
            output_errors.append(f"missing output: {logical}")
            continue
        if len(payload) != record.get("bytes"):
            output_errors.append(f"size mismatch: {logical}")
        if sha256_bytes(payload) != record.get("sha256"):
            output_errors.append(f"SHA-256 mismatch: {logical}")
    # 包内不能有多余文件，也不能有"只有清单没有文件"的记录
    actual_paths = set(payloads) - {"repro_manifest.json"}
    unrecorded = sorted(actual_paths - recorded_paths)
    missing_records = sorted(recorded_paths - actual_paths)
    if unrecorded:
        output_errors.append("unrecorded package files: " + ", ".join(unrecorded))
    if missing_records:
        output_errors.append("manifest-only package files: " + ", ".join(missing_records))
    checks.append(
        _check(
            "packaged_output_hashes",
            not output_errors,
            "; ".join(output_errors) or "all packaged output hashes match",
        )
    )

    expected_canonical = canonical_output_record(manifest.get("outputs", []))
    canonical_ok = manifest.get("canonical_outputs") == expected_canonical
    checks.append(
        _check(
            "packaged_canonical_digest",
            canonical_ok,
            "sanitized package digest matches"
            if canonical_ok
            else "sanitized package digest is missing or changed",
        )
    )
    # 包里必须带"完整参考摘要"（作者侧 freeze 时写入的比对基准）
    comparison = manifest.get("reference_comparison")
    comparison_ok = (
        isinstance(comparison, dict)
        and comparison.get("schema_version") == REFERENCE_COMPARISON_SCHEMA_VERSION
        and isinstance(comparison.get("canonical_outputs"), dict)
        and bool(comparison["canonical_outputs"].get("sha256"))
    )
    checks.append(
        _check(
            "full_reference_digest",
            comparison_ok,
            "full canonical reference digest is present"
            if comparison_ok
            else "full canonical reference digest is missing",
        )
    )
    status = "Green" if all(check["passed"] for check in checks) else "Red"
    return {
        "schema_version": REFERENCE_VERIFICATION_SCHEMA_VERSION,
        "status": status,
        "reference": reference.name,
        "manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "packaged_file_count": len(payloads),
        "checks": checks,
        "errors": [check["detail"] for check in checks if not check["passed"]],
    }


def compare_run_to_reference(run_dir: pathlib.Path, reference: pathlib.Path) -> dict[str, Any]:
    """Compare a verified full paper run with the immutable reference digest."""
    # （英文原注释）把已验证的完整论文 run 与"不可变参考摘要"比对。
    # 这是"别人复现你"的核心机制：读者跑完自己的 paper run 后，下载作者
    # 发布的参考包，这里逐项比对——参考包自洽、profile 一致、config 哈希
    # 一致、子集哈希一致、canonical 后端无回退、canonical 输出摘要与文件
    # 清单完全一致。全部通过 = match=True（Green）。
    run_dir = run_dir.resolve()
    reference = reference.resolve()
    run_report = verify_run(run_dir)  # 先自证 run 合法
    run_manifest = load_json(run_dir / "repro_manifest.json")
    reference_manifest = _load_reference_manifest(reference)
    comparison_record = reference_manifest.get("reference_comparison", {})
    expected_canonical = (
        comparison_record.get("canonical_outputs")
        if isinstance(comparison_record, dict)
        else None
    ) or reference_manifest.get("canonical_outputs", {})  # 优先用完整参考摘要
    observed_canonical = run_manifest.get("canonical_outputs", {})

    # 逐文件清单比对（路径 → 记录）
    expected_files = {
        str(record.get("path")): record
        for record in expected_canonical.get("files", [])
        if isinstance(record, dict)
    }
    observed_files = {
        str(record.get("path")): record
        for record in observed_canonical.get("files", [])
        if isinstance(record, dict)
    }
    file_differences: list[dict[str, Any]] = []
    for path in sorted(set(expected_files) | set(observed_files)):
        expected = expected_files.get(path)
        observed = observed_files.get(path)
        if expected != observed:
            file_differences.append(
                {
                    "path": path,
                    "expected": expected,
                    "observed": observed,
                }
            )

    # 若参考是归档/目录而非裸 JSON，先做包自校验
    reference_package_ok = True
    reference_package_detail = "reference manifest loaded"
    if not reference.is_file() or reference.suffix.lower() != ".json":
        package_report = verify_reference_archive(reference)
        reference_package_ok = package_report["status"] == "Green"
        reference_package_detail = (
            "reference package self-verifies"
            if reference_package_ok
            else "; ".join(package_report["errors"])
        )

    config_match = (
        run_manifest.get("config", {}).get("sha256")
        == reference_manifest.get("config", {}).get("sha256")
    )
    subset_match = (
        run_manifest.get("data", {}).get("subset_sha256")
        == reference_manifest.get("data", {}).get("subset_sha256")
    )
    canonical_match = (
        bool(expected_canonical.get("sha256"))
        and observed_canonical.get("sha256") == expected_canonical.get("sha256")
        and not file_differences
    )

    checks = [
        _check(
            "run_release_ready",
            bool(run_report.get("release_ready")),
            "full paper run passes all release gates"
            if run_report.get("release_ready")
            else "; ".join(run_report.get("errors", [])),
        ),
        _check("reference_package", reference_package_ok, reference_package_detail),
        _check(
            "profile",
            run_manifest.get("profile") == reference_manifest.get("profile") == "paper",
            f"run/reference profiles: {run_manifest.get('profile')}/{reference_manifest.get('profile')}",
        ),
        _check(
            "config_sha256",
            config_match,
            "config snapshots match" if config_match else "config snapshot hashes differ",
        ),
        _check(
            "subset_sha256",
            subset_match,
            "fixed sample subsets match" if subset_match else "fixed sample subset hashes differ",
        ),
        _check(
            "canonical_backend",
            run_manifest.get("backend", {}).get("selected") == "mps"
            and run_manifest.get("backend", {}).get("fallback_used") is False,
            "MPS selected with no fallback",
        ),
        _check(
            "canonical_output_digest",
            canonical_match,
            "canonical output digest and file inventory match"
            if canonical_match
            else (
                "canonical output digest differs; "
                f"{len(file_differences)} file inventory difference(s)"
            ),
        ),
    ]
    match = all(check["passed"] for check in checks)
    return {
        "schema_version": REFERENCE_COMPARISON_SCHEMA_VERSION,
        "status": "Green" if match else "Red",
        "match": match,
        "reference": reference.name,
        "reference_manifest_sha256": sha256_bytes(canonical_json_bytes(reference_manifest)),
        "expected_canonical_sha256": expected_canonical.get("sha256", ""),
        "observed_canonical_sha256": observed_canonical.get("sha256", ""),
        "checks": checks,
        "file_differences": file_differences,
        "errors": [check["detail"] for check in checks if not check["passed"]],
    }


def write_reference_comparison_report(
    run_dir: pathlib.Path, reference: pathlib.Path, output: pathlib.Path | None = None
) -> dict[str, Any]:
    """Compare a run and write the reviewer-facing machine-readable report."""
    # （英文原注释）比对 run 并写出"面向审稿人的"机器可读报告。
    # 默认写到 run_dir/reference_comparison_report.json；也可用 --output 指定。
    report = compare_run_to_reference(run_dir, reference)
    write_json(output or (run_dir / "reference_comparison_report.json"), report)
    return report


def _reference_member(relative: str) -> bool:
    """判断某个相对路径是否属于"reference 树"（要复制进 reference/ 的成员）。

    包含：config 快照、manifest、输入子集、tables/ 下全部文件、
    以及 run 根目录下的 *_manifest.json。这些是审稿比对的关键材料。

    :param relative: run 内相对路径。
    :return: True 表示该文件属于 reference 树。
    """
    if relative in {"config_snapshot.json", "repro_manifest.json", "inputs/subset.csv"}:
        return True
    if relative.startswith("tables/"):
        return True
    if "/" not in relative and relative.endswith("_manifest.json"):
        return True
    return False


def _archive_member(relative: str) -> bool:
    """判断某个相对路径是否要打进"发布归档"（比 reference 树稍大）。

    在 _reference_member 基础上，额外包含 raw/ 下的 CSV/JSON（每样本
    置信度等原始细节），供读者深挖。

    :param relative: run 内相对路径。
    :return: True 表示该文件要打包。
    """
    if _reference_member(relative):
        return True
    if relative.startswith("raw/") and pathlib.PurePosixPath(relative).suffix.lower() in {
        ".csv",
        ".json",
    }:
        return True
    return False


def _deterministic_tar_gz(
    source_root: pathlib.Path,
    members: Sequence[str],
    destination: pathlib.Path,
    *,
    archive_root: str,
) -> None:
    """生成"确定性 tar.gz"：所有成员的时间戳/UID/GID/权限全部归零。

    为什么重要：普通 tar 会把文件 mtime 写进去，导致"同一份内容"打两次包
    哈希不同，无法用于可复现比对。这里把 mtime=0、uid/gid=0、uname/gname
    置空、权限固定 0644，并用 PAX 格式 + gzip mtime=0，使"同内容→同哈希"
    成立。

    :param source_root: 源目录（run 目录）。
    :param members: 要打包的相对路径列表。
    :param destination: 归档输出路径。
    :param archive_root: 归档内顶层目录名。
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle:
            with tarfile.open(fileobj=gzip_handle, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for relative in sorted(members):
                    path = _safe_join(source_root, relative)
                    info = archive.gettarinfo(str(path), arcname=f"{archive_root}/{relative}")
                    info.uid = 0  # 元数据归零（确定性关键）
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)


def _write_sha256sums(output_dir: pathlib.Path, paths: Sequence[pathlib.Path]) -> pathlib.Path:
    """生成 SHA256SUMS 校验清单文件（每行：哈希 + 相对路径）。"""
    sums_path = output_dir / "SHA256SUMS"
    lines = [f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}" for path in sorted(paths)]
    sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return sums_path


def freeze_run(
    *,
    run_dir: pathlib.Path,
    output_dir: pathlib.Path,
    version: str = "v1.0.1",
) -> dict[str, Any]:
    """Freeze a verified canonical run into a small reference tree and release archive."""
    # （英文原注释）把已验证的 canonical run 冻结成"小参考树 + 发布归档"。
    # 这是发布侧的最后一环：
    #   1) verify_run 必须 release_ready，否则拒绝冻结；
    #   2) 按 allowlist 挑文件：reference 树（tables/manifest/subset 等）
    #      复制成可浏览的 reference/ 目录；archive 成员打成确定性 tar.gz；
    #   3) 写 freeze_manifest.json 记录全部哈希，再写 SHA256SUMS。
    # 产物不含数据集与权重（许可证原因），故 manifest 标记
    # excludes_dataset_and_weights=True。
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    verification = verify_run(run_dir)
    if not verification["release_ready"]:
        raise ReproducibilityError(
            "Run is not release-ready: " + "; ".join(verification.get("errors", []))
        )
    _require_empty_output(output_dir)
    all_members = [record["path"] for record in _output_records(run_dir)]
    reference_members = sorted(relative for relative in all_members if _reference_member(relative))
    archive_members = sorted(relative for relative in all_members if _archive_member(relative))
    if not reference_members or not archive_members:
        raise ReproducibilityError("Freeze allowlist selected no artifact files")

    reference_dir = output_dir / "reference"
    copied: list[pathlib.Path] = []
    for relative in reference_members:
        source = _safe_join(run_dir, relative)
        target = _safe_join(reference_dir, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied.append(target)
    archive_path = output_dir / f"hpat-artifact-{version}.tar.gz"
    _deterministic_tar_gz(
        run_dir,
        archive_members,
        archive_path,
        archive_root=f"hpat-artifact-{version}",
    )
    manifest = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "created_utc": utc_now(),
        "version": version,
        "status": "release_ready_with_limitations",
        "run_manifest_sha256": sha256_file(run_dir / "repro_manifest.json"),
        "verification": verification,
        "reference_files": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(copied)
        ],
        "release_archive": {
            "path": archive_path.name,
            "sha256": sha256_file(archive_path),
            "bytes": archive_path.stat().st_size,
            "members": archive_members,
            "excludes_dataset_and_weights": True,  # 明确：包内不含数据集与权重
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    freeze_manifest_path = output_dir / "freeze_manifest.json"
    write_json(freeze_manifest_path, manifest)
    checksum_inputs = [*copied, archive_path, freeze_manifest_path]
    _write_sha256sums(output_dir, checksum_inputs)
    return manifest
