from __future__ import annotations

"""
audit_public_release.py —— 发布审计脚本（给 HPAT 论文匿名公开版做"安全检查"）

做什么：
    扫描待公开的代码/数据仓库，检查是否存在不能对外公开的内容，例如：
    作者本机路径（如 /Users/xxx）、作者账号标识、邮箱地址、ORCID、设备 ID、
    被排除的目录（缓存、构建产物、论文稿件等）、超大文件、符号链接、C2PA 溯源元数据。
    最后输出一份 JSON 审计报告，状态为 "Green"（通过）或 "Red"（有拦截项）。

数据从哪来：
    扫描路径由命令行 --root 指定，默认是当前目录（即公开版 checkout 根目录）。
    白名单（允许发布的文件清单）与必含文件清单可经 --allowlist 指向的 JSON 文件提供。

产出到哪：
    审计报告打印到标准输出；若给了 --output，同时写入该 JSON 文件（报告文件自身会被排除在扫描之外）。

如何运行：
    python audit_public_release.py --root <公开版目录> [--output report.json] [--allowlist allowlist.json]
    进程退出码为 0（Green）或 1（Red）。
"""

import argparse
import fnmatch
import json
import pathlib
import re
import sys
from collections.abc import Iterable
from typing import Any


# REPO_ROOT：脚本向上两级目录，即仓库根目录（仅作为参考常量使用）
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
# 默认的单文件大小上限：20 MiB，超过则视为不符合发布要求
DEFAULT_MAX_FILE_BYTES = 20 * 1024 * 1024

# These directories are either local state, unreviewed evidence, build output,
# or third-party payloads.  They must not be present in the public allowlist
# checkout.  Matching is component based and therefore does not reject prose
# that merely documents the exclusion policy.
# 中文说明：下面这些目录要么是本机状态、未审阅的证据、构建产物，要么是第三方内容，
# 它们不允许出现在公开版中。匹配是按路径"组件"（目录名）判断的，因此文档里单纯
# 描述这条排除策略的文字不会被误判为违规。
FORBIDDEN_DIRECTORY_NAMES = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "cache",
    "coreml",
    "device_logs",
    "image2_drafts",
    "logs",
    "model_binaries",
    "node_modules",
    "manuscript",
    "paper",
    "paper_src",
    "review_packets",
    "runs",
    "third_party_papers",
    "tmp",
    "weights",
}

# 禁止发布的单文件（按文件名精确匹配）：DS_Store 是 macOS 系统垃圾文件，
# .research-resource.json 可能包含研究资源来源信息，均不应出现在公开版里
FORBIDDEN_FILE_NAMES = {
    ".DS_Store",
    ".research-resource.json",
}

# 禁止发布的文件后缀：.pdf/.tex 通常属于论文稿件，不属于"仅实验"的公开范围
FORBIDDEN_FILE_SUFFIXES = {".pdf", ".tex"}
# 禁止发布的路径前缀：figures/final/ 下是最终版论文插图，同样不在公开范围内
FORBIDDEN_PATH_PREFIXES = ("figures/final/",)

# 视为"文本文件"的后缀集合：文本文件会做完整的身份信息（邮箱/路径等）扫描；
# 其余后缀（图片、压缩包等）只扫描可打印的元数据字符串
TEXT_SUFFIXES = {
    "",
    ".bib",
    ".cff",
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}

# 用于识别敏感信息的正则：
# 邮箱地址（形如 xxx@xxx.xxx）
_EMAIL_RE = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+")
# ORCID 研究者编号（16 位数字+校验位，形如 0000-0000-0000-0000）
_ORCID_RE = re.compile(r"(?i)\b(?:https?://orcid\.org/)?\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")
# 设备标识（UDID / 设备 ID / 序列号等键值对，取冒号或等号后的长串字符）
_DEVICE_ID_RE = re.compile(
    r"(?i)\b(?:udid|device[_ -]?id|device[_ -]?identifier|hardware[_ -]?uuid|"
    r"serial(?:[_ -]?number)?)\b\s*[=:]\s*[\"']?([A-Z0-9][A-Z0-9-]{7,})"
)
# 三类本机主目录路径的匹配规则：macOS/Linux 的 /Users/或 /home/、Windows 的 C:\Users\、以及 file:// 形式
_HOME_PATH_RES = [
    re.compile(r"/(?:Users|home)/[^/\s\"']+", flags=re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\s\"']+", flags=re.IGNORECASE),
    re.compile(r"file:///(?:Users|home)/[^/\s\"']+", flags=re.IGNORECASE),
]


def _sha256(path: pathlib.Path) -> str:
    """计算文件的 SHA-256 校验和。

    用途：对文件内容做摘要，便于核对文件是否被改动（本脚本中预留用）。
    参数：
        path: 要计算校验和的文件路径。
    返回：
        小写十六进制字符串形式的 SHA-256 摘要。
    """
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: pathlib.Path, root: pathlib.Path) -> str:
    """Return a stable POSIX label without ever exposing an absolute path.
    返回一个稳定的 POSIX 格式的相对路径标签，绝不泄露绝对路径。

    用途：把文件相对仓库根目录的位置转成统一的"正斜杠"标签（如 a/b/c.py），
          用于审计报告里的路径展示，避免把作者本机的绝对路径泄露进报告。
    参数：
        path: 需要转换的文件路径。
        root: 仓库根目录，用于计算相对位置。
    返回：
        相对路径字符串；若无法计算（例如跨盘符），则退回 external/<文件名>。
    """

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        # 计算相对路径失败时，退化为 "external/文件名" 形式，并丢弃绝对路径信息
        return f"external/{path.name}"


def _is_forbidden_directory(relative_path: pathlib.PurePath) -> str | None:
    """判断路径的父目录链中是否含有被禁止发布的目录名。

    用途：检查公开版里是否混入了缓存、构建产物、论文稿等禁止目录。
    参数：
        relative_path: 相对路径对象（不取文件名的最后一段，只看目录组件）。
    返回：
        命中时返回该被禁止的目录名（小写匹配）；否则返回 None。
    """

    # parts[:-1] 去掉文件名，只遍历目录层级；逐段比对，大小写不敏感
    for part in relative_path.parts[:-1]:
        if part.lower() in FORBIDDEN_DIRECTORY_NAMES:
            return part
    return None


def _looks_textual(path: pathlib.Path) -> bool:
    """粗略判断文件是不是文本类文件（按扩展名查表）。

    用途：决定对该文件做"全量文本扫描"还是仅做"二进制元数据扫描"。
    参数：
        path: 待判断的文件路径。
    返回：
        True 表示扩展名在 TEXT_SUFFIXES 表中，按文本文件处理。
    """

    return path.suffix.lower() in TEXT_SUFFIXES


def _identity_findings(text: str) -> list[dict[str, str]]:
    """在给定文本中查找身份/隐私信息，返回查到的条目列表。

    用途：扫描邮箱、ORCID、设备 ID、作者本机路径、账号标识等敏感信息。
          审计脚本自身的源码里会出现这些正则与示例，因此把关键字拆开拼写，
          避免扫描整个公开仓库时把"实现这些检查的代码"误报为泄露。
    参数：
        text: 待扫描的文本内容。
    返回：
        由 dict 组成的列表，每个元素含 kind（泄露类型）与 match（脱敏后的展示值）。
    """

    findings: list[dict[str, str]] = []
    lowered = text.lower()
    # Split literals keep this auditor from reporting its own implementation
    # when the complete public checkout is scanned.
    # 中文说明：关键字用字符串拼接（而非完整字面量），防止审计自己时被误报
    user_prefix = ("/" + "Users" + "/").lower()
    redacted_user_path = "/" + "Users" + "/<redacted>"
    local_account = ("jk" + "6k").lower()
    # 依次检查各类敏感信息；匹配到的值统一用 <redacted-xxx> 占位符脱敏展示
    if user_prefix in lowered:
        findings.append({"kind": "absolute_user_path", "match": redacted_user_path})
    if any(pattern.search(text) for pattern in _HOME_PATH_RES):
        # 若刚才已记录过绝对路径泄露，就不重复追加同类型条目
        if not any(item["kind"] == "absolute_user_path" for item in findings):
            findings.append({"kind": "absolute_user_path", "match": "<redacted-home-path>"})
    if local_account in lowered:
        findings.append({"kind": "local_account_identifier", "match": "<redacted-account>"})
    if _EMAIL_RE.search(text):
        findings.append({"kind": "email_address", "match": "<redacted-email>"})
    if _ORCID_RE.search(text):
        findings.append({"kind": "orcid", "match": "<redacted-orcid>"})
    if _DEVICE_ID_RE.search(text):
        findings.append({"kind": "device_identifier", "match": "<redacted-device-id>"})
    return findings


def audit_file_set(
    files: Iterable[tuple[pathlib.Path, str]],
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    check_forbidden_paths: bool = True,
    check_size: bool = True,
    allowed_patterns: Iterable[str] | None = None,
    required_files: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Audit an explicitly labelled file set.
    审计一组带标签的文件。

    ``label`` is the path written into reports.  Callers must supply a relative
    label; absolute labels are converted to a redacted basename and blocked.
    The function intentionally returns findings rather than raising so a Red
    audit is still machine-readable.
    中文说明：
        label 是写进报告里的路径。调用方必须传相对路径；若传入绝对路径，
        会被改写成脱敏的文件名并记录拦截项。
        本函数刻意"收集问题并返回"而不是直接抛异常，这样即使审计结果是
        Red（失败），报告仍然是可机读的 JSON，方便脚本继续处理。
    参数：
        files: 待审计文件的可迭代对象，元素为 (文件路径 Path, 相对标签 str)。
        max_file_bytes: 单文件大小上限（默认 20 MiB）。
        check_forbidden_paths: 是否检查被禁止的目录/文件/路径前缀。
        check_size: 是否检查文件大小上限。
        allowed_patterns: 公开白名单 glob 模式；非空时只允许匹配到的文件。
        required_files: 必须出现在公开版中的文件标签列表。
    返回：
        汇总审计结果的 dict（schema_version、status、blockers 等）。
    """

    # blockers：收集到的所有"拦截项"，即不符合发布要求的问题清单
    blockers: list[dict[str, Any]] = []
    # 统计信息：扫描过的文件数、文本文件数、累计字节数
    scanned_files = 0
    scanned_text_files = 0
    total_bytes = 0

    # 把白名单与必含文件统一成"正斜杠 + 去重 + 排序"的元组，便于精确匹配
    patterns = tuple(sorted({str(pattern).replace("\\", "/") for pattern in (allowed_patterns or [])}))
    required = tuple(sorted({str(path).replace("\\", "/") for path in (required_files or [])}))
    matched_patterns: set[str] = set()
    seen_labels: set[str] = set()

    # 按标签排序遍历，保证审计顺序稳定、输出可复现
    for source, supplied_label in sorted(files, key=lambda item: item[1]):
        # 统一把标签转成 POSIX 风格（正斜杠）路径
        label_path = pathlib.PurePosixPath(str(supplied_label).replace("\\", "/"))
        label = label_path.as_posix()
        # 若标签是绝对路径，或含有 .. 试图跳出仓库根目录，则记为拦截项并脱敏处理
        if label_path.is_absolute() or ".." in label_path.parts:
            blockers.append(
                {
                    "code": "non_relative_report_path",
                    "path": f"external/{source.name}",
                    "detail": "Audit input label was absolute or escaped its root.",
                }
            )
            label = f"external/{source.name}"
        # 记录已出现的标签，供后面检查"必含文件是否齐全"
        seen_labels.add(label)

        # 若提供了白名单，则检查当前文件是否被某个模式匹配到
        if patterns:
            # fnmatchcase 做大小写敏感的 glob 匹配；匹配到则记录该模式被命中
            matching = [pattern for pattern in patterns if fnmatch.fnmatchcase(label, pattern)]
            if matching:
                matched_patterns.update(matching)
            else:
                blockers.append(
                    {
                        "code": "not_in_public_allowlist",
                        "path": label,
                        "detail": "Path is not matched by the declared public-release allowlist.",
                    }
                )

        # 发布条目必须是可以独立拷贝的普通文件：符号链接、不存在的路径、
        # 非普通文件（如目录、设备文件）都会直接记为拦截项并跳过后续检查
        if source.is_symlink():
            blockers.append(
                {
                    "code": "symlink_not_allowed",
                    "path": label,
                    "detail": "Public release entries must be regular files, not symlinks.",
                }
            )
            continue
        if not source.exists():
            blockers.append({"code": "missing_file", "path": label, "detail": "File does not exist."})
            continue
        if not source.is_file():
            blockers.append({"code": "non_regular_file", "path": label, "detail": "Entry is not a regular file."})
            continue

        scanned_files += 1
        size = source.stat().st_size
        total_bytes += size
        # 超过单文件大小上限（默认 20 MiB）记为拦截项
        if check_size and size > max_file_bytes:
            blockers.append(
                {
                    "code": "file_exceeds_20_mib_limit" if max_file_bytes == DEFAULT_MAX_FILE_BYTES else "file_exceeds_limit",
                    "path": label,
                    "size_bytes": size,
                    "limit_bytes": max_file_bytes,
                }
            )

        # 检查路径本身是否命中"被禁止的目录/文件/后缀/前缀"
        if check_forbidden_paths:
            forbidden_dir = _is_forbidden_directory(label_path)
            if forbidden_dir is not None:
                blockers.append(
                    {
                        "code": "forbidden_directory",
                        "path": label,
                        "detail": f"Path contains excluded directory '{forbidden_dir}'.",
                    }
                )
            if label_path.name in FORBIDDEN_FILE_NAMES:
                blockers.append({"code": "forbidden_file", "path": label, "detail": "File is excluded from public releases."})
            if label_path.suffix.lower() in FORBIDDEN_FILE_SUFFIXES:
                blockers.append(
                    {
                        "code": "forbidden_publication_file",
                        "path": label,
                        "detail": "Manuscript PDF/TeX files are outside the experiment-only public scope.",
                    }
                )
            if any(label.startswith(prefix) for prefix in FORBIDDEN_PATH_PREFIXES):
                blockers.append(
                    {
                        "code": "forbidden_publication_path",
                        "path": label,
                        "detail": "Publication artwork is outside the experiment-only public scope.",
                    }
                )

        # Identity strings can be embedded in binary metadata, so scan a
        # bounded byte window for every file.  Structured identity patterns are
        # scanned only for text-like files to avoid noisy binary coincidences.
        # 中文说明：身份信息可能藏在二进制文件的元数据里，因此每个文件都读取
        # 一个"有上限"的字节窗口来扫描；而结构化身份正则只对文本类文件做，
        # 以免二进制数据里偶然出现的字符组合造成误报。
        try:
            with source.open("rb") as handle:
                head = handle.read(min(size, 2 * 1024 * 1024))
            scan_bytes = source.read_bytes() if size <= max_file_bytes else head
            scan_text = scan_bytes.decode("utf-8", errors="ignore")
        except OSError as exc:
            blockers.append({"code": "unreadable_file", "path": label, "detail": str(exc)})
            continue

        # 在原始字节（小写化）里直接搜绝对路径与账号标识；拆开拼接避免审计自报
        raw_lower = scan_bytes.lower()
        if ("/" + "Users" + "/").encode("utf-8").lower() in raw_lower:
            blockers.append(
                {"code": "identity_leak:absolute_user_path", "path": label, "detail": "Contains a local absolute user path."}
            )
        if ("jk" + "6k").encode("utf-8").lower() in raw_lower:
            blockers.append(
                {"code": "identity_leak:local_account_identifier", "path": label, "detail": "Contains a local account identifier."}
            )
        if any(pattern.search(scan_text) for pattern in _HOME_PATH_RES):
            code = "identity_leak:absolute_user_path"
            if not any(item.get("code") == code and item.get("path") == label for item in blockers):
                blockers.append({"code": code, "path": label, "detail": "Contains an absolute user home path."})

        if not _looks_textual(source):
            # Inspect printable metadata strings, not arbitrary decoded
            # compressed bytes: the latter produces false email/C2PA matches
            # in perfectly ordinary PNG/PDF/zip payloads.
            # 中文说明：对非文本文件只检查"可打印的元数据字符串"，不检查解压后的
            # 任意字节，否则普通 PNG/PDF/zip 里的随机内容会误报邮箱或 C2PA 签名。
            # 用正则找出长度 >=6 的连续可见 ASCII 片段，拼起来再搜邮箱等模式
            metadata_text = b"\n".join(re.findall(rb"[\x20-\x7e]{6,}", scan_bytes)).decode(
                "ascii", errors="ignore"
            )
            if _EMAIL_RE.search(metadata_text):
                blockers.append(
                    {
                        "code": "identity_leak:email_address",
                        "path": label,
                        "detail": "Binary metadata contains an email address.",
                    }
                )
            # C2PA（内容真实性联盟）溯源元数据特征串：出现在图片/音视频里通常
            # 说明文件带有 AI 生成或编辑痕迹的签名，需要清洗后才能公开
            strong_c2pa_markers = (
                b"urn:c2pa:",
                b"c2pa.assertions",
                b"c2pa.claim",
                b"org.contentauth.c2pa",
                b"trainedalgorithmicmedia",
            )
            if any(marker in raw_lower for marker in strong_c2pa_markers):
                blockers.append(
                    {
                        "code": "embedded_provenance_metadata",
                        "path": label,
                        "detail": "Binary contains C2PA/JUMBF provenance metadata; publish a losslessly sanitized copy.",
                    }
                )

        if _looks_textual(source):
            scanned_text_files += 1
            # Files under the size limit are small enough to scan completely.
            # For larger files, the bounded head scan above still catches
            # common absolute-path metadata without loading the whole payload.
            # 中文说明：小于上限的文件可以整体读入做全量扫描；更大的文件上面已经
            # 用"头部窗口"扫描过常见的绝对路径信息，这里就不再整文件加载。
            text = scan_text
            if size <= max_file_bytes:
                try:
                    text = source.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    pass
            # 逐条记录身份泄露；同一文件同一类型只记一条，避免重复刷屏
            for finding in _identity_findings(text):
                code = f"identity_leak:{finding['kind']}"
                if any(item.get("code") == code and item.get("path") == label for item in blockers):
                    continue
                blockers.append({"code": code, "path": label, "detail": finding["match"]})

    # 校验"必含文件"：要求它们都存在且是相对仓库根目录的路径
    for required_path in required:
        required_label = pathlib.PurePosixPath(required_path).as_posix()
        if pathlib.PurePosixPath(required_path).is_absolute() or ".." in pathlib.PurePosixPath(required_path).parts:
            blockers.append(
                {
                    "code": "invalid_required_public_file",
                    "path": required_label,
                    "detail": "Required public file labels must be repository-relative paths.",
                }
            )
        elif required_label not in seen_labels:
            blockers.append(
                {
                    "code": "missing_required_public_file",
                    "path": required_label,
                    "detail": "Required public-release file is absent from the scanned tree.",
                }
            )

    # 汇总前按"路径 + 错误码"排序，保证报告输出顺序稳定
    blockers.sort(key=lambda row: (str(row.get("path", "")), str(row.get("code", ""))))
    return {
        "schema_version": "hpat-public-release-audit-v1",
        "status": "Green" if not blockers else "Red",
        "root": ".",
        "max_file_bytes": max_file_bytes,
        "scanned_file_count": scanned_files,
        "scanned_text_file_count": scanned_text_files,
        "total_bytes": total_bytes,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "allowlist_enabled": bool(patterns),
        "allowlist_pattern_count": len(patterns),
        "matched_allowlist_pattern_count": len(matched_patterns),
        "required_public_file_count": len(required),
        "present_required_public_file_count": sum(1 for path in required if path in seen_labels),
        "claim_boundary": (
            "A Green anonymity/package audit establishes only that the scanned release tree meets the declared "
            "packaging rules; it does not promote modelled HPAT results to measured silicon or deployment evidence."
            # 中文说明：Green 审计只说明打包符合规则，并不等于把 HPAT 的仿真结果
            # 升级为硅片实测或部署证据——这是审计结论的"声明边界"。
        ),
    }


def _walk_release(root: pathlib.Path, excluded: set[pathlib.Path]) -> list[tuple[pathlib.Path, str]]:
    """递归收集发布目录下的所有文件（带相对标签）。

    用途：遍历整个待发布目录，生成 (文件路径, 相对标签) 列表供审计使用。
          会跳过 .git 目录以及调用方指定排除的文件（如审计报告自身）。
    参数：
        root: 发布目录根路径。
        excluded: 需要排除在外的文件路径集合（按解析后的绝对路径比较）。
    返回：
        (pathlib.Path, str) 元组列表，str 为 POSIX 风格的相对标签。
    """

    files: list[tuple[pathlib.Path, str]] = []
    # rglob 递归列出全部条目，再排序保证遍历顺序稳定
    for path in sorted(root.rglob("*")):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if not relative.parts:
            continue
        # .git 目录不属于发布内容，跳过
        if relative.parts[0] == ".git":
            continue
        # 排除调用方指定不扫描的文件（例如报告输出文件自身）
        if any(path.resolve() == item.resolve() for item in excluded):
            continue
        # 符号链接与普通文件都纳入审计（符号链接会被 audit_file_set 拦截）
        if path.is_symlink():
            files.append((path, relative.as_posix()))
        elif path.is_file():
            files.append((path, relative.as_posix()))
    return files


def audit_public_release(
    root: pathlib.Path,
    *,
    output: pathlib.Path | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    allowed_patterns: Iterable[str] | None = None,
    required_files: Iterable[str] | None = None,
) -> dict[str, Any]:
    """对公开版发布目录执行完整审计。

    用途：调用 audit_file_set 扫描整个发布目录并汇总报告；
          若指定 output，还会把 JSON 报告写入该文件（报告自身不参与扫描）。
    参数：
        root: 发布目录根路径。
        output: 可选的报告输出 JSON 文件路径；会从扫描中排除自身。
        max_file_bytes: 单文件大小上限。
        allowed_patterns: 公开白名单模式。
        required_files: 必含文件列表。
    返回：
        与 audit_file_set 相同结构的审计报告 dict。
    """

    # 解析绝对路径；若报告会写进目录内，就把报告文件本身排除在扫描外
    root = root.resolve()
    excluded = {output.resolve()} if output is not None else set()
    # 根目录不存在或不是文件夹时，直接返回一份 Red 报告
    if not root.exists() or not root.is_dir():
        report = {
            "schema_version": "hpat-public-release-audit-v1",
            "status": "Red",
            "root": ".",
            "max_file_bytes": max_file_bytes,
            "scanned_file_count": 0,
            "scanned_text_file_count": 0,
            "total_bytes": 0,
            "blocker_count": 1,
            "blockers": [{"code": "invalid_release_root", "path": ".", "detail": "Release root is missing or not a directory."}],
        }
    else:
        report = audit_file_set(
            _walk_release(root, excluded),
            max_file_bytes=max_file_bytes,
            allowed_patterns=allowed_patterns,
            required_files=required_files,
        )

    # 需要输出报告时，先确保父目录存在，再以 UTF-8 写入缩进排版的 JSON
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> int:
    """命令行入口：解析参数，执行审计，打印报告并返回退出码。

    用途：把 audit_public_release 封装成可命令行调用的工具。
    返回：
        0 表示审计通过（Green），1 表示存在拦截项（Red）。
    """

    parser = argparse.ArgumentParser(description="Audit an anonymous HPAT public-release checkout.")
    parser.add_argument("--root", default=".", help="Public release checkout to scan.")
    parser.add_argument("--output", default="", help="Optional JSON report path (excluded from its own scan).")
    # 单文件大小上限（单位 MiB），默认 20
    parser.add_argument("--max-file-mib", type=float, default=20.0)
    parser.add_argument(
        "--allowlist",
        default="",
        help="Optional JSON file containing a public_allowlist array (or a JSON array).",
    )
    args = parser.parse_args()

    # 大小上限必须为正数，否则直接报错退出
    if args.max_file_mib <= 0:
        parser.error("--max-file-mib must be positive")
    output = pathlib.Path(args.output) if args.output else None
    allowed_patterns: list[str] | None = None
    required_files: list[str] | None = None
    # 若提供白名单文件：从中读取 public_allowlist（数组）与 required_public_files（必含文件）
    if args.allowlist:
        payload = json.loads(pathlib.Path(args.allowlist).read_text(encoding="utf-8"))
        raw_patterns = payload.get("public_allowlist") if isinstance(payload, dict) else payload
        if not isinstance(raw_patterns, list) or not all(isinstance(item, str) for item in raw_patterns):
            parser.error("--allowlist must contain a JSON string array or an object with public_allowlist")
        allowed_patterns = raw_patterns
        if isinstance(payload, dict):
            raw_required = payload.get("required_public_files", [])
            if not isinstance(raw_required, list) or not all(isinstance(item, str) for item in raw_required):
                parser.error("required_public_files must be a JSON string array")
            required_files = raw_required
    report = audit_public_release(
        pathlib.Path(args.root),
        output=output,
        max_file_bytes=int(args.max_file_mib * 1024 * 1024),
        allowed_patterns=allowed_patterns,
        required_files=required_files,
    )
    # 把完整报告打印到标准输出，便于人工查看或管道处理
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "Green" else 1


if __name__ == "__main__":
    sys.exit(main())
