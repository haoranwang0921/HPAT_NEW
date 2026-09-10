"""公共工具模块（_common.py）。

本模块是所有实验脚本共用的基础设施，不直接跑实验本身，而是提供一组
"通用小工具"，让其它脚本省去重复造轮子：

- 路径约定：定义仓库根目录、实验结果输出目录（results）、src 源码目录、
  配置文件的统一位置，并自动把 src 目录加进 Python 搜索路径，方便 import 库。
- 文件读写：读写 JSON / CSV / 文本的封装（带 UTF-8、目录自动创建、安全保护）。
- 清单（manifest）生成：每次实验运行都会生成一份"运行元信息"（如什么时间、
  什么命令、什么 Python 版本跑的），供可复现性审计使用。
- 环境信息：Git 状态、依赖包可用性检测等辅助函数。

本模块被其它脚本 import 使用，本身不产出任何结果文件。
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import pathlib
import platform
import subprocess
import sys
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "experiments" / "src"
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json"
PROJECT_TABLES_DIR = REPO_ROOT / "tables"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hpat_eval.p0_contract import MANIFEST_SCHEMA_VERSION, P0_GATE_VERSION


def utc_now() -> str:
    """返回当前 UTC 时间的字符串（精确到秒）。"""
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def run_id() -> str:
    """生成本次运行的时间戳 ID，例如 run_20260719_103000。

    用途：作为输出目录/文件名前缀，让每次实验的结果互不覆盖。
    """
    return _dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")


def ensure_dir(path: pathlib.Path) -> pathlib.Path:
    """确保目录存在，不存在则递归创建，返回该目录路径。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_writes_enabled() -> bool:
    """Return whether repository-facing tables may be updated.

    判断"是否允许把结果写入仓库根目录下的 tables 目录"。

    Project promotion is deliberately opt-in.  A normal experiment invocation
    must remain output-directory isolated; only the exact value ``1`` grants
    permission to update ``<repo>/tables``.

    背景：普通实验只允许把结果写进各自的 results 目录（相互隔离），
    只有当环境变量 HPAT_WRITE_PROJECT_TABLES 恰好等于 "1" 时，
    才允许更新仓库级表格（tables/），避免实验互相污染。
    """

    return os.environ.get("HPAT_WRITE_PROJECT_TABLES", "").strip() == "1"


def is_project_table_path(path: pathlib.Path) -> bool:
    """判断给定路径是否位于仓库级 tables 目录内。

    参数 path：要检查的文件路径。
    返回：True 表示该路径属于 tables 目录（写入它会改动仓库公共产物）。
    """
    try:
        resolved = path.resolve()
        root = PROJECT_TABLES_DIR.resolve()
        return resolved == root or root in resolved.parents
    except FileNotFoundError:
        try:
            return path.absolute().is_relative_to(PROJECT_TABLES_DIR.absolute())
        except ValueError:
            return False


def project_write_skipped(path: pathlib.Path) -> bool:
    """判断对某路径的写入是否应被跳过。

    规则：若目标是 tables 目录下的文件、但当前没有开启"项目写入"权限
    （见 project_writes_enabled），则跳过写入——保证默认情况下实验只写
    自己的 results 目录，不会悄悄改动仓库公共表格。
    """
    return is_project_table_path(path) and not project_writes_enabled()


def load_json(path: pathlib.Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """读取 JSON 文件为字典，默认读实验总配置文件。

    参数 path：JSON 文件路径（默认是 hpat_experiment_config.json）。
    返回：解析出的字典。
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    """把字典写成格式化 JSON 文件。

    注意：写入前先检查 project_write_skipped，若被保护则静默跳过；
    否则自动创建父目录并写入（键按字母排序，缩进 2 格，便于 diff 对比）。
    """
    if project_write_skipped(path):
        return
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    """读取 CSV 文件，把每一行转成一个字典（表头作为键）。

    返回：由每行字典组成的列表。
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """把行字典列表写成 CSV 文件（带表头）。

    参数：
        path：目标 CSV 路径。
        rows：要写入的行，每行是一个字典。
        fieldnames：列顺序（CSV 表头）。
    """
    if project_write_skipped(path):
        return
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        # Keep repository-facing artifacts byte-stable across macOS, Linux, and
        # Windows.  The default csv dialect emits CRLF regardless of host,
        # which previously made hashes depend on which helper produced a table.
        # （中文说明）强制用 "\n" 作为行结尾：默认 CSV 方言在 Windows 上会输出
        # CRLF，导致同一内容在不同操作系统上字节不同、哈希对不上，破坏可复现性。
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: pathlib.Path, text: str) -> None:
    """把纯文本写入文件（UTF-8 编码）。"""
    if project_write_skipped(path):
        return
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: pathlib.Path) -> str | None:
    """计算文件的 SHA-256 校验和，用于可复现性核验。

    参数 path：文件路径。
    返回：十六进制哈希字符串；若文件不存在返回 None。
    """
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        # 按 1MB 分块读取，避免大文件一次读入内存
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_status(names: list[str]) -> dict[str, dict[str, Any]]:
    """检测一批 Python 包是否可被 import，返回每个包的可用状态。

    参数 names：包名列表。
    返回：字典，形如 {包名: {"available": 是否可用, "origin": 来源位置}}。
    origin 会被转换成相对仓库根的路径，避免泄露本机绝对路径。
    """
    status: dict[str, dict[str, Any]] = {}
    for name in names:
        spec = importlib.util.find_spec(name)
        origin: str | None = None
        if spec and getattr(spec, "origin", None):
            origin_path = pathlib.Path(str(spec.origin))
            try:
                # 转成相对仓库根的路径，方便跨机器复现且不泄露本机位置
                origin = str(origin_path.resolve().relative_to(REPO_ROOT.resolve()))
            except (OSError, ValueError):
                origin = origin_path.name
        status[name] = {
            "available": spec is not None,
            "origin": origin,
        }
    return status


def git_short_status() -> str:
    """获取仓库 Git 状态摘要（只含已跟踪文件的改动，不含未跟踪文件）。

    返回：git status --short 的输出字符串；git 不可用时返回错误提示。
    用途：写进实验清单，记录"跑实验时代码处于什么版本/有没有改动"。
    """
    try:
        proc = subprocess.run(
            ["git", "status", "--short", "--untracked-files=no"],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        return proc.stdout.strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"git status unavailable: {exc}"


def base_manifest(name: str, evidence_label: str, command: list[str] | None = None) -> dict[str, Any]:
    """生成一份"实验运行清单"（manifest）的基础字典，记录本次运行的环境信息。

    参数：
        name：清单名称（如 "energy_accounting"）。
        evidence_label：证据标签（如 "P1"/"P2"），用于说明该实验支撑哪一档结论。
        command：本次运行的命令；默认取 sys.argv（脚本启动时的命令行）。
    返回：包含 schema 版本、时间戳、命令、Python 版本、平台、Git 状态等的字典。
    """
    recorded_command = list(command or sys.argv)
    if recorded_command:
        first = pathlib.Path(recorded_command[0])
        try:
            # 把命令第一个参数（通常是 python 解释器路径）转成相对仓库根的路径
            recorded_command[0] = str(first.resolve().relative_to(REPO_ROOT.resolve()))
        except (OSError, ValueError):
            # A Python executable outside the checkout is useful only by name;
            # recording its absolute path leaks the local account/workspace.
            # （中文说明）若解释器在仓库外，只记录文件名即可：
            # 记录绝对路径会泄露本机账号/工作区信息，对复现也没有帮助。
            recorded_command[0] = first.name
    return {
        "name": name,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "gate_version": P0_GATE_VERSION,
        "created_utc": utc_now(),
        "evidence_label": evidence_label,
        "cwd": ".",
        "command": recorded_command,
        "python": {
            "executable": pathlib.Path(sys.executable).name,
            "version": sys.version,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "git_status_tracked_only": git_short_status(),
        "claim_boundary": (
            # 免责声明：本结果既不是真实流片的硅片、也不是作者实测的
            # HPAT 边缘部署数据、更不是最终标定过的能耗，除非另行提升证据级别。
            "Not fabricated silicon, not author-measured HPAT edge deployment, "
            "and not final calibrated energy unless separately promoted."
        ),
    }


def relative(path: pathlib.Path) -> str:
    """把路径转成相对仓库根的字符串，方便在清单/日志中记录。

    参数 path：任意路径。
    返回：相对仓库根的路径字符串；若不在仓库内则原样返回。
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
