"""清单（manifest）文件生成工具：为论文实验产物打上"可溯源"标签。

背景：为了满足可复现性要求，每个生成的图表/数据文件都应附带一个
"元数据清单"（.meta.json），记录它由哪些源代码和数据生成、当时用什么
命令、作者对它的证据强度声明是什么。这样读者可以逐级回溯到原始证据，
防止"图上画的结论无法验证"。本文件就是构造这些清单的底层工具。

关键概念：
- evidence_label / evidence_tier：证据强度标签（如模拟建模 / 实测 / 仿真）。
- claim_boundary：声明边界——说明这份证据"不是"什么，防止夸大结论。
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from typing import Any

from hpat_eval.p0_contract import MANIFEST_SCHEMA_VERSION, P0_GATE_VERSION


# 声明边界说明：这里产生的都只是"架构级或建模级"的 HPAT 证据，
# 既不是流片后的真实硅片，也不是作者实测的 HPAT 端侧/移动端部署结果，
# 更不是实测的 HPAT 加速比。写清楚这一点，读者就不会误把模拟当实测。
CLAIM_BOUNDARY_NOTE = (
    "Architecture-level or modelled HPAT evidence only. Not fabricated silicon, "
    "not author-measured HPAT edge/mobile deployment, and not measured HPAT speedup."
)


def sha256_file(path: pathlib.Path) -> str | None:
    """计算文件的 SHA-256 哈希值（用于内容校验 / 溯源）。

    :param path: 待计算的文件路径。
    :return: 文件的十六进制 SHA-256 摘要；文件不存在时返回 None。
    """
    if not path.exists():
        return None
    import hashlib  # 局部导入，避免文件头部 import 过多

    h = hashlib.sha256()
    with path.open("rb") as f:
        # 分块读取（每块 1 MiB），避免一次性把大文件读进内存
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    """把 dict 以格式化 JSON 写入文件（自动建父目录）。

    :param path: 目标文件路径。
    :param payload: 要写入的字典内容。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)  # sort_keys 保证输出稳定可 diff
        f.write("\n")  # 文件末尾补一个换行，符合 Unix 文本规范


def command_string(argv: list[str] | None = None) -> str:
    """把命令行参数拼成一个空格分隔的字符串，用于在清单中记录"生成命令"。

    :param argv: 参数列表；为 None 时取当前进程的 sys.argv。
    :return: 拼接后的命令字符串。
    """
    return " ".join(argv or sys.argv)


def source_record(path: pathlib.Path, repo_root: pathlib.Path) -> dict[str, Any]:
    """生成某个源文件在仓库中的"溯源记录"。

    :param path: 源文件路径。
    :param repo_root: 仓库根目录，用于把绝对路径折算成仓库内相对路径。
    :return: 形如 {"path": 相对路径, "sha256": 哈希, "exists": 是否存在} 的字典。
    """
    try:
        # 尽量用相对路径（可移植性好）；若文件不在仓库内则退回绝对路径
        rel = str(path.relative_to(repo_root))
    except ValueError:
        rel = str(path)
    return {
        "path": rel,
        "sha256": sha256_file(path),
        "exists": path.exists(),
    }


def p0_readiness_record(repo_root: pathlib.Path) -> dict[str, Any]:
    """读取 P0 就绪度总结文件，打包成清单里的一节。

    P0 就绪度文件（tables/p0_readiness_summary.json）汇总了 P0 证据门
    （gate）的通过情况。这里把它嵌入每个产物清单，方便读者一眼看到
    当前证据达到了哪个门槛。

    :param repo_root: 仓库根目录。
    :return: 包含文件溯源信息、gate 版本、整体就绪状态的字典。
    """
    path = repo_root / "tables" / "p0_readiness_summary.json"
    record = source_record(path, repo_root)  # 先记录该文件本身是否存在/哈希
    record["overall_g1_ready"] = None   # G1 门（第一道证据门槛）是否整体通过，默认未知
    record["gate_version"] = P0_GATE_VERSION  # 记录当前 P0 门定义的版本号
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            record["overall_g1_ready"] = data.get("overall_g1_ready")
            record["schema_version"] = data.get("schema_version")
        except json.JSONDecodeError:
            # 文件存在但解析失败：至少把 schema 版本标成 None，避免整个清单报错
            record["schema_version"] = None
    return record


def figure_sidecar(
    *,
    output_path: pathlib.Path,
    repo_root: pathlib.Path,
    source_paths: list[pathlib.Path],
    evidence_label: str,
    caption: str,
    render_command: str | None = None,
    notes: list[str] | None = None,
) -> pathlib.Path:
    """为一张生成图（figure）写出配套的 .meta.json 清单文件。

    这是本文件的核心入口。用法示例：画完图后调用
        figure_sidecar(output_path=fig, repo_root=REPO_ROOT,
                       source_paths=[plotting.py, data.csv],
                       evidence_label="modelled", caption="能耗 vs 位宽")
    就会在同目录生成 fig.meta.json，记录图的所有来源。

    :param output_path: 图文件路径（清单文件放在它旁边，追加 .meta.json 后缀）。
    :param repo_root: 仓库根目录，用于把路径折算为相对路径。
    :param source_paths: 生成该图所依赖的源代码/数据文件列表。
    :param evidence_label: 证据强度标签（如 modelled / measured）。
    :param caption: 图的标题/说明文字。
    :param render_command: 生成该图的命令；不传则用当前命令行字符串。
    :param notes: 附加说明列表。
    :return: 生成的清单文件路径。
    """
    sidecar = output_path.with_suffix(output_path.suffix + ".meta.json")
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,  # 清单自身的格式版本
        "gate_version": P0_GATE_VERSION,            # P0 证据门的版本
        "figure": source_record(output_path, repo_root),              # 图本身的信息
        "sources": [source_record(path, repo_root) for path in source_paths],  # 全部依赖来源
        "render_command": render_command or command_string(),  # 记录复现命令
        "evidence_label": evidence_label,
        "evidence_tier": evidence_label,  # 与 evidence_label 冗余存储，方便下游按 tier 过滤
        "p0_readiness": p0_readiness_record(repo_root),  # 附上 P0 就绪度
        "caption": caption,
        "claim_boundary": CLAIM_BOUNDARY_NOTE,          # 声明边界（旧字段名）
        "claim_boundary_note": CLAIM_BOUNDARY_NOTE,     # 声明边界（新字段名）
        "notes": notes or [],
    }
    write_json(sidecar, payload)
    return sidecar


def copy_to(path: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
    """把一个文件原样复制到目标路径（自动建父目录）。

    :param path: 源文件。
    :param target: 目标路径。
    :return: 目标路径。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())  # 按字节复制，保证二进制文件不被破坏
    return target


def maybe_caffeinate_prefix(is_long_run: bool) -> list[str]:
    """为长时间运行的命令自动加 caffeinate 前缀（macOS 防睡眠）。

    caffeinate 是 macOS 自带的命令，可让系统在运行期间不休眠。
    Windows/Linux 上没有该命令，本函数会检测失败并返回空列表，
    因此这段逻辑在两个平台都能安全运行。

    :param is_long_run: 是否为长耗时任务；False 时直接返回空列表。
    :return: 需要加在命令前的参数列表（如 ["caffeinate", "-dimsu"]），否则为空。
    """
    if not is_long_run:
        return []
    try:
        # 探测系统上是否存在 caffeinate 命令（存在则 returncode == 0）
        proc = subprocess.run(["command", "-v", "caffeinate"], text=True, capture_output=True, shell=True)
        if proc.returncode == 0:
            return ["caffeinate", "-dimsu"]  # -d 显示器、-i 系统、-m 硬盘、-s 交流电、-u 用户活跃时
    except Exception:
        pass  # 任何异常都静默降级：不加前缀，不影响主流程
    return []
