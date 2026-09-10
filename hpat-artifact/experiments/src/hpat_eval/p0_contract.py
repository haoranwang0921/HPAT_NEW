"""P0 契约（contract）：定义 P0 档证据的"门槛、状态机与校验规则"。

背景：论文把证据分成 P0/P1/P2 三档，P0 是最初级的"模拟器/模型推导"
证据档。为了不让作者自说自话，这里用代码强制了"契约"：
- P0_GATE_VERSION：P0 门的版本号（契约改了版本号也要变，防止新旧混淆）；
- P0_EXPERIMENTS：四个 P0 实验（E1 能耗、E2 端侧基线、E3 非理想性精度、
  E4 桌面基线溯源），每个实验有配套的 manifest 文件和状态判定规则；
- 状态机：blocked（被阻塞）→ partial（部分满足）→ ready（就绪）→
  paper_eligible_with_limitations（可入论文但有局限）→ proxy（代理）→ smoke；
- readiness_summary 汇总出"整体 G1 门是否通过"；
- validate_evidence_package 校验证据包的字段是否齐全。

易混淆点：本文件的"契约"是给 P0 用的；P1（真实硬件实测）和
P2（物理代理/仿真）各有自己的管线（p1.py / p2.py），不在此处。
"""

from __future__ import annotations

import json
import pathlib
from typing import Any


# 各 schema/门的版本号：改动格式时必须同步升版本，保证新旧数据可区分
P0_GATE_VERSION = "p0-gate-v2"                     # P0 证据门版本
MANIFEST_SCHEMA_VERSION = "hpat-manifest-v2"       # 清单文件 schema 版本
P0_READINESS_SCHEMA_VERSION = "p0-readiness-v1"    # 就绪度总结 schema 版本
P0_EVIDENCE_PACKAGE_SCHEMA_VERSION = "p0-evidence-package-v1"  # 证据包 schema 版本

# 一个合法的 P0 证据包必须包含的字段（缺一即校验失败）
P0_REQUIRED_PACKAGE_FIELDS = [
    "package_kind",                      # 包的种类
    "schema_version",                    # schema 版本
    "files",                             # 文件清单
    "source_description",                # 来源说明
    "collection_command_or_method",      # 采集命令/方法
    "hashes",                            # 文件哈希（防篡改）
    "claim_scope",                       # 声明的适用范围
    "known_limitations",                 # 已知局限
]

# 四种证据包种类及其对应的实验编号
P0_PACKAGE_KINDS = {
    "hpat-activity": "P0-E1",        # HPAT 模拟器活动数据
    "edge-baseline": "P0-E2",        # 端侧/移动端基线数据
    "nonideality-subset": "P0-E3",   # 非理想性精度的标注子集
    "desktop-provenance": "P0-E4",   # 桌面基线溯源
}

# 每个实验从"就绪"升格为"可写进论文"所需满足的额外要求
# （这里写的是论文署名前的承诺，防止用本地数据冒充端侧实测）
P0_PROMOTION_REQUIREMENTS = {
    "P0-E1": [
        "Provide HPAT simulator-exported activity counts for MobileViT-XXS/XS/S.",
        "Provide complete unit-cost metadata: value, unit, source, technology_assumption, precision, and scope_note.",
        "Keep calibrated energy labelled modelled from simulator activity, not silicon-measured.",
    ],
    "P0-E2": [
        "Provide raw edge/mobile latency samples for MobileViT-XXS/XS/S with device/runtime/precision metadata.",
        "Provide synchronized power logs only if power/energy claims are requested.",
        "Keep edge rows as MobileViT baseline context, not measured HPAT deployment.",
    ],
    "P0-E3": [
        "Provide a fixed labeled validation subset and recorded clean/perturbed logits.",
        "Use HPAT mapping-aligned injection boundary with sufficient coverage.",
        "Do not use all-linear-smoke or unlabeled logit drift for accuracy robustness claims.",
    ],
    "P0-E4": [
        "Keep retained CPU/GPU rows labelled as measured desktop references only; do not use them as mobile/edge evidence.",
        "Local MPS rows support desktop reference context only.",
    ],
}

# 四个 P0 实验的定义：各自要检查的 manifest 文件名、状态集合与默认阻塞原因
P0_EXPERIMENTS = [
    {
        "experiment": "P0-E1",  # 实验编号
        "manifest": "hpat_energy_manifest.json",  # 对应的清单文件名
        "meaning": "HPAT simulator activity and unit-cost gate",  # 含义：HPAT 模拟器活动+单位成本门
        "ready_statuses": {"ready"},
        "claim_eligible_statuses": {"ready"},        # 可入论文的状态集合
        "proxy_statuses": {"proxy"},                 # 代理状态集合
        "default_blockers": ["No HPAT simulator activity CSV and complete unit-cost metadata were provided."],
    },
    {
        "experiment": "P0-E2",
        "manifest": "edge_baseline_import_manifest.json",
        "meaning": "Edge/mobile baseline import gate",  # 端侧/移动端基线导入门
        "ready_statuses": {"ready"},
        "claim_eligible_statuses": {"ready"},
        "proxy_statuses": set(),
        "default_blockers": ["No external edge/mobile baseline package was provided."],
    },
    {
        "experiment": "P0-E3",
        "manifest": "nonideality_accuracy_sweep_manifest.json",
        "meaning": "Accuracy-coupled non-ideality gate",  # 与精度挂钩的非理想性门
        "ready_statuses": {"ready", "paper_eligible_with_limitations"},
        "claim_eligible_statuses": {"ready", "paper_eligible_with_limitations"},
        "proxy_statuses": {"proxy", "smoke"},
        "default_blockers": ["No fixed labeled validation subset and HPAT mapping-aligned logits were provided."],
    },
    {
        "experiment": "P0-E4",
        "manifest": "baseline_provenance_manifest.json",
        "meaning": "Desktop baseline provenance gate",  # 桌面基线溯源门
        "ready_statuses": {"ready"},
        "claim_eligible_statuses": {"ready"},
        "proxy_statuses": set(),
        "default_blockers": ["No desktop baseline provenance manifest was available."],
    },
]


def normalize_status(status: Any) -> str:
    """把各种写法（ok/OK/空/大小写混杂）归一成标准状态字符串。

    特例：写 "ok" 的归一成 "ready"（两者等价）。

    :param status: 原始状态值。
    :return: 归一化后的小写状态字符串；缺失时视为 "blocked"。
    """
    text = str(status or "blocked").strip().lower()
    return "ready" if text == "ok" else text


def evidence_tier_for_status(experiment: str, status: str) -> str:
    """把状态字符串翻译成人类可读的"证据等级"标签。

    :param experiment: 实验编号（P0-E1~E4）。
    :param status: 归一化后的状态。
    :return: 证据等级描述（如 "claim-ready bounded evidence"）。
    """
    status = normalize_status(status)
    if experiment == "P0-E4" and status == "ready":
        return "measured desktop reference"  # E4 就绪 = 已实测的桌面参考数据
    if status == "ready":
        return "claim-ready bounded evidence"  # 就绪 = 可支撑有边界的结论
    if status == "paper_eligible_with_limitations":
        return "local/modelled diagnostic; paper-eligible with limitations"  # 有局限但可入论文
    if status == "partial":
        return "partial"
    if status == "proxy":
        return "local/modelled proxy"  # 代理证据
    if status == "smoke":
        return "smoke"  # 冒烟测试
    return "not claimable"  # 不可支撑结论


def manifest_status_for_experiment(experiment: str, manifest: dict[str, Any]) -> str:
    """根据 manifest 内容，把每个实验的"表面状态"校准成"真实状态"。

    原因：manifest 可能自称 ready，但内容不达标（如 E1 的活动数据不是
    模拟器导出的、E3 的注入边界不对）。本函数逐实验检查关键字段，
    不达标就降级到 proxy / partial / blocked。

    :param experiment: 实验编号。
    :param manifest: 读出的 manifest 字典。
    :return: 校准后的状态。
    """
    status = normalize_status(manifest.get("status", "blocked"))
    if experiment == "P0-E1" and status == "ready":
        # E1 必须要求：活动数据来自模拟器导出 + 单位成本元信息齐全
        if manifest.get("activity_source") != "simulator_export":
            return "proxy"
        if manifest.get("unit_cost_status") != "complete_metadata_claim_eligible":
            return "blocked"
    if experiment == "P0-E2" and status == "ready":
        # E2 必须要求：基线覆盖满足"可声明"条件
        coverage = manifest.get("coverage", {})
        if isinstance(coverage, dict) and not coverage.get("claim_eligible", False):
            return "partial"
    if experiment == "P0-E3" and status == "ready":
        # E3 必须要求：注入边界是 hpat-mapping 且标注子集非空
        dataset = manifest.get("dataset", {})
        label_count = 0
        if isinstance(dataset, dict):
            try:
                label_count = int(dataset.get("label_count") or 0)
            except (TypeError, ValueError):
                label_count = 0
        if manifest.get("injection_boundary") != "hpat-mapping" or label_count <= 0:
            return "partial"
    if experiment == "P0-E3" and status == "paper_eligible_with_limitations":
        # 带局限入论文时，边界可放宽到 reasonable-max，但标注子集仍必须非空
        dataset = manifest.get("dataset", {})
        try:
            label_count = int(dataset.get("label_count") or 0) if isinstance(dataset, dict) else 0
        except (TypeError, ValueError):
            label_count = 0
        if manifest.get("injection_boundary") not in {"hpat-mapping", "reasonable-max"} or label_count <= 0:
            return "partial"
    return status


def read_manifest(path: pathlib.Path) -> dict[str, Any]:
    """读取一个 manifest JSON 文件；不存在或损坏时返回空字典。

    返回空字典的好处：调用方统一走"manifest 不存在 → blocked"的路径，
    不需要到处判断文件是否存在。

    :param path: manifest 文件路径。
    :return: 解析出的字典；失败时 {}。
    """
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}  # JSON 损坏时按"无此文件"处理


def _blockers(spec: dict[str, Any], manifest: dict[str, Any], status: str) -> list[str]:
    """为每个实验生成"阻塞原因"列表。

    优先级：ready 类状态无阻塞 → manifest 里写了 blocked_reason 就用它 →
    是 partial 就给通用解释 → 否则用实验的默认阻塞原因。

    :param spec: P0_EXPERIMENTS 里的一个实验定义。
    :param manifest: 该实验的 manifest。
    :param status: 校准后的状态。
    :return: 阻塞原因字符串列表。
    """
    if status in {"ready", "paper_eligible_with_limitations"}:
        return []  # 就绪 = 无阻塞
    explicit = str(manifest.get("blocked_reason", "")).strip()
    if explicit:
        return [explicit]
    if status == "partial":
        return ["Evidence package is present but coverage or provenance is incomplete."]
    return list(spec["default_blockers"])


def readiness_rows(run_dir: pathlib.Path) -> list[dict[str, Any]]:
    """逐个检查四个 P0 实验的就绪状态，返回一行一个实验的状态表。

    :param run_dir: 存放 manifest 文件的运行目录。
    :return: 状态表（含 status / claim_eligible / evidence_tier / blockers 等）。
    """
    rows: list[dict[str, Any]] = []
    for spec in P0_EXPERIMENTS:
        manifest_path = run_dir / spec["manifest"]
        manifest = read_manifest(manifest_path)
        experiment = str(spec["experiment"])
        # 没有 manifest 直接判 blocked；有则按内容校准状态
        status = "blocked" if not manifest else manifest_status_for_experiment(experiment, manifest)
        claim_eligible = status in spec["claim_eligible_statuses"]  # 是否具备"入论文"资格
        rows.append(
            {
                "experiment": experiment,
                "status": status,
                "claim_eligible": claim_eligible,
                "evidence_tier": evidence_tier_for_status(experiment, status),
                "meaning": spec["meaning"],
                "blockers": _blockers(spec, manifest, status),
                "source_manifest": str(manifest_path.name),
                "promotion_requirements": P0_PROMOTION_REQUIREMENTS[experiment],
            }
        )
    return rows


def readiness_summary(run_dir: pathlib.Path) -> dict[str, Any]:
    """生成"P0 就绪度总结"，核心是 overall_g1_ready 这个总开关。

    :param run_dir: 存放 manifest 文件的运行目录。
    :return: 就绪度总结字典（含 schema/gate 版本与逐实验明细）。
    """
    rows = readiness_rows(run_dir)
    return {
        "schema_version": P0_READINESS_SCHEMA_VERSION,
        "gate_version": P0_GATE_VERSION,
        "overall_g1_ready": all(bool(row["claim_eligible"]) for row in rows),  # 全部可入论文才算 G1 通过
        "experiments": rows,
    }


def validate_evidence_package(kind: str, manifest: dict[str, Any]) -> tuple[str, list[str]]:
    """校验一个"证据包"manifest 是否合法完整。

    :param kind: 证据包种类（须在 P0_PACKAGE_KINDS 中）。
    :param manifest: 证据包 manifest 字典。
    :return: (状态 "ok"/"blocked", 错误原因列表)。
    """
    errors: list[str] = []
    if kind not in P0_PACKAGE_KINDS:
        errors.append(f"unknown package kind: {kind}")
    if manifest.get("package_kind") != kind:
        errors.append(f"package_kind must be {kind}")
    # 逐字段检查必填字段是否为空（None / "" / [] / {} 都算缺失）
    for field in P0_REQUIRED_PACKAGE_FIELDS:
        value = manifest.get(field)
        if value is None or value == "" or value == [] or value == {}:
            errors.append(f"missing required field: {field}")
    if manifest.get("schema_version") != P0_EVIDENCE_PACKAGE_SCHEMA_VERSION:
        errors.append(f"schema_version must be {P0_EVIDENCE_PACKAGE_SCHEMA_VERSION}")
    files = manifest.get("files")
    if files is not None and not isinstance(files, list):
        errors.append("files must be a list")
    hashes = manifest.get("hashes")
    if hashes is not None and not isinstance(hashes, dict):
        errors.append("hashes must be an object")
    return ("ok" if not errors else "blocked", errors)
