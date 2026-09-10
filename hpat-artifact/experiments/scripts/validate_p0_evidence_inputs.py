"""校验 P0 档证据的输入文件是否合法、完整（P0：证据等级中最严谨的一档）。

本脚本做什么：
    读取一份"证据包 manifest"（JSON 清单，声明了引用的文件列表及其 SHA256 摘要），
    依次做三件事：
      1) JSON 本身能否解析；
      2) 交给 hpat_eval.p0_contract.validate_evidence_package 校验包结构与
         各契约约束（schema 版本、gate 版本、包种类等）；
      3) 逐文件核对：清单里声明的每个文件是否真实存在、其实际 SHA256 是否
         与清单里记录的摘要一致（防止内容被改动或缺失）。
    任一环节失败，最终 status 置为 blocked（阻塞，不满足证据要求）。

数据从哪来：
    --input-manifest 指定的 JSON 文件（P0 证据包清单），加上清单内引用
    的各个数据文件。

产出什么：
    --output-dir/p0_evidence_input_validation_manifest.json，包含状态
    （ok/blocked）、错误列表、输入文件摘要、claim_eligible（是否可写进论文
    结论）等，供上游实验脚本决定是否放行。

怎么运行：
    python experiments/scripts/validate_p0_evidence_inputs.py \
        --kind <包种类> --input-manifest <JSON路径> --output-dir <输出目录>
    注意：本脚本本身不联网，可以安全本地运行。
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from _common import base_manifest, ensure_dir, load_json, relative, sha256_file, write_json
from hpat_eval.p0_contract import (
    P0_EVIDENCE_PACKAGE_SCHEMA_VERSION,
    P0_GATE_VERSION,
    P0_PACKAGE_KINDS,
    P0_PROMOTION_REQUIREMENTS,
    validate_evidence_package,
)


def _resolve_package_path(manifest_path: pathlib.Path, file_ref: str) -> pathlib.Path:
    """把清单里引用的相对文件路径解析为绝对路径。

    参数：
        manifest_path: 清单 JSON 文件本身所在的路径。
        file_ref: 清单里 files 字段中的文件引用（相对或绝对路径）。
    返回：
        pathlib.Path：绝对路径；绝对引用原样返回，相对引用以清单所在目录为基准。
    """
    candidate = pathlib.Path(file_ref)
    if candidate.is_absolute():
        return candidate
    # 相对路径按"清单文件所在目录"解析，保证无论从哪运行都能找到文件
    return manifest_path.parent / candidate


def _file_errors(manifest_path: pathlib.Path, payload: dict[str, Any]) -> list[str]:
    """校验清单中所有被引用文件的存在性与 SHA256 一致性。

    参数：
        manifest_path: 清单 JSON 文件路径（用于解析相对引用）。
        payload: 已解析的清单字典，需含 files（文件引用列表）与 hashes
            （引用 -> 期望 SHA256 摘要）。
    返回：
        list[str]：收集到的错误描述列表；没问题则为空列表。
    """
    errors: list[str] = []
    files = payload.get("files", [])
    hashes = payload.get("hashes", {})
    # 字段类型不对（不是列表/字典）就跳过文件级检查，让契约层去报告结构问题
    if not isinstance(files, list) or not isinstance(hashes, dict):
        return errors
    for file_ref in files:
        file_key = str(file_ref)
        path = _resolve_package_path(manifest_path, file_key)
        # 文件不存在 = 证据包不完整
        if not path.exists():
            errors.append(f"referenced file does not exist: {file_key}")
            continue
        expected_hash = str(hashes.get(file_key, "")).strip()
        actual_hash = sha256_file(path)
        # 清单没给哈希 = 无法校验完整性；哈希对不上 = 文件内容被动过
        if not expected_hash:
            errors.append(f"missing hash for file: {file_key}")
        elif actual_hash != expected_hash:
            errors.append(f"sha256 mismatch for file: {file_key}")
    return errors


def run(kind: str, input_manifest: pathlib.Path, output_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """执行一次 P0 证据输入校验，并写出校验 manifest。

    参数：
        kind: P0 包种类（必须是 P0_PACKAGE_KINDS 的键之一）。
        input_manifest: 待校验的证据包清单 JSON 路径。
        output_dir: 校验结果 manifest 的输出目录。
    返回：
        dict：{"manifest": 校验 manifest 的路径}。
    """
    ensure_dir(output_dir)
    manifest_path = output_dir / "p0_evidence_input_validation_manifest.json"
    # base_manifest 生成带基本元数据（名称、描述、时间戳等）的清单骨架
    validation = base_manifest("p0_evidence_input_validation", "P0 external evidence package validation")
    if not input_manifest.exists():
        # 输入清单文件本身就不存在：直接判 blocked
        status = "blocked"
        errors = [f"input manifest does not exist: {input_manifest}"]
        payload: dict[str, Any] = {}
    else:
        try:
            payload = load_json(input_manifest)
            # 第一道闸：契约层校验结构/约束；第二道闸：文件存在性与哈希核对
            status, errors = validate_evidence_package(kind, payload)
            errors.extend(_file_errors(input_manifest, payload))
            status = "ok" if not errors else "blocked"
        except json.JSONDecodeError as exc:
            # 清单不是合法 JSON：同样判 blocked，并保留解析错误信息
            payload = {}
            status = "blocked"
            errors = [f"input manifest is not valid JSON: {exc}"]

    experiment = P0_PACKAGE_KINDS.get(kind, "")
    validation.update(
        {
            "status": status,
            "kind": kind,
            "expected_schema_version": P0_EVIDENCE_PACKAGE_SCHEMA_VERSION,
            "gate_version": P0_GATE_VERSION,
            "input_manifest": relative(input_manifest),
            # 记录输入清单自身的 SHA256，便于追踪用的是哪一版
            "input_manifest_sha256": sha256_file(input_manifest) if input_manifest.exists() else None,
            "experiment": experiment,
            # 只有 ok 状态才允许进入论文结论（claim）
            "claim_eligible": status == "ok",
            "errors": errors,
            "blocked_reason": "; ".join(errors) if errors else "",
            "package_claim_scope": payload.get("claim_scope", ""),
            "promotion_requirements": P0_PROMOTION_REQUIREMENTS.get(experiment, []),
            "outputs": [relative(manifest_path)],
            "promotion_note": (
                "Validation only checks package completeness and hashes. Downstream experiment scripts still "
                "must enforce their own schema and claim gates before producing paper-facing evidence."
            ),
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; this is a short evidence-package validation.",
        }
    )
    write_json(manifest_path, validation)
    return {"manifest": manifest_path}


def main() -> None:
    """命令行入口：解析 --kind/--input-manifest/--output-dir 并执行校验。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=sorted(P0_PACKAGE_KINDS))
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    outputs = run(args.kind, pathlib.Path(args.input_manifest), pathlib.Path(args.output_dir))
    # 以相对路径形式打印产出，便于阅读
    print(json.dumps({k: relative(v) for k, v in outputs.items()}, indent=2))


if __name__ == "__main__":
    # 直接执行本文件时调用命令行入口
    main()
