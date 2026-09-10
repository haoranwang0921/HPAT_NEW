"""在线校验外部边缘上下文数据（会联网访问外部数据源！）。

本脚本做什么：
    论文会把一些"外部边缘/移动场景"的公开数据（文献或官方 benchmark）作为
    背景上下文引用。本脚本逐个去访问这些外部链接，确认它们现在还活着
    （HTTP 状态是否正常）、内容是否有变化（取响应前若干字节算 SHA256 摘要
    记录在案，供日后比对），从而防止论文引用的是已失效或被改动的数据。

数据从哪来：
    --source-csv 指定的 CSV（默认仓库 tables/external_edge_context_verified.csv），
    每行一个外部来源，含 source_url 与来源描述字段。

产出什么：
    - 本轮输出目录 tables/external_edge_context_online_verification.csv；
    - 同时把同一份表写到仓库 tables/ 下；
    - 以及一份校验 manifest，含状态、警告数、claim_boundary（结论边界）。
    注意：所有数据一律只当"公开背景上下文"使用，不参与计算 HPAT 加速比。

怎么运行：
    python experiments/scripts/verify_external_edge_context_online.py \
        --output-dir <输出目录> [--source-csv <CSV>] [--timeout 20] [--max-bytes 262144]
    警告：本脚本会联网，运行前请确认网络可用；只做语法检查时勿直接运行。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import urllib.error
import urllib.request
from typing import Any

from _common import REPO_ROOT, base_manifest, ensure_dir, read_csv, relative, sha256_file, utc_now, write_csv, write_json


# 结论边界：这些只是公开背景信息，不是作者实测的 HPAT 边缘/移动基线，
# 不是实测加速比，也不是硅片级验证——防止下游误用为实测证据
CLAIM_BOUNDARY = (
    "External edge/mobile context is verified public context only. It is not an author-measured "
    "HPAT edge/mobile baseline, not measured HPAT speedup, and not silicon validation."
)

# 校验结果表固定列：前半段是抓取结果（HTTP 状态/最终 URL/摘要），
# 后半段沿用来源 CSV 的描述字段与统一的安全使用说明
FIELDS = [
    "source_id",
    "source_url",
    "http_status",
    "final_url",
    "retrieved_utc",
    "content_type",
    "content_length_header",
    "hash_scope",
    "content_sha256",
    "verification_status",
    "error",
    "source_type",
    "metric_scope",
    "safe_use",
    "must_not_imply",
    "external_data_label",
    "claim_boundary",
]


def _fetch(url: str, timeout: int, max_bytes: int) -> dict[str, Any]:
    """抓取一个 URL 并提取校验所需的信息（仅元数据，不解析正文）。

    参数：
        url: 待访问的外部来源链接。
        timeout: 请求超时秒数。
        max_bytes: 最多读取的响应字节数（只看前 N 字节算摘要，省流量）。
    返回：
        dict：包含 HTTP 状态、最终 URL、content-type、content-length、
        摘要范围说明、内容 SHA256、校验状态、错误信息。
    """
    request = urllib.request.Request(
        url,
        headers={
            # 显式声明自己是公开来源元数据核验工具，避免被当作异常爬虫
            "User-Agent": "HPAT evidence-context-verifier/1.0 (public-source metadata only)",
            "Accept": "*/*",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(getattr(response, "status", 0) or response.getcode())
            final_url = response.geturl()  # 跟随重定向后的最终地址
            headers = response.headers
            # 只读前 max_bytes 字节：既可校验内容一致性，又不下载整个文件
            data = response.read(max_bytes)
            return {
                "http_status": status,
                "final_url": final_url,
                "content_type": headers.get("content-type", ""),
                "content_length_header": headers.get("content-length", ""),
                # 摘要范围说明：明确是"前 N 字节"的摘要，避免将来误以为全量摘要
                "hash_scope": f"first_{len(data)}_bytes_max_{max_bytes}",
                "content_sha256": hashlib.sha256(data).hexdigest(),
                # 2xx/3xx 且有内容才算在线验证通过，否则算"取到了但状态/内容异常"
                "verification_status": "online_verified" if 200 <= status < 400 and data else "online_retrieved_non_ok_or_empty",
                "error": "",
            }
    except urllib.error.HTTPError as exc:
        # HTTP 层错误（404/403 等）：把错误页内容也摘录下来便于归档
        payload = exc.read(max_bytes) if exc.fp else b""
        return {
            "http_status": int(exc.code),
            "final_url": exc.geturl(),
            "content_type": exc.headers.get("content-type", "") if exc.headers else "",
            "content_length_header": exc.headers.get("content-length", "") if exc.headers else "",
            "hash_scope": f"first_{len(payload)}_bytes_max_{max_bytes}",
            "content_sha256": hashlib.sha256(payload).hexdigest() if payload else "",
            "verification_status": "http_error",
            "error": str(exc),
        }
    except Exception as exc:  # pragma: no cover - network dependent
        # 网络不可达/超时等其他异常：归为 retrieval_failed，不带任何内容
        return {
            "http_status": "",
            "final_url": "",
            "content_type": "",
            "content_length_header": "",
            "hash_scope": f"first_0_bytes_max_{max_bytes}",
            "content_sha256": "",
            "verification_status": "retrieval_failed",
            "error": repr(exc),
        }


def run(output_dir: pathlib.Path, source_csv: pathlib.Path, timeout: int, max_bytes: int) -> dict[str, pathlib.Path]:
    """对来源清单里的每个外部链接做一次在线校验并落盘结果。

    参数：
        output_dir: 本轮校验产物输出目录。
        source_csv: 来源 CSV（必须含 source_url 列）。
        timeout: 每个请求的超时秒数。
        max_bytes: 每个请求最多读取的响应字节数。
    返回：
        dict：{"csv", "project_csv", "manifest"} 三个产出的路径。
    """
    ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    rows = read_csv(source_csv)
    out_rows: list[dict[str, Any]] = []
    # 逐个来源抓取并拼装结果行（在线部分只在这里发生）
    for row in rows:
        result = _fetch(row["source_url"], timeout=timeout, max_bytes=max_bytes)
        out_rows.append(
            {
                "source_id": row.get("source_id", ""),
                "source_url": row.get("source_url", ""),
                "http_status": result["http_status"],
                "final_url": result["final_url"],
                "retrieved_utc": utc_now(),  # 记录抓取时刻（UTC），便于追踪时效
                "content_type": result["content_type"],
                "content_length_header": result["content_length_header"],
                "hash_scope": result["hash_scope"],
                "content_sha256": result["content_sha256"],
                "verification_status": result["verification_status"],
                "error": result["error"],
                "source_type": row.get("source_type", ""),
                "metric_scope": row.get("metric_scope", ""),
                "safe_use": row.get("safe_use", ""),
                "must_not_imply": row.get("must_not_imply", ""),
                "external_data_label": row.get("external_data_label", "external_public_context; not author-measured HPAT baseline"),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )

    csv_path = tables_dir / "external_edge_context_online_verification.csv"
    # 本轮产出 + 仓库正式表各写一份，保证仓库版本与本次核验结果一致
    project_csv = REPO_ROOT / "tables" / "external_edge_context_online_verification.csv"
    write_csv(csv_path, out_rows, FIELDS)
    write_csv(project_csv, out_rows, FIELDS)
    # 统计没有"在线验证通过"的行作为警告数
    failures = [row for row in out_rows if row["verification_status"] != "online_verified"]
    manifest_path = output_dir / "external_edge_context_online_verification_manifest.json"
    manifest = base_manifest("external_edge_context_online_verification", "online verification for external edge/mobile context")
    manifest.update(
        {
            # 有失败行时状态降级为"带警告验证通过"
            "status": "context_verified_with_warnings" if failures else "context_verified",
            "source_csv": relative(source_csv),
            "source_csv_sha256": sha256_file(source_csv),
            "outputs": [relative(csv_path), relative(project_csv)],
            "row_count": len(out_rows),
            "warning_count": len(failures),
            "timeout_seconds": timeout,
            "max_bytes_per_source": max_bytes,
            "claim_boundary": CLAIM_BOUNDARY,
            "promotion_note": "Use only as external motivation/related context. Do not compute HPAT speedup from these rows.",
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; online context verification is short and metadata-only.",
        }
    )
    write_json(manifest_path, manifest)
    return {"csv": csv_path, "project_csv": project_csv, "manifest": manifest_path}


def main() -> None:
    """命令行入口：解析参数并执行在线校验。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-csv", default=str(REPO_ROOT / "tables" / "external_edge_context_verified.csv"))
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--max-bytes", type=int, default=262144)
    parser.add_argument("--config", default=None, help="Accepted for run_all compatibility; unused.")
    args = parser.parse_args()
    outputs = run(pathlib.Path(args.output_dir), pathlib.Path(args.source_csv), args.timeout, args.max_bytes)
    # 以相对路径形式打印产出
    print(json.dumps({key: relative(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    # 直接执行本文件时调用命令行入口（会触发联网抓取）
    main()
