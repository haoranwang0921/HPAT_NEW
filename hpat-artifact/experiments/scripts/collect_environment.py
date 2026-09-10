"""采集运行环境信息（collect_environment.py）。

本脚本不跑任何实验，只做"环境体检"：记录当前 Python 环境里装了哪些
关键库（torch/torchvision/timm/numpy/pandas/matplotlib/PIL）、torch 的
MPS 加速（Apple 芯片的 GPU）是否可用、macOS 版本等，写成清单 JSON。
这些信息用于复现时核验"结果是在什么环境下跑出来的"。

- 输入：无（不读数据文件，只探测本机环境）。
- 产出：<输出目录>/environment_manifest.json。
- 命令：python collect_environment.py --output-dir <输出目录>
"""

from __future__ import annotations

import argparse
import json
import subprocess

from _common import base_manifest, ensure_dir, import_status, relative, write_json


def collect(output_dir):
    """探测并记录运行环境信息。

    参数 output_dir：结果输出目录（pathlib.Path）。
    返回：写出的清单文件路径 environment_manifest.json。
    """
    ensure_dir(output_dir)
    # 探测一组关键依赖是否可被 import 及来源位置
    packages = import_status(["torch", "torchvision", "timm", "numpy", "pandas", "matplotlib", "PIL"])
    # 默认状态：torch 不可用（占位，后面若可导入再更新）
    torch_mps = {
        "available": False,
        "is_built": False,
        "reason": "torch is not importable in this Python environment",
    }
    if packages["torch"]["available"]:
        try:
            import torch  # type: ignore

            # MPS 是 Apple Silicon 上的 GPU 后端；available 表示可用，is_built 表示是否编译进 torch
            torch_mps = {
                "available": bool(torch.backends.mps.is_available()),
                "is_built": bool(torch.backends.mps.is_built()),
                "reason": "",
            }
        except Exception as exc:  # pragma: no cover - diagnostic only
            # 导入失败时记录原因，便于排查环境问题
            torch_mps = {"available": False, "is_built": False, "reason": str(exc)}

    # sw_vers 是 macOS 特有的系统版本查询命令；非 mac 平台会走到 except
    sw_vers = None
    try:
        proc = subprocess.run(["sw_vers"], check=False, text=True, capture_output=True)
        sw_vers = proc.stdout.strip()
    except Exception as exc:  # pragma: no cover
        sw_vers = f"sw_vers unavailable: {exc}"

    # 生成基础清单后补充环境专属字段
    manifest = base_manifest("environment_manifest", "smoke / reproducibility")
    manifest.update(
        {
            "packages": packages,
            "torch_mps": torch_mps,
            "macos_sw_vers": sw_vers,
            "caffeinate_used": False,
            "caffeinate_reason": "Not used; environment collection is a short smoke command.",
        }
    )
    out = output_dir / "environment_manifest.json"
    write_json(out, manifest)
    return out


def main() -> None:
    """命令行入口：接收 --output-dir 并调用 collect()。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = collect(__import__("pathlib").Path(args.output_dir))
    print(json.dumps({"wrote": relative(out)}, indent=2))


if __name__ == "__main__":
    main()
