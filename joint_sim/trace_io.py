"""联合仿真 Trace 的严格读写工具。

Trace 使用 JSONL：第一行是导出说明（manifest），其余每行才是一条算子记录。
统一在这里校验，可避免不同实验对同一份 trace 作出不同解释。
"""

import hashlib
import json
from pathlib import Path
from typing import List, Tuple


def sha256_file(path: Path) -> str:
    # 分块读取（每次 1 MB）计算 SHA-256，避免把超大文件一次性读进内存。
    """计算文件指纹，用来确认 trace 或 checkpoint 没有被悄悄替换。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest_jsonl(path: Path) -> Tuple[dict, List[dict]]:
    """读取并校验带 manifest 的算子 trace。

    Validation is deliberately performed at the shared input boundary so
    every experiment uses identical dimensions and DAG semantics.
    """
    # 逐行把 JSON 读成对象（跳过空行）。第一行必须是 manifest（说明），
    # 其余每一行是一条算子记录。把校验集中在这里，可保证所有实验
    # 对同一份 trace 的解释完全一致（避免各实验自行解析产生分歧）。
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        objects = [json.loads(line) for line in handle if line.strip()]
    if len(objects) < 2:
        raise ValueError(f"Trace must contain a manifest and operator records: {path}")
    manifest, records = objects[0], objects[1:]
    if not isinstance(manifest, dict) or not isinstance(records, list):
        raise ValueError(f"Malformed trace container: {path}")
    # manifest 里声明了算子总数（total_operators），逐条核对是否一致
    declared = manifest.get("total_operators")
    if declared is not None and declared != len(records):
        raise ValueError(
            f"Trace manifest says {declared} operators, but file contains {len(records)}"
        )

    # 每条记录必须含这 4 个基础字段，且 op_id/order 全局唯一，
    # dependencies 里的每个依赖都必须指向更早（order 更小）的算子。
    # 这些约束保证了 DAG（有向无环图）语义在文件层面就是合法的。
    required = {"op_id", "order", "op_type", "dependencies"}
    op_by_id = {}
    order_owner = {}
    for index, rec in enumerate(records, start=2):
        if not isinstance(rec, dict):
            raise ValueError(f"Trace line {index} is not a JSON object")
        missing = sorted(required - rec.keys())
        if missing:
            raise ValueError(f"Trace line {index} lacks required fields: {missing}")
        op_id = rec["op_id"]
        if not isinstance(op_id, str) or not op_id:
            raise ValueError(f"Trace line {index} has an invalid op_id")
        if op_id in op_by_id:
            raise ValueError(f"Duplicate op_id in trace: {op_id}")
        order = rec["order"]
        if not isinstance(order, int) or order < 0:
            raise ValueError(f"Operator {op_id} has invalid order: {order!r}")
        if order in order_owner:
            raise ValueError(
                f"Duplicate order {order} for {order_owner[order]} and {op_id}"
            )
        deps = rec["dependencies"]
        if not isinstance(deps, (list, tuple)) or any(
            not isinstance(dep, str) for dep in deps
        ):
            raise ValueError(f"Operator {op_id} has invalid dependencies")
        if len(deps) != len(set(deps)):
            raise ValueError(f"Operator {op_id} has duplicate dependencies")

        # 非 Phase 算子必须给出正的 M/K/N 和位宽；Phase 只是阶段标记，
        # 不代表具体计算，因此跳过这组检查。
        if rec["op_type"] != "Phase":
            for field in ("M", "K", "N", "input_bits", "output_bits"):
                value = rec.get(field)
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(
                        f"Operator {op_id} has invalid {field}: {value!r}"
                    )
            # batch_repetitions：一次调用里批矩阵乘被重复执行的次数
            repetitions = rec.get("batch_repetitions", 1)
            if not isinstance(repetitions, int) or repetitions <= 0:
                raise ValueError(
                    f"Operator {op_id} has invalid batch_repetitions: {repetitions!r}"
                )
        # 固定权重（weight_static）的 Linear 算子必须有 weight_id（权重的
        # 稳定哈希）和正的 weight_bits，否则无法在光子阵列中定位/量化权重。
        if rec.get("op_type") == "Linear" and rec.get("weight_static"):
            if not rec.get("weight_id"):
                raise ValueError(f"Static Linear operator {op_id} lacks weight_id")
            if not isinstance(rec.get("weight_bits"), (int, float)) or rec["weight_bits"] <= 0:
                raise ValueError(f"Static Linear operator {op_id} has invalid weight_bits")

        op_by_id[op_id] = rec
        order_owner[order] = op_id

    # 第二轮检查：所有依赖都必须指向已存在的算子，且被依赖者必须
    # 比依赖者更早出现（order 更小），否则 DAG 里会出现"指向未来"的边。
    for op_id, rec in op_by_id.items():
        for dep in rec["dependencies"]:
            if dep not in op_by_id:
                raise ValueError(f"Operator {op_id} depends on unknown operator {dep}")
            if op_by_id[dep]["order"] >= rec["order"]:
                raise ValueError(
                    f"Operator {op_id} has non-preceding dependency {dep}"
                )
    return manifest, records
