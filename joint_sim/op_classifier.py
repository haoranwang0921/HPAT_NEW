"""
决定算子走光子还是电子路径。

当前规则刻意保守：只有“权重固定”的 Linear 才进入 MRR 阵列；QKᵀ、AV 等
两个输入都在运行时变化的矩阵乘，以及非线性算子，都留在电子侧。

Rules (Plan Section 8 D5):
  - eligible=True  → static_weight_linear
  - eligible=False → both_operands_dynamic / nonlinear / control_flow
"""

from typing import Dict, List


# Operator types that can be mapped to the photonic core
# 有资格上光子阵列的算子类型：目前只有 Linear（全连接层）。因为 MRR 阵列
# 做的是"权重固定、输入变化"的矩阵乘法，其他算子类型一律走电子后端。
PHOTONIC_ELIGIBLE_OPS = {"Linear"}

# Operator roles that use static (pre-trained) weights
# 使用固定（预训练后不变）权重的角色集合。Q/K/V/O 是注意力四个投影，
# FFN_up/FFN_down 是前馈网络，c_proj 是上下文投影。这些角色权重固定，
# 适合长期驻留在 MRR 上复用。
STATIC_WEIGHT_ROLES = {
    "Q", "K", "V", "O",              # attention projections
    "FFN_up", "FFN_down",            # feed-forward projections
    "c_proj", "c_proj_up",          # context projections
    "proj",                           # generic projection (STN, CNN heads)
}

# Dynamic operators: both operands change each call
# 动态算子：两个输入（例如 QKᵀ 里的 Q 和 K）每次调用都在变，权重无法
# 预先驻留到 MRR，因此不能走光子路径。
DYNAMIC_OPS = {"MatMul", "BatchedMatMul", "QMatMul"}

# Non-linear / control-flow operators
# 非线性算子与控制流算子：Softmax/LayerNorm/GELU 等在光子阵列上难以
# 高效实现，统一交给电子后端处理。
NONLINEAR_OPS = {
    "LayerNorm", "RMSNorm", "Softmax", "GELU", "ReLU",
    "Dropout", "Conv2d", "ConvTranspose2d", "Upsample",
    "Sample", "Argmax", "TopK", "Reshape", "Permute",
    "Add", "Mul", "Concat", "Split", "Identity",
}


def classify_operator(op_type: str, op_role: str, weight_static: bool) -> dict:
    """判断单个算子是否可走光子路径。

    All fixed-weight Linear layers are photonic candidates (Plan Section 1):
    "第一版只将固定权重的Q/K/V/O、FFN、c_proj和其他固定Linear映射到MRR"

    Returns:
        dict with keys: eligible (bool), reason (str)
    """
    # 判断顺序（从上到下）：
    # 1) 动态矩阵乘（两个操作数都在变）→ 拒绝
    if op_type in DYNAMIC_OPS:
        return {"eligible": False, "reason": "both_operands_dynamic"}

    # 2) 非线性/控制流算子 → 拒绝
    if op_type in NONLINEAR_OPS:
        return {"eligible": False, "reason": "nonlinear"}

    # 3) 固定权重的 Linear → 通过；known_static_role 标记其角色是否在
    #    已知的固定权重角色表里（便于审计，不代表可否执行）。
    if op_type == "Linear" and weight_static:
        return {
            "eligible": True,
            "reason": "static_weight_linear",
            "known_static_role": op_role in STATIC_WEIGHT_ROLES,
        }

    # 4) 权重会动态变化的 Linear → 拒绝（权重不能驻留，光路没有优势）
    if op_type == "Linear" and not weight_static:
        return {"eligible": False, "reason": "dynamic_weight"}

    # 5) 其他未知类型 → 保守拒绝，reason 里带上类型名便于排查
    return {"eligible": False, "reason": f"unclassified:{op_type}"}


def classify_trace(
    operators: List[dict],
) -> List[dict]:
    """Classify all operators in a trace.

    Each operator dict must have: op_type, op_role, weight_static.

    Returns a list of classification results with op_id, eligible, reason.
    """
    # 对整个 trace 逐条分类，并在结果里附上 op_id，方便下游按算子核对。
    results = []
    for op in operators:
        result = classify_operator(
            op_type=op.get("op_type", ""),
            op_role=op.get("op_role", "unknown"),
            weight_static=op.get("weight_static", False),
        )
        result["op_id"] = op.get("op_id", "unknown")
        results.append(result)
    return results


def build_weight_id(module_path: str, weight_shape: tuple) -> str:
    """Generate a stable weight_id from module path and weight shape.

    Uses SHA-256; never uses Python object addresses (Plan Section 4.1).
    """
    import hashlib
    # 用"模块路径 + 权重形状"做 SHA-256 前 16 位，保证同一个权重在
    # 不同运行、不同机器上都得到相同的 id（可复现）。
    raw = f"{module_path}|{weight_shape}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
