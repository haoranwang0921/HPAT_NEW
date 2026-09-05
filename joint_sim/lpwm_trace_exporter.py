"""
LPWM operator trace exporter (Phase D).

Hooks into an LPWM model's forward pass to record every operator
invocation as an OperatorRecord. Exports a versioned JSON-lines trace
that can be replayed by LLMCompass + SimPhony.

Usage (requires LPWM model code):
    from lpwm_trace_exporter import TraceExporter
    exporter = TraceExporter(
        model=lpwm_model,
        trace_version="1.0.0",
        cond_steps=1,
        horizon=16,
    )
    exporter.install_hooks()
    output = lpwm_model(input_video, actions)
    exporter.save("trace.jsonl")

Design (Plan Section 8):
  - All nn.Linear layers get a forward hook.
  - MatMul (QK^T, AV) get record_function wrappers.
  - Non-linear ops (Softmax, LayerNorm, GELU, etc.) are recorded.
  - Every record has a stable weight_id, rollout step, phase, and role.

中文阅读提示：这是“模型世界”到“硬件世界”的入口。它只观察并记录真实执行，
不做任何光子/电子成本判断；分类和调度在 joint_sim 的其他文件完成。
"""
# =============================================================================
# 本文件角色一句话：把 LPWM 模型的一次真实前向传播，录成算子执行轨迹(trace)。
# 做法：给模型里所有 nn.Linear 层挂 forward hook（前向钩子），每次调用时把
# 该算子的类型/形状/权重信息记成一条 OperatorRecord，最后导出 JSONL 文件。
# 注意：它只做"记录"，不判断算子是走光子还是电子后端（那是 op_classifier
# 的事），也不算成本（那是 scheduler + 两个 backend 的事）。
# 依赖 torch；若环境里没有 torch，模块仍可被 import（HAS_TORCH=False），
# 但 TraceExporter 会被禁用（schema 等基础能力不受影响）。
# =============================================================================

import json
import hashlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# Try torch import — will fail if torch not installed, which is fine
# for environments where only the schema is needed.
# torch 不是必需依赖：只在需要挂 hook 导出轨迹时才要它
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Trace schema version
# ---------------------------------------------------------------------------
# trace 文件的格式版本号；版本变化时旧文件不能被新代码直接复用
TRACE_SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Operator role inference from module path
# ---------------------------------------------------------------------------

def infer_op_role(module_path: str, op_type: str) -> str:
    """Infer an operator's semantic role from its LPWM module path.

    Covers the LPWM PINT transformer naming conventions:
      - Attention blocks: *.attn.key, *.attn.query, *.attn.value, *.attn.proj
      - FFN blocks: *.mlp.fc_1, *.mlp.proj (SwiGLU: w1, w2, w3)
      - Context projections: *.c_proj, context_proj, context_projection
      - Particle projections: *.xy_projection, *.scale_projection, etc.
      - Output heads: *.head, *head.{0,2}
    """
    # 中文说明：根据模块路径猜算子的"语义角色"（Q/K/V/O、FFN_up 等）。
    # 为什么要猜：模型里层名是自由的，而硬件侧需要知道每个 Linear 是
    # 注意力投影还是前馈层（决定它是否权重固定、能否驻留光阵列）。
    # 规则从上到下按优先级匹配，靠路径子串判断（如 .attn.key -> "K"）。
    # 识别不了就返回 "other_linear"（保守兜底）。
    path_lower = module_path.lower()

    # Attention Q/K/V/O projections (PINT-style: *.attn.key/query/value/proj)
    # 注意力四投影（新版命名）：.attn.key/.query/.value/.proj
    if ".attn.key" in path_lower:
        return "K"
    if ".attn.query" in path_lower:
        return "Q"
    if ".attn.value" in path_lower:
        return "V"
    if ".attn.proj" in path_lower:
        return "O"
    # Legacy naming: q_proj, k_proj, etc.
    # 旧版命名：q_proj/wq、k_proj/wk、v_proj/wv、o_proj/out_proj/wo
    if "q_proj" in path_lower or "wq" in path_lower:
        return "Q"
    if "k_proj" in path_lower or "wk" in path_lower:
        return "K"
    if "v_proj" in path_lower or "wv" in path_lower:
        return "V"
    if "o_proj" in path_lower or "out_proj" in path_lower or "wo" in path_lower:
        return "O"

    # FFN (SwiGLU: fc_1=w1, proj=w2; or w1/w2/w3)
    # 前馈网络（SwiGLU）：fc_1/w1=第一层，proj/w2=第二层，w3=门控层
    if ".mlp." in path_lower or "ffn" in path_lower:
        if "fc_1" in path_lower or "w1" in path_lower:
            return "FFN_up"
        if "proj" in path_lower and "c_proj" not in path_lower:
            return "FFN_down"
        if "w3" in path_lower:
            return "FFN_gate"
        return "FFN"

    # Context projections
    # 上下文投影
    if "c_proj" in path_lower or "context_proj" in path_lower:
        return "c_proj"

    # Particle projection heads
    # 粒子投影头：按输出内容细分（xy/scale/feature/obj_on/depth/bg/...）
    if "projection" in path_lower or "_proj" in path_lower:
        if "xy" in path_lower:
            return "proj_xy"
        if "scale" in path_lower:
            return "proj_scale"
        if "feature" in path_lower:
            return "proj_features"
        if "obj_on" in path_lower:
            return "proj_obj_on"
        if "depth" in path_lower:
            return "proj_depth"
        if "bg" in path_lower:
            return "proj_bg"
        if "origin" in path_lower:
            return "proj_origin"
        if "particle" in path_lower:
            return "proj_particle"
        if "score" in path_lower:
            return "proj_score"
        if "action" in path_lower:
            return "proj_action"
        if "goal" in path_lower:
            return "proj_goal"
        if "cond" in path_lower:
            return "proj_cond"
        if "lang" in path_lower:
            return "proj_lang"
        return "proj"

    # Output heads
    # 输出头
    if ".head" in path_lower:
        if "context" in path_lower:
            return "head_context"
        if "obj_on" in path_lower:
            return "head_obj_on"
        if "depth" in path_lower:
            return "head_depth"
        if "feature" in path_lower:
            return "head_features"
        if "xy" in path_lower:
            return "head_xy"
        if "scale" in path_lower:
            return "head_scale"
        if "score" in path_lower:
            return "head_score"
        if "bg" in path_lower:
            return "head_bg"
        if "offset" in path_lower:
            return "head_offset"
        if "to_action" in path_lower:
            return "head_action"
        return "head"

    # Decoder backbone
    # 解码器主干
    if "decoder" in path_lower or "from_latent" in path_lower:
        return "decoder_proj"
    if "backbone" in path_lower:
        return "backbone"
    if "bg_" in path_lower.replace(".", "_"):
        return "bg_proj"

    # Embedding / conditioning
    # 嵌入 / 条件注入
    if "embed" in path_lower:
        return "embed"
    if "proj" in path_lower:
        return "projection"

    # 没识别出来的 Linear：保守归类
    return "other_linear"


def infer_phase(module_path: str) -> str:
    """Infer the LPWM phase from module path.

    LPWM hierarchy:
      encoder_module.*          → encode
      encoder_module.ctx_enc.*  → context
      encoder_module.prior_*    → encode (prior)
      decoder_module.*          → decode
      dyn_module.*              → dynamics
      ctx_module.*              → context
    """
    # 中文说明：根据模块路径猜算子在模型流水线的哪个"阶段"。
    # 阶段顺序：encode(编码) -> context(上下文) -> dynamics(动态演化)
    # -> decode(解码)。硬件侧用阶段信息决定何时释放一次性权重（见调度器）。
    pl = module_path.lower()

    # Decoder explicitly
    if "decoder_module" in pl or ".decoder" in pl or ".bg_dec" in pl:
        return "decode"

    # Dynamics
    if "dyn_module" in pl or ".dyn." in pl:
        return "dynamics"

    # Context module (inside encoder)
    if "ctx_enc" in pl or "ctx_module" in pl or "context" in pl:
        return "context"

    # Spatial / Temporal blocks → semantics depend on parent
    # 空间/时间块：属于哪个阶段取决于它的父模块
    if "spatio" in pl or "spatial" in pl:
        if "ctx_enc" in pl or "dyn_module" in pl:
            return "dynamics" if "dyn_module" in pl else "context"
        return "dynamics"

    if "temp_block" in pl or "temporal" in pl:
        if "ctx_enc" in pl or "dyn_module" in pl:
            return "dynamics" if "dyn_module" in pl else "context"
        return "dynamics"

    # Encoder
    if "encoder_module" in pl or ".enc." in pl or "prior_module" in pl or "prior_enc" in pl:
        return "encode"

    # Heads, projections, embeddings → part of encode/decode depending on location
    if ".head" in pl or "projection" in pl:
        if "decoder" in pl:
            return "decode"
        return "encode"

    # Sample
    if "sample" in pl:
        return "decode"

    return "unknown"


def infer_block_kind(module_path: str) -> str:
    """判断模块属于空间块还是时间块（用于区分世界模型的空/时维度）。"""
    pl = module_path.lower()
    if "spatial" in pl:
        return "spatial"
    if "temporal" in pl:
        return "temporal"
    return "other"


def make_weight_id(module_path: str, weight_shape: tuple) -> str:
    """Stable weight_id from module path and weight shape (Plan Section 4.1)."""
    # 中文说明：由"模块路径 + 权重形状"生成稳定的 16 位哈希 id。
    # 同一权重在多次调用/多次运行间 id 不变，是 SRAM/MRR 驻留的键。
    raw = f"{module_path}|{tuple(weight_shape)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Hook-based trace exporter
# ---------------------------------------------------------------------------

class TraceExporter:
    # hooks 记录算子调用；records 是最终写入 JSONL 的顺序列表。
    """Register forward hooks on an LPWM model and export operator traces.

    Parameters
    ----------
    model : torch.nn.Module
        The LPWM model to instrument.
    trace_version : str
        Version string for this trace.
    input_bits : int
    weight_bits : int
    output_bits : int
    dtype_map : dict
        Mapping from torch dtype names to bit widths.
    """
    # 中文说明：轨迹导出器。用法：
    #   exporter = TraceExporter(model)
    #   exporter.install_hooks()      # 给所有 nn.Linear 挂前向钩子
    #   model(input)                  # 正常跑一次前向
    #   exporter.save("trace.jsonl")  # 导出成 JSONL 文件
    # 记录列表 self.records 是最终写入文件的顺序列表；_order 是全局序号，
    # _call_index 记录每个模块第几次被调用（同一层多次调用要有不同 op_id）。

    def __init__(
        self,
        model,
        trace_version: str = "1.0.0",
        input_bits: int = 8,
        weight_bits: int = 8,
        output_bits: int = 8,
        dtype_map: Optional[Dict[str, int]] = None,
    ):
        if not HAS_TORCH:
            raise ImportError(
                "TraceExporter requires PyTorch. Install with: pip install torch"
            )
        self.model = model
        self.trace_version = trace_version
        self.input_bits = input_bits
        self.weight_bits = weight_bits
        self.output_bits = output_bits
        # torch dtype 名 -> 位宽 的映射（float32=32 位，fp16/bf16=16 位...）
        self.dtype_map = dtype_map or {"float32": 32, "float16": 16, "bfloat16": 16, "int8": 8}
        self.records: List[dict] = []
        self._order = 0
        self._call_index: Dict[str, int] = {}
        self._handles = []          # 已注册的 hook 句柄（remove_hooks 时用）

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def install_hooks(self):
        """Register forward hooks on all nn.Linear and key ops."""
        # 给模型里每一个 nn.Linear 模块注册前向钩子；
        # 钩子在每次前向计算后触发，把算子信息记录下来
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(
                    self._make_linear_hook(name)
                )
                self._handles.append(h)

    def remove_hooks(self):
        """卸载全部钩子（跑完后清理，避免影响其它代码）。"""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------

    def _make_linear_hook(self, module_path: str):
        # 闭包：把模块路径"绑定"进钩子函数（钩子本身只拿到 module/input/output）
        def hook(module, input_, output):
            self._record_linear(module_path, module, input_, output)
        return hook

    def _record_linear(
        self,
        module_path: str,
        module: nn.Linear,
        input_: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ):
        # 把一次 Linear 调用整理成一条 OperatorRecord 字典。
        # 关键换算：
        #   - M = 输入张量去掉最后一维后所有维度的乘积（=批 x 序列长度等）
        #   - K = in_features，N = out_features
        #   - weight_id = make_weight_id(module_path, weight_shape)，稳定可复现
        #   - weight_static 恒为 True：Linear 权重训练完就固定（光子后端的入场券）
        inp = input_[0]
        out = output
        in_features = module.in_features
        out_features = module.out_features

        # M = product of all batch dims, K = in_features, N = out_features
        if inp.dim() >= 2:
            M = int(torch.prod(torch.tensor(inp.shape[:-1])).item())
        else:
            M = 1
        K = in_features
        N = out_features

        # Linear 的权重形状约定：(out_features, in_features)
        weight_shape = (out_features, in_features)
        weight_id = make_weight_id(module_path, weight_shape)

        self._order += 1
        self._call_index[module_path] = self._call_index.get(module_path, 0) + 1

        rec = {
            "trace_version": self.trace_version,
            "op_id": f"{module_path}_{self._call_index[module_path]}",
            "order": self._order,
            "module_path": module_path,
            "op_type": "Linear",
            "op_role": infer_op_role(module_path, "Linear"),
            "phase": infer_phase(module_path),
            "block_kind": infer_block_kind(module_path),
            "layer_index": None,
            "rollout_step": None,
            "call_index": self._call_index[module_path],
            "input_shapes": tuple(inp.shape),
            "output_shape": tuple(out.shape),
            "M": M,
            "K": K,
            "N": N,
            "batch_repetitions": 1,
            "dtype": str(inp.dtype).replace("torch.", ""),
            "input_bits": self._dtype_bits(inp.dtype),
            "weight_bits": self.weight_bits,
            "output_bits": self.output_bits,
            "weight_id": weight_id,
            "weight_static": True,
            # 各张量的字节数：元素数 x 位宽 / 8
            "input_bytes": int(M * K * self._dtype_bits(inp.dtype) // 8),
            "weight_bytes": int(K * N * self.weight_bits // 8),
            "output_bytes": int(M * N * self.output_bits // 8),
            "dependencies": tuple(),
        }
        self.records.append(rec)

    def _dtype_bits(self, dtype) -> int:
        # 把 torch dtype 名映射到位宽；没见过的类型退回 input_bits
        name = str(dtype).replace("torch.", "")
        return self.dtype_map.get(name, self.input_bits)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Write the trace as a JSON-lines file."""
        # 第一行写 manifest（版本、算子总数、导出时间），之后每条记录一行。
        # trace_io.load_manifest_jsonl 会按这个格式读取并校验。
        with open(path, "w", encoding="utf-8") as f:
            # Header comment with metadata
            f.write(json.dumps({
                "trace_version": self.trace_version,
                "schema_version": TRACE_SCHEMA_VERSION,
                "total_operators": len(self.records),
                "exported_at": time.time(),
            }) + "\n")
            for rec in self.records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def to_json(self) -> str:
        """Return the full trace as a JSON-lines string."""
        # 同上但不落盘，直接返回字符串（便于内存传递）
        lines = []
        for rec in self.records:
            lines.append(json.dumps(rec, ensure_ascii=False))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def linear_count(self) -> int:
        """Linear 算子的总调用次数。"""
        return sum(1 for r in self.records if r["op_type"] == "Linear")

    @property
    def unique_weight_count(self) -> int:
        """去重后的权重个数（多少个不同权重被访问过）。"""
        return len(set(r["weight_id"] for r in self.records if r.get("weight_id")))

    def summary(self) -> dict:
        """返回 trace 的统计摘要（算子总数、Linear 数、阶段/角色分布）。"""
        return {
            "trace_version": self.trace_version,
            "total_operators": len(self.records),
            "linear_count": self.linear_count,
            "unique_weights": self.unique_weight_count,
            "phases": self._count_by("phase"),
            "roles": self._count_by("op_role"),
        }

    def _count_by(self, key: str) -> dict:
        """按某个字段（如 phase/op_role）统计各类出现次数。"""
        counts: Dict[str, int] = {}
        for r in self.records:
            v = r.get(key, "unknown")
            counts[v] = counts.get(v, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Standalone: export from a generic nn.Module (without LPWM-specific code)
# ---------------------------------------------------------------------------

def export_trace_from_module(
    model,
    dummy_input: Any,
    trace_path: str,
    trace_version: str = "1.0.0",
    **kwargs,
) -> TraceExporter:
    """Run one forward pass and export the trace.

    This is the main entry point for Phase D.
    """
    # 中文说明：最常用的入口函数——给定任意 nn.Module 和一个示例输入，
    # 自动完成"挂钩子 -> 跑一次前向 -> 卸载钩子 -> 存 trace 文件"全流程。
    # 返回 exporter 对象（里面还保留着 records，可继续查询）。
    exporter = TraceExporter(model, trace_version=trace_version, **kwargs)
    exporter.install_hooks()
    try:
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)
    finally:
        # 无论前向是否报错，都要保证钩子被卸载，避免污染模型
        exporter.remove_hooks()
    exporter.save(trace_path)
    return exporter
