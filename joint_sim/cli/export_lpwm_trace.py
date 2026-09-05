#!/usr/bin/env python
"""
Export LPWM operator trace for joint LLMCompass-SimPhony simulation (Step 7).

Captures every operator (Linear, MatMul, Softmax, LayerNorm, GELU, etc.)
during a full LPWM rollout, with rollout_step, dependencies, phase, and role.

Usage:
    python joint_sim/cli/export_lpwm_trace.py \
        --config third_party/lpwm/configs/bair64.json \
        --checkpoint path/to/checkpoint.pth \
        --output traces/lpwm_bair64_trace.jsonl \
        --horizon 30 --cond-steps 1

【中文说明】
这是"轨迹导出"工具（整个仿真流水线的第 7 步）：真正运行一遍 LPWM 视频预测模型
的完整推理（encode → 动态 rollout → decode），用钩子（hook）拦截每个算子
（Linear/MatMul/Softmax/GELU/LayerNorm 等），把它们记录成一行一行的 JSON 轨迹，
供后面的 joint_sim 仿真器读取并调度。

核心思路：
- LPWM 用 PyTorch 实现，PyTorch 允许给子模块注册"前向钩子"（forward hook），
  模块每次前向计算时钩子就会被调用。本脚本利用这一点"偷看"每个 Linear 层的
  输入输出形状，从而知道这个算子是做什么的（M×K 乘以 K×N）。
- 注意力里的 QK^T、Softmax、AV 不是独立模块，而是藏在融合算子
  F.scaled_dot_product_attention 里，所以脚本用"猴子补丁"（monkey-patch）替换
  该函数，在它执行时手动拆出 QK^T → Softmax → AV 三段并记录。
- 为了给仿真器提供"谁先谁后"的依赖信息（数据流图），脚本给每个张量打上
  "由哪个算子产生"的标签，通过一个 TorchDispatchMode 在 reshape/cat 等操作中
  自动传播这个标签。

产出：一个 .jsonl 文件（JSON Lines 格式，每行一个算子的完整记录），
包含 op_type、M/K/N 形状、weight_id、rollout_step（预测步编号）、phase（编码/滚动/解码）、
dependencies（该算子的上游算子 id 列表）等字段。

命令行常用参数：
- --config      LPWM 模型配置文件（json）
- --checkpoint  预训练权重 .pth（可选；不填则用随机权重）
- --output      轨迹输出文件路径
- --horizon     要预测多少个未来帧（rollout 步数）
- --cond-steps  用几帧历史帧作为条件输入
- --input/weight/output-bits  量化位宽（默认 8 位）
- --stochastic  是否随机采样（默认确定性推理）
- --cpu         强制用 CPU
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LPWM_ROOT = os.path.join(_PROJECT_ROOT, "third_party", "lpwm")
_JOINT_SIM = os.path.join(_PROJECT_ROOT, "joint_sim")
for p in [_LPWM_ROOT, _JOINT_SIM, _PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from models import DLP
from lpwm_trace_exporter import make_weight_id, infer_op_role, infer_phase, infer_block_kind


# ---------------------------------------------------------------------------
# Global state for rollout tracking
# ---------------------------------------------------------------------------
# 下面的全局变量用来在整个导出过程中跨函数共享状态。
_order = 0  # 全局算子序号：每记录一个算子就 +1，保证所有算子的先后顺序是确定的
_call_index = defaultdict(int)  # 每个模块路径被调用了多少次（用于给同名算子编号）
_rollout_step = 0  # 当前处于 rollout（自回归预测）的第几步
_records = []  # 收集到的所有算子记录，最后统一写入输出文件
_attention_count = 0  # counter for QK^T/AV naming  # 注意力块计数器，用于给 QK^T/AV 命名
_attention_stack = []  # 当前正在执行的注意力模块路径栈（配合前向钩子知道此刻在哪个注意力块里）


# ---------------------------------------------------------------------------
# Tensor-producer dependency tracking
# ---------------------------------------------------------------------------
# 为了让仿真器知道算子之间的数据依赖（数据流 DAG），给每个张量挂一个
# "_lpwm_trace_sources" 属性，记录"这个张量是由哪些算子产生的"。

_SOURCE_ATTR = "_lpwm_trace_sources"


def _iter_tensors(value):
    """Yield every tensor contained in a nested eager-mode value."""
    # 中文说明：递归遍历嵌套结构（dict/list/tuple/单个张量），把里面所有
    # torch.Tensor 逐个"吐出来"。用于统一处理各种函数返回值。
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)


def _tensor_sources(value):
    """Return the nearest captured producer op_ids for ``value``."""
    # 中文说明：收集 value 里所有张量的生产者算子 id 的并集。
    # 即"这个值依赖哪些算子"，用于填写算子的 dependencies 字段。
    sources = set()
    for tensor in _iter_tensors(value):
        sources.update(getattr(tensor, _SOURCE_ATTR, ()))
    return sources


def _tag_sources(value, sources):
    """Attach producer provenance to all tensors in a nested value."""
    # 中文说明：把生产者 id 集合（sources）写进 value 里所有张量的属性上，
    # 相当于给张量"盖章"，标记它的来源算子。
    frozen = frozenset(sources)
    for tensor in _iter_tensors(value):
        setattr(tensor, _SOURCE_ATTR, frozen)


def _ordered_dependencies(sources):
    """Return deterministic, already-emitted producer ids."""
    # 中文说明：把生产者 id 集合按"它们被记录的顺序"排序，保证依赖列表是
    # 确定性的（两次运行结果一致），且只保留已经记录过的算子。
    emitted_order = {record["op_id"]: record["order"] for record in _records}
    return sorted(
        (source for source in sources if source in emitted_order),
        key=emitted_order.__getitem__,
    )


class _DependencyPropagationMode(TorchDispatchMode):
    """Propagate captured producer ids through unrecorded tensor operations.

    Reshape, transpose, residual addition, concatenation and other operators
    are not separate Trace records, but they must not break data provenance.
    This mode conservatively carries the union of their input producers to
    their tensor outputs, including while ``torch.no_grad()`` is active.

    中文说明：
    这个类是一个"张量调度模式"（TorchDispatchMode）。PyTorch 每次执行底层算子时
    都会先经过它。这里做两件事：先正常执行算子（func(*args, **kwargs)），然后把
    "输入张量来自哪些算子"的标签复制到输出张量上。

    为什么要这么做？reshape、transpose、加法、cat 这类算子本身不会被记录成一条
    轨迹记录（它们太琐碎了），但如果不管它们，经过这些操作后张量会"丢失身世"，
    下游真正的算子就不知道自己的输入是从哪来的了。这个模式保证依赖信息在
    "未被记录的操作"中也能连续传递。
    """

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # 先取出本次操作所有输入张量的来源标签（并集）
        sources = _tensor_sources((args, kwargs))
        # 正常执行这个底层算子
        output = func(*args, **kwargs)
        # 如果输入有来源标签，就把标签传播到输出张量上
        if sources:
            _tag_sources(output, sources)
        return output


def _record_qk_softmax_av(query, key, value, output, module_path="attn"):
    """Record the real SDPA dataflow: QK^T -> Softmax -> AV.

    中文说明：
    PyTorch 的 F.scaled_dot_product_attention 把"QK^T 矩阵乘 → Softmax →
    与 V 相乘（AV）"三段融成了一个底层算子，普通钩子看不到内部结构。
    但这个脚本必须把三段分开记录——因为仿真器需要知道光子阵列具体在做哪个
    矩阵乘法。所以这里手动按公式拆成三条记录：
      1) QK^T：MatMul，形状 [bs, n_heads, seq_len, d_head]×[d_head, seq_len]
          → [bs, n_heads, seq_len, seq_len]
      2) Softmax：归一化注意力权重（让每行加起来等于 1）
      3) AV：MatMul，注意力权重 × V → [bs, n_heads, seq_len, d_head]
    参数：query/key/value 分别是注意力三件套 Q、K、V；output 是融合算子真正的
    输出张量；module_path 是所在注意力模块的路径名。
    """
    global _order, _rollout_step, _attention_count
    _attention_count += 1

    # 从 Q 的形状推断出 batch 大小、注意力头数、序列长度、每个头的维度
    q_shape = tuple(query.shape)
    k_shape = tuple(key.shape)
    v_shape = tuple(value.shape)

    # q: [bs, n_heads, seq_len, d_head]
    # k: [bs, n_heads, seq_len, d_head]
    # v: [bs, n_heads, seq_len, d_head]
    # 注意：若形状不足 4 维，就按 1 补全（兼容各种维度写法）
    bs = q_shape[0] if len(q_shape) >= 4 else 1
    n_heads = q_shape[1] if len(q_shape) >= 3 else 1
    seq_len = q_shape[-2] if len(q_shape) >= 3 else q_shape[0]
    d_head = q_shape[-1]

    # QK^T: [bs, n_heads, seq_len, seq_len]
    # --- 记录第 1 段：QK^T 矩阵乘 ---
    order = _next_order()
    # 依赖 = 产生 Q 和 K 的那些算子（Q、K 必须算完才能做点积）
    dep = _ordered_dependencies(_tensor_sources((query, key)))
    op_id = f"{module_path}_QKT_{_attention_count}"
    # 每条记录都是一个小字典，字段含义：
    #   op_type 算子种类；op_role 在模型里的角色；M/K/N 矩阵乘的三维尺寸
    #   batch_repetitions 批量重复次数（batch×头数，多个同样的矩阵乘并行做）
    #   weight_id / weight_static：光子阵列只适合算"权重固定的线性算子"，
    #     这里注意力权重不固定，所以 weight_static=False、weight_id=None
    #   dependencies：这个算子的上游算子 id 列表（数据流依赖）
    _records.append({
        "trace_version": "1.1.0", "op_id": op_id, "order": order,
        "module_path": module_path, "op_type": "MatMul",
        "op_role": "QK_T",
        "phase": infer_phase(module_path),
        "block_kind": infer_block_kind(module_path),
        "layer_index": None, "rollout_step": _rollout_step,
        "call_index": _attention_count,
        "input_shapes": q_shape, "output_shape": (bs, n_heads, seq_len, seq_len),
        "M": int(seq_len), "K": int(d_head), "N": int(seq_len),
        "batch_repetitions": int(bs * n_heads),
        "dtype": "float32", "input_bits": 8, "weight_bits": None,
        "output_bits": 8, "weight_id": None, "weight_static": False,
        "input_bytes": 0, "weight_bytes": 0, "output_bytes": 0,
        "dependencies": dep,
    })

    # The fused PyTorch kernel contains a real Softmax between QK^T and AV.
    # --- 记录第 2 段：Softmax（PyTorch 融合内核里确实夹在 QK^T 和 AV 之间）---
    order = _next_order()
    softmax_id = f"{module_path}_Softmax_{_attention_count}"
    _records.append({
        "trace_version": "1.1.0", "op_id": softmax_id, "order": order,
        "module_path": module_path, "op_type": "Softmax",
        "op_role": "attention_softmax",
        "phase": infer_phase(module_path),
        "block_kind": infer_block_kind(module_path),
        "layer_index": None, "rollout_step": _rollout_step,
        "call_index": _attention_count,
        "input_shapes": (bs, n_heads, seq_len, seq_len),
        "output_shape": (bs, n_heads, seq_len, seq_len),
        "M": int(bs * n_heads * seq_len), "K": int(seq_len), "N": 1,
        "batch_repetitions": int(bs * n_heads),
        "dtype": str(query.dtype).replace("torch.", ""),
        "input_bits": 8, "weight_bits": None, "output_bits": 8,
        "weight_id": None, "weight_static": False,
        "input_bytes": 0, "weight_bytes": 0, "output_bytes": 0,
        "dependencies": [op_id],  # Softmax 只依赖 QK^T 的输出
    })

    # AV depends on the attention probabilities and V, not on log order.
    # --- 记录第 3 段：AV 矩阵乘 ---
    # 注意：AV 依赖的是"V 的来源算子"+"Softmax 的输出"，和记录顺序无关
    order = _next_order()
    dep = _ordered_dependencies(_tensor_sources(value))
    dep = _ordered_dependencies(set(dep) | {softmax_id})
    av_id = f"{module_path}_AV_{_attention_count}"
    _records.append({
        "trace_version": "1.1.0", "op_id": av_id, "order": order,
        "module_path": module_path, "op_type": "MatMul",
        "op_role": "AV",
        "phase": infer_phase(module_path),
        "block_kind": infer_block_kind(module_path),
        "layer_index": None, "rollout_step": _rollout_step,
        "call_index": _attention_count,
        "input_shapes": (bs, n_heads, seq_len, seq_len),
        "output_shape": v_shape, "M": int(seq_len), "K": int(seq_len),
        "N": int(d_head), "batch_repetitions": int(bs * n_heads),
        "dtype": "float32", "input_bits": 8, "weight_bits": None,
        "output_bits": 8, "weight_id": None, "weight_static": False,
        "input_bytes": 0, "weight_bytes": 0, "output_bytes": 0,
        "dependencies": dep,
    })
    # 整个注意力模块的真正输出就是 AV 的输出，把标签贴回 output
    _tag_sources(output, {av_id})


def _patch_attention():
    """Monkey-patch F.scaled_dot_product_attention to intercept MatMul ops.

    中文说明：
    猴子补丁：临时把 torch.nn.functional.scaled_dot_product_attention 换成我们
    自己的包装函数。这样 LPWM 模型调用注意力时，先正常执行原函数拿到结果，
    再调用 _record_qk_softmax_av 手动拆解记录 QK^T/Softmax/AV 三条算子记录。
    返回值：被替换掉的原始函数（供以后恢复用）。
    """
    import torch.nn.functional as _F
    _original_sdpa = _F.scaled_dot_product_attention

    def _sdpa_wrapper(query, key, value, attn_mask=None, dropout_p=0.0,
                      is_causal=False, scale=None, **kwargs):
        # 先用原始实现算出真正的注意力输出
        output = _original_sdpa(query, key, value, attn_mask=attn_mask,
                                dropout_p=dropout_p, is_causal=is_causal,
                                scale=scale, **kwargs)
        # 判断当前正在哪个注意力模块里执行（从 _attention_stack 栈顶取路径名）
        module_path = _attention_stack[-1] if _attention_stack else "attention"
        # 拆解并记录 QK^T → Softmax → AV
        _record_qk_softmax_av(query, key, value, output, module_path)
        return output

    _F.scaled_dot_product_attention = _sdpa_wrapper
    return _original_sdpa


def _unpatch_attention(original_fn):
    """Restore original F.scaled_dot_product_attention.

    中文说明：导出结束后把被替换的注意力函数恢复原样，避免影响后续代码。
    """
    import torch.nn.functional as _F
    _F.scaled_dot_product_attention = original_fn


def _next_order() -> int:
    global _order
    _order += 1
    return _order


def _record_linear(module_path: str, module: nn.Linear, input_, output, input_bits=8, weight_bits=8, output_bits=8):
    """记录一次 nn.Linear 前向计算，生成一条轨迹记录。

    中文说明：
    nn.Linear 就是"y = xW + b"：输入形状 [..., in_features]，权重 W 形状
    [out_features, in_features]，输出形状 [..., out_features]。把它看成矩阵乘
    是 M×K 乘 K×N：
      M = 输入展平后的行数（batch 内元素数），K = in_features，N = out_features。
    参数：
      module_path  该层的完整路径名（如 "dyn.pint_dyn.0.spatio_block.attn.q"）
      module       nn.Linear 模块对象（用来取 in_features / out_features）
      input_       前向输入（可能是元组，取第一个元素）
      output       前向输出
      *_bits       输入/权重/输出的量化位宽（默认 8 位，光子计算按低位宽估算）
    """
    global _order, _rollout_step
    inp = input_[0] if isinstance(input_, tuple) else input_
    out = output[0] if isinstance(output, tuple) else output
    in_features = module.in_features
    out_features = module.out_features
    # M = 输入张量除最后一维外的所有元素个数相乘（如 [batch, seq] → batch×seq）
    if inp.dim() >= 2:
        M = int(torch.prod(torch.tensor(inp.shape[:-1])).item())
    else:
        M = 1
    K = in_features
    N = out_features
    weight_shape = (out_features, in_features)
    # 生成该层权重的唯一标识（光子阵列靠它判断权重是否已就位、能否复用）
    wid = make_weight_id(module_path, weight_shape)
    order = _next_order()
    op_id = f"{module_path}_{_call_index[module_path]}"
    # 依赖 = 产生当前输入张量的那些算子
    dep = _ordered_dependencies(_tensor_sources(inp))

    _records.append({
        "trace_version": "1.1.0", "op_id": op_id, "order": order,
        "module_path": module_path, "op_type": "Linear",
        "op_role": infer_op_role(module_path, "Linear"),  # 判断是 Q/K/V/输出投影/FFN 等角色
        "phase": infer_phase(module_path),  # 属于编码/滚动/解码哪个阶段
        "block_kind": infer_block_kind(module_path),
        "layer_index": None, "rollout_step": _rollout_step,
        "call_index": _call_index[module_path],
        "input_shapes": tuple(int(d) for d in inp.shape),
        "output_shape": tuple(int(d) for d in out.shape),
        "M": int(M), "K": int(K), "N": int(N),
        "batch_repetitions": 1,
        "dtype": str(inp.dtype).replace("torch.", ""),
        "input_bits": int(input_bits), "weight_bits": int(weight_bits),
        "output_bits": int(output_bits),
        # 权重固定 → 光子阵列可做（weight_static=True），并给出权重 id 供复用
        "weight_id": wid, "weight_static": True,
        # 按位宽换算字节数（bit ÷ 8 = byte），供仿真器估算能耗/带宽
        "input_bytes": int(M * K * input_bits // 8),
        "weight_bytes": int(K * N * weight_bits // 8),
        "output_bytes": int(M * N * output_bits // 8),
        "dependencies": dep,
    })
    # A captured operator becomes the nearest producer for its output.
    # 被记录的算子成为它输出张量的"最近生产者"，供下游算子记录依赖
    _tag_sources(output, {op_id})


def _record_dynamic(op_type: str, op_role: str, input_shapes, output_shape,
                    M=None, K=None, N=None, module_path="", input_bits=8,
                    output_bits=8, dependencies=None, output=None):
    """记录一个"动态算子"（权重不固定、无法用光子阵列算的算子）。

    中文说明：
    像 Softmax、GELU、LayerNorm、激活这类算子没有固定权重，只有数据输入，
    仿真器会把它们分派给电子后端（类 GPU 张量核心）。本函数把它们也记成
    一条轨迹记录。注意这些算子的 weight_id 为 None、weight_static=False，
    表示"不能上光子阵列"。
    参数：op_type 算子类型名（如 "Softmax"），op_role 角色名，input_shapes/
    output_shape 输入输出形状列表，M/K/N 近似矩阵乘维度（用于能耗估算），
    dependencies 上游算子 id，output 若是实际张量则给其贴来源标签。
    返回值：本条记录的 op_id。
    """
    global _order, _rollout_step
    order = _next_order()
    op_id = f"{module_path}_{op_type}_{order}"
    dep = _ordered_dependencies(dependencies or ())
    _records.append({
        "trace_version": "1.1.0", "op_id": op_id, "order": order,
        "module_path": module_path, "op_type": op_type,
        "op_role": op_role, "phase": infer_phase(module_path),
        "block_kind": infer_block_kind(module_path),
        "layer_index": None, "rollout_step": _rollout_step,
        "call_index": 0,
        "input_shapes": input_shapes,
        "output_shape": output_shape,
        "M": M, "K": K, "N": N,
        "batch_repetitions": 1,
        "dtype": "float32",
        "input_bits": int(input_bits), "weight_bits": None,
        "output_bits": int(output_bits),
        "weight_id": None, "weight_static": False,
        "input_bytes": 0, "weight_bytes": 0, "output_bytes": 0,
        "dependencies": dep,
    })
    # 如果传入了真实输出张量，就把本条 op_id 作为它的来源标签
    if output is not None:
        _tag_sources(output, {op_id})
    return op_id


def _record_activation(module_path, module, input_, output, input_bits=8, output_bits=8):
    """Capture an activation at its actual execution point."""
    # 中文说明：记录一次激活函数（如 GELU）的执行。激活函数的输入输出同形状，
    # 在轨迹里用 M=元素总数、K=1、N=1 表示"逐元素操作"。
    inp = input_[0] if isinstance(input_, tuple) else input_
    out = output[0] if isinstance(output, tuple) else output
    shape = tuple(int(d) for d in inp.shape)
    # GELU 单独标注角色，其他激活统一叫 "activation"
    role = "gelu" if isinstance(module, nn.GELU) else "activation"
    _record_dynamic(
        module.__class__.__name__, role, shape,
        tuple(int(d) for d in out.shape),
        M=int(inp.numel()), K=1, N=1, module_path=module_path,
        input_bits=input_bits, output_bits=output_bits,
        dependencies=_tensor_sources(inp), output=output,
    )


# ---------------------------------------------------------------------------
# Model loading with checkpoint
# ---------------------------------------------------------------------------
# 模型加载区：把 LPWM 世界模型从配置 + 权重文件恢复出来，供轨迹导出使用。

def load_lpwm_model(config_path: str, checkpoint_path: str = None, device: str = "cpu"):
    """Load LPWM model from a config JSON, optionally with checkpoint weights.

    中文说明：
    从 json 配置文件读出所有超参数（层数、通道数、特征维度等），据此构造一个
    DLP 模型（LPWM 的神经网络主体）。如果给了 checkpoint 权重文件就加载权重，
    否则警告并用随机权重。
    参数：
      config_path      LPWM 配置 json 的路径
      checkpoint_path  权重 .pth 路径（可选）
      device           模型放到的设备（cpu/cuda）
    返回值：五元组 (model, config, ckpt_loaded, ckpt_hash, ckpt_epoch)，
      分别是模型对象、原始配置字典、是否成功加载权重、权重文件哈希前 16 位、
      权重保存时的训练轮数（可能为 None）。
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    # DLP 是 LPWM 的核心网络类，这里把配置里的键一一对应传给构造函数
    # （每个参数都给了默认值，所以配置缺项时也能跑）
    model = DLP(
        cdim=config.get("ch", 3),
        image_size=config.get("image_size", 64),
        normalize_rgb=config.get("normalize_rgb", False),
        n_views=config.get("n_views", 1),
        n_kp_per_patch=config.get("n_kp_per_patch", 1),
        patch_size=config.get("patch_size", 4),
        anchor_s=config.get("anchor_s", 0.125),
        n_kp_enc=config.get("n_kp_enc", 80),
        n_kp_prior=config.get("n_kp_prior", 256),
        pad_mode=config.get("pad_mode", "zeros"),
        dropout=config.get("dropout", 0.1),
        features_dist=config.get("features_dist", "gauss"),
        learned_feature_dim=config.get("learned_feature_dim", 5),
        learned_bg_feature_dim=config.get("learned_bg_feature_dim", 5),
        obj_res_from_fc=config.get("obj_res_from_fc", 4),
        obj_ch_mult_prior=config.get("obj_ch_mult_prior", [4]),
        obj_ch_mult=config.get("obj_ch_mult", [1, 4]),
        obj_base_ch=config.get("obj_base_ch", 32),
        obj_final_cnn_ch=config.get("obj_final_cnn_ch", 32),
        bg_res_from_fc=config.get("bg_res_from_fc", 8),
        bg_ch_mult=config.get("bg_ch_mult", [1, 1, 2, 4]),
        bg_base_ch=config.get("bg_base_ch", 32),
        bg_final_cnn_ch=config.get("bg_final_cnn_ch", 32),
        use_resblock=config.get("use_resblock", False),
        num_res_blocks=config.get("num_res_blocks", 1),
        cnn_mid_blocks=config.get("cnn_mid_blocks", False),
        mlp_hidden_dim=config.get("mlp_hidden_dim", 256),
        attn_norm_type=config.get("attn_norm_type", "rms"),
        pint_enc_layers=config.get("pint_enc_layers", 1),
        pint_enc_heads=config.get("pint_enc_heads", 1),
        timestep_horizon=config.get("timestep_horizon", 16),
        n_static_frames=config.get("num_static_frames", 1),
        predict_delta=config.get("predict_delta", False),
        context_dim=config.get("context_dim", 7),
        ctx_dist=config.get("context_dist", "gauss"),
        ctx_pool_mode=config.get("ctx_pool_mode", "none"),
        pint_dyn_layers=config.get("pint_dyn_layers", 6),
        pint_dyn_heads=config.get("pint_dyn_heads", 8),
        pint_dim=config.get("pint_dim", 512),
        pint_ctx_layers=config.get("pint_ctx_layers", 4),
        pint_ctx_heads=config.get("pint_ctx_heads", 8),
        action_condition=config.get("action_condition", False),
        action_dim=config.get("action_dim", 0),
        null_action_embed=config.get("null_action_embed", False),
        random_action_condition=config.get("random_action_condition", False),
        random_action_dim=config.get("random_action_dim", 0),
        language_condition=config.get("language_condition", False),
        language_embed_dim=config.get("language_embed_dim", 0),
        img_goal_condition=config.get("image_goal_condition", False),
        scale_std=config.get("scale_std", 0.15),
        offset_std=config.get("offset_std", 0.1),
        obj_on_alpha=config.get("obj_on_alpha", 0.01),
        obj_on_beta=config.get("obj_on_beta", 0.01),
        n_fg_categories=config.get("n_fg_categories", 8),
        n_fg_classes=config.get("n_fg_classes", 4),
        n_bg_categories=config.get("n_bg_categories", 4),
        n_bg_classes=config.get("n_bg_classes", 4),
        n_ctx_categories=config.get("n_ctx_categories", 8),
        n_ctx_classes=config.get("n_ctx_classes", 4),
    )

    # --- 尝试加载预训练权重 ---
    ckpt_loaded = False
    ckpt_hash = None
    ckpt_epoch = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}", file=sys.stderr)
        # 兼容三种权重保存格式：含 "model_state_dict" 键 / 含 "state_dict" 键 / 直接就是权重字典
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        ckpt_loaded = True
        ckpt_epoch = ckpt.get("epoch")
        # 记录权重文件的哈希，方便后续对比"同一份权重"导出的一致性
        ckpt_hash = hashlib.sha256(
            open(checkpoint_path, "rb").read()
        ).hexdigest()[:16]
        print(f"Checkpoint loaded OK (epoch={ckpt_epoch}, hash={ckpt_hash})", file=sys.stderr)
    else:
        print(f"WARNING: No checkpoint at '{checkpoint_path}', using random weights",
              file=sys.stderr)

    model = model.to(device)
    return model, config, ckpt_loaded, ckpt_hash, ckpt_epoch


# ---------------------------------------------------------------------------
# Traced LPWM wrapper
# ---------------------------------------------------------------------------

class TracedLPWM:
    """Wraps LPWM model to intercept all operators during sample_from_x.

    中文说明：
    一个"带钩子的 LPWM 包装器"。LPWM 自回归推理入口是 sample_from_x
    （一次调用内完成"编码 → 动态预测 → 解码"整个 rollout）。本类在调用
    sample_from_x 期间给模型的所有 nn.Linear 层挂上前向钩子，从而把每一步
    的算子都拦截下来记录。记录结果放在全局 _records 列表里。
    """

    def __init__(self, model: DLP, config: dict, input_bits=8, weight_bits=8, output_bits=8):
        # 保存模型、配置和量化位宽；_handles 用来记录所有已注册的钩子
        self.model = model
        self.config = config
        self.input_bits = input_bits
        self.weight_bits = weight_bits
        self.output_bits = output_bits
        self._handles = []

    def install_hooks(self):
        """Register forward hooks on all nn.Linear modules.

        中文说明：遍历模型的每一个子模块，凡遇到 nn.Linear 就注册前向钩子
        （每次该层前向计算后自动调用）。注意用闭包（make_hook 捕获 path）
        保存各自的路径名，避免循环里共用同一个变量。
        """
        self._handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                _call_index[name] = 0

                def make_hook(path):
                    def hook(m, inp, out):
                        _call_index[path] += 1  # 该层第几次被调用
                        _record_linear(path, m, inp, out,
                                       self.input_bits, self.weight_bits, self.output_bits)
                    return hook
                self._handles.append(module.register_forward_hook(make_hook(name)))

    def remove_hooks(self):
        """卸载所有已注册的钩子（避免泄漏到其他运行中）。"""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def run_rollout(self, x, num_steps=30, cond_steps=1, deterministic=True):
        """Run sample_from_x and capture all operators.

        The LPWM sample_from_x does:
          1. encode cond_steps input frames → particles
          2. iterate num_steps dynamics → predicted particles
          3. decode each predicted step → image
        We intercept nn.Linear hooks throughout, plus record
        the implicit Softmax/GELU/LayerNorm inside the model.

        中文说明：
        LPWM 的 sample_from_x 内部做三件事：
          1) 编码：把条件帧编码成粒子（particles，即关键点特征）
          2) 动态预测：自回归迭代 num_steps 步，逐步预测未来粒子
          3) 解码：把每一步的预测粒子解码成图像
        本方法挂上钩子后调用 sample_from_x，期间拦截所有 nn.Linear；
        Softmax/GELU/LayerNorm 等隐式算子则靠注意力补丁 + 激活钩子记录。
        参数：x 输入视频张量，num_steps 要预测的步数，cond_steps 条件帧数，
        deterministic 是否确定性推理。返回值：模型输出的预测视频。
        """
        global _rollout_step

        self.install_hooks()
        try:
            with torch.no_grad():
                # The LPWM model internally uses:
                #  - Attention: Q=query, K=key, V=value, proj
                #  - FFN: mlp.fc_1, mlp.proj (SwiGLU)
                #  - c_proj in each transformer block
                #  - Various projection heads
                #  - Softmax in attention (implicit in F.scaled_dot_product_attention)
                #  - GELU in FFN (implicit in F.gelu or nn.GELU)
                #  - LayerNorm/RMSNorm (implicit)
                #
                # We can't easily hook F.softmax or F.gelu, but we can
                # estimate them from the attention shapes.
                # （LPWM 内部结构：注意力 Q/K/V/输出投影、FFN 的 fc_1/proj、
                #  各投影头；Softmax/GELU/LayerNorm 都是隐式的，没法直接挂钩子，
                #  只能靠形状估算或补丁方式记录。）

                # sample_from_x owns the recurrent state; calling it once per
                # step would restart the rollout.  Run the complete horizon in
                # one call and let hooks observe every internal invocation.
                # 注意：sample_from_x 内部自带循环状态，如果一步一调就会不断
                # 重启 rollout。所以必须一次调用跑完整个预测周期，让钩子观察到
                # 内部每一次调用。
                _rollout_step = 0
                return self.model.sample_from_x(
                    x, num_steps=num_steps, deterministic=deterministic,
                    cond_steps=cond_steps, decode=True,
                )

        finally:
            # 无论成功失败都卸载钩子，保证状态干净
            self.remove_hooks()


# ---------------------------------------------------------------------------
# Post-processing: infer dynamic ops from attention architecture
# ---------------------------------------------------------------------------

def inject_dynamic_ops(records: list, config: dict):
    """After hook-based capture, inject dynamic ops that can't be hooked.

    For each attention block in the LPWM PINT, add:
      - Softmax (per attention head)
      - LayerNorm (per block)
      - GELU activation (per FFN block)

    The LPWM architecture (from config):
      - pint_ctx_layers=4 context transformer layers
      - pint_dyn_layers=6 dynamics transformer layers
      - Each has spatio_block + temp_block, with attn + mlp + c_proj

    中文说明：
    这是一个预留的补洞函数：钩子能抓到 Linear，但 Softmax / LayerNorm / GELU
    这类藏在融合算子里的动态算子难以直接挂钩，本函数设想在钩子捕获完成后，
    根据 LPWM 的固定结构（PINT 变换器有 context 4 层、dynamics 6 层，每层包含
    空间块和时间块，各带 attn + mlp + c_proj）把这些动态算子补插进 records。
    目前实现为 pass（空操作），因为注意力部分实际已由 _patch_attention 在
    运行时拆解记录，故暂无必要。
    """
    pass  # The hooks capture Linear layers; dynamic ops are implicit in F.* calls
    # 说明：钩子已抓到 Linear；动态算子藏在 F.* 融合调用里，暂时不需要额外注入。


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export_full_trace(model, config, output_path, num_steps=30, cond_steps=1,
                      input_bits=8, weight_bits=8, output_bits=8,
                      deterministic=True, device="cpu",
                      ckpt_loaded=False, ckpt_hash=None, ckpt_epoch=None):
    """真正执行一次完整的 LPWM 推理并导出算子轨迹。

    中文说明：
    这是本脚本的主流程函数，按 LPWM 推理的三个阶段（编码→动态预测→解码）
    逐步执行：
      1) 造一批随机输入视频张量 x（形状 [1, T, 通道, 高, 宽]）。
      2) 给所有 nn.Linear 挂钩子记录矩阵乘；给 nn.GELU 挂钩子记录激活；
         给注意力模块挂前后钩子维护 _attention_stack。
      3) 在 torch.no_grad() 和依赖传播模式下执行编码、num_steps 步自回归
         动态预测、解码，期间钩子自动把算子一条条记进 _records。
      4) 把所有记录按 JSON Lines 格式写入 output_path，同时写入文件头元信息
         （轨迹版本、模型结构超参、checkpoint 信息、算子总数等）。
    参数：model/config 由 load_lpwm_model 得到；num_steps 预测步数；
    cond_steps 条件帧数；各 *_bits 量化位宽；deterministic 是否确定性推理。
    返回值：记录列表 _records。
    """
    global _order, _rollout_step, _records, _call_index
    global _attention_count, _attention_stack
    # 重置所有全局状态，保证多次调用互不干扰
    _order = 0
    _rollout_step = 0
    _records = []
    _call_index = defaultdict(int)
    _attention_count = 0
    _attention_stack = []

    model.eval()  # 切到评估模式（关闭 dropout 等）
    image_size = config.get("image_size", 64)
    cdim = config.get("ch", 3)
    timestep_horizon = config.get("timestep_horizon", 16)
    cond_steps = cond_steps or 1

    # 造随机输入：T = 条件帧 + 预测帧，模拟"看过 T 帧视频"
    T = cond_steps + num_steps
    x = torch.randn(1, T, cdim, image_size, image_size, device=device)

    # --- Phase 1: Encode cond_steps frames ---
    # --- 阶段 1：编码条件帧 ---
    _rollout_step = 0
    phase_start = _record_dynamic(
        "Phase", "encode_start", [(1, T, cdim, image_size, image_size)],
        [(1,)], M=1, K=T, N=1, module_path="encode",
        input_bits=input_bits,
    )
    # 把"开始编码"标记为 x 的来源，使后续所有算子依赖它
    _tag_sources(x, {phase_start})

    # Install hooks for Linear capture
    # 注册钩子：nn.Linear 记录矩阵乘，nn.GELU 记录激活，
    # 注意力模块记录进栈/出栈（让补丁知道当前模块路径）
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            _call_index[name] = 0

            def make_hook(path):
                def hook(m, inp, out):
                    _call_index[path] += 1
                    _record_linear(path, m, inp, out, input_bits, weight_bits, output_bits)
                return hook
            handles.append(module.register_forward_hook(make_hook(name)))

        elif isinstance(module, nn.GELU):
            def make_activation_hook(path):
                def hook(m, inp, out):
                    _record_activation(path, m, inp, out, input_bits, output_bits)
                return hook
            handles.append(module.register_forward_hook(make_activation_hook(name)))

        # Keep the exact module path available to the fused SDPA wrapper.
        # 注意力模块：进入前把路径压栈，退出后弹栈，供注意力补丁查询
        if "Attention" in module.__class__.__name__:
            def make_attention_pre_hook(path):
                def hook(_module, _inputs):
                    _attention_stack.append(path)
                return hook

            def attention_post_hook(_module, _inputs, output):
                if _attention_stack:
                    _attention_stack.pop()
                return output

            handles.append(module.register_forward_pre_hook(make_attention_pre_hook(name)))
            handles.append(module.register_forward_hook(attention_post_hook))

    # Run the entire workload under tensor-provenance propagation. This works
    # with no_grad and preserves dependencies through view/add/cat operations.
    # 整个推理在"依赖传播模式"下运行：no_grad 下也能工作，且保证依赖信息
    # 穿过 view/add/cat 等操作不丢失。
    with torch.no_grad(), _DependencyPropagationMode():
        # 阶段 1 执行：编码条件帧（前 cond_steps 帧）得到粒子状态
        enc_dict = model.encode_all(x[:, :cond_steps].contiguous(), deterministic=True)

        # --- Phase 2: Dynamics rollout ---
        # --- 阶段 2：动态自回归预测 ---
        # 从编码结果取出各种粒子状态（潜在变量 z、缩放、目标开关、深度、特征…）
        z = enc_dict['z']
        z_scale = enc_dict['z_scale']
        z_obj_on = enc_dict['obj_on']
        z_depth = enc_dict['z_depth']
        z_features = enc_dict['z_features']
        z_bg_features = enc_dict['z_bg_features']
        z_context = enc_dict['z_context']
        z_score = enc_dict.get('z_score', None)
        filter_key = enc_dict.get('z_base_var', None)
        if filter_key is not None:
            filter_key = filter_key.sum(-1)

        # 逐步预测：每一步基于"最新的粒子状态"预测下一步，再把新状态拼回去
        for step in range(num_steps):
            _rollout_step = step + 1  # step 1..N (cond steps = step 0)  # 当前预测步编号（从 1 开始）
            # 调用动态模块推进一步（steps=1），只喂最后一个时刻的状态
            dyn_out = model.dyn_module.sample(
                z[:, -1:], z_scale[:, -1:], z_obj_on[:, -1:],
                z_depth[:, -1:], z_features[:, -1:], z_bg_features[:, -1:],
                z_context, z_score, steps=1, deterministic=deterministic)

            # Next iteration slices only the newly appended state. Retag the
            # concatenated tensors with that segment's producers to avoid a
            # false all-history dependency caused by conservative cat union.
            # 下一步只需要"新拼上的那段"状态。这里把拼接后张量的来源标签重新
            # 标成"新段的来源"，避免因为 cat 的保守并集而产生"依赖全部历史"的
            # 假依赖（否则整个 rollout 会变成一个巨大的串行链）。
            state_pairs = [
                ("z", z), ("z_scale", z_scale), ("z_obj_on", z_obj_on),
                ("z_depth", z_depth), ("z_features", z_features),
                ("z_bg_features", z_bg_features),
            ]
            updated = {}
            for key, previous in state_pairs:
                # 把旧状态和新预测状态在时间维上拼接起来
                value = torch.cat([previous, dyn_out[key]], dim=1)
                # 关键：重贴来源标签，只认新段的来源
                _tag_sources(value, _tensor_sources(dyn_out[key]))
                updated[key] = value
            z = updated["z"]
            z_scale = updated["z_scale"]
            z_obj_on = updated["z_obj_on"]
            z_depth = updated["z_depth"]
            z_features = updated["z_features"]
            z_bg_features = updated["z_bg_features"]
            z_context = dyn_out.get('z_context', z_context)
            z_score = dyn_out.get('z_score', z_score)

        # --- Phase 3: Decode final state ---
        # --- 阶段 3：解码最终的预测粒子为图像 ---
        _rollout_step = num_steps
        decoded = model.decode_all(
            z[:, -1:], z_scale[:, -1:], z_features[:, -1:],
            z_obj_on[:, -1:], z_depth[:, -1:], z_bg_features[:, -1:],
            z_context, filter_key=filter_key)

    # 记录"rollout 完成"标记算子，依赖解码输出
    _record_dynamic("Phase", "rollout_complete",
                    [(1, num_steps)], [(1,)],
                    M=1, K=num_steps, N=1,
                    module_path="decode",
                    dependencies=_tensor_sources(decoded))

    # Remove hooks  # 卸载所有钩子
    for h in handles:
        h.remove()

    # Save trace  # --- 保存轨迹文件 ---
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # 先写文件头：一行 JSON，记录整体元信息（版本、模型结构、checkpoint、总数等）
        f.write(json.dumps({
            "trace_version": "1.1.0",
            "schema_version": "1.1.0",
            "dependency_model": "tensor_producer_dag_v1",
            "config_file": config.get("ds", "unknown"),
            "checkpoint_loaded": ckpt_loaded,
            "checkpoint_hash": ckpt_hash,
            "checkpoint_epoch": ckpt_epoch,
            "total_operators": len(_records),
            "num_steps": num_steps,
            "cond_steps": cond_steps,
            "deterministic": deterministic,
            "pint_dim": config.get("pint_dim", 512),
            "pint_ctx_layers": config.get("pint_ctx_layers", 4),
            "pint_dyn_layers": config.get("pint_dyn_layers", 6),
            "n_kp_enc": config.get("n_kp_enc", 80),
            "n_particles": config.get("n_kp_enc", 80),
            "n_heads": config.get("pint_dyn_heads", 8),
            "exported_at": time.time(),
        }) + "\n")
        # 然后逐条写每个算子的记录（每行一个 JSON）
        for rec in _records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return _records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """命令行入口：解析参数 → 加载模型 → 导出轨迹 → 打印统计。

    中文说明：
    命令行调用流程（对应模块 docstring 里的 Usage）：
      1) 解析命令行参数。
      2) 先补丁注意力函数（必须在导入 LPWM 之前做，否则模型内部用的可能不是
         被替换的那个函数引用，补丁就不生效）。
      3) 把相对路径拼成项目根目录下的绝对路径。
      4) 加载模型（可带 checkpoint）。
      5) 强制所有注意力块使用 F.scaled_dot_product_attention（让补丁能捕获）。
      6) 调用 export_full_trace 执行推理并导出轨迹文件。
      7) 打印统计摘要：算子类型分布、角色分布、阶段分布、光子候选矩阵乘等。
    输出结果文件：--output 指定的 .jsonl 轨迹文件；统计打印到 stderr。
    """
    parser = argparse.ArgumentParser(description="Export LPWM operator trace")
    parser.add_argument("--config", default="third_party/lpwm/configs/bair64.json")
    parser.add_argument("--checkpoint", default="", help="Path to LPWM checkpoint .pth")
    parser.add_argument("--output", default="traces/lpwm_bair64_trace.jsonl")
    parser.add_argument("--horizon", type=int, default=30, help="Rollout steps to predict")
    parser.add_argument("--cond-steps", type=int, default=1, help="Conditioning frames")
    parser.add_argument("--input-bits", type=int, default=8)
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--output-bits", type=int, default=8)
    parser.add_argument("--stochastic", action="store_true", help="Non-deterministic")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    # Patch F.scaled_dot_product_attention BEFORE importing LPWM (Fix 3)
    # 关键顺序：必须先补丁注意力函数再导入 LPWM 模块
    #（否则 LPWM 内部拿到的可能是补丁前的函数引用，捕获会失效）
    orig_sdpa = _patch_attention()

    # 把相对路径统一转换为项目根目录下的绝对路径
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_PROJECT_ROOT, config_path)
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(_PROJECT_ROOT, output_path)
    checkpoint_path = args.checkpoint
    if checkpoint_path and not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(_PROJECT_ROOT, checkpoint_path)

    # 选择运行设备：--cpu 强制 CPU；否则有 CUDA 用 CUDA，都没有则回退 CPU
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}", file=sys.stderr)
    print(f"Config: {config_path}", file=sys.stderr)
    print(f"Horizon: {args.horizon}, Cond steps: {args.cond_steps}", file=sys.stderr)

    # 加载模型和配置
    model, config, ckpt_loaded, ckpt_hash, ckpt_epoch = load_lpwm_model(
        config_path, checkpoint_path, device)

    # Force all attention blocks to use F.scaled_dot_product_attention
    # so our monkey-patch captures QK^T and AV (Fix 3)
    # 强制所有注意力块改用 F.scaled_dot_product_attention，保证补丁能捕获 QK^T 和 AV
    attn_forced = 0
    for m in model.modules():
        if hasattr(m, "torch_attn") and not m.torch_attn:
            m.torch_attn = True
            attn_forced += 1
    print(f"Forced torch_attn=True on {attn_forced} attention blocks", file=sys.stderr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} parameters", file=sys.stderr)

    # 执行完整的"推理 + 轨迹导出"
    records = export_full_trace(
        model, config, output_path,
        num_steps=args.horizon,
        cond_steps=args.cond_steps,
        input_bits=args.input_bits,
        weight_bits=args.weight_bits,
        output_bits=args.output_bits,
        deterministic=not args.stochastic,
        device=device,
        ckpt_loaded=ckpt_loaded,
        ckpt_hash=ckpt_hash,
        ckpt_epoch=ckpt_epoch,
    )

    # Summary  # --- 打印统计摘要 ---
    print(f"\nExported {len(records)} operators to {output_path}", file=sys.stderr)

    from collections import Counter
    # 按算子类型 / 角色 / 阶段 / 预测步 分别计数
    types = Counter(r["op_type"] for r in records)
    roles = Counter(r["op_role"] for r in records)
    phases = Counter(r["phase"] for r in records)
    steps = Counter(r.get("rollout_step", 0) for r in records)

    print(f"\nOp types:", file=sys.stderr)
    for t, c in types.most_common():
        print(f"  {t}: {c}", file=sys.stderr)

    print(f"\nTop roles:", file=sys.stderr)
    for role, count in roles.most_common(15):
        print(f"  {role}: {count}", file=sys.stderr)

    print(f"\nPhases:", file=sys.stderr)
    for p, c in phases.most_common(10):
        print(f"  {p}: {c}", file=sys.stderr)

    print(f"\nRollout steps: {len(steps)} unique steps, range [{min(steps)}..{max(steps)}]", file=sys.stderr)

    n_deps = sum(1 for r in records if r.get("dependencies") and len(r["dependencies"]) > 0)
    print(f"Records with dependencies: {n_deps}/{len(records)}", file=sys.stderr)

    # Unique MKN for photonic candidates
    # 统计"光子候选算子"的独特 (M,K,N) 组合：Linear 权重固定，可上光子阵列
    sig_set = set()
    for r in records:
        if r["op_type"] == "Linear":
            sig_set.add((r["M"], r["K"], r["N"]))
    print(f"\nUnique photonic (M,K,N): {len(sig_set)}", file=sys.stderr)
    # 按计算量（M×K×N）从大到小列出前 10 种，便于看出光子阵列的主要负载
    for sig in sorted(sig_set, key=lambda x: -(x[0]*x[1]*x[2]))[:10]:
        print(f"  M={sig[0]}, K={sig[1]}, N={sig[2]}", file=sys.stderr)


if __name__ == "__main__":
    main()
