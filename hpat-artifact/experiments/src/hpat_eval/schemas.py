"""P0 证据等级活动数据（activity）的字段 schema 与校验/归一化工具。

背景：论文把证据分为 P0/P1/P2 三档（P0=模拟器/模型推导证据，
P1=真实硬件实测证据，P2=物理代理/仿真证据）。本文件管的是 P0 档：
模拟器导出的"活动轨迹 CSV"（记录每个模型算子在一颗光子芯片上
执行时用了多少次 DAC/ADC/PD/TIA 采样、激活了多少 MRR 等），
在进入下游（能耗计算、可复现性打包）之前，必须先经过这里的
字段校验与格式归一化，保证数据形状统一、可被后续管线依赖。
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# P0 活动数据 schema：把 CSV 每一行应包含的字段分成三类
# ---------------------------------------------------------------------------
# 字符串字段：文本性质，不能为空。
# 例如 model_variant=模型变体（如 MobileViT-XXS）、
#      layer_id=层编号、op_type=算子类型（如 matmul / attention）。
P0_ACTIVITY_STRING_FIELDS = ["model_variant", "layer_id", "op_type"]

# 整数字段：计数/字节类数值，必须是非负整数。
# 包括 DAC/ADC/PD/TIA 采样次数、激活/编程的 MRR 微环数量、
# 缓冲区与总线读写字节数、非线性算子个数、校准事件次数等。
P0_ACTIVITY_INTEGER_FIELDS = [
    "n_dac_samples",        # DAC（数模转换器）采样次数
    "n_adc_samples",        # ADC（模数转换器）采样次数
    "n_pd_samples",         # PD（光电探测器）采样次数
    "n_tia_samples",        # TIA（跨阻放大器）采样次数
    "n_mrr_active",         # 处于激活工作状态的 MRR 微环数量
    "n_mrr_program",        # 被编程/写入权重的 MRR 微环数量
    "buffer_read_bytes",    # 片上缓冲区读出的字节数
    "buffer_write_bytes",   # 写入片上缓冲区的字节数
    "bus_bytes",            # 总线搬运的字节数
    "nonlinear_ops",        # 非线性算子（如激活函数）执行次数
    "calibration_events",   # 热校准事件次数
]

# 浮点字段：带小数的时间类数值（单位：纳秒）。
# active_optical_time_ns=光路实际参与运算的时长；
# thermal_tracking_time_ns=热跟踪/校准占用的时长。
P0_ACTIVITY_FLOAT_FIELDS = [
    "active_optical_time_ns",
    "thermal_tracking_time_ns",
]

# 必填字段 = 三类字段的总和。校验时若缺少其中任何一个，整行数据作废。
P0_ACTIVITY_REQUIRED_FIELDS = P0_ACTIVITY_STRING_FIELDS + P0_ACTIVITY_INTEGER_FIELDS + P0_ACTIVITY_FLOAT_FIELDS


def _row_label(index: int) -> str:
    """生成第 index 行的人类可读标签，用于错误提示（如 "row 3"）。

    :param index: 行号，从 0 开始。
    :return: 形如 "row N" 的字符串（N 从 1 开始，方便人读）。
    """
    return f"row {index + 1}"


def _as_nonnegative_int(value: Any, *, field: str, label: str) -> int:
    """把一个原始值转成"非负整数"，并给出清晰的中文风格报错。

    作用：统一校验 CSV 里整数字段，把字符串/浮点形式的数字安全转成 int，
    同时杜绝三类非法情况——缺失、负数、带小数的非整数。

    :param value: 原始取值（可能来自 CSV 的字符串，也可能是 None/空串）。
    :param field: 字段名，用于拼错误信息。
    :param label: 行标签（如 "row 1"），用于拼错误信息。
    :return: 转换后的非负整数。
    :raises ValueError: 当值为空、非数字、为负数或非整数时抛出。
    """
    if value in ("", None):
        # 空值 = 该字段缺失，直接判非法
        raise ValueError(f"{label}: {field} is required")
    try:
        numeric = float(value)  # 先统一转 float，兼容 "8" 与 8 两种形态
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: {field} must be numeric, got {value!r}") from exc
    if numeric < 0:
        raise ValueError(f"{label}: {field} must be nonnegative, got {value!r}")
    if not numeric.is_integer():
        # 例：计数类字段不允许出现 3.7 这种非整数值
        raise ValueError(f"{label}: {field} must be an integer count/byte value, got {value!r}")
    return int(numeric)


def _as_nonnegative_float(value: Any, *, field: str, label: str) -> float:
    """把一个原始值转成"非负浮点数"。

    与 _as_nonnegative_int 的区别：这里允许带小数（用于纳秒时间字段）。

    :param value: 原始取值。
    :param field: 字段名，用于拼错误信息。
    :param label: 行标签（如 "row 1"），用于拼错误信息。
    :return: 转换后的非负浮点数。
    :raises ValueError: 当值为空、非数字或为负数时抛出。
    """
    if value in ("", None):
        raise ValueError(f"{label}: {field} is required")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: {field} must be numeric, got {value!r}") from exc
    if numeric < 0:
        raise ValueError(f"{label}: {field} must be nonnegative, got {value!r}")
    return numeric


def normalize_hpat_activity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate the P0 simulator-export schema and return normalized rows."""
    # （英文原注释）校验 P0 模拟器导出数据的 schema，返回归一化后的行列表。
    # 通俗说：这是活动数据进入下游前的"关卡"——先检查表头有没有漏字段，
    # 再逐行把每个字段转成标准类型（字符串去空格、计数转 int、时间格式化），
    # 最后补上 source_operator_group / evidence_label 两个元信息字段。
    if not rows:
        raise ValueError("activity CSV contains no data rows")

    # 只检查第一行的表头字段是否齐全（CSV 表头对所有行一致，检查一行即可）
    missing = [field for field in P0_ACTIVITY_REQUIRED_FIELDS if field not in rows[0]]
    if missing:
        raise ValueError(f"activity CSV missing required columns: {', '.join(missing)}")

    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        label = _row_label(index)  # 生成 "row N" 标签，供错误信息定位行号
        out: dict[str, Any] = {}   # 存放本行归一化后的结果
        # 1) 字符串字段：转 str 并去首尾空格，空字符串视为非法
        for field in P0_ACTIVITY_STRING_FIELDS:
            value = str(row.get(field, "")).strip()
            if not value:
                raise ValueError(f"{label}: {field} must be non-empty")
            out[field] = value
        # 2) 整数字段：必须是非负整数（采样次数、字节数等）
        for field in P0_ACTIVITY_INTEGER_FIELDS:
            out[field] = _as_nonnegative_int(row.get(field), field=field, label=label)
        # 3) 浮点字段：必须是非负浮点，并统一格式化为 6 位小数，
        #    保证下游读取时数值格式一致、可稳定比较/写文件
        for field in P0_ACTIVITY_FLOAT_FIELDS:
            out[field] = f"{_as_nonnegative_float(row.get(field), field=field, label=label):.6f}"
        # 4) 元信息字段：标记本行数据来源（默认是模拟器导出）与证据标签
        out["source_operator_group"] = str(row.get("source_operator_group") or "simulator_export")
        out["evidence_label"] = str(
            row.get("evidence_label")
            or "modelled from simulator activity counts and explicit unit costs"
        )
        normalized.append(out)
    return normalized


P0_ACTIVITY_INTEGER_FIELDS = [
    "n_dac_samples",
    "n_adc_samples",
    "n_pd_samples",
    "n_tia_samples",
    "n_mrr_active",
    "n_mrr_program",
    "buffer_read_bytes",
    "buffer_write_bytes",
    "bus_bytes",
    "nonlinear_ops",
    "calibration_events",
]

P0_ACTIVITY_FLOAT_FIELDS = [
    "active_optical_time_ns",
    "thermal_tracking_time_ns",
]

P0_ACTIVITY_REQUIRED_FIELDS = P0_ACTIVITY_STRING_FIELDS + P0_ACTIVITY_INTEGER_FIELDS + P0_ACTIVITY_FLOAT_FIELDS


def _row_label(index: int) -> str:
    return f"row {index + 1}"


def _as_nonnegative_int(value: Any, *, field: str, label: str) -> int:
    if value in ("", None):
        raise ValueError(f"{label}: {field} is required")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: {field} must be numeric, got {value!r}") from exc
    if numeric < 0:
        raise ValueError(f"{label}: {field} must be nonnegative, got {value!r}")
    if not numeric.is_integer():
        raise ValueError(f"{label}: {field} must be an integer count/byte value, got {value!r}")
    return int(numeric)


def _as_nonnegative_float(value: Any, *, field: str, label: str) -> float:
    if value in ("", None):
        raise ValueError(f"{label}: {field} is required")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: {field} must be numeric, got {value!r}") from exc
    if numeric < 0:
        raise ValueError(f"{label}: {field} must be nonnegative, got {value!r}")
    return numeric


def normalize_hpat_activity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate the P0 simulator-export schema and return normalized rows."""
    if not rows:
        raise ValueError("activity CSV contains no data rows")

    missing = [field for field in P0_ACTIVITY_REQUIRED_FIELDS if field not in rows[0]]
    if missing:
        raise ValueError(f"activity CSV missing required columns: {', '.join(missing)}")

    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        label = _row_label(index)
        out: dict[str, Any] = {}
        for field in P0_ACTIVITY_STRING_FIELDS:
            value = str(row.get(field, "")).strip()
            if not value:
                raise ValueError(f"{label}: {field} must be non-empty")
            out[field] = value
        for field in P0_ACTIVITY_INTEGER_FIELDS:
            out[field] = _as_nonnegative_int(row.get(field), field=field, label=label)
        for field in P0_ACTIVITY_FLOAT_FIELDS:
            out[field] = f"{_as_nonnegative_float(row.get(field), field=field, label=label):.6f}"
        out["source_operator_group"] = str(row.get("source_operator_group") or "simulator_export")
        out["evidence_label"] = str(
            row.get("evidence_label")
            or "modelled from simulator activity counts and explicit unit costs"
        )
        normalized.append(out)
    return normalized
