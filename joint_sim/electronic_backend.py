"""非光子算子的 Roofline 时延与能耗模型。

中文阅读提示：当前主路径是 roofline 近似，不是完整 GPU/NPU 微架构仿真。
它的任务是让未映射到光子阵列的算子也进入同一条系统时间线。
"""
# =============================================================================
# 本文件角色一句话：给"走电子后端的算子"（MatMul/Softmax/GELU/LayerNorm 等
# 动态算子）估计时延与能耗。
# 时延用 roofline 模型：把"计算时间"和"访存时间"都算出来，取较大者——
# 类比：一道菜能不能上桌，取决于"锅(算力)"和"备菜(带宽)"哪个先到瓶颈。
#   计算时间 = 浮点运算量 / (每周期浮点次数 x 时钟频率)
#   访存时间 = 读写字节数 / 带宽
# 能耗优先复用 LLMCompass 的 ElectronicEnergyModel；若该模型不可用则返回 0，
# 并给出警告（避免悄悄漏计）。
# =============================================================================

import math
import os
import sys
import warnings


def _llmcompass_energy_model():
    """Load the nominal LLMCompass energy model when it is available."""
    # 中文说明：尝试把 LLMCompass 的电子能耗模型加载进来（用于算能耗）。
    # LLMCompass 是本项目配套的电子侧仿真器；找不到时返回 None 并告警，
    # 后续 operator_energy_j 会按"能耗未知=0"处理（显式降级，不静默出错）。
    try:
        llmc_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "LLMCompass")
        )
        if llmc_path not in sys.path:
            sys.path.insert(0, llmc_path)
        from hardware_model.energy_model import ElectronicEnergyModel
        return ElectronicEnergyModel.nominal()
    except (ImportError, ModuleNotFoundError) as exc:
        warnings.warn(
            f"LLMCompass energy model unavailable; electronic energy is zero: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


class ElectronicBackend:
    # 电子侧与光子侧共享同一份 trace，但使用完全不同的成本公式。
    """Electronic backend using a compute/memory roofline.

    ``flops_per_cycle`` describes the complete electronic accelerator, not a
    single ALU. Vector kernels use 10% of peak throughput and are also bounded
    by HBM traffic.
    """
    # 中文说明：电子后端。注意几个易混淆点：
    #   - flops_per_cycle 描述"整台电子加速器"的吞吐，不是单个 ALU；
    #   - 向量类算子（Softmax/GELU/LayerNorm）假设只能用到峰值吞吐的 10%，
    #     同时受 HBM 带宽限制（访存密集）；
    #   - 单位：时延一律秒(s)，能耗一律焦耳(J)，带宽是字节/秒(B/s)。
    # 默认参数大致对应一块高端 HBM GPU：1.3 GHz、312 TFLOPS、2 TB/s。

    def __init__(
        self,
        device=None,
        energy_model=None,
        clock_freq_hz: float = 1.3e9,
        flops_per_cycle: float = 312e12 / 1.3e9,
        bandwidth_bytes_per_s: float = 2.0e12,
    ):
        # 参数校验：时钟/算力/带宽必须为正的有限数
        for name, value in (
            ("clock_freq_hz", clock_freq_hz),
            ("flops_per_cycle", flops_per_cycle),
            ("bandwidth_bytes_per_s", bandwidth_bytes_per_s),
        ):
            if value <= 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and positive")

        self.device = device
        self.energy_model = (
            energy_model if energy_model is not None else _llmcompass_energy_model()
        )
        self.clock_freq_hz = clock_freq_hz
        self.flops_per_cycle = flops_per_cycle
        self.bandwidth_bytes_per_s = bandwidth_bytes_per_s

        # 若传入真实设备对象，尝试从其模块配置里读取更准确的时钟/带宽
        if device is not None:
            try:
                self.clock_freq_hz = device.compute_module.clock_freq
            except AttributeError:
                pass
            try:
                self.bandwidth_bytes_per_s = device.io_module.bandwidth
            except AttributeError:
                pass

    @staticmethod
    def _positive(rec: dict, name: str, default: int = 1):
        # 从算子记录里取一个必须为正数的字段；缺失用默认值，非法值报错
        value = rec.get(name, default) or default
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be positive for {rec.get('op_id')}")
        return value

    def matmul_latency_s(
        self,
        M: int,
        K: int,
        N: int,
        input_bits: int = 16,
        weight_bits: int = 16,
        output_bits: int = 16,
        repetitions: int = 1,
    ) -> float:
        """Roofline latency for one or more independent matrix products."""
        # 中文说明：矩阵乘时延的 roofline 公式。
        # 浮点运算量：一次 [M,K]x[K,N] 乘加约 2*M*K*N 次 FLOP（乘和加各一次），
        # repetitions 是批重复次数（一次调用里做多少遍）。
        flop_count = 2 * M * K * N * repetitions
        # 访存字节数（位转字节要除以 8）：读输入(M*K)、读权重(K*N)、写输出(M*N)
        io_bits = repetitions * (
            M * K * input_bits + K * N * weight_bits + M * N * output_bits
        )
        # 计算时间 = FLOP / (每周期FLOP x 时钟频率)
        compute_time = flop_count / (self.flops_per_cycle * self.clock_freq_hz)
        # 访存时间 = 字节数 / 带宽
        memory_time = (io_bits / 8.0) / self.bandwidth_bytes_per_s
        # roofline 取两者较大者：算力和带宽谁先耗尽，谁决定时延
        return max(compute_time, memory_time)

    def _vector_latency_s(
        self,
        elements: int,
        ops_per_element: int,
        input_bits: int,
        output_bits: int,
    ) -> float:
        # 中文说明：向量类算子（逐元素操作）的统一时延公式。
        # 向量核假设只用 10% 峰值吞吐（访存密集/低 ILP），因此乘 0.1。
        compute_time = (
            elements * ops_per_element
            / (self.flops_per_cycle * 0.1 * self.clock_freq_hz)
        )
        io_bytes = elements * (input_bits + output_bits) / 8.0
        return max(compute_time, io_bytes / self.bandwidth_bytes_per_s)

    def softmax_latency_s(
        self,
        rows: int,
        row_length: int,
        input_bits: int = 16,
        output_bits: int = 16,
    ) -> float:
        # 每个元素约 5 次操作（指数、求和、归一化等）
        return self._vector_latency_s(
            rows * row_length, 5, input_bits, output_bits
        )

    def layernorm_latency_s(
        self,
        M: int,
        N: int,
        input_bits: int = 16,
        output_bits: int = 16,
    ) -> float:
        # 每个元素约 6 次操作（均值、方差、归一化等）
        return self._vector_latency_s(M * N, 6, input_bits, output_bits)

    def gelu_latency_s(
        self,
        M: int,
        N: int,
        input_bits: int = 16,
        output_bits: int = 16,
    ) -> float:
        # GELU 激活每个元素约 8 次操作
        return self._vector_latency_s(M * N, 8, input_bits, output_bits)

    def operator_latency_s(self, rec: dict) -> float:
        """Route a trace record to its electronic cost model."""
        # 中文说明：按算子类型分发到对应的时延模型。Phase 是阶段标记
        # （不代表实际计算），时延记 0。
        op_type = rec.get("op_type", "")
        if op_type == "Phase":
            return 0.0

        M = self._positive(rec, "M")
        K = self._positive(rec, "K")
        N = self._positive(rec, "N")
        input_bits = self._positive(rec, "input_bits", 16)
        weight_bits = rec.get("weight_bits") or input_bits
        output_bits = self._positive(rec, "output_bits", 16)
        repetitions = self._positive(rec, "batch_repetitions")

        if op_type in ("Linear", "MatMul", "Conv2d"):
            return self.matmul_latency_s(
                M, K, N, input_bits, weight_bits, output_bits, repetitions
            )
        if op_type == "Softmax":
            # In trace schema v1.1, M is the number of rows (already including
            # batch and heads) and K is the reduction length.
            # trace 里 M=行数（已含批和注意力头），K=归一化方向的长度
            return self.softmax_latency_s(M, K, input_bits, output_bits)
        if op_type in ("LayerNorm", "RMSNorm"):
            return self.layernorm_latency_s(M, N, input_bits, output_bits)
        if op_type == "GELU":
            return self.gelu_latency_s(M, N, input_bits, output_bits)
        raise ValueError(f"Unsupported electronic operator type: {op_type!r}")

    def operator_energy_j(self, rec: dict, latency_s: float) -> float:
        """Estimate dynamic compute, HBM traffic, and chip-static energy."""
        # 中文说明：电子算子能耗 = 动态计算能耗 + HBM 读写能耗 + 芯片静态能耗。
        # 三项都由 LLMCompass 的 ElectronicEnergyModel 提供；模型不可用时
        # 返回 0（已在 _llmcompass_energy_model 里给出过警告）。
        if self.energy_model is None:
            return 0.0
        op_type = rec.get("op_type", "")
        if op_type == "Phase":
            return 0.0

        M = self._positive(rec, "M")
        K = self._positive(rec, "K")
        N = self._positive(rec, "N")
        input_bits = self._positive(rec, "input_bits", 16)
        weight_bits = rec.get("weight_bits") or input_bits
        output_bits = self._positive(rec, "output_bits", 16)
        repetitions = self._positive(rec, "batch_repetitions")

        # 各算子类型：算出 MAC 数/元素数（动态能耗的依据）和访存字节数
        if op_type in ("Linear", "MatMul", "Conv2d"):
            mac_count = M * K * N * repetitions
            io_bytes = repetitions * (
                M * K * input_bits + K * N * weight_bits + M * N * output_bits
            ) / 8.0
            dynamic = self.energy_model.matmul_energy(mac_count)
        elif op_type == "Softmax":
            elements, ops_per_element = M * K, 5
            io_bytes = elements * (input_bits + output_bits) / 8.0
            dynamic = self.energy_model.vector_energy(elements * ops_per_element)
        elif op_type in ("LayerNorm", "RMSNorm"):
            elements, ops_per_element = M * N, 6
            io_bytes = elements * (input_bits + output_bits) / 8.0
            dynamic = self.energy_model.vector_energy(elements * ops_per_element)
        elif op_type == "GELU":
            elements, ops_per_element = M * N, 8
            io_bytes = elements * (input_bits + output_bits) / 8.0
            dynamic = self.energy_model.vector_energy(elements * ops_per_element)
        else:
            raise ValueError(f"Unsupported electronic operator type: {op_type!r}")

        # 总能耗 = 动态 + HBM 读写（按字节）+ 静态（按时延）
        return (
            dynamic
            + self.energy_model.hbm_read_energy(io_bytes)
            + self.energy_model.static_energy(latency_s)
        )

    def summary(self) -> dict:
        """返回电子后端的配置摘要（时钟、算力、带宽、峰值 TFLOPS）。"""
        return {
            "clock_freq_hz": self.clock_freq_hz,
            "flops_per_cycle": self.flops_per_cycle,
            "bandwidth_bytes_per_s": self.bandwidth_bytes_per_s,
            "peak_tflops": self.flops_per_cycle * self.clock_freq_hz / 1e12,
        }
