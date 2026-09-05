"""
SimPhony backend for the joint simulation framework.

Wraps GemmCostSimulator and exposes three functions:
  - architecture_cost(arch_id)     → ArchitectureCost
  - kernel_cost(request)           → PhotonicKernelCost
  - programming_cost(weight_spec)  → WeightProgrammingCost

Supports both in-process (import) and subprocess (JSON) modes so that
LLMCompass and SimPhony can run in separate Python environments.

中文阅读提示：此处是联合层与 SimPhony 的“翻译器”。上层只给出 GEMM 维度，
本文件返回统一的延迟/能耗字段，不把 SimPhony 的内部对象泄漏给调度器。
"""
# =============================================================================
# 本文件角色一句话：把"光子 GEMM 的请求"翻译成 SimPhony（光子器件级仿真器）
# 能听懂的话，再把 SimPhony 返回的器件级成本转成统一的字段返回给调度器。
# 对外暴露三个函数：
#   architecture_cost(arch_id)   一次性架构成本（面积、器件数、激光功率）
#   kernel_cost(request)         单个光子 GEMM 的成本（时延/能耗分解）
#   programming_cost(weight_spec)全系统权重编程（调谐）的一次性成本
# 支持两种运行模式：
#   in-process（进程内 import，速度快）；
#   subprocess（子进程 JSON 通信，适用于 SimPhony 与主程序在不同 Python
#   环境中运行的情形）。
# =============================================================================

import json
import os
import subprocess
import sys
from typing import Optional

# Ensure SimPhony is on the path for in-process mode
# 把项目根目录下的 SimPhony 加入 Python 搜索路径，进程内模式才能 import
_SIMPHONY_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "SimPhony")
)
if _SIMPHONY_ROOT not in sys.path:
    sys.path.insert(0, _SIMPHONY_ROOT)


class SimPhonyBackend:
    # 注意：它回答“单次光子 GEMM 的器件成本”，不回答权重是否已经调谐到 MRR。
    """Backend that queries SimPhony for photonic kernel costs.

    Parameters
    ----------
    arch_cfg : str
        Path to the hetero-architecture config file.
    device_root : str
        Path to device configuration directory.
    arch_version : str
        Architecture version string.
    sub_arch_name : str
        Name of the sub-architecture to use.
    use_subprocess : bool
        If True, invoke SimPhony via JSON subprocess (isolated env).
        If False, import GemmCostSimulator in-process.
    """
    # 中文说明：光子后端的"翻译器"。请务必区分两个概念：
    #   - kernel_cost 回答"光算一次 GEMM 花多久/耗多少能"（器件行为）；
    #   - 权重是否已在 MRR 上、要不要重新编程，是调度器 + mrr_residency
    #     的事，不在这里回答。

    def __init__(
        self,
        arch_cfg: str = "configs/design/architectures/HPAT_hetero.yml",
        device_root: str = "configs/devices",
        device_cfg_globs: str = "*/*.yml",
        arch_version: str = "v1",
        sub_arch_name: str = "HPAT",
        use_subprocess: bool = False,
    ):
        self._arch_cfg = os.path.join(_SIMPHONY_ROOT, arch_cfg)
        self._device_root = os.path.join(_SIMPHONY_ROOT, device_root)
        self._device_cfg_globs = device_cfg_globs
        self._arch_version = arch_version
        self._sub_arch_name = sub_arch_name
        self._use_subprocess = use_subprocess
        self._simulator = None  # lazy init（首次使用时才真正创建，省启动时间）

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def architecture_cost(self):
        """Return one-time architecture-level costs (area, IL, device counts)."""
        # 中文说明：一次性架构成本。这些数字（面积、插入损耗、激光功率、
        # 各类器件数量）在整个仿真过程中只算一次，与具体算子无关。
        sim = self._get_simulator()
        arch = sim.get_architecture_cost()
        return {
            "pic_area_um2": arch.pic_area_um2,
            "rf_eic_area_um2": arch.rf_eic_area_um2,
            "total_area_um2": arch.total_area_um2,
            "core_insertion_loss_db": arch.core_insertion_loss_db,
            "laser_wall_plug_power_w": arch.laser_wall_plug_power_w,
            "mrr_count": arch.mrr_count,
            "pd_count": arch.pd_count,
            "dac_count": arch.dac_count,
            "adc_count": arch.adc_count,
        }

    def kernel_cost(
        self,
        M: int,
        K: int,
        N: int,
        input_bits: int = 8,
        weight_bits: int = 8,
        output_bits: int = 8,
        dataflow: str = "weight_stationary",
    ) -> dict:
        """Return the photonic cost for a single GEMM.

        Returns a dict matching PhotonicKernelCost fields.
        """
        # 中文说明：单个光子 GEMM 的成本。调用前先校验参数（必须为正整数），
        # 且只支持 weight_stationary 数据流（权重驻留，本项目的校准目标）。
        # 返回 dict 里关键的能耗项：dynamic_energy_j 是总动态能耗，还拆出
        # DAC/ADC/激光/MRR 调谐/MRR 保持等子项；latency 拆成
        # operand_encoding(编码)/compute(计算)/conversion(转换) 三段。
        for name, value in (
            ("M", M), ("K", K), ("N", N),
            ("input_bits", input_bits), ("weight_bits", weight_bits),
            ("output_bits", output_bits),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if dataflow != "weight_stationary":
            raise ValueError(
                f"Unsupported dataflow {dataflow!r}; only 'weight_stationary' is calibrated"
            )
        if self._use_subprocess:
            # 子进程模式：把请求序列化成 JSON 发给 simphony_cli.py
            return self._kernel_cost_subprocess(
                M, K, N, input_bits, weight_bits, output_bits, dataflow
            )
        sim = self._get_simulator()
        cost = sim.simulate_gemm(
            M=M, K=K, N=N,
            input_bits=input_bits,
            weight_bits=weight_bits,
            output_bits=output_bits,
            dataflow=dataflow,
        )
        # 把 SimPhony 的对象字段搬进普通 dict，避免把内部对象泄漏给上层
        return {
            "compute_latency_s": cost.compute_latency_s,
            "operand_encoding_latency_s": cost.operand_encoding_latency_s,
            "conversion_latency_s": cost.conversion_latency_s,
            "programming_latency_s": cost.programming_latency_s,
            "dynamic_energy_j": cost.dynamic_energy_j,
            "dac_energy_j": cost.dac_energy_j,
            "adc_energy_j": cost.adc_energy_j,
            "laser_energy_j": cost.laser_energy_j,
            "mrr_tuning_energy_j": cost.mrr_tuning_energy_j,
            "mrr_hold_energy_j": cost.mrr_hold_energy_j,
            "input_bytes": cost.input_bytes,
            "output_bytes": cost.output_bytes,
            "iter_M": cost.iter_M,
            "iter_K": cost.iter_K,
            "iter_N": cost.iter_N,
            "switching_cycles": cost.switching_cycles,
            "max_cycles": cost.max_cycles,
            "utilization": cost.utilization,
        }

    def programming_cost(self, weight_id: Optional[str] = None) -> dict:
        """Return the one-time cost to program all MRR weights in the system."""
        # 中文说明：把整个系统的权重全部写入（调谐到）MRR 的一次性成本。
        # 注意"一次性"：仿真中实际每次编程的开销由调度器按事件逐次记账，
        # 这个函数给出的是"冷启动时全部配好"的参考值。
        sim = self._get_simulator()
        wp = sim.get_weight_programming_cost()
        return {
            "tile_count": wp.tile_count,
            "programmed_mrr_count": wp.programmed_mrr_count,
            "programming_latency_s": wp.programming_latency_s,
            "programming_energy_j": wp.programming_energy_j,
            "hold_power_w": wp.hold_power_w,
        }

    @property
    def schema_version(self) -> str:
        # 当前 SimPhony 的成本字段格式版本（缓存键的组成部分之一）
        sim = self._get_simulator()
        return sim.schema_version

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_simulator(self):
        # 懒加载：首次调用时切换工作目录到 SimPhony 根目录再 import
        # （SimPhony 的配置是相对路径），用完切回原目录。
        if self._simulator is None:
            import os as _os
            _cwd = _os.getcwd()
            try:
                _os.chdir(_SIMPHONY_ROOT)
                from onnarchsim.gemm_simulator import GemmCostSimulator
                self._simulator = GemmCostSimulator(
                    arch_cfg=self._arch_cfg,
                    device_root=self._device_root,
                    device_cfg_globs=self._device_cfg_globs,
                    arch_version=self._arch_version,
                    sub_arch_name=self._sub_arch_name,
                )
            finally:
                _os.chdir(_cwd)
        return self._simulator

    def _kernel_cost_subprocess(
        self, M, K, N, input_bits, weight_bits, output_bits, dataflow
    ) -> dict:
        """Invoke SimPhony via JSON subprocess (isolated environment)."""
        # 中文说明：子进程模式——把请求写成 JSON 通过标准输入传给
        # cli/simphony_cli.py，从标准输出读 JSON 结果。适用于 SimPhony 与
        # 主程序不在同一个 Python 环境的情况（环境隔离）。
        script = os.path.join(os.path.dirname(__file__), "cli", "simphony_cli.py")
        request = {
            "command": "kernel_cost",
            "M": M, "K": K, "N": N,
            "input_bits": input_bits,
            "weight_bits": weight_bits,
            "output_bits": output_bits,
            "dataflow": dataflow,
            "arch_cfg": self._arch_cfg,
            "device_root": self._device_root,
            "device_cfg_globs": self._device_cfg_globs,
            "arch_version": self._arch_version,
            "sub_arch_name": self._sub_arch_name,
        }
        proc = subprocess.run(
            [sys.executable, script],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"SimPhony subprocess failed (rc={proc.returncode}): {proc.stderr}"
            )
        return json.loads(proc.stdout)
