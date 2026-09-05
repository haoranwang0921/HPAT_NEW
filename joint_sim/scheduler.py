"""
Hybrid event scheduler with DAG + resource pool (Experiment 4, Section 7.2-7.6).

Key changes from the prototype:
  - Tile-granularity weight lookup (via SramManager)
  - Non-zero DMA latency with configurable bandwidth
  - Program lanes with configurable parallelism (P_prog)
  - DAG-based dependency scheduling
  - Per-resource ready-time tracking (96 cores, DACs, ADCs, lanes, HBM)
  - Pipeline modes: serial / dma_program_overlap / triple_pipeline
  - Resource conservation checks

中文阅读提示：这是系统仿真的主时钟。它不执行神经网络计算，而是把每个算子
拆成 DMA、写权重、DAC、光计算、ADC 等事件，并放到可争用的硬件资源上。
"""
# =============================================================================
# 本文件是 joint_sim 的"主调度器"，角色一句话：把算子 trace 排成一条真实
# 可执行的时间线，模拟各硬件资源（光子核心、DAC/ADC 通道、编程通道、HBM、
# 电子核心）之间的争用，最终算出端到端时延和各类能耗事件。
# 两个核心类：
#   ResourcePool        硬件资源池：记录每个资源"下一次空闲"的时刻，
#                       每次请求分配最早空闲的那个实例。
#   ResourceScheduler   调度器：按 DAG 依赖 + 流水线模式把每个算子拆成
#                       若干事件放到资源池上执行。
# 三种流水线模式：
#   serial                 全串行：同一时刻只能有一个算子在光路里跑
#   dma_program_overlap    允许权重搬运(DMA)与编程重叠，但光路仍是一条
#   triple_pipeline        三重流水：DMA/编程/计算三个阶段各自独立重叠
# =============================================================================

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict
import math

try:
    from .mrr_residency import MrrResidencyManager
except ImportError:  # direct CLI/module execution
    from mrr_residency import MrrResidencyManager


class EventType(Enum):
    # 时间线中的原子动作。部分类型为后续扩展预留，未必在每次运行中都会出现。
    # 中文说明：仿真时间线上的"最小动作"类型。一个算子会被拆成若干事件：
    #   WEIGHT_DMA       把权重从 HBM 搬到 SRAM
    #   WEIGHT_PROGRAM   把权重写入（调谐到）MRR 微环
    #   ACTIVATION_DMA   把激活值从 HBM 搬到片上
    #   DAC_ENCODE       把数字输入编码成模拟光信号（电转光）
    #   PHOTONIC_COMPUTE 光阵列上的矩阵乘法（光路计算）
    #   ADC_DECODE       把模拟光信号转回数字（光转电）
    #   ELECTRONIC_COMPUTE 电子核心上的计算（动态算子）
    #   PARTIAL_SUM_REDUCE 部分和累加（多核心拼结果时）
    #   SAMPLE / CALIBRATE 采样 / 校准（防漂移）
    WEIGHT_DMA = "weight_dma"
    WEIGHT_PROGRAM = "weight_program"
    ACTIVATION_DMA = "activation_dma"
    DAC_ENCODE = "dac_encode"
    PHOTONIC_COMPUTE = "photonic_compute"
    ADC_DECODE = "adc_decode"
    ELECTRONIC_COMPUTE = "electronic_compute"
    PARTIAL_SUM_REDUCE = "partial_sum_reduce"
    SAMPLE = "sample"
    CALIBRATE = "calibrate"


@dataclass
class Event:
    """A single event on the execution timeline."""
    # 中文说明：时间线上的一个事件。字段：
    #   event_type    事件类型（见 EventType）
    #   op_id         属于哪个算子
    #   start_time_s  开始时刻（秒）
    #   duration_s    持续时长（秒）
    #   resource      占用哪个资源（如 "photonic_core_0"、"program_lane_1"）
    #   energy_j      该事件消耗的动态能量（焦耳）
    #   metadata      附加信息（tile_id、被踢掉的 tile 等）
    event_type: EventType
    op_id: str
    start_time_s: float
    duration_s: float
    resource: str = "default"
    energy_j: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        # 初始化后自动校验时间与能量的合法性（非法值说明账本有 bug），
        # 并顺手算出结束时刻 end_time_s = 开始 + 持续。
        if not math.isfinite(self.start_time_s) or self.start_time_s < 0:
            raise ValueError("event start_time_s must be finite and non-negative")
        if not math.isfinite(self.duration_s) or self.duration_s < 0:
            raise ValueError("event duration_s must be finite and non-negative")
        if not math.isfinite(self.energy_j) or self.energy_j < 0:
            raise ValueError("event energy_j must be finite and non-negative")
        self.end_time_s = self.start_time_s + self.duration_s


class ResourcePool:
    # 用每个资源“下一次可用时间”代替复杂的周期级波形，是本项目的离散事件抽象。
    """Tracks availability of all hardware resources.

    Each resource type has N independent instances, each with its own
    ready_time. The pool allocates the earliest-available instance for
    each request.
    """
    # 中文说明：硬件资源池。每类资源有若干独立实例，每个实例记录自己的
    # "下次空闲时刻"（ready_time）。请求某类资源时，取"最早空闲"的那个实例，
    # 占用后把它推进到"占用结束时刻"。这样所有资源争用都反映在时间线上。
    # 类比：就像打印机共享——先排队先打印，谁最早空闲谁接活。

    def __init__(
        self,
        num_photonic_cores: int = 1,
        num_dac_channels: int = 1,
        num_adc_channels: int = 1,
        num_program_lanes: int = 1,
        hbm_bandwidth_bytes_per_s: float = 512e9,
        dma_fixed_latency_s: float = 100e-9,
        weight_tile_bytes: int = 4096,
    ):
        # 注意：这些实例代表"整机级别的执行引擎/流水级"。SimPhony 的
        # 内核成本已经用满 96 个 HPAT 核心，所以默认 1 个光子引擎、1 个
        # DAC 级、1 个 ADC 级；数值 >1 表示仿真"复制整套加速器"的场景。
        for name, value in (
            ("num_photonic_cores", num_photonic_cores),
            ("num_dac_channels", num_dac_channels),
            ("num_adc_channels", num_adc_channels),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if num_program_lanes <= 0:
            raise ValueError("num_program_lanes must be positive")
        if hbm_bandwidth_bytes_per_s <= 0:
            raise ValueError("hbm_bandwidth_bytes_per_s must be positive")
        if dma_fixed_latency_s < 0:
            raise ValueError("dma_fixed_latency_s must be non-negative")
        if weight_tile_bytes <= 0:
            raise ValueError("weight_tile_bytes must be positive")

        # These instances are full-architecture execution engines/stages.
        # A SimPhony kernel cost already uses all 96 HPAT cores, so the normal
        # configuration has one photonic engine, one DAC stage and one ADC
        # stage. Values >1 model replicated complete accelerators.
        self._photonic_ready: List[float] = [0.0] * num_photonic_cores
        self._dac_ready: List[float] = [0.0] * num_dac_channels
        self._adc_ready: List[float] = [0.0] * num_adc_channels
        self._program_lane_ready: List[float] = [0.0] * num_program_lanes
        self._hbm_ready: float = 0.0
        self._electronic_ready: float = 0.0

        self.hbm_bandwidth = hbm_bandwidth_bytes_per_s
        self.dma_fixed_latency = dma_fixed_latency_s
        self.weight_tile_bytes = weight_tile_bytes

        # Store capacities for queries
        self.num_photonic_cores = num_photonic_cores
        self.num_dac_channels = num_dac_channels
        self.num_adc_channels = num_adc_channels
        self.num_program_lanes = num_program_lanes

        # Allocation counters for conservation checks
        self._allocations: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Resource acquisition
    # ------------------------------------------------------------------

    def acquire_dma(self, earliest_s: float) -> Tuple[float, float]:
        """Acquire HBM DMA for one tile transfer.

        Returns (start_time_s, latency_s).
        """
        # 中文说明：申请一次 HBM DMA（把一块 tile 从 HBM 搬到 SRAM）。
        # 时延 = 传输时间（tile 字节数/带宽）+ 固定开销；
        # 开始时刻 = max(请求的最早时刻, HBM 上次空闲时刻)（串行排队）。
        if not math.isfinite(earliest_s) or earliest_s < 0:
            raise ValueError("DMA earliest time must be finite and non-negative")
        lat = self.weight_tile_bytes / self.hbm_bandwidth + self.dma_fixed_latency
        start = max(earliest_s, self._hbm_ready)
        self._hbm_ready = start + lat
        self._allocations["dma"] += 1
        return start, lat

    def acquire_program_lane(self, earliest_s: float, prog_latency_s: float) -> Tuple[float, int]:
        """Acquire the earliest-available program lane.

        Returns (start_time_s, lane_index).
        """
        # 中文说明：申请一条编程通道（把权重写进 MRR 用）。从全部通道里
        # 挑"最早空闲"的，返回开始时刻和通道编号（事件里要用编号标注资源）。
        self._validate_request(earliest_s, prog_latency_s, "program")
        lane_idx = min(range(len(self._program_lane_ready)),
                       key=lambda i: self._program_lane_ready[i])
        start = max(earliest_s, self._program_lane_ready[lane_idx])
        self._program_lane_ready[lane_idx] = start + prog_latency_s
        self._allocations["program"] += 1
        return start, lane_idx

    def acquire_photonic_core(self, earliest_s: float, duration_s: float) -> Tuple[float, int]:
        """Acquire the earliest-available photonic core.

        Returns (start_time_s, core_index).
        """
        # 中文说明：申请光子计算核心（光路矩阵乘法）。
        self._validate_request(earliest_s, duration_s, "photonic")
        if not self._photonic_ready:
            raise RuntimeError("no photonic execution engine is configured")
        core_idx = min(range(len(self._photonic_ready)),
                       key=lambda i: self._photonic_ready[i])
        start = max(earliest_s, self._photonic_ready[core_idx])
        self._photonic_ready[core_idx] = start + duration_s
        self._allocations["photonic_compute"] += 1
        return start, core_idx

    def acquire_dac(self, earliest_s: float, duration_s: float) -> Tuple[float, int]:
        """Acquire a DAC channel."""
        # 中文说明：申请一个 DAC（数模转换）通道，把数字输入编码成光信号。
        self._validate_request(earliest_s, duration_s, "DAC")
        if not self._dac_ready:
            raise RuntimeError("no DAC stage is configured")
        idx = min(range(len(self._dac_ready)),
                  key=lambda i: self._dac_ready[i])
        start = max(earliest_s, self._dac_ready[idx])
        self._dac_ready[idx] = start + duration_s
        return start, idx

    def acquire_adc(self, earliest_s: float, duration_s: float) -> Tuple[float, int]:
        """Acquire an ADC channel."""
        # 中文说明：申请一个 ADC（模数转换）通道，把光结果转回数字。
        self._validate_request(earliest_s, duration_s, "ADC")
        if not self._adc_ready:
            raise RuntimeError("no ADC stage is configured")
        idx = min(range(len(self._adc_ready)),
                  key=lambda i: self._adc_ready[i])
        start = max(earliest_s, self._adc_ready[idx])
        self._adc_ready[idx] = start + duration_s
        return start, idx

    def acquire_electronic(self, earliest_s: float, duration_s: float) -> float:
        """Acquire the electronic compute core."""
        # 中文说明：申请电子计算核心（动态算子如 MatMul/Softmax 等用）。
        self._validate_request(earliest_s, duration_s, "electronic")
        start = max(earliest_s, self._electronic_ready)
        self._electronic_ready = start + duration_s
        self._allocations["electronic_compute"] += 1
        return start

    @staticmethod
    def _validate_request(earliest_s: float, duration_s: float, resource: str) -> None:
        # 通用参数校验：开始时刻与时长都必须有限且非负
        if not math.isfinite(earliest_s) or earliest_s < 0:
            raise ValueError(f"{resource} earliest time must be finite and non-negative")
        if not math.isfinite(duration_s) or duration_s < 0:
            raise ValueError(f"{resource} duration must be finite and non-negative")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def max_time(self) -> float:
        """Latest ready time across all resources (E2E latency)."""
        # 中文说明：取所有资源"最后一次空闲"中的最晚值 = 系统端到端时延。
        # 只要还有任何资源没忙完，整条流水就不能算结束。
        return max(
            max(self._photonic_ready, default=0.0),
            max(self._dac_ready, default=0.0),
            max(self._adc_ready, default=0.0),
            max(self._program_lane_ready, default=0.0),
            self._hbm_ready,
            self._electronic_ready,
        )

    def reset(self):
        """把所有资源重置到 0 时刻空闲，清空分配计数（新一轮仿真）。"""
        self._photonic_ready = [0.0] * self.num_photonic_cores
        self._dac_ready = [0.0] * self.num_dac_channels
        self._adc_ready = [0.0] * self.num_adc_channels
        self._program_lane_ready = [0.0] * self.num_program_lanes
        self._hbm_ready = 0.0
        self._electronic_ready = 0.0
        self._allocations = defaultdict(int)


class ResourceScheduler:
    # 先满足 DAG 前驱，再申请资源；因此结果同时反映数据依赖和硬件争用。
    """Schedules operator trace onto a resource pool with DAG dependencies.

    Parameters
    ----------
    resource_pool : ResourcePool
    electronic_latency_fn : callable
        Function(rec) → latency_s for electronic operators.
    photonic_latency_fn : callable
        Function(rec) → latency_s for the photonic compute kernel.
    pipeline_mode : str
        "serial" | "dma_program_overlap" | "triple_pipeline"
    """
    # 中文说明：联合调度的主类。它把 trace 里的每个算子：
    #   1) 先检查 DAG 依赖（所有前驱都完成才能开始）；
    #   2) 分类：光子算子（固定权重 Linear）拆成
    #      DMA搬权重 -> 编程调谐MRR -> DAC -> 光计算 -> ADC 一串事件；
    #      电子算子（动态算子）直接在电子核心上算；
    #   3) 所有事件放到 ResourcePool 上，按资源争用推进时间。
    # 结果 self.events 就是完整时间线，供 report 输出和守恒性检查。

    def __init__(
        self,
        resource_pool: ResourcePool = None,
        electronic_latency_fn: Callable = None,
        photonic_latency_fn: Callable = None,
        pipeline_mode: str = "serial",
        program_response_time_s: float = 1000e-9,
        program_energy_j_per_tile: float = 0.093e-3 * 1000e-9 * 64 * 64,
        sram_manager=None,
        hbm_bandwidth_bytes_per_s: float = 512e9,
        mrr_program_mode: str = "reprogram_each_use",
        mrr_capacity_tiles: Optional[int] = None,
        mrr_refresh_interval_s: Optional[float] = None,
    ):
        # 说明几个默认值的来源：
        #   program_response_time_s = 1000e-9：每个 tile 编程（调谐）耗 1 us
        #   program_energy_j_per_tile = 0.093e-3 W * 1e-6 s * 64 * 64：
        #     按每微环 0.093 mW 保持功率、4096 个微环算出的编程能耗
        #   mrr_program_mode 三种模式：
        #     reprogram_each_use            每次用都重新编程（无驻留）
        #     program_on_miss               只在未命中时编程（LRU 驻留）
        #     program_on_miss_with_refresh  驻留 + 周期刷新（防漂移）
        self.pool = resource_pool or ResourcePool()
        self._electronic_latency = electronic_latency_fn or self._missing_electronic_cost
        self._photonic_latency = photonic_latency_fn or self._missing_photonic_cost
        valid_pipeline_modes = {"serial", "dma_program_overlap", "triple_pipeline"}
        if pipeline_mode not in valid_pipeline_modes:
            raise ValueError(
                f"Unknown pipeline_mode={pipeline_mode!r}; "
                f"expected one of {sorted(valid_pipeline_modes)}"
            )
        self.pipeline_mode = pipeline_mode
        self._program_response_time_s = program_response_time_s
        self._program_energy_j_per_tile = program_energy_j_per_tile

        # Tile SRAM mode
        self._sram_manager = sram_manager
        self._hbm_bandwidth = hbm_bandwidth_bytes_per_s
        valid_mrr_modes = {
            "reprogram_each_use",
            "program_on_miss",
            "program_on_miss_with_refresh",
        }
        if mrr_program_mode not in valid_mrr_modes:
            raise ValueError(
                f"Unknown mrr_program_mode={mrr_program_mode!r}; "
                f"expected one of {sorted(valid_mrr_modes)}"
            )
        if mrr_program_mode == "program_on_miss_with_refresh" and not mrr_refresh_interval_s:
            raise ValueError(
                "program_on_miss_with_refresh requires mrr_refresh_interval_s"
            )
        self.mrr_program_mode = mrr_program_mode
        # 只要不是"每次重编程"，就建一个 MRR 驻留管理器（默认 96 个槽位）
        self._mrr_manager = (
            MrrResidencyManager(
                capacity_tiles=(
                    mrr_capacity_tiles
                    if mrr_capacity_tiles is not None
                    else 96
                ),
                refresh_interval_s=(
                    mrr_refresh_interval_s
                    if mrr_program_mode == "program_on_miss_with_refresh"
                    else None
                ),
            )
            if mrr_program_mode != "reprogram_each_use"
            else None
        )

        # 本轮仿真的结果容器：
        #   events         全部时间线事件
        #   _op_complete   每个算子完成的时刻（DAG 依赖判断用）
        #   _tile_last_accessed 每个 tile 上次被访问的时刻
        #   _records_by_id 按 op_id 索引的算子记录
        self.events: List[Event] = []
        self._op_complete: Dict[str, float] = {}
        self._tile_last_accessed: Dict[str, float] = {}
        self._records_by_id: Dict[str, dict] = {}

        # Phase tracking
        self._current_phase: str = ""
        # Stage barriers used to distinguish the three Experiment-4 modes.
        # ``serial`` has one end-to-end barrier; ``dma_program_overlap``
        # allows weight supply to overlap but keeps one optical pipeline;
        # ``triple_pipeline`` relies on the per-resource pools for overlap.
        # 两种"流水线闸门"：
        #   _serial_pipeline_ready 串行模式下，上一算子结束才能开始下一算子
        #   _optical_pipeline_ready 光路（DAC->计算->ADC）同一时刻只跑一个算子
        self._serial_pipeline_ready: float = 0.0
        self._optical_pipeline_ready: float = 0.0

        # Per-step latency tracking
        self._step_latency: Dict[int, float] = {}

        # Streaming stats
        # 流式统计：编程次数/时延/能耗、DMA 时延/字节（能耗账本的输入之一）
        self._streaming_stats = {
            "programs": 0,
            "programming_latency_s": 0.0,
            "programming_energy_j": 0.0,
            "dma_latency_s": 0.0,
            "dma_bytes": 0,
        }

    @staticmethod
    def _missing_electronic_cost(rec: dict):
        # 兜底函数：没配置电子成本模型时直接报错，防止"静默用 0"掩盖错误
        raise RuntimeError(
            f"No electronic latency model configured for {rec.get('op_id', 'unknown')}"
        )

    @staticmethod
    def _missing_photonic_cost(rec: dict):
        # 兜底函数：没配置光子成本模型时直接报错
        raise RuntimeError(
            f"No photonic cost model configured for {rec.get('op_id', 'unknown')}"
        )

    def _photonic_stage_latencies(self, rec: dict) -> Tuple[float, float, float]:
        """Return aggregate-architecture DAC, compute and ADC stage latencies.

        Modern callbacks return the complete SimPhony cost dictionary. Scalar
        callbacks remain supported for old tests and are split 10/60/30.
        """
        # 中文说明：把光子内核成本拆成三段流水时延：
        #   DAC（数字->光编码）、计算（光路矩阵乘）、ADC（光->数字）。
        # 新版回调返回完整 dict（取三个字段）；旧版返回一个标量总时延，
        # 按 10%/60%/30% 经验比例拆分（保持旧测试兼容）。
        cost = self._photonic_latency(rec)
        if isinstance(cost, dict):
            stages = (
                float(cost.get("operand_encoding_latency_s", 0.0)),
                float(cost.get("compute_latency_s", 0.0)),
                float(cost.get("conversion_latency_s", 0.0)),
            )
        else:
            total = float(cost)
            stages = (total * 0.1, total * 0.6, total * 0.3)
        if any(not math.isfinite(value) or value < 0 for value in stages):
            raise ValueError(
                f"Invalid photonic stage latency for {rec.get('op_id', 'unknown')}: {stages}"
            )
        if sum(stages) <= 0:
            raise ValueError(
                f"Photonic latency must be positive for {rec.get('op_id', 'unknown')}"
            )
        return stages

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def schedule_trace(
        self,
        records: List[dict],
        classifier_fn: Callable = None,
        tile_mapper=None,
        tile_accesses=None,
    ):
        """Schedule all operators from a trace onto the resource pool.

        Parameters
        ----------
        records : list of dict
            Operator records sorted by dependency order.
        classifier_fn : callable, optional
            Function(rec) → {"eligible": bool, "reason": str}.
        tile_mapper : TileMapper, optional
            For tile-level weight lookup. If None, falls back to weight-level.
        tile_accesses : list of TileAccess, optional
            Pre-built tile trace with future-use metadata.
        """
        # 中文说明：整条 trace 的调度入口。步骤：
        #   0) 清空上一轮的状态（事件、依赖表、资源池等），准备新的一轮；
        #   1) 按 order 排序，校验 op_id 无重复；
        #   2) 逐算子：先验证所有 DAG 依赖已完成，再分类（光子/电子），
        #      遇到阶段切换时把已结束阶段的权重从 SRAM/MRR 里清出去；
        #   3) 逐算子记录"该 rollout step 末尾的时刻"（per-step 时延）。
        self.events = []
        self._op_complete = {}
        self._tile_last_accessed = {}
        self._records_by_id = {}
        self._step_latency = {}
        self._current_phase = ""
        self._serial_pipeline_ready = 0.0
        self._optical_pipeline_ready = 0.0
        self.pool.reset()

        if self._sram_manager:
            self._sram_manager.reset()
            if tile_accesses:
                # 预先把未来访问信息喂给 SRAM 管理器（先知型策略用）
                self._sram_manager.register_future_info(tile_accesses)
        if self._mrr_manager:
            self._mrr_manager.reset()

        # Sort by order, then schedule respecting DAG dependencies
        ordered = sorted(records, key=lambda r: r.get("order", 0))
        op_ids = [r.get("op_id", "unknown") for r in ordered]
        if len(op_ids) != len(set(op_ids)):
            raise ValueError("Trace contains duplicate op_id values")
        self._records_by_id = {rec["op_id"]: rec for rec in ordered}

        for rec in ordered:
            # Validate DAG dependencies
            deps = rec.get("dependencies", [])
            # 依赖校验：所有前驱必须已在本算子之前调度完成
            missing = [d for d in deps if d not in self._op_complete]
            if missing:
                raise ValueError(
                    f"Operator {rec.get('op_id')}: missing/forward deps: {missing}"
                )

            # Determine eligibility
            eligible = True
            if classifier_fn is not None:
                result = classifier_fn(
                    rec.get("op_type", ""),
                    rec.get("op_role", "unknown"),
                    rec.get("weight_static", False),
                )
                eligible = result.get("eligible", False)

            # Track phase transitions
            # 阶段切换时（encode/context 结束），这些阶段一次性使用的权重
            # 不再需要，立刻从 SRAM 逐出并失效 MRR 上的调谐，否则容量实验
            # 里"死权重"会一直占着缓存直到仿真结束（结果偏大）。
            phase = rec.get("phase", "")
            if phase and phase != self._current_phase:
                previous_phase = self._current_phase
                # Context/encode weights are one-shot working-set data for
                # this trace. Release them at the actual phase boundary so
                # capacity experiments do not retain dead tiles until EOF.
                if previous_phase in ("encode", "context"):
                    if self._sram_manager:
                        released = self._sram_manager.evict_phase_tiles(previous_phase)
                        if self._mrr_manager:
                            self._mrr_manager.invalidate_many(released)
                self._current_phase = phase
                if self._sram_manager:
                    self._sram_manager.notify_phase(phase)

            if eligible:
                self._schedule_photonic(rec, tile_mapper)
            else:
                self._schedule_electronic(rec)

            # Track per-step completion
            # 记录每个 rollout step 结束时的时刻（取该 step 内最晚完成时刻）
            step = rec.get("rollout_step") or 0
            op_end = self._op_complete.get(rec.get("op_id", ""), 0.0)
            self._step_latency[step] = max(self._step_latency.get(step, 0.0), op_end)

        # Phase-cleanup: release Context tiles after Context phase
        # 收尾：context 阶段结束时清掉它的权重（若之前漏做）
        if self._sram_manager:
            released = self._sram_manager.evict_phase_tiles("context")
            if self._mrr_manager:
                self._mrr_manager.invalidate_many(released)

    def _schedule_photonic(self, rec: dict, tile_mapper=None):
        """安排一个光子 Linear：先让权重就绪，再串联 DAC、光计算和 ADC。

        统一走 Tile SRAM 驻留路径：有 SRAM 管理器时按 tile 命中/未命中
        决定 HBM DMA，否则视为无限 SRAM（100% 命中）。
        """
        # 中文说明：单个光子算子的完整调度流程（本文件最重要的方法）：
        #   1) 计算本算子可以开始的最早时刻 = max(DAG 前驱完成时刻,
        #      流水线闸门时刻)；
        #   2) 权重准备阶段（逐 tile）：查 SRAM —— 命中则不搬，未命中则
        #      安排 HBM->SRAM DMA；再查 MRR —— 未命中或模式要求重编程则
        #      安排编程通道把权重调谐进微环；
        #   3) 光路计算阶段：DAC(电转光) -> 光子计算 -> ADC(光转电) 三段，
        #      triple_pipeline 模式下三段可各自重叠，否则串成一条光路。
        op_id = rec.get("op_id", "unknown")
        ready = self._dependency_ready(rec)
        if self.pipeline_mode == "serial":
            # 串行模式：还要等上一个算子整体结束
            ready = max(ready, self._serial_pipeline_ready)
        weight_id = rec.get("weight_id")
        K = rec.get("K", 0) or 0
        N = rec.get("N", 0) or 0
        step = rec.get("rollout_step") or 0

        # Get tiles for this weight
        tiles = None
        if tile_mapper and weight_id and weight_id in tile_mapper.tile_map:
            tiles = tile_mapper.tile_map[weight_id]

        # --- Weight DMA + Programming ---
        # dma_chain: next DMA starts when previous DMA finishes (HBM is free)
        # compute_ready: compute phase starts when ALL programs finish
        # dma_chain 记录"下一次 DMA 可以开始"（HBM 是单口，串行排队）；
        # compute_ready 记录"所有 tile 都编程完"的最晚时刻（光路才开始）
        dma_chain = ready
        compute_ready = ready
        total_prog_lat = 0.0
        total_prog_energy = 0.0
        total_dma_lat = 0.0
        tile_count = len(tiles) if tiles else 0

        if tiles and len(tiles) > 0:
            # --------------- TILE SRAM ---------------
            # 12 Tile-local SRAMs (up to 96 MB total).
            # With SRAM manager: per-tile hit/miss -> DMA on miss.
            # Without SRAM manager: infinite SRAM -> 100% hit (legacy bulk).
            phase = rec.get("phase", "")
            prog_lat_per_tile = self._program_response_time_s

            if self._sram_manager:
                # ---- Managed SRAM: per-tile hit/miss ----
                self._sram_manager.sync_time(ready)
                program_count = 0
                dma_count = 0

                for tile in tiles:
                    # 逐 tile：查 SRAM 是否命中
                    result = self._sram_manager.lookup_tile(
                        tile.tile_id, ready,
                        metadata={"phase": phase, "drift_risk": 0.01 * step},
                    )
                    tile_ready = max(
                        ready, result.get("ready_time_s", ready)
                    )
                    if not result["hit"]:
                        # MRR residency is a subset of SRAM residency:
                        # evicting a backing tile invalidates any tune.
                        # 中文：MRR 驻留是 SRAM 驻留的子集——SRAM 里这块 tile
                        # 被逐出（backing 数据没了），MRR 上对应调谐必须作废
                        if self._mrr_manager:
                            self._mrr_manager.invalidate(tile.tile_id)
                            self._mrr_manager.invalidate_many(
                                result["evicted_tile_ids"]
                            )
                        # HBM -> SRAM DMA on miss
                        dma_lat = result["dma_latency_s"]
                        dma_start, _ = self.pool.acquire_dma(dma_chain)
                        self.events.append(Event(
                            EventType.WEIGHT_DMA, op_id, dma_start,
                            dma_lat, "hbm",
                            metadata={
                                "tile_id": tile.tile_id, "target": "sram",
                                "evicted_tile_ids": result["evicted_tile_ids"],
                            },
                        ))
                        dma_end = dma_start + dma_lat
                        # 回填 DMA 完成时刻（数据真正就绪）
                        self._sram_manager.mark_tile_ready(
                            tile.tile_id, dma_end
                        )
                        tile_ready = dma_end
                        dma_chain = dma_end
                        total_dma_lat += dma_lat
                        dma_count += 1

                    # 再查 MRR：这块权重在光路上还能不能用
                    mrr_result = {
                        "hit": False,
                        "refresh": False,
                        "evicted_tile_id": None,
                    }
                    if self._mrr_manager:
                        mrr_result = self._mrr_manager.lookup_tile(
                            tile.tile_id, tile_ready
                        )

                    # 需要编程的条件：模式要求"每次都用"或 MRR 未命中
                    needs_program = (
                        self.mrr_program_mode == "reprogram_each_use"
                        or not mrr_result["hit"]
                    )
                    if needs_program:
                        prog_ready = max(
                            tile_ready,
                            mrr_result.get("ready_time_s", tile_ready),
                        )
                        prog_start, lane = self.pool.acquire_program_lane(
                            prog_ready, prog_lat_per_tile,
                        )
                        program_end = prog_start + prog_lat_per_tile
                        # 编程事件：写入能耗按每 tile 常量计（能量账本一项）
                        self.events.append(Event(
                            EventType.WEIGHT_PROGRAM, op_id, prog_start,
                            prog_lat_per_tile, f"program_lane_{lane}",
                            energy_j=self._program_energy_j_per_tile,
                            metadata={
                                "tile_id": tile.tile_id,
                                "source": "tile_sram",
                                "mrr_mode": self.mrr_program_mode,
                                "mrr_refresh": mrr_result["refresh"],
                                "mrr_evicted_tile_id": mrr_result["evicted_tile_id"],
                            },
                        ))
                        if self._mrr_manager:
                            self._mrr_manager.record_program(tile.tile_id, program_end)
                        compute_ready = max(compute_ready, program_end)
                        total_prog_lat += prog_lat_per_tile
                        total_prog_energy += self._program_energy_j_per_tile
                        program_count += 1
                    else:
                        # MRR 命中：直接复用上次调谐，不用编程
                        compute_ready = max(
                            compute_ready,
                            tile_ready,
                            mrr_result.get("ready_time_s", tile_ready),
                        )

            else:
                # ---- Infinite SRAM (100% hit, no per-tile DMA) ----
                # 无 SRAM 管理器 = 假设缓存无限大（传统整权重模式），
                # 每个 tile 都只需编程，没有 DMA 搬移
                for tile in tiles:
                    prog_start, lane = self.pool.acquire_program_lane(
                        ready, prog_lat_per_tile,
                    )
                    self.events.append(Event(
                        EventType.WEIGHT_PROGRAM, op_id, prog_start,
                        prog_lat_per_tile, f"program_lane_{lane}",
                        energy_j=self._program_energy_j_per_tile,
                        metadata={"tile_id": tile.tile_id, "source": "tile_sram"},
                    ))
                    compute_ready = max(compute_ready, prog_start + prog_lat_per_tile)
                    total_prog_lat += prog_lat_per_tile
                    total_prog_energy += self._program_energy_j_per_tile

            # 累计流式统计（编程/DMA 的次数、时延、能耗、字节）
            self._streaming_stats["programs"] += (
                program_count if self._sram_manager else tile_count
            )
            self._streaming_stats["programming_latency_s"] += total_prog_lat
            self._streaming_stats["programming_energy_j"] += total_prog_energy
            self._streaming_stats["dma_latency_s"] += total_dma_lat
            self._streaming_stats["dma_bytes"] += (
                (dma_count if self._sram_manager else tile_count)
                * self.pool.weight_tile_bytes
            )

            ready = compute_ready
        # --- Photonic compute pipeline ---
        dac_lat, compute_lat, adc_lat = self._photonic_stage_latencies(rec)

        # A single optical pipeline is the intermediate mode. In the full
        # triple pipeline DAC/compute/ADC may use independent instances and
        # overlap across operators.
        optical_ready = ready
        if self.pipeline_mode != "triple_pipeline":
            # 非三重流水：整条光路同一时刻只能有一个算子
            optical_ready = max(optical_ready, self._optical_pipeline_ready)

        if self.pipeline_mode == "triple_pipeline":
            # Triple pipeline: DMA, program, and compute can all overlap.
            # Each sub-event acquires its own resource independently.
            # 三重流水：DAC/光计算/ADC 各自独立申请资源，可与其它算子重叠
            dac_start, dac_idx = self.pool.acquire_dac(optical_ready, dac_lat)
            self.events.append(Event(
                EventType.DAC_ENCODE, op_id, dac_start, dac_lat,
                f"dac_{dac_idx}",
            ))
            compute_start, core_idx = self.pool.acquire_photonic_core(
                dac_start + dac_lat, compute_lat,
            )
            self.events.append(Event(
                EventType.PHOTONIC_COMPUTE, op_id, compute_start, compute_lat,
                f"photonic_core_{core_idx}",
            ))
            adc_start, adc_idx = self.pool.acquire_adc(
                compute_start + compute_lat, adc_lat,
            )
            self.events.append(Event(
                EventType.ADC_DECODE, op_id, adc_start, adc_lat,
                f"adc_{adc_idx}",
            ))
            ready = adc_start + adc_lat
        else:
            # Serial compute pipeline: DAC → photonic → ADC, all on
            # the same timing chain (resources still tracked separately).
            # 串行光路：DAC -> 光计算 -> ADC 首尾相接（资源仍单独记账）
            dac_start, dac_idx = self.pool.acquire_dac(optical_ready, dac_lat)
            self.events.append(Event(
                EventType.DAC_ENCODE, op_id, dac_start, dac_lat,
                f"dac_{dac_idx}",
            ))
            compute_start, core_idx = self.pool.acquire_photonic_core(
                dac_start + dac_lat, compute_lat,
            )
            self.events.append(Event(
                EventType.PHOTONIC_COMPUTE, op_id, compute_start, compute_lat,
                f"photonic_core_{core_idx}",
            ))
            adc_start, adc_idx = self.pool.acquire_adc(
                compute_start + compute_lat, adc_lat,
            )
            self.events.append(Event(
                EventType.ADC_DECODE, op_id, adc_start, adc_lat,
                f"adc_{adc_idx}",
            ))
            ready = adc_start + adc_lat

        # 记录本算子完成时刻（DAG 后续算子依赖它）
        self._op_complete[op_id] = ready

        # 推进流水线闸门：非三重流水时整条光路被本算子占用到 ready
        if self.pipeline_mode != "triple_pipeline":
            self._optical_pipeline_ready = ready
        if self.pipeline_mode == "serial":
            self._serial_pipeline_ready = ready

    def _schedule_electronic(self, rec: dict):
        """安排电子算子；它会等待相同的 DAG 前驱和电子资源。"""
        # 中文说明：电子算子的调度最简单——等 DAG 前驱完成后，
        # 在电子核心上顺序执行（电子核心是单个资源，天然串行）。
        op_id = rec.get("op_id", "unknown")
        ready = self._dependency_ready(rec)
        lat = self._electronic_latency(rec)
        start = self.pool.acquire_electronic(ready, lat)
        self.events.append(Event(
            EventType.ELECTRONIC_COMPUTE, op_id, start, lat, "electronic_core",
        ))
        self._op_complete[op_id] = start + lat

    def _dependency_ready(self, rec: dict) -> float:
        """返回所有直接前驱完成的最晚时刻；这是 DAG 正确性的核心约束。"""
        # 中文说明：DAG 调度核心公式——本算子能开始的最早时刻 = 所有前驱
        # 完成时刻的最大值（最慢的那个前驱决定）。没有前驱则从 0 开始。
        deps = rec.get("dependencies", [])
        if not deps:
            return 0.0
        return max(self._op_complete.get(d, 0.0) for d in deps)

    # ------------------------------------------------------------------
    # Conservation checks (Section 7.6 item 4)
    # ------------------------------------------------------------------

    def check_conservation(self) -> List[dict]:
        """检查同一物理资源的事件是否重叠，用于发现调度账本错误。

        Returns list of {check, passed, detail} dicts.
        """
        # 中文说明：守恒性检查（自检）。调度的账本如果写错，同一资源上会
        # 出现两个事件时间重叠（相当于两个任务同时占一台机器，物理上不可能）。
        # 这里做 6 项检查，每项返回 {check, passed, detail}：
        #   1) 光子核心不重叠    2) 时间守恒（总时长=最晚事件结束）
        #   3) 编程通道不重叠    4) 辅助资源（DAC/ADC/HBM/电子）不重叠
        #   5) DAG 依赖成立     6) 驻留管理器不超容量
        results = []

        # 1. Photonic core non-overlap
        # 把事件按资源名分组：photonic_core_<编号> -> 该核心上的事件列表
        core_events: Dict[int, List[Event]] = defaultdict(list)
        for e in self.events:
            if e.resource.startswith("photonic_core_"):
                core_idx = int(e.resource.split("_")[-1])
                core_events[core_idx].append(e)

        # 按开始时刻排序后逐个检查：前一事件的结束时刻不得晚于后一事件开始
        core_overlap = False
        for core_idx, evts in core_events.items():
            evts.sort(key=lambda e: e.start_time_s)
            for i in range(len(evts) - 1):
                if evts[i].end_time_s > evts[i + 1].start_time_s + 1e-12:
                    core_overlap = True
                    break
        results.append({
            "check": "photonic_core_non_overlap",
            "passed": not core_overlap,
            "detail": f"{len(core_events)} cores checked" if not core_overlap
                      else "OVERLAP DETECTED",
        })

        # 2. Time conservation: total_time ≈ max_end_time
        # 时间守恒：事件的跨度应等于资源池报告的最晚完成时刻（两者一致说明
        # 事件时间线与资源占用时间线没有脱节）
        if self.events:
            max_end = max(e.end_time_s for e in self.events)
            min_start = min(e.start_time_s for e in self.events)
            total_span = max_end - min_start
            pool_end = self.pool.max_time()
            consistent = math.isclose(max_end, pool_end, rel_tol=1e-10, abs_tol=1e-12)
            results.append({
                "check": "time_conservation",
                "passed": consistent,
                "detail": (
                    f"span={total_span:.9e}s, max_event_end={max_end:.9e}s, "
                    f"pool_end={pool_end:.9e}s"
                ),
            })
        else:
            results.append({
                "check": "time_conservation",
                "passed": True,
                "detail": "no events",
            })

        # 3. Program lane non-overlap
        # 编程通道不重叠（同理由：同一通道不能同时编两块权重）
        lane_events: Dict[int, List[Event]] = defaultdict(list)
        for e in self.events:
            if e.resource.startswith("program_lane_"):
                lane_idx = int(e.resource.split("_")[-1])
                lane_events[lane_idx].append(e)

        lane_overlap = False
        for lane_idx, evts in lane_events.items():
            evts.sort(key=lambda e: e.start_time_s)
            for i in range(len(evts) - 1):
                if evts[i].end_time_s > evts[i + 1].start_time_s + 1e-12:
                    lane_overlap = True
                    break
        results.append({
            "check": "program_lane_non_overlap",
            "passed": not lane_overlap,
            "detail": f"{len(lane_events)} lanes checked" if not lane_overlap
                      else "OVERLAP DETECTED",
        })

        # 4. All remaining serialized resources: DAC, ADC, HBM and electronic.
        # 其余串行资源：DAC/ADC/HBM/电子核心，各自内部不得重叠
        serial_prefixes = ("dac_", "adc_", "hbm", "electronic_core")
        other_events: Dict[str, List[Event]] = defaultdict(list)
        for event in self.events:
            if event.resource.startswith(serial_prefixes):
                other_events[event.resource].append(event)
        overlaps = []
        for resource, events in other_events.items():
            events.sort(key=lambda event: event.start_time_s)
            for left, right in zip(events, events[1:]):
                if left.end_time_s > right.start_time_s + 1e-12:
                    overlaps.append(resource)
                    break
        results.append({
            "check": "auxiliary_resource_non_overlap",
            "passed": not overlaps,
            "detail": (
                f"{len(other_events)} resources checked"
                if not overlaps else f"overlap on {sorted(set(overlaps))}"
            ),
        })

        # 5. Every operation must begin after all declared predecessors finish.
        # DAG 依赖检查：每个算子的第一个事件都必须晚于所有前驱的完成时刻
        first_event = {}
        for event in self.events:
            first_event[event.op_id] = min(
                first_event.get(event.op_id, float("inf")), event.start_time_s
            )
        dag_violations = []
        for op_id, rec in self._records_by_id.items():
            op_start = first_event.get(op_id, self._op_complete.get(op_id, 0.0))
            for dep in rec.get("dependencies", []):
                if op_start + 1e-12 < self._op_complete.get(dep, float("inf")):
                    dag_violations.append(f"{dep}->{op_id}")
        results.append({
            "check": "dag_dependencies",
            "passed": not dag_violations,
            "detail": (
                f"{len(self._records_by_id)} operators checked"
                if not dag_violations else f"violations={dag_violations[:5]}"
            ),
        })

        # 6. Residency managers must never exceed their declared capacities.
        # 驻留容量检查：SRAM 里实际占用数不得超过声明的容量
        capacity_violations = []
        if self._sram_manager is not None:
            if self._sram_manager.occupied > self._sram_manager.capacity_tiles:
                capacity_violations.append(
                    f"sram={self._sram_manager.occupied}/"
                    f"{self._sram_manager.capacity_tiles}"
                )
        results.append({
            "check": "residency_capacity",
            "passed": not capacity_violations,
            "detail": "within capacity" if not capacity_violations else ", ".join(capacity_violations),
        })

        return results

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def end_to_end_latency_s(self) -> float:
        """端到端时延 = 资源池里最晚的完成时刻。"""
        return self.pool.max_time()

    def total_energy_by_type(self) -> Dict[str, float]:
        """把事件能量按事件类型累加，得到各类动态能耗的小计。"""
        result: Dict[str, float] = defaultdict(float)
        for e in self.events:
            result[e.event_type.value] += e.energy_j
        return dict(result)

    def timeline_csv(self) -> str:
        """把事件时间线导出成 CSV 字符串（report 会写入 event_timeline.csv）。"""
        lines = ["event_type,op_id,start_s,duration_s,end_s,resource,energy_j"]
        for e in sorted(self.events, key=lambda ev: ev.start_time_s):
            lines.append(
                f"{e.event_type.value},{e.op_id},{e.start_time_s:.9f},"
                f"{e.duration_s:.9f},{e.end_time_s:.9f},"
                f"{e.resource},{e.energy_j:.9e}"
            )
        return "\n".join(lines)

    def per_step_latency(self) -> Dict[int, float]:
        """Latency at the end of each rollout step."""
        # 每个 rollout step 末尾的系统时刻（按 step 排序）
        return dict(sorted(self._step_latency.items()))

    def summary(self) -> dict:
        # 汇总本轮调度结果：事件数、端到端时延、分类型能耗、分步时延、
        # SRAM/MRR 驻留摘要、流式统计、编程关键路径、守恒性检查结果。
        # programming_critical_path 是最后一个编程事件结束的时刻，
        # 反映"全部权重就位"需要多久（若它是瓶颈，优化驻留才有价值）。
        program_events = [
            e for e in self.events if e.event_type == EventType.WEIGHT_PROGRAM
        ]
        programming_critical_path_s = (
            max((e.end_time_s for e in program_events), default=0.0)
        )
        return {
            "total_events": len(self.events),
            "end_to_end_latency_s": self.end_to_end_latency_s,
            "energy_by_type": self.total_energy_by_type(),
            "per_step_latency_s": self.per_step_latency(),
            "pipeline_mode": self.pipeline_mode,
            "sram_manager_summary": (
                self._sram_manager.summary() if self._sram_manager else None
            ),
            "mrr_residency_summary": (
                {"mode": self.mrr_program_mode, **self._mrr_manager.summary()}
                if self._mrr_manager else {
                    # 每次重编程模式没有 MRR 管理器：命中率恒 0，全部算缺失
                    "mode": "reprogram_each_use",
                    "accesses": self._streaming_stats["programs"],
                    "hits": 0,
                    "misses": self._streaming_stats["programs"],
                    "evictions": 0,
                    "refreshes": 0,
                    "hit_rate": 0.0,
                }
            ),
            "streaming_stats": dict(self._streaming_stats),
            "programming_critical_path_s": programming_critical_path_s,
            "conservation_checks": self.check_conservation(),
        }
