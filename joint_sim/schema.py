"""
Joint-simulation data types for the LLMCompass + SimPhony co-simulation.

Defines OperatorRecord (trace entries), joint requests/responses, and
the ExecutionCost type that LLMCompass needs from the photonic backend.

All fields use explicit SI units as required by the implementation plan (Section 4).

中文阅读提示：本文件不执行仿真，只定义各模块交换的数据格式。阅读其他代码时，
先回到这里确认字段含义；时间统一用秒、能量统一用焦耳、面积统一用平方微米。
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Operator Record — one entry in a trace (Plan Section 4.1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperatorRecord:
    # 中文说明：trace（执行轨迹）中的一条算子记录 = 模型前向传播中真实发生的
    # 一次调用（同一层被调用多次会得到多条记录，靠 order/call_index 区分）。
    # 关键字段含义：
    #   op_type   算子类型，如 Linear/MatMul/Softmax/GELU/LayerNorm；
    #             其中固定权重的 Linear 才有资格走光子后端。
    #   op_role   算子的语义角色：Q/K/V/O（注意力四个投影）、FFN_up/FFN_down
    #             （前馈网络）、c_proj（上下文投影）等。
    #   phase     所属阶段：encode(编码)/context(上下文)/dynamics(动态演化)/
    #             sample(采样)/decode(解码)。
    #   M/K/N     矩阵乘法 [M,K]x[K,N] 的三个维度：M=输入向量个数(含批)，
    #             K=内维(输入特征数)，N=输出特征数。读不懂时回到这条注释。
    #   weight_static 权重是否固定(训练完不再变)；固定权重才可长期驻留在
    #             MRR(微环谐振器)阵列上复用。
    #   weight_id 权重的稳定哈希标识，同一权重在多次调用间复用同一个 id。
    #   dependencies 本算子依赖的前置算子 op_id 列表（调度器按它建 DAG）。
    """A single operator invocation captured from LPWM forward pass."""
    # 一条记录对应模型前向传播中实际发生的一次算子调用，而非网络层的静态定义。
    trace_version: str
    op_id: str
    order: int
    module_path: str
    op_type: str                  # "Linear", "MatMul", "Conv", "LayerNorm", etc.
    op_role: str                  # "Q", "K", "V", "O", "FFN_up", "FFN_down", "c_proj", ...
    phase: str                    # "encode" / "context" / "dynamics" / "sample" / "decode"
    block_kind: str               # "spatial" / "temporal" / "other"
    layer_index: Optional[int]
    rollout_step: Optional[int]
    call_index: int
    input_shapes: tuple
    output_shape: tuple
    M: Optional[int]              # batch / number of input vectors
    K: Optional[int]              # inner dimension (input features)
    N: Optional[int]              # output features
    batch_repetitions: int        # for batched matmul: repetitions per call
    dtype: str
    input_bits: int
    weight_bits: Optional[int]
    output_bits: int
    weight_id: Optional[str]      # stable hash of (module_path, weight_shape)
    weight_static: bool           # True for fixed Linear weights
    input_bytes: int
    weight_bytes: int
    output_bytes: int
    dependencies: tuple = ()      # op_ids that must complete before this one


# ---------------------------------------------------------------------------
# Classification result (Phase D5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassificationResult:
    # 中文说明：op_classifier 的输出。eligible=True 表示该算子可映射到
    # 光子后端；reason 记录原因，便于审计为什么某个算子被拒。
    """Whether an operator is eligible for photonic execution and why."""
    op_id: str
    eligible: bool
    reason: str                   # "static_weight_linear" / "both_operands_dynamic" / ...


# ---------------------------------------------------------------------------
# Execution cost — what LLMCompass gets back (Plan Section 10 F3)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionCost:
    # 中文说明：单个算子（无论光子还是电子）的核算结果，是能量账本的最小单位。
    #   latency_s            该算子耗时（秒）
    #   dynamic_energy_j     动态能耗（秒内做功的能量，焦耳）
    #   memory_energy_j      访存能耗（读写数据消耗的能量）
    #   programming_energy_j 编程能耗（把权重写进 MRR 的能耗；电子算子恒为 0）
    #   resource             该算子实际使用的后端："photonic"(光子)/"electronic"(电子)
    #   breakdown            可选的细分账本（dict），用于报表逐项展示
    """Structured cost for one operator, photonic or electronic."""
    latency_s: float
    dynamic_energy_j: float
    memory_energy_j: float
    programming_energy_j: float     # zero for electronic ops
    resource: str                   # "photonic" / "electronic"
    breakdown: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cache manifest — versioning info stored alongside cached costs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CacheManifest:
    # 中文说明：缓存的"版本防伪标签"。缓存命中的前提是这些指纹全部一致，
    # 任何一项变化（换了 SimPhony 代码、配置、设备或格式版本）都会失效缓存。
    """Metadata stored in the SQLite cache for reproducibility."""
    simphony_commit: str
    config_hash: str
    device_hash: str
    schema_version: str


# ---------------------------------------------------------------------------
# Tile types — Experiment 4 weight decomposition (Section 7.2)
# ---------------------------------------------------------------------------

# 单个光子计算核心（core）的边长：64x64 个 MRR（微环谐振器）。
# 类比：一个 64x64 的 tile 就像一张 64x64 像素的小图，正好铺满一块光子矩阵乘法单元。
CORE_SIZE = 64

@dataclass(frozen=True)
class TileRecord:
    # 中文说明：一个大权重矩阵 KxN 被切成若干 64x64 的 tile（小块），
    # 每个 tile 对应光子阵列上一个 64x64 的核心。
    #   effective_K / effective_N  该 tile 实际用到的行列数。矩阵边沿可能
    #                              不够 64，所以"尾巴 tile"实际有效尺寸 <64
    #                              （is_tail=True）。
    #   valid_mrrs    该 tile 真正点亮参与计算的微环数 = 有效K x 有效N
    #   physical_mrrs 该核心物理上的微环总数 = 64x64 = 4096
    #   utilization   valid/physical，反映该核心被利用了多少
    #   storage_bytes 该 tile 在 SRAM 里占多少字节（8bit 权重 x 有效元素数，
    #                 有符号扩展模式还会翻倍）
    #   mapping_mode  unsigned_64x64：普通无符号映射（64 个输出）；
    #                 signed_row_pair_fixed_area：固定面积有符号映射，两个物理
    #                 行表示一个有符号逻辑输出，故每个核心只剩 32 个逻辑输出。
    """One 64×64 weight tile split from a K×N Linear weight matrix."""
    tile_id: str               # f"{weight_id}_k{k_idx}_n{n_idx}"
    weight_id: str
    k_idx: int
    n_idx: int
    effective_K: int           # actual rows (≤64, smaller for tail tiles)
    effective_N: int           # actual cols (≤64, smaller for tail tiles)
    valid_mrrs: int            # effective_K * effective_N (active MRRs)
    physical_mrrs: int         # CORE_SIZE * CORE_SIZE = 4096 (total MRRs)
    is_tail: bool              # True if effective_K < 64 or effective_N < 64
    mapping_mode: str = "unsigned_64x64"
    storage_mode: str = "compact"
    storage_bytes: int = 0
    logical_capacity_n: int = CORE_SIZE

    @property
    def utilization(self) -> float:
        # 利用率 = 实际用到的微环数 / 核心总的微环数（0~1）
        return self.valid_mrrs / self.physical_mrrs


@dataclass(frozen=True)
class TileAccess:
    # 中文说明：trace 里某一次算子调用对某个 tile 的一次"访问"。
    # 核心是为缓存/驻留策略准备未来信息：
    #   next_use_order 这次访问之后，该 tile 下一次被用到的全局 order；
    #                  None 表示以后不再用（"最后一次访问"）。
    #   remaining_uses 这次访问之后还剩多少次访问（不含本次）。
    # 这些字段让调度器可以"预知未来"，从而做出 LRU/Belady 等替换决策。
    """One scheduled access to a tile, derived from the operator trace."""
    tile_id: str
    op_id: str
    phase: str                 # "encode" / "context" / "dynamics" / "decode"
    rollout_step: int
    order: int                 # global order in the original trace
    next_use_order: Optional[int]  # order of next access to this tile (None if last)
    remaining_uses: int        # how many more times this tile will be accessed


@dataclass
class TilePlacement:
    # 中文说明：运行时状态——某个 tile 现在"放在哪、状态如何"。
    #   weight_bank_id 所在权重库（bank）
    #   core_id        库内的哪个 64x64 核心
    #   ready_time_s   该 tile 数据就绪（DMA 完成）的时刻
    #   last_used_time_s 上次被访问的时刻（LRU 替换要用）
    #   hold_energy_j  驻留期间累计的保持能耗
    #   drift_risk     累积漂移风险（0=全新未漂移）；漂移会让光计算精度下降
    """Runtime state: where a tile is placed and its residency metadata."""
    tile_id: str
    weight_bank_id: str
    core_id: int               # which 64×64 core within the weight bank
    ready_time_s: float
    last_used_time_s: float
    hold_energy_j: float = 0.0
    drift_risk: float = 0.0    # accumulated drift risk (0=pristine)


@dataclass
class ProgramLane:
    # 中文说明：一条"编程通道"（数模转换器 DAC + 保持电路）。
    # 把权重写入 MRR 需要占用一条编程通道；P_prog 就是这类通道的总数，
    # 通道越多，可同时编程的 tile 越多（并行度越高）。
    #   ready_time_s     该通道"下一次可用"的时刻（被占用到这一刻）
    #   current_tile_id  当前正在写入/驻留的 tile
    """An independent programming lane (DAC + hold circuit)."""
    lane_id: str
    ready_time_s: float = 0.0
    current_tile_id: Optional[str] = None

    def busy_until(self, duration_s: float) -> float:
        """占用该通道 duration_s 秒，返回写完后的完成时刻（秒）。"""
        finish = self.ready_time_s + duration_s
        self.ready_time_s = finish
        return finish


@dataclass
class ResidencyConfig:
    # 中文说明：一次驻留策略实验的完整参数配置。
    #   strategy            驻留/替换策略：no_residency(不驻留，每次重新编程)、
    #                       lru(最近最少使用)、belady(最优离线)、rollout_aware(利用未来信息打分)
    #   total_tile_slots    SRAM 最多能同时放多少个 tile（容量）
    #   program_parallelism 编程通道数 P_prog（并行度）
    #   pipeline_mode       流水线模式：serial(全串行)/dma_program_overlap(搬运
    #                       与编程重叠)/triple_pipeline(三重流水：搬运+编程+计算全重叠)
    #   dma_bandwidth_bytes_per_s HBM 到 SRAM 的搬移带宽（字节/秒）
    #   hold_power_nominal  是否计入 MRR 驻留保持功率
    #   drift_rate          器件漂移速率档位：low/nominal/high
    #   calibrate_interval_steps 每隔多少步做一次校准（0=从不校准）
    """Configuration for a residency strategy experiment run."""
    strategy: str               # "no_residency" | "lru" | "belady" | "rollout_aware"
    total_tile_slots: int       # 96, 384, 1536, 6144, 14712
    program_parallelism: int    # P_prog: 1, 2, 4, 8, 16, 32, 64, 96
    pipeline_mode: str          # "serial" | "dma_program_overlap" | "triple_pipeline"
    dma_bandwidth_bytes_per_s: float
    hold_power_nominal: bool    # True → include hold energy
    drift_rate: str             # "low" | "nominal" | "high"
    calibrate_interval_steps: int  # calibrate every N steps (0 = never)
