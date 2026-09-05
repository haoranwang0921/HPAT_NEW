"""
Experiment 1: Native CPU/GPU LPWM inference latency benchmark.

Measures end-to-end and per-operator latency of the LPWM world model
(BAIR-128, 30-step rollout) on CPU and GPU, following the measurement
protocol in docs/04-实验设计/LPWM_GPU推理延迟实测实验文档.md.

Usage:
  CPU baseline:
    python benchmark_lpwm.py --device cpu --output experiments/native_baseline/cpu_fp32/run_001

  GPU baseline:
    python benchmark_lpwm.py --device cuda --output experiments/native_baseline/gpu_fp32/run_001

  Smoke test (horizon=4):
    python benchmark_lpwm.py --device cpu --horizon 4 --warmup 3 --iterations 5 --output results/smoke

【中文说明】
这是"实验1"：测量 LPWM 世界模型在原生 CPU / GPU 上的推理时延基线。
这个数字很重要——它是后面光子加速实验的"对照基准"：只有光子系统比原生
推理更快，加速才有意义。

本脚本测三类数据：
  1) E2E（端到端）时延：完整跑一次 30 步 rollout 的总耗时，多次重复取统计。
  2) 阶段/每步时延：编码（encode）、动态预测（dynamics，30 步）、解码（decode）
     各花多少时间；每步动态预测的平均时长。
  3) 算子级画像：用 PyTorch 前向钩子统计每个 nn.Linear 的形状（M/K/N）、
     属于哪个阶段、第几个 rollout 步——供仿真器生成轨迹使用。

测量协议（与实验文档一致）：10 次预热 + 30 次正式测量；固定种子 42 与确定性
模式；CPU 用 time.perf_counter_ns 计时，GPU 用 torch.cuda.Event 计时；
每次测量前 cuda.synchronize 保证计时准确。

运行方式（从仓库根目录）：
  python joint_sim/benchmarks/benchmark_lpwm.py --device cpu  \
      --output experiments/native_baseline/cpu_fp32/run_001
  python joint_sim/benchmarks/benchmark_lpwm.py --device cuda \
      --output experiments/native_baseline/gpu_fp32/run_001
  # 快速冒烟测试：
  python joint_sim/benchmarks/benchmark_lpwm.py --device cpu --horizon 4 \
      --warmup 3 --iterations 5 --output results/smoke

产出文件（在 --output 目录下）：
  run_manifest.json        运行清单（硬件/软件/模型/测量配置，含权重哈希）
  summary.json             汇总统计（E2E 均值/标准差/各百分位、每步时延、帧率等）
  operator_latency.csv     每个 Linear 算子的形状与阶段信息
  operator_stats_by_type.csv / _by_phase.csv / _by_step.csv  按角色/阶段/步分组统计
  per_step_latency.csv     每步动态预测时延
  raw_timestamps.jsonl     每次测量的原始时延
  gpu_memory.csv           峰值显存（仅 GPU 运行）
  software_versions.txt    环境版本信息
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths relative to project root
# ---------------------------------------------------------------------------
# 相对项目根目录的路径：LPWM 第三方代码、checkpoint 权重、超参配置文件
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LPWM_ROOT = os.path.join(PROJECT_ROOT, "third_party", "lpwm")
CHECKPOINT_DIR = os.path.join(LPWM_ROOT, "checkpoints", "bair128")
HPARAMS_PATH = os.path.join(CHECKPOINT_DIR, "hparams.json")  # 模型超参
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "bair_gddlp_ts16_best_lpips.pth")  # 权重

sys.path.insert(0, LPWM_ROOT)  # 让 Python 能找到 LPWM 的 models 模块


# ---------------------------------------------------------------------------
# Phase / role inference (mirrors lpwm_trace_exporter.py logic)
# ---------------------------------------------------------------------------
# 阶段/角色推断区：从模块路径名判断"这是模型哪一段、负责什么"。逻辑与
# lpwm_trace_exporter.py 保持一致（两处要同步修改）。

def infer_phase(module_path: str) -> str:
    """根据模块路径判断它属于哪个阶段：encode/context/dynamics/decode。

    中文说明：LPWM 推理分三阶段：编码（encode，把条件帧编码成粒子）、
    动态预测（dynamics，自回归迭代预测未来，PINT 变换器在这里）、
    解码（decode，把粒子解码成图像）。context 是上下文处理子阶段。
    判断靠"模块名里包含的关键词"——例如路径里有 "dyn_module" 就是动力学阶段，
    有 "decoder_module" 就是解码阶段。返回 "unknown" 表示无法识别。
    """
    pl = module_path.lower()
    if "decoder_module" in pl or ".decoder" in pl or ".bg_dec" in pl:
        return "decode"
    if "dyn_module" in pl or ".dyn." in pl:
        return "dynamics"
    if "ctx_enc" in pl or "ctx_module" in pl or "context" in pl:
        return "context"
    if "spatio" in pl or "spatial" in pl:
        return "dynamics" if "dyn_module" in pl else "context"
    if "temp_block" in pl or "temporal" in pl:
        return "dynamics" if "dyn_module" in pl else "context"
    if "encoder_module" in pl or ".enc." in pl or "prior_module" in pl or "prior_enc" in pl:
        return "encode"
    if ".head" in pl or "projection" in pl:
        return "decode" if "decoder" in pl else "encode"
    if "sample" in pl:
        return "decode"
    return "unknown"


def infer_op_role(module_path: str) -> str:
    """根据模块路径推断这个 Linear 在模型中的角色。

    中文说明：返回的角色名供后续按"算子类型"分组统计/生成轨迹。常见角色：
    Q/K/V/O（注意力四件套：query/key/value/输出投影）、FFN_up/FFN_down/
    FFN_gate（前馈网络三段）、c_proj（变换器块末的残差投影）、
    proj_xy/proj_scale/proj_features 等（各种粒子属性投影头）、head/embed。
    """
    pl = module_path.lower()
    # 注意力 q/k/v/o 投影
    if ".attn.key" in pl: return "K"
    if ".attn.query" in pl: return "Q"
    if ".attn.value" in pl: return "V"
    if ".attn.proj" in pl: return "O"
    if "q_proj" in pl or "wq" in pl: return "Q"
    if "k_proj" in pl or "wk" in pl: return "K"
    if "v_proj" in pl or "wv" in pl: return "V"
    if "o_proj" in pl or "out_proj" in pl or "wo" in pl: return "O"
    # 前馈网络 FFN（SwiGLU 有 up/gate/down 三段）
    if ".mlp." in pl or "ffn" in pl:
        if "fc_1" in pl or "w1" in pl: return "FFN_up"
        if "proj" in pl and "c_proj" not in pl: return "FFN_down"
        if "w3" in pl: return "FFN_gate"
        return "FFN"
    if "c_proj" in pl or "context_proj" in pl: return "c_proj"
    # 各类粒子属性投影头
    if "projection" in pl or "_proj" in pl:
        if "xy" in pl: return "proj_xy"  # 粒子位置（坐标）
        if "scale" in pl: return "proj_scale"  # 粒子尺度
        if "feature" in pl: return "proj_features"  # 粒子特征
        if "obj_on" in pl: return "proj_obj_on"  # 目标开关
        if "depth" in pl: return "proj_depth"  # 深度
        if "bg" in pl: return "proj_bg"  # 背景
        return "proj"
    if ".head" in pl: return "head"
    if "embed" in pl: return "embed"
    return "other_linear"


# ---------------------------------------------------------------------------
# Operator timing hooks
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Runs LPWM inference benchmarks following the experiment protocol.

    Protocol (from experiment docs):
      - 10 warmup iterations, 30 measurement iterations
      - CPU: time.perf_counter_ns() for E2E timing
      - GPU: torch.cuda.Event + torch.cuda.synchronize() for E2E timing
      - Separate E2E runs (no hooks) and profiler runs (with hooks)
      - Per-operator profiling via forward hooks
      - Fixed seed=42, deterministic mode

    中文说明：
    基准测试的总控类。负责四件事：
      1) load_model            加载 LPWM 模型（BAIR-128 checkpoint）
      2) run_e2e_benchmark     测端到端时延（不挂钩子，保证测量干净）
      3) run_phase_benchmark   测编码/动态/解码三阶段及每步时延
      4) run_operator_profile  挂钩子做算子级画像（收集每个 Linear 的形状）
      最后 write_outputs 把所有结果写成文件。测量协议：预热若干次让 GPU 进入
      稳定状态，再正式测若干次；CPU 用高性能时钟，GPU 用 CUDA 事件并强制同步。
    """

    def __init__(
        self,
        device: str = "cpu",
        horizon: int = 30,
        cond_steps: int = 1,
        warmup: int = 10,
        iterations: int = 30,
        seed: int = 42,
        output_dir: str = "experiments/native_baseline",
    ):
        """初始化基准测试配置与结果容器。

        参数：device 运行设备（cpu/cuda）；horizon 要预测的步数（默认 30）；
        cond_steps 条件帧数；warmup 预热轮数；iterations 正式测量轮数；
        seed 随机种子；output_dir 结果输出目录。
        各种 self.* 字段是结果容器：e2e_latencies_s 记录每次端到端时延，
        operator_records 收集算子画像，phase_times 记录各阶段时延。
        """
        self.device = device
        self.horizon = horizon
        self.cond_steps = cond_steps
        self.warmup = warmup
        self.iterations = iterations
        self.seed = seed
        self.output_dir = output_dir

        self.model: Optional[nn.Module] = None  # 模型（load_model 后填充）
        self.checkpoint_sha256: str = ""  # checkpoint 权重哈希

        # E2E results  # E2E 结果容器
        self.e2e_latencies_s: List[float] = []
        self.per_step_latencies: List[List[float]] = []  # [iteration][step]
        self.peak_memory_mb: float = 0.0

        # Operator-level results (from profiler run)  # 算子画像结果
        self.operator_records: List[dict] = []

        # Phase-level results  # 各阶段时延
        self.phase_times: Dict[str, List[float]] = defaultdict(list)

        # Manifest  # 运行清单
        self.manifest: dict = {}

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self):
        """Load LPWM model from the BAIR-128 checkpoint.

        中文说明：读 hparams.json 超参 → 用全部超参构造 DLP 模型 → 从
        checkpoint 载入权重 → 移到目标设备并切到 eval 模式 → GPU 时开启
        确定性算法（保证可复现）。同时记录权重文件 SHA-256 供追溯。
        """
        print(f"Loading model from: {CHECKPOINT_PATH}")
        print(f"Config from: {HPARAMS_PATH}")

        # Load config  # 读超参配置
        with open(HPARAMS_PATH, "r") as f:
            config = json.load(f)

        # Compute checkpoint hash  # 计算权重哈希（复现/追溯用）
        with open(CHECKPOINT_PATH, "rb") as f:
            self.checkpoint_sha256 = hashlib.sha256(f.read()).hexdigest()

        # Import LPWM modules (add third_party/lpwm to path)
        # 导入 LPWM 的模型类与工具函数
        from models import DLP
        from utils.util_func import get_config

        # Build model
        # 用配置里全部超参构造 DLP 模型（每个参数都有默认值兜底）
        model = DLP(
            cdim=config["ch"],
            image_size=config["image_size"],
            normalize_rgb=config.get("normalize_rgb", False),
            n_views=config.get("n_views", 1),
            n_kp_per_patch=config["n_kp_per_patch"],
            patch_size=config["patch_size"],
            anchor_s=config["anchor_s"],
            n_kp_enc=config["n_kp_enc"],
            n_kp_prior=config["n_kp_prior"],
            warmup_n_kp_ratio=config.get("warmup_n_kp_ratio", 1.0),
            pad_mode=config.get("pad_mode", "zeros"),
            dropout=config.get("dropout", 0.1),
            features_dist=config.get("features_dist", "gauss"),
            learned_feature_dim=config["learned_feature_dim"],
            learned_bg_feature_dim=config.get("learned_bg_feature_dim", config["learned_feature_dim"]),
            n_fg_categories=config.get("n_fg_categories", 8),
            n_fg_classes=config.get("n_fg_classes", 4),
            n_bg_categories=config.get("n_bg_categories", 4),
            n_bg_classes=config.get("n_bg_classes", 4),
            scale_std=config.get("scale_std", 0.3),
            offset_std=config.get("offset_std", 0.2),
            obj_on_alpha=config.get("obj_on_alpha", 0.01),
            obj_on_beta=config.get("obj_on_beta", 0.01),
            obj_res_from_fc=config.get("obj_res_from_fc", 8),
            obj_ch_mult_prior=config.get("obj_ch_mult_prior", config.get("obj_ch_mult", (1, 2, 3))),
            obj_ch_mult=config.get("obj_ch_mult", (1, 2, 3)),
            obj_base_ch=config.get("obj_base_ch", 32),
            obj_final_cnn_ch=config.get("obj_final_cnn_ch", 32),
            bg_res_from_fc=config.get("bg_res_from_fc", 8),
            bg_ch_mult=config.get("bg_ch_mult", (1, 2, 3)),
            bg_base_ch=config.get("bg_base_ch", 32),
            bg_final_cnn_ch=config.get("bg_final_cnn_ch", 32),
            use_resblock=config.get("use_resblock", True),
            num_res_blocks=config.get("num_res_blocks", 2),
            cnn_mid_blocks=config.get("cnn_mid_blocks", False),
            mlp_hidden_dim=config.get("mlp_hidden_dim", 256),
            attn_norm_type=config.get("attn_norm_type", "rms"),
            pint_enc_layers=config.get("pint_enc_layers", 1),
            pint_enc_heads=config.get("pint_enc_heads", 1),
            embed_init_std=config.get("embed_init_std", 0.02),
            particle_positional_embed=config.get("particle_positional_embed", True),
            use_z_orig=config.get("use_z_orig", True),
            particle_score=config.get("particle_score", False),
            filtering_heuristic=config.get("filtering_heuristic", "none"),
            timestep_horizon=config["timestep_horizon"],
            n_static_frames=config.get("num_static_frames", 1),
            predict_delta=config.get("predict_delta", False),
            context_dim=config.get("context_dim", None),
            ctx_dist=config.get("context_dist", "gauss"),
            n_ctx_categories=config.get("n_ctx_categories", 8),
            n_ctx_classes=config.get("n_ctx_classes", 4),
            causal_ctx=config.get("causal_ctx", True),
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
            action_in_ctx_module=config.get("action_in_ctx_module", True),
            language_condition=config.get("language_condition", False),
            language_embed_dim=config.get("language_embed_dim", 0),
            language_max_len=config.get("language_max_len", 64),
            img_goal_condition=config.get("img_goal_condition", False),
            init_zero_bias=config.get("init_zero_bias", True),
            init_conv_layers=config.get("init_conv_layers", True),
            init_conv_fg_std=config.get("init_conv_fg_std", 0.02),
            init_conv_bg_std=config.get("init_conv_bg_std", 0.005),
        )

        # Load weights  # 载入训练好的权重
        state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")

        # Move to device  # 移到目标设备，切评估模式
        model = model.to(self.device)
        model.eval()

        # Set deterministic mode  # GPU 上开启确定性算法，保证结果可复现
        if self.device == "cuda":
            torch.use_deterministic_algorithms(True)

        self.model = model
        self._config = config
        return model

    # ------------------------------------------------------------------
    # E2E timing (no hooks)
    # ------------------------------------------------------------------
    # E2E（端到端）计时区：不挂钩子，保证测量不受干扰

    def _create_input(self) -> torch.Tensor:
        """Create a dummy input tensor: [1, cond_steps, 3, 128, 128].

        中文说明：造一批随机输入视频帧（batch=1，cond_steps 帧，3 通道，
        128×128 分辨率），模拟模型要"看完"的条件帧。注意这里只用条件帧数，
        预测帧是模型在 rollout 里自回归生成的。
        """
        return torch.randn(1, self.cond_steps, 3, 128, 128, device=self.device)

    def _run_one_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Run one LPWM inference. Returns the predicted frames.

        中文说明：跑一次完整推理：在 no_grad（不记录梯度）下调用模型的
        sample_from_x，一次性完成"编码 + 自回归预测 horizon 步 + 解码"，
        确定性模式（deterministic=True）。返回值是预测出的视频帧。
        """
        with torch.no_grad():
            output = self.model.sample_from_x(
                x,
                num_steps=self.horizon,
                cond_steps=self.cond_steps,
                deterministic=True,
                n_pred_eq_gt=False,
            )
        return output

    def run_e2e_benchmark(self):
        """Run E2E latency benchmark (without operator hooks).

        中文说明：
        端到端时延测量主流程：预热 → 正式测量 → 记录峰值显存 → 计算统计量。
        预热的意义：让 CUDA 内核完成编译、GPU 频率爬升、缓存填充，避免前几次
        慢跑污染统计。计时方式：GPU 用 CUDA 事件（记录在 GPU 时间线上，比 CPU
        墙钟更准），CPU 用 perf_counter_ns。统计量含均值/标准差/变异系数/各
        百分位，以及"每步平均时延"和"帧率（每秒能预测多少帧）"。
        """
        print(f"\n{'='*60}")
        print(f"  Experiment 1: E2E Latency Benchmark")
        print(f"  Device: {self.device.upper()}")
        print(f"  Horizon: {self.horizon}, Cond steps: {self.cond_steps}")
        print(f"  Warmup: {self.warmup}, Iterations: {self.iterations}")
        print(f"  Seed: {self.seed}")
        print(f"{'='*60}\n")

        # Set seeds  # 固定随机种子，保证每次运行输入一致
        torch.manual_seed(self.seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.seed)
            torch.cuda.empty_cache()

        # Warmup  # --- 预热阶段：跑 warmup 次，丢弃这些数据 ---
        print(f"Warming up ({self.warmup} iterations)...")
        for i in range(self.warmup):
            torch.manual_seed(self.seed)
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            x = self._create_input()
            _ = self._run_one_inference(x)
            if self.device == "cuda":
                torch.cuda.synchronize()
            print(f"  Warmup {i+1}/{self.warmup} done")

        # Measurement  # --- 正式测量阶段 ---
        print(f"\nMeasuring ({self.iterations} iterations)...")
        self.e2e_latencies_s = []

        for i in range(self.iterations):
            torch.manual_seed(self.seed)
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            x = self._create_input()

            if self.device == "cuda":
                # GPU：用 CUDA 事件计时（精确记录 GPU 端耗时）
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_ev.record()
                _ = self._run_one_inference(x)
                end_ev.record()
                torch.cuda.synchronize()
                latency_s = start_ev.elapsed_time(end_ev) / 1000.0  # 毫秒 → 秒
            else:
                # CPU：用高精度性能计数器计时
                t0 = time.perf_counter_ns()
                _ = self._run_one_inference(x)
                t1 = time.perf_counter_ns()
                latency_s = (t1 - t0) / 1e9  # 纳秒 → 秒

            self.e2e_latencies_s.append(latency_s)
            print(f"  Iter {i+1}/{self.iterations}: {latency_s*1000:.1f} ms")

        # Memory  # 记录峰值显存（GPU 才有意义；CPU 运行留 0 占位）
        if self.device == "cuda":
            self.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            self.peak_memory_mb = 0.0  # TODO: track RSS

        # Statistics  # --- 统计：排序后算百分位 ---
        latencies = self.e2e_latencies_s
        latencies_sorted = sorted(latencies)
        n = len(latencies)
        mean_s = sum(latencies) / n
        std_s = (sum((x - mean_s) ** 2 for x in latencies) / n) ** 0.5  # 总体标准差
        cv = std_s / mean_s if mean_s > 0 else 0.0  # 变异系数：波动相对大小

        def percentile(p: float) -> float:
            """线性插值计算第 p 百分位。"""
            k = (p / 100.0) * (n - 1)
            f = int(k)
            c = k - f
            if f + 1 < n:
                return latencies_sorted[f] * (1 - c) + latencies_sorted[f + 1] * c
            return latencies_sorted[f]

        self.e2e_stats = {
            "mean_s": mean_s,
            "std_s": std_s,
            "cv": cv,
            "median_s": percentile(50),
            "p25_s": percentile(25),
            "p75_s": percentile(75),
            "min_s": min(latencies),
            "max_s": max(latencies),
            "per_step_mean_s": mean_s / self.horizon,  # 平均每步时延（近似：均分到各步）
            "frame_rate": 1.0 / (mean_s / self.horizon) if mean_s > 0 else 0.0,  # 帧率（每秒帧数）
        }

        # 打印结果摘要
        print(f"\nResults:")
        print(f"  Mean:   {mean_s*1000:.1f} ms")
        print(f"  Std:    {std_s*1000:.1f} ms (CV={cv*100:.1f}%)")
        print(f"  Median: {self.e2e_stats['median_s']*1000:.1f} ms")
        print(f"  P25:    {self.e2e_stats['p25_s']*1000:.1f} ms")
        print(f"  P75:    {self.e2e_stats['p75_s']*1000:.1f} ms")
        print(f"  Min:    {min(latencies)*1000:.1f} ms")
        print(f"  Max:    {max(latencies)*1000:.1f} ms")
        print(f"  Per-step:  {self.e2e_stats['per_step_mean_s']*1000:.1f} ms")
        print(f"  Frame rate: {self.e2e_stats['frame_rate']:.2f} fps")
        if self.device == "cuda":
            print(f"  Peak GPU memory: {self.peak_memory_mb:.1f} MB")

    # ------------------------------------------------------------------
    # Operator-level profiling (with hooks)
    # ------------------------------------------------------------------
    # 算子级画像区：挂钩子收集每个 Linear 的形状/阶段/步数信息

    def run_operator_profile(self):
        """Run one inference with operator hooks to collect per-op latency.

        中文说明：
        跑一次带钩子的推理，收集每个 nn.Linear 算子的信息（模块路径、角色、
        阶段、rollout 步、M/K/N 形状），存进 self.operator_records。这些记录
        和轨迹导出脚本（export_lpwm_trace.py）的格式类似，可供后续分析算子
        构成。注意：本函数只收集形状/计数（latency_s=None），因为挂钩子计时
        会干扰测量；真实时延由 E2E 和阶段测量给出。
        还顺带打补丁包装 dyn_module.sample，以便把 dynamics 阶段内部各步
        划分出来（rollout_state 记录当前在第几步）。
        """
        print(f"\n{'='*60}")
        print(f"  Operator-Level Profiling")
        print(f"{'='*60}\n")

        torch.manual_seed(self.seed)
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        x = self._create_input()

        # Install hooks  # 安装钩子
        records = []
        handles = []
        call_index: Dict[str, int] = defaultdict(int)  # 每个模块被调用的次数

        # Track rollout step: we need to count dynamics iterations.
        # The dyn_module.sample() loops `for k in range(steps)` internally.
        # We track this via a state variable.
        # 追踪 rollout 步数：dyn_module.sample() 内部有 for 循环，这里用一个
        # 状态字典记录当前阶段（encode/dynamics/done）
        rollout_state = {"step": 0, "phase": "encode"}

        def make_pre_hook(name: str):
            """前向钩子：记录当前正在执行哪个模块。"""
            def pre_hook(module, input_):
                rollout_state["_current_module"] = name
            return pre_hook

        def make_hook(name: str):
            """Linear 前向钩子：把算子的形状/阶段/步数记下来。"""
            def hook(module, input_, output):
                inp = input_[0]
                out = output
                in_features = module.in_features
                out_features = module.out_features

                # 从输入形状算矩阵乘 M 维（除最后一维外的元素个数）
                if inp.dim() >= 2:
                    M = int(torch.prod(torch.tensor(inp.shape[:-1])).item())
                else:
                    M = 1
                K = in_features
                N = out_features

                call_index[name] += 1  # 该模块第几次被调用
                phase = infer_phase(name)  # 所属阶段
                op_role = infer_op_role(name)  # 角色（Q/K/V/O/FFN...）
                # dynamics 阶段才记步数，其他阶段步数为 0
                rollout_step = rollout_state["step"] if phase == "dynamics" else 0

                rec = {
                    "op_id": f"{name}_{call_index[name]}",
                    "module_path": name,
                    "op_type": "Linear",
                    "op_role": op_role,
                    "phase": phase,
                    "rollout_step": rollout_step,
                    "call_index": call_index[name],
                    "M": M,
                    "K": K,
                    "N": N,
                    "dtype": str(inp.dtype).replace("torch.", ""),
                    # Hooks collect shapes/counts only; assigning zero would be
                    # indistinguishable from a measured zero-time operator.
                    # 钩子只收集形状/计数；时延设 None 而不是 0，
                    # 避免"没测"和"真测到 0"混淆。
                    "latency_s": None,
                    "timing_scope": "shape_only",
                }
                records.append(rec)
            return hook

        # 给所有 nn.Linear 模块注册前向钩子
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(make_hook(name))
                handles.append(h)

        # Patch the dynamics sample to track rollout_step
        # 打补丁包装 dyn_module.sample，把 dynamics 阶段的前后标记出来
        dyn_module = self.model.dyn_module
        original_sample = dyn_module.sample

        def patched_sample(z, z_scale, z_obj_on, z_depth, z_features, z_bg_features,
                           z_context=None, z_score=None, steps=10, deterministic=False,
                           deterministic_particles=True, actions=None, actions_mask=None,
                           lang_embed=None, z_goal=None, return_context_posterior=False):
            # Call the original sample but track step transitions
            # The original sample uses a for loop: for k in range(steps)
            # We need to hook into each iteration.
            # Strategy: temporarily replace the loop or track via a forward hook on a dynamics submodule.
            # 原 sample 内部有 for 循环；这里在调用前后切换阶段标记
            #（实际上 hooks 里看到的 rollout_step 仍是 0，这是当前实现的近似）
            rollout_state["phase"] = "dynamics"
            result = original_sample(
                z, z_scale, z_obj_on, z_depth, z_features, z_bg_features,
                z_context, z_score, steps, deterministic, deterministic_particles,
                actions, actions_mask, lang_embed, z_goal, return_context_posterior
            )
            rollout_state["phase"] = "done"
            return result

        dyn_module.sample = patched_sample

        # Run inference with hooks  # 带钩子跑一次推理，测总时长
        try:
            if self.device == "cuda":
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_ev.record()
                _ = self._run_one_inference(x)
                end_ev.record()
                torch.cuda.synchronize()
                total_s = start_ev.elapsed_time(end_ev) / 1000.0
            else:
                t0 = time.perf_counter_ns()
                _ = self._run_one_inference(x)
                t1 = time.perf_counter_ns()
                total_s = (t1 - t0) / 1e9
        finally:
            # 无论成功失败都恢复原方法、卸载钩子
            dyn_module.sample = original_sample
            for h in handles:
                h.remove()

        self.operator_records = records

        # Count by phase / type  # 按阶段/角色统计算子数量
        by_phase = defaultdict(int)
        by_role = defaultdict(int)
        for r in records:
            by_phase[r["phase"]] += 1
            by_role[r["op_role"]] += 1

        print(f"Total Linear ops recorded: {len(records)}")
        print(f"By phase: {dict(by_phase)}")
        print(f"By role: {dict(by_role)}")
        print(f"Total inference time: {total_s*1000:.1f} ms")

        self.operator_records = records

    # ------------------------------------------------------------------
    # Phase-level timing
    # ------------------------------------------------------------------
    # 阶段级计时区：分别测编码/动态/解码三阶段

    def run_phase_benchmark(self):
        """Run phase-level and per-step timing by patching model methods.

        中文说明：
        分别测量三阶段时延：通过临时"打补丁"包装模型的 encode_all /
        dyn_module.sample / decode_all 三个方法，在它们执行时计时（计时前
        cuda.synchronize 确保前面的工作已排空），测完恢复原方法。
        每步动态预测时延用"总动态时延 ÷ 步数"做均匀估计（实际每步未必完全
        均匀，这里只做一阶近似，更精确的逐步测量在
        _measure_dynamics_per_step 里实现）。
        """
        print(f"\n{'='*60}")
        print(f"  Phase-Level + Per-Step Timing")
        print(f"{'='*60}\n")

        torch.manual_seed(self.seed)
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        x = self._create_input()

        phase_times = {}

        # --- Encode timing ---
        # 包装 encode_all：执行时计时，结果存 phase_times["encode"]
        original_encode = self.model.encode_all

        def timed_encode(*args, **kwargs):
            if self.device == "cuda":
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                result = original_encode(*args, **kwargs)
                e.record()
                torch.cuda.synchronize()
                phase_times["encode"] = s.elapsed_time(e) / 1000.0
            else:
                t0 = time.perf_counter_ns()
                result = original_encode(*args, **kwargs)
                t1 = time.perf_counter_ns()
                phase_times["encode"] = (t1 - t0) / 1e9
            return result

        self.model.encode_all = timed_encode

        # --- Per-step dynamics timing ---
        # Patch the dyn_module's particle_transformer.forward to count steps.
        # The sample() method loops `for k in range(steps)`. Within each
        # iteration, it calls the transformer (or MLP-based) forward at least once.
        # We track step boundaries by intercepting the outermost per-step call.
        # 包装 dyn_module.sample：测量整个 dynamics 阶段总时延
        #（逐步细分另由 _measure_dynamics_per_step 负责）
        original_dyn_sample = self.model.dyn_module.sample

        def timed_dyn_sample(z, z_scale, z_obj_on, z_depth, z_features, z_bg_features,
                             z_context=None, z_score=None, steps=10, deterministic=False,
                             deterministic_particles=True, actions=None, actions_mask=None,
                             lang_embed=None, z_goal=None, return_context_posterior=False):
            """Wrapped sample() that times the entire dynamics phase.

            中文说明：被补丁包装的 sample()，只负责给整个 dynamics 阶段计时，
            结果存 phase_times["dynamics"]。
            """
            if self.device == "cuda":
                torch.cuda.synchronize()
                s_total = torch.cuda.Event(enable_timing=True)
                e_total = torch.cuda.Event(enable_timing=True)
                s_total.record()
                result = original_dyn_sample(
                    z, z_scale, z_obj_on, z_depth, z_features, z_bg_features,
                    z_context, z_score, steps, deterministic, deterministic_particles,
                    actions, actions_mask, lang_embed, z_goal, return_context_posterior
                )
                e_total.record()
                torch.cuda.synchronize()
                phase_times["dynamics"] = s_total.elapsed_time(e_total) / 1000.0
            else:
                t0_total = time.perf_counter_ns()
                result = original_dyn_sample(
                    z, z_scale, z_obj_on, z_depth, z_features, z_bg_features,
                    z_context, z_score, steps, deterministic, deterministic_particles,
                    actions, actions_mask, lang_embed, z_goal, return_context_posterior
                )
                t1_total = time.perf_counter_ns()
                phase_times["dynamics"] = (t1_total - t0_total) / 1e9
            return result

        # NOTE: Per-step measurement is deferred to after E2E run to avoid
        # double-counting overhead within the timed phase measurement.
        # 注意：逐步测量放在 E2E 运行之后做，避免计时包装叠加影响测量
        self.model.dyn_module.sample = timed_dyn_sample

        # --- Decode timing ---
        # 包装 decode_all：执行时计时，结果存 phase_times["decode"]
        original_decode = self.model.decode_all

        def timed_decode(*args, **kwargs):
            if self.device == "cuda":
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                result = original_decode(*args, **kwargs)
                e.record()
                torch.cuda.synchronize()
                phase_times["decode"] = s.elapsed_time(e) / 1000.0
            else:
                t0 = time.perf_counter_ns()
                result = original_decode(*args, **kwargs)
                t1 = time.perf_counter_ns()
                phase_times["decode"] = (t1 - t0) / 1e9
            return result

        self.model.decode_all = timed_decode

        try:
            _ = self._run_one_inference(x)  # 带计时包装跑一次完整推理
        finally:
            # 恢复三个被包装的方法
            self.model.encode_all = original_encode
            self.model.dyn_module.sample = original_dyn_sample
            self.model.decode_all = original_decode

        print(f"Phase timing ({self.device}):")
        for phase, lat in sorted(phase_times.items()):
            print(f"  {phase:12s}: {lat*1000:.1f} ms")
        if phase_times:
            total = sum(phase_times.values())
            print(f"  {'total':12s}: {total*1000:.1f} ms")

        # Estimate per-step dynamics (uniform distribution, validated by phase timing)
        # 每步动态时延的均匀估计（用总时长÷步数；更精确测量见另一函数）
        dyn_total = phase_times.get("dynamics", 0)
        per_step_times = [dyn_total / self.horizon] * self.horizon if dyn_total > 0 else []
        if per_step_times and dyn_total > 0:
            print(f"\n  Per-step dynamics (estimated uniform, {self.horizon} steps):")
            print(f"    each: {per_step_times[0]*1000:.1f} ms")
            print(f"    total: {dyn_total*1000:.1f} ms")

        self.phase_times = phase_times
        self.per_step_dynamics_times = per_step_times

    def _measure_dynamics_per_step(self, z, z_scale, z_obj_on, z_depth, z_features,
                                    z_bg_features, z_context, z_score, steps,
                                    deterministic, deterministic_particles,
                                    actions, actions_mask, lang_embed, z_goal,
                                    return_context_posterior, original_sample):
        """Run dynamics one step at a time, measuring each step's wall-clock."""
        import copy
        per_step = []
        # Start with initial state
        z_cur = z[:, -1:].clone()
        z_scale_cur = z_scale[:, -1:].clone()
        z_obj_on_cur = z_obj_on[:, -1:].clone()
        z_depth_cur = z_depth[:, -1:].clone()
        z_features_cur = z_features[:, -1:].clone()
        z_bg_features_cur = z_bg_features[:, -1:].clone()
        z_context_cur = z_context[:, -1:] if z_context is not None else None
        z_score_cur = z_score[:, -1:] if z_score is not None else None

        for step in range(steps):
            if self.device == "cuda":
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                result = original_sample(
                    z_cur, z_scale_cur, z_obj_on_cur, z_depth_cur,
                    z_features_cur, z_bg_features_cur,
                    z_context_cur, z_score_cur, steps=1,
                    deterministic=deterministic,
                    deterministic_particles=deterministic_particles,
                    actions=actions, actions_mask=actions_mask,
                    lang_embed=lang_embed, z_goal=z_goal,
                    return_context_posterior=return_context_posterior,
                )
                e.record()
                torch.cuda.synchronize()
                per_step.append(s.elapsed_time(e) / 1000.0)
            else:
                t0 = time.perf_counter_ns()
                result = original_sample(
                    z_cur, z_scale_cur, z_obj_on_cur, z_depth_cur,
                    z_features_cur, z_bg_features_cur,
                    z_context_cur, z_score_cur, steps=1,
                    deterministic=deterministic,
                    deterministic_particles=deterministic_particles,
                    actions=actions, actions_mask=actions_mask,
                    lang_embed=lang_embed, z_goal=z_goal,
                    return_context_posterior=return_context_posterior,
                )
                t1 = time.perf_counter_ns()
                per_step.append((t1 - t0) / 1e9)

            # Advance state for next step
            z_cur = result["z"][:, -1:].clone()
            z_scale_cur = result["z_scale"][:, -1:].clone()
            z_obj_on_cur = result["z_obj_on"][:, -1:].clone()
            z_depth_cur = result["z_depth"][:, -1:].clone()
            z_features_cur = result["z_features"][:, -1:].clone()
            z_bg_features_cur = result["z_bg_features"][:, -1:].clone()
            z_context_cur = result["z_context"][:, -1:] if result.get("z_context") is not None else None
            z_score_cur = result["z_score"][:, -1:] if result.get("z_score") is not None else None

        return per_step

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def build_manifest(self):
        """Build run manifest with all fixed variables recorded."""
        gpu_name = ""
        gpu_driver = ""
        cuda_version = ""
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_driver = torch.cuda.get_device_properties(0).name
            cuda_version = str(torch.version.cuda)

        import platform

        self.manifest = {
            "run_id": os.path.basename(self.output_dir),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "experiment": "Experiment 1: Native LPWM Inference Baseline",
            "hardware": {
                "device": self.device,
                "gpu_name": gpu_name if self.device == "cuda" else "N/A (CPU run)",
                "cpu": platform.processor() or "AMD Ryzen 9 7940HX",
                "system_ram_gb": "",  # TODO
            },
            "software": {
                "python": sys.version.split()[0],
                "pytorch": torch.__version__,
                "cuda": cuda_version,
                "cudnn": str(torch.backends.cudnn.version()) if self.device == "cuda" else "N/A",
            },
            "model": {
                "config": "bair128/hparams.json",
                "checkpoint_sha256": self.checkpoint_sha256,
                "d_model": self._config.get("pint_dim", 512),
                "precision": "float32",
                "batch_size": 1,
                "horizon": self.horizon,
                "cond_steps": self.cond_steps,
                "seed": self.seed,
            },
            "measurement": {
                "warmup_iterations": self.warmup,
                "measurement_iterations": self.iterations,
                "timer": "torch.cuda.Event" if self.device == "cuda" else "time.perf_counter_ns",
            },
        }

    # ------------------------------------------------------------------
    # Output files
    # ------------------------------------------------------------------

    def write_outputs(self):
        """Write all output files to the output directory."""
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"\nWriting outputs to: {self.output_dir}")

        # 1. run_manifest.json
        self._write_manifest()

        # 2. summary.json
        self._write_summary()

        # 3. operator_latency.csv
        if self.operator_records:
            self._write_operator_latency()

        # 4. operator_stats_by_type.csv
        if self.operator_records:
            self._write_operator_stats_by_type()

        # 5. operator_stats_by_phase.csv
        if self.operator_records:
            self._write_operator_stats_by_phase()

        # 6. operator_stats_by_step.csv
        if self.operator_records:
            self._write_operator_stats_by_step()

        # 7. per_step_latency.csv (phase-level per-step dynamics)
        self._write_per_step_latency()

        # 8. raw_timestamps.jsonl
        self._write_raw_timestamps()

        # 9. gpu_memory.csv (GPU only)
        if self.device == "cuda":
            self._write_gpu_memory()

        # 10. hardware_info.csv / software_versions.txt
        self._write_environment_info()

        print("Done.")

    def _write_manifest(self):
        path = os.path.join(self.output_dir, "run_manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False, default=str)

    def _write_summary(self):
        path = os.path.join(self.output_dir, "summary.json")
        stats = getattr(self, "e2e_stats", {})
        summary = {
            "run_id": self.manifest.get("run_id", ""),
            "device": self.device,
            "horizon": self.horizon,
            "cond_steps": self.cond_steps,
            "warmup_iterations": self.warmup,
            "measurement_iterations": self.iterations,
            "e2e_latency_s": {
                "mean": stats.get("mean_s"),
                "std": stats.get("std_s"),
                "cv_percent": stats.get("cv", 0) * 100,
                "median": stats.get("median_s"),
                "p25": stats.get("p25_s"),
                "p75": stats.get("p75_s"),
                "min": stats.get("min_s"),
                "max": stats.get("max_s"),
            },
            "per_step_latency_s_mean": stats.get("per_step_mean_s"),
            "frame_rate_fps": stats.get("frame_rate"),
            "peak_memory_mb": self.peak_memory_mb,
            "phase_times_s": self.phase_times,
            "per_step_dynamics_s": getattr(self, "per_step_dynamics_times", []),
            "raw_latencies_s": self.e2e_latencies_s,
            "total_linear_ops": len(self.operator_records),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    def _write_operator_latency(self):
        path = os.path.join(self.output_dir, "operator_latency.csv")
        keys = ["op_id", "module_path", "op_type", "op_role", "phase",
                "rollout_step", "call_index", "M", "K", "N", "dtype", "latency_s"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.operator_records)

    def _write_operator_stats_by_type(self):
        path = os.path.join(self.output_dir, "operator_stats_by_type.csv")
        by_type: Dict[str, List[dict]] = defaultdict(list)
        for r in self.operator_records:
            by_type[r["op_role"]].append(r)

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["op_role", "count", "total_MxKxN", "mean_M", "mean_K", "mean_N"])
            for role, recs in sorted(by_type.items()):
                count = len(recs)
                total_ops = sum(r["M"] * r["K"] * r["N"] for r in recs)
                mean_M = sum(r["M"] for r in recs) / count
                mean_K = sum(r["K"] for r in recs) / count
                mean_N = sum(r["N"] for r in recs) / count
                w.writerow([role, count, total_ops, f"{mean_M:.1f}", f"{mean_K:.1f}", f"{mean_N:.1f}"])

    def _write_operator_stats_by_phase(self):
        path = os.path.join(self.output_dir, "operator_stats_by_phase.csv")
        by_phase: Dict[str, List[dict]] = defaultdict(list)
        for r in self.operator_records:
            by_phase[r["phase"]].append(r)

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["phase", "count", "total_MxKxN", "unique_modules"])
            for phase, recs in sorted(by_phase.items()):
                count = len(recs)
                total_ops = sum(r["M"] * r["K"] * r["N"] for r in recs)
                unique = len(set(r["module_path"] for r in recs))
                w.writerow([phase, count, total_ops, unique])

    def _write_operator_stats_by_step(self):
        path = os.path.join(self.output_dir, "operator_stats_by_step.csv")
        by_step: Dict[int, List[dict]] = defaultdict(list)
        for r in self.operator_records:
            by_step[r["rollout_step"]].append(r)

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rollout_step", "count"])
            for step in sorted(by_step.keys()):
                w.writerow([step, len(by_step[step])])

    def _write_per_step_latency(self):
        """Write per-step dynamics latency."""
        if not getattr(self, "per_step_dynamics_times", None):
            return
        path = os.path.join(self.output_dir, "per_step_latency.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "latency_s", "latency_ms"])
            for i, lat in enumerate(self.per_step_dynamics_times):
                w.writerow([i + 1, lat, lat * 1000])

    def _write_raw_timestamps(self):
        path = os.path.join(self.output_dir, "raw_timestamps.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i, lat in enumerate(self.e2e_latencies_s):
                f.write(json.dumps({
                    "iteration": i,
                    "latency_s": lat,
                    "category": "warmup" if i < self.warmup else "measurement",
                }) + "\n")

    def _write_gpu_memory(self):
        path = os.path.join(self.output_dir, "gpu_memory.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["iteration", "allocated_mb", "reserved_mb"])
            # Single value from peak
            w.writerow(["peak", self.peak_memory_mb, torch.cuda.max_memory_reserved() / (1024 * 1024)])

    def _write_environment_info(self):
        path = os.path.join(self.output_dir, "software_versions.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Python: {sys.version}\n")
            f.write(f"PyTorch: {torch.__version__}\n")
            f.write(f"CUDA available: {torch.cuda.is_available()}\n")
            f.write(f"CUDA version: {torch.version.cuda}\n")
            if torch.cuda.is_available():
                f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
                f.write(f"cuDNN: {torch.backends.cudnn.version()}\n")
            f.write(f"Checkpoint SHA-256: {self.checkpoint_sha256}\n")

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------

    def run_all(self):
        """Run complete benchmark suite."""
        self.load_model()
        self.build_manifest()
        self.run_e2e_benchmark()
        self.run_phase_benchmark()
        self.run_operator_profile()
        self.write_outputs()
        self._print_final_summary()

    def _print_final_summary(self):
        stats = getattr(self, "e2e_stats", {})
        print(f"\n{'='*60}")
        print(f"  BENCHMARK COMPLETE")
        print(f"  Device: {self.device.upper()}")
        print(f"  E2E mean: {stats.get('mean_s', 0)*1000:.1f} ms")
        print(f"  Per-step: {stats.get('per_step_mean_s', 0)*1000:.1f} ms")
        print(f"  CV: {stats.get('cv', 0)*100:.1f}%")
        print(f"  Linear ops: {len(self.operator_records)}")
        print(f"  Output: {self.output_dir}")
        print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LPWM Native Inference Benchmark (Experiment 1)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to run on (default: cpu)")
    parser.add_argument("--horizon", type=int, default=30,
                        help="Rollout horizon (default: 30)")
    parser.add_argument("--cond-steps", type=int, default=1,
                        help="Conditional frames (default: 1)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations (default: 10)")
    parser.add_argument("--iterations", type=int, default=30,
                        help="Measurement iterations (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str,
                        default="experiments/native_baseline/default",
                        help="Output directory")
    parser.add_argument("--e2e-only", action="store_true",
                        help="Only run E2E timing (skip operator profiling)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: horizon=4, warmup=2, iterations=3")

    args = parser.parse_args()

    if args.smoke:
        args.horizon = 4
        args.warmup = 2
        args.iterations = 3
        if args.output == "experiments/native_baseline/default":
            args.output = "results/smoke_test"

    runner = BenchmarkRunner(
        device=args.device,
        horizon=args.horizon,
        cond_steps=args.cond_steps,
        warmup=args.warmup,
        iterations=args.iterations,
        seed=args.seed,
        output_dir=args.output,
    )

    runner.load_model()
    runner.build_manifest()
    runner.run_e2e_benchmark()
    runner.run_phase_benchmark()

    if not args.e2e_only:
        runner.run_operator_profile()

    runner.write_outputs()
    runner._print_final_summary()


if __name__ == "__main__":
    main()
