"""
Experiment 3: HPAT-style non-ideality injection on LPWM (BAIR-128, 10-step rollout).

Usage (from project root):
    conda run -n 4DLangSplat python joint_sim/nonidealities/run_experiment3.py

Output: results/experiment3/{run_id}/summary.json
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import time
from datetime import datetime
from typing import Any

# ── Path setup ───────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_LPWM_ROOT = os.path.join(_PROJECT_ROOT, "third_party", "lpwm")
for p in [_LPWM_ROOT, os.path.join(_PROJECT_ROOT, "joint_sim"), _PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn

from models import DLP
from nonidealities.perturb import make_perturb_fn, SWEEP_PLAN, STOCHASTIC_EFFECTS

# ── Configuration ────────────────────────────────────────────────────────────
# 实验 3 的完整配置：BAIR-128 数据集，10 步 rollout 预测
HPARAMS_PATH = os.path.join(
    _PROJECT_ROOT, "third_party", "lpwm", "checkpoints", "bair128", "hparams.json"
)
CHECKPOINT_PATH = os.path.join(
    _PROJECT_ROOT, "third_party", "lpwm", "checkpoints", "bair128",
    "bair_gddlp_ts16_best_lpips.pth",
)

ROLLOUT_STEPS = 10       # Generate 10 predicted frames（要预测的帧数=10）
COND_STEPS = 1           # 1 conditioning frame（1 个条件帧）
IMAGE_SIZE = 128
REPEAT_SEEDS = [20260706, 20260707, 20260708, 20260709, 20260710]  # 随机类误差重复 5 个种子
DETERMINISTIC = True     # Remove sampling randomness from comparisons
                         # 关闭采样随机性，保证对比时只反映器件误差

# ── Helpers ──────────────────────────────────────────────────────────────────

def _sha256_hex(data: bytes) -> str:
    """文件内容 SHA-256 指纹（前 16 位），用于记录 checkpoint 身份。"""
    return hashlib.sha256(data).hexdigest()[:16]


def _count_linear_modules(model: nn.Module) -> int:
    """统计模型里 nn.Linear 层总数（=误差注入点位数量）。"""
    return sum(1 for _m in model.modules() if isinstance(_m, nn.Linear))


def _psnr(clean: torch.Tensor, perturbed: torch.Tensor) -> float:
    """PSNR in dB. Expects tensors in [0, 1] or [-1, 1] range."""
    # PSNR（峰值信噪比，dB）：两帧图像越像越大。MSE 小于 1e-12 视为完全一致。
    mse = torch.mean((clean - perturbed) ** 2).item()
    if mse < 1e-12:
        return 100.0
    return float(10.0 * torch.log10(torch.tensor(1.0 / mse)).item())


def _ssim(clean: torch.Tensor, perturbed: torch.Tensor) -> float:
    """Simple SSIM approximation (luminance + contrast terms only)."""
    # 简化的结构相似度指数(SSIM)：只取亮度项 + 对比度项（省略结构项）。
    # 常数 C1/C2 防止除以 0；范围约 [0,1]，越接近 1 越相似。
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x = clean.mean().item()
    mu_y = perturbed.mean().item()
    sigma_x = clean.std().item()
    sigma_y = perturbed.std().item()
    sigma_xy = ((clean - mu_x) * (perturbed - mu_y)).mean().item()
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2)
    return float(num / max(den, 1e-12))


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(device: str = "cuda") -> tuple[nn.Module, dict]:
    """Load BAIR-128 DLP model with checkpoint weights."""
    # 中文说明：按 hparams.json 重建 DLP 视频预测模型并加载 BAIR-128 权重，
    # 返回 (model, config 字典)。
    with open(HPARAMS_PATH, "r") as f:
        config = json.load(f)

    model = DLP(
        cdim=config.get("ch", 3),
        image_size=config.get("image_size", 128),
        normalize_rgb=config.get("normalize_rgb", False),
        n_views=config.get("n_views", 1),
        n_kp_per_patch=config.get("n_kp_per_patch", 1),
        patch_size=config.get("patch_size", 8),
        anchor_s=config.get("anchor_s", 0.125),
        n_kp_enc=config.get("n_kp_enc", 90),
        n_kp_prior=config.get("n_kp_prior", 256),
        pad_mode=config.get("pad_mode", "zeros"),
        dropout=config.get("dropout", 0.1),
        features_dist=config.get("features_dist", "gauss"),
        learned_feature_dim=config.get("learned_feature_dim", 5),
        learned_bg_feature_dim=config.get("learned_bg_feature_dim", 5),
        obj_res_from_fc=config.get("obj_res_from_fc", 4),
        obj_ch_mult_prior=config.get("obj_ch_mult_prior", [2, 4, 8]),
        obj_ch_mult=config.get("obj_ch_mult", [2, 4, 8]),
        obj_base_ch=config.get("obj_base_ch", 32),
        obj_final_cnn_ch=config.get("obj_final_cnn_ch", 32),
        bg_res_from_fc=config.get("bg_res_from_fc", 8),
        bg_ch_mult=config.get("bg_ch_mult", [1, 1, 1, 2, 4]),
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

    # Load checkpoint
    # 加载权重，兼容三种 checkpoint 格式；strict=False 允许键不完全对齐
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model = model.to(device)
    model.eval()
    print(f"Model loaded: {_count_linear_modules(model)} nn.Linear modules on {device}")
    return model, config


# ── Rollout with hooks ───────────────────────────────────────────────────────

def run_rollout(
    model: nn.Module,
    x_input: torch.Tensor,
    perturb_fn: Any | None,
    device: str,
) -> torch.Tensor:
    """Run a 10-step rollout, optionally with perturbation hooks.

    Args:
        model: DLP model (eval mode).
        x_input: [B, T_total, C, H, W] — T_total = cond_steps + rollout_steps.
                  We use cond_steps frames as context, predict the rest.
        perturb_fn: Callable(output, module_name) or None for clean baseline.

    Returns:
        preds: [B, T_total, C, H, W] — full sequence (context + generated).
    """
    # 中文说明：跑 10 步 rollout 推理。有扰动函数时给每个 nn.Linear 挂前向
    # 钩子，在输出后注入器件误差。GPU 上跑完要 cuda.synchronize() 等计算
    # 真正完成（否则计时不准）。try/finally 保证钩子一定卸载。
    hooks = []
    if perturb_fn is not None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(
                    lambda _m, _inp, out, n=name: perturb_fn(out, n)
                )
                hooks.append(hook)

    try:
        with torch.no_grad():
            preds = model.sample_from_x(
                x_input,
                num_steps=ROLLOUT_STEPS,
                deterministic=DETERMINISTIC,
                cond_steps=COND_STEPS,
                use_all_ctx=True,
            )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    finally:
        for hook in hooks:
            hook.remove()

    return preds


# ── Main sweep ───────────────────────────────────────────────────────────────

def run_experiment(output_dir: str) -> dict:
    # 中文说明：完整实验流程——
    #   1) 加载模型（优先 GPU，否则 CPU）；
    #   2) 生成随机输入（本地没有 BAIR 测试集）；
    #   3) 先跑一次干净基线（无误差）；
    #   4) 按 SWEEP_PLAN 扫描：5 种误差 x 每档严重度；随机类误差用 5 个
    #      种子重复实验取平均，确定性误差只跑 1 次；
    #   5) 对每个生成的帧计算 PSNR/SSIM，写 summary.json。
    # 对照组：value=0 的结果应与干净基线几乎一致（PSNR 应 >80），否则说明
    # 钩子/注入本身引入了额外误差（需检查）。
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Rollout steps: {ROLLOUT_STEPS}, Cond steps: {COND_STEPS}")

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    model, config = load_model(device)
    load_time = time.perf_counter() - t0
    print(f"Model load time: {load_time:.1f}s")

    # ── Create input tensor ───────────────────────────────────────────────
    # Use random tensor as input (we don't have BAIR test data locally).
    # Shape: [B=1, T=cond_steps+rollout_steps, C=3, H=128, W=128]
    # 用随机张量当输入（本地没有 BAIR 测试集），形状与真实输入一致
    total_frames = COND_STEPS + ROLLOUT_STEPS
    rng = torch.Generator(device="cpu").manual_seed(20260706)
    x_input = torch.randn(1, total_frames, 3, IMAGE_SIZE, IMAGE_SIZE,
                          generator=rng)
    x_input = x_input.to(device)

    # ── Count injection sites ─────────────────────────────────────────────
    # 注入点位 = 所有 nn.Linear 的输出位置
    linear_module_names = sorted(
        name for name, m in model.named_modules() if isinstance(m, nn.Linear)
    )
    site_count = len(linear_module_names)
    print(f"Injection sites (nn.Linear): {site_count}")

    # ── Clean baseline ────────────────────────────────────────────────────
    print("\n=== Clean baseline ===")
    t0 = time.perf_counter()
    clean_preds = run_rollout(model, x_input, None, device)
    clean_time = time.perf_counter() - t0
    print(f"Clean rollout time: {clean_time:.2f}s")
    print(f"Clean output shape: {tuple(clean_preds.shape)}")

    # ── Sweep ─────────────────────────────────────────────────────────────
    results: list[dict] = []
    total_runs = 0
    completed_runs = 0
    sweep_start = time.perf_counter()

    # 第一遍循环只算总实验次数（用于进度显示与 ETA）
    for effect, variable, values in SWEEP_PLAN:
        for value in values:
            is_stochastic = effect in STOCHASTIC_EFFECTS
            seeds = REPEAT_SEEDS if is_stochastic else [REPEAT_SEEDS[0]]
            for trial_idx, seed in enumerate(seeds):
                total_runs += 1

    print(f"\n=== Sweep: {total_runs} runs ===")
    run_idx = 0

    for effect, variable, values in SWEEP_PLAN:
        for value in values:
            is_stochastic = effect in STOCHASTIC_EFFECTS
            # 随机类误差用多个种子重复，确定性误差只跑第一个种子
            seeds = REPEAT_SEEDS if is_stochastic else [REPEAT_SEEDS[0]]
            for trial_idx, seed in enumerate(seeds):
                run_idx += 1
                # Set torch RNG for reproducibility
                # 固定随机种子，保证同一次实验可复现
                torch.manual_seed(seed)
                if device.startswith("cuda"):
                    torch.cuda.manual_seed(seed)

                perturb_fn = make_perturb_fn(effect, float(value), site_count)

                t0 = time.perf_counter()
                perturbed_preds = run_rollout(model, x_input, perturb_fn, device)
                elapsed = time.perf_counter() - t0

                # Compute metrics per predicted frame (skip cond frame)
                # 逐帧（跳过条件帧）计算 PSNR/SSIM
                step_metrics = []
                for step in range(ROLLOUT_STEPS):
                    frame_idx = COND_STEPS + step
                    clean_frame = clean_preds[0, frame_idx]
                    pert_frame = perturbed_preds[0, frame_idx]
                    psnr_val = _psnr(clean_frame, pert_frame)
                    ssim_val = _ssim(clean_frame, pert_frame)
                    step_metrics.append({
                        "step": step,
                        "psnr_db": round(psnr_val, 4),
                        "ssim": round(ssim_val, 6),
                    })

                # 汇总：取各帧 PSNR 的平均值；quality_drop 用 100 减去平均
                # PSNR 粗估"质量损失"（只对 value>0 的注入有意义）
                psnr_values = [m["psnr_db"] for m in step_metrics]
                avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0.0
                quality_drop = max(0.0, 100.0 - avg_psnr) if value > 0.0 else 0.0

                result = {
                    "effect": effect,
                    "sweep_variable": variable,
                    "sweep_value": value,
                    "trial_index": trial_idx,
                    "seed": seed,
                    "stochastic": is_stochastic,
                    "elapsed_s": round(elapsed, 3),
                    "avg_psnr_db": round(avg_psnr, 4),
                    "quality_drop": round(quality_drop, 4),
                    "per_step": step_metrics,
                }
                results.append(result)

                # ETA 估算：已耗时长 / 已完成次数 * 剩余次数
                eta = (time.perf_counter() - sweep_start) / run_idx * (total_runs - run_idx)
                print(
                    f"[{run_idx}/{total_runs}] {effect} {variable}={value} "
                    f"seed={seed} | {elapsed:.1f}s | avg_PSNR={avg_psnr:.1f}dB "
                    f"| ETA={eta:.0f}s"
                )

    sweep_time = time.perf_counter() - sweep_start

    # ── Summary ───────────────────────────────────────────────────────────
    # 汇总 JSON：实验配置、checkpoint 指纹、计时、干净基线范围、全部结果
    summary = {
        "experiment": "experiment3_nonideality_sweep",
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "config": {
            "hparams_path": HPARAMS_PATH,
            "checkpoint_path": CHECKPOINT_PATH,
            "checkpoint_sha256": _sha256_hex(open(CHECKPOINT_PATH, "rb").read()),
            "rollout_steps": ROLLOUT_STEPS,
            "cond_steps": COND_STEPS,
            "image_size": IMAGE_SIZE,
            "deterministic": DETERMINISTIC,
            "repeat_seeds": REPEAT_SEEDS,
            "linear_module_count": site_count,
        },
        "timing": {
            "model_load_s": round(load_time, 2),
            "clean_rollout_s": round(clean_time, 3),
            "total_sweep_s": round(sweep_time, 1),
            "total_runs": total_runs,
            "avg_per_run_s": round(sweep_time / total_runs, 3),
            "estimated_cpu_hours": round(sweep_time / 3600 * 5, 1),  # rough CPU factor
        },
        "clean_baseline": {
            "output_shape": list(clean_preds.shape),
            "frame_range": [float(clean_preds.min()), float(clean_preds.max())],
        },
        "results": results,
    }

    # ── Write outputs ─────────────────────────────────────────────────────
    summary_path = out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")

    # Quick check: severity=0 should match clean baseline
    # 自检：value=0（无误差）的结果应与干净基线一致，PSNR 应 >80dB；
    # 若不满足，多半是钩子本身引入了开销/误差
    zero_results = [r for r in results if r["sweep_value"] == 0.0]
    psnr_checks = [r["avg_psnr_db"] for r in zero_results]
    if psnr_checks:
        min_psnr = min(psnr_checks)
        print(f"Severity=0 PSNR range: {min(psnr_checks):.1f}–{max(psnr_checks):.1f} dB "
              f"({'PASS' if min_psnr > 80 else 'CHECK — may indicate hook overhead'})")

    return summary


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 命令行入口：以时间戳作为 run_id，结果写到 results/experiment3/run_<时间戳>
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        _PROJECT_ROOT, "results", "experiment3", f"run_{run_id}"
    )
    print(f"Output directory: {output_dir}")
    run_experiment(output_dir)
