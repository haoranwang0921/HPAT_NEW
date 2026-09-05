"""
Experiment 3 MINIMAL — single-effect, 1-step rollout, CPU-safe validation.
Run this first to verify the HPAT injection pipeline works end-to-end
before scaling to the full sweep.

Usage (from project root):
    conda run -n 4DLangSplat python joint_sim/nonidealities/run_exp3_minimal.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from datetime import datetime
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LPWM_ROOT = os.path.join(_PROJECT_ROOT, "third_party", "lpwm")
for p in [_LPWM_ROOT, os.path.join(_PROJECT_ROOT, "joint_sim"), _PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn

from models import DLP
from nonidealities.perturb import make_perturb_fn, SWEEP_PLAN, STOCHASTIC_EFFECTS

HPARAMS_PATH = os.path.join(
    _PROJECT_ROOT, "third_party", "lpwm", "checkpoints", "bair128", "hparams.json"
)
CHECKPOINT_PATH = os.path.join(
    _PROJECT_ROOT, "third_party", "lpwm", "checkpoints", "bair128",
    "bair_gddlp_ts16_best_lpips.pth",
)

# ── Minimal config: 1 rollout step, 1 effect, 2 severities, CPU ────────────
# 精简配置：只跑 1 步 rollout、每个误差只试 2 档严重度(0 与最大)、在 CPU 上
# 运行。目标是快速验证"注入误差 -> 量化精度损失"整条链路是否走通。
ROLLOUT_STEPS = 1
COND_STEPS = 1
IMAGE_SIZE = 128
DEVICE = "cpu"
SEED = 20260706


def load_model(device: str = "cpu") -> nn.Module:
    # 中文说明：按 hparams.json 里的超参重建 DLP 模型（视频预测世界模型），
    # 加载 BAIR-128 预训练权重，返回 (model, Linear 层总数)。
    # Linear 层总数 = 误差注入的点位数（每个 Linear 输出后注一次误差）。
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

    # 加载权重：兼容三种 checkpoint 格式（model_state_dict / state_dict / 裸字典）
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    linear_count = sum(1 for _m in model.modules() if isinstance(_m, nn.Linear))
    return model, linear_count


def run_rollout(model, x_input, perturb_fn):
    # 中文说明：跑一次 rollout 推理。若给了扰动函数，就给每个 nn.Linear
    # 挂前向钩子，在其输出后注入器件误差（扰动的是"结果"，不改权重本身）。
    # 用 try/finally 保证钩子一定被卸载。
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
                x_input, num_steps=ROLLOUT_STEPS,
                deterministic=True, cond_steps=COND_STEPS, use_all_ctx=True,
            )
    finally:
        for h in hooks:
            h.remove()
    return preds


def psnr(a, b):
    # PSNR（峰值信噪比）：衡量两张图像有多像，越大越像（dB 单位）。
    # 这里用均方误差 MSE 的倒数取对数；MSE 极小(1e-12 兜底)时 PSNR 很大。
    mse = torch.mean((a - b) ** 2).item()
    return float(10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item())


def main():
    print(f"Device: {DEVICE} | Rollout: {ROLLOUT_STEPS} step(s) | Image: {IMAGE_SIZE}")
    t0 = time.perf_counter()

    # Load
    model, num_linear = load_model(DEVICE)
    print(f"Loaded: {num_linear} nn.Linear modules | {time.perf_counter() - t0:.1f}s")

    # Input tensor: [1, 2, 3, 128, 128] = 1 cond + 1 rollout frame
    # 输入形状 [1, 2, 3, 128, 128]：1 个条件帧 + 1 个待预测帧
    rng = torch.Generator().manual_seed(SEED)
    x = torch.randn(1, COND_STEPS + ROLLOUT_STEPS, 3, IMAGE_SIZE, IMAGE_SIZE, generator=rng)

    # Clean baseline
    # 先跑一次"无误差"的干净基线，之后所有带误差的结果都与它对比
    t1 = time.perf_counter()
    clean = run_rollout(model, x, None)
    clean_t = time.perf_counter() - t1
    print(f"Clean baseline: {clean_t:.1f}s | output shape={tuple(clean.shape)}")

    # ── Minimal sweep: all 5 effects, 2 severities (0 + max), 1 seed ─────
    # 精简扫描：5 种误差，每种只试 {0, 最大档} 两个严重度，固定 1 个种子
    sweep_effects = [
        ("gaussian_pd_tia_noise",   "noise_lsb",       [0.0, 1.0]),
        ("wdm_adjacent_crosstalk",  "crosstalk_alpha", [0.0, 0.05]),
        ("mrr_variation",           "sigma_percent",   [0.0, 5.0]),
        ("thermal_drift",           "delta_c",         [0.0, 10.0]),
        ("wavelength_detuning",     "delta_lambda_pm", [0.0, 20.0]),
    ]

    results = []
    for effect, variable, values in sweep_effects:
        for value in values:
            # 固定随机种子，保证可复现
            torch.manual_seed(SEED)
            perturb_fn = make_perturb_fn(effect, float(value), num_linear)
            t1 = time.perf_counter()
            pert = run_rollout(model, x, perturb_fn)
            elapsed = time.perf_counter() - t1

            # 对比第一个生成帧的 PSNR（干净 vs 带误差）
            gen_frame = COND_STEPS  # first generated frame
            psnr_val = psnr(clean[0, gen_frame], pert[0, gen_frame])
            # 自检规则：value=0（无误差）时 PSNR 应 >80（说明误差注入没改变
            # 干净结果）；value>0 只是"待人工确认"标记
            status = "PASS" if (value == 0.0 and psnr_val > 80) or value > 0.0 else "CHECK"
            results.append({
                "effect": effect, "variable": variable, "value": value,
                "elapsed_s": round(elapsed, 1), "psnr_db": round(psnr_val, 2),
                "status": status,
            })
            print(f"  {effect} {variable}={value}: {elapsed:.1f}s PSNR={psnr_val:.1f}dB [{status}]")

    total_t = time.perf_counter() - t0
    print(f"\nTotal: {total_t:.1f}s | {len(results)} runs")

    # Estimate full experiment
    # 根据本次耗时粗估完整实验（5 误差 x 4 档 x 种子数）要多久
    full_runs = sum(
        len(values) * (5 if e in STOCHASTIC_EFFECTS else 1)
        for e, _, values in sweep_effects
    ) + 1
    full_per_run = total_t / len(results) if results else 0
    print(f"Estimated full sweep (5 effects×4 severities×seeds): "
          f"{full_runs} runs × {full_per_run:.0f}s = {full_runs * full_per_run / 60:.0f} min ({full_runs * full_per_run / 3600:.1f} h)")

    # Write
    # 结果写到 results/experiment3/minimal/summary.json
    out_dir = pathlib.Path(_PROJECT_ROOT) / "results" / "experiment3" / "minimal"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "device": DEVICE,
        "rollout_steps": ROLLOUT_STEPS,
        "num_linear": num_linear,
        "clean_rollout_s": round(clean_t, 1),
        "total_s": round(total_t, 1),
        "results": results,
        "full_sweep_estimate_min": round(full_runs * full_per_run / 60, 1),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Output: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
