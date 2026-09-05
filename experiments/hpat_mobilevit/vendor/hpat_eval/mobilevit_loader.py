from __future__ import annotations

import importlib.util
import statistics
from typing import Any


DEFAULT_MODEL_SPECS = [
    {
        "variant": "MobileViT-XXS",
        "timm_model": "mobilevit_xxs",
        "input_resolution": 192,
        "embedding_dim": 128,
    },
    {
        "variant": "MobileViT-XS",
        "timm_model": "mobilevit_xs",
        "input_resolution": 224,
        "embedding_dim": 192,
    },
    {
        "variant": "MobileViT-S",
        "timm_model": "mobilevit_s",
        "input_resolution": 256,
        "embedding_dim": 256,
    },
]


def available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def missing_packages(names: list[str]) -> list[str]:
    return [name for name in names if not available(name)]


def variants_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_name = {row["variant"]: row for row in DEFAULT_MODEL_SPECS}
    for item in config.get("mobilevit_variants", []):
        base = dict(by_name.get(item["name"], {}))
        base.update(
            {
                "variant": item["name"],
                "timm_model": item.get("timm_model", base.get("timm_model", "")),
                "input_resolution": int(item.get("input_resolution", base.get("input_resolution", 224))),
                "embedding_dim": int(item.get("d", item.get("embedding_dim", base.get("embedding_dim", 128)))),
                "token_count_sweep": list(item.get("token_count_sweep", [196])),
                "draft_group": item.get("draft_group", ""),
                "parameter_count_m": item.get("parameter_count_m", ""),
            }
        )
        rows.append(base)
    return rows or DEFAULT_MODEL_SPECS


def primary_token_count(variant: dict[str, Any], config: dict[str, Any]) -> int:
    configured = config.get("qkv", {}).get("primary_token_count")
    if configured:
        return int(configured)
    sweep = variant.get("token_count_sweep") or [196]
    if 196 in sweep:
        return 196
    return int(sweep[len(sweep) // 2])


def select_device(torch: Any, requested: str) -> tuple[Any, str, str]:
    if requested == "auto":
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return torch.device("mps"), "mps", "mps available and selected by auto policy"
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda", "cuda available and selected by auto policy"
        reason = "mps unavailable; selected cpu"
        if not torch.backends.mps.is_built():
            reason = "torch was not built with mps; selected cpu"
        elif not torch.backends.mps.is_available():
            reason = "torch mps built but unavailable at runtime; selected cpu"
        return torch.device("cpu"), "cpu", reason
    if requested == "mps":
        if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
            raise RuntimeError("Requested mps but torch.backends.mps is unavailable")
        return torch.device("mps"), "mps", "mps explicitly requested"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested cuda but torch.cuda.is_available() is false")
        return torch.device("cuda"), "cuda", "cuda explicitly requested"
    return torch.device("cpu"), "cpu", "cpu explicitly requested"


def sync_device(torch: Any, device: Any) -> None:
    device_type = getattr(device, "type", str(device))
    if device_type == "mps":
        torch.mps.synchronize()
    elif device_type == "cuda":
        torch.cuda.synchronize()


def latency_stats(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    if not ordered:
        return {key: 0.0 for key in ["mean", "median", "p05", "p95", "min", "max"]}
    p05 = ordered[int(0.05 * (len(ordered) - 1))]
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p05": p05,
        "p95": p95,
        "min": min(ordered),
        "max": max(ordered),
    }
