from __future__ import annotations

"""
export_mobilevit_coreml.py —— 导出 MobileViT 的 Core ML 模型（给 iPhone 测延迟用）

做什么：
    把 timm 库里的 MobileViT 视觉模型（若干变体）转换为 Apple 的 Core ML 格式
    （.mlpackage），供 iPhone 真机测量推理延迟，作为论文中"边缘移动端基线"
    （Edge MobileViT baseline）的证据资产。

数据从哪来：
    模型权重来自 timm 库（可指定 --pretrained 加载预训练权重），模型结构配置
    来自 --config 指向的 JSON 配置文件（变体、输入分辨率等）。

产出到哪：
    每个变体导出一个 .mlpackage 文件到 --output-dir/models/ 下，同时在输出目录
    生成导出摘要 CSV（mobilevit_coreml_export_summary.csv）与清单 JSON
    （mobilevit_coreml_export_manifest.json）。

如何运行：
    python export_mobilevit_coreml.py --output-dir <输出目录> [--config config.json] [--pretrained] \
        [--minimum-deployment-target iOS16|iOS17|iOS18]
    注意：依赖 torch、timm、coremltools；缺包时脚本会记录 blocked 而不强行执行。

声明边界：这些 Core ML 包只用于测量 MobileViT 延迟基线，不是 HPAT 部署产物，
也不是 HPAT 硅片证据或精度证据（除非另有标注的验证运行）。
"""

import argparse
import hashlib
import json
import math
import pathlib
import types
from typing import Any

from _common import base_manifest, ensure_dir, load_json, relative, sha256_file, write_csv, write_json
from hpat_eval.mobilevit_loader import variants_from_config


# 导出摘要 CSV 的列顺序：包含模型变体、输入尺寸、部署目标、校验和、状态、版本号等
SUMMARY_FIELDS = [
    "variant",
    "timm_model",
    "input_shape",
    "batch_size",
    "precision",
    "minimum_deployment_target",
    "output_path",
    "sha256",
    "status",
    "blocked_reason",
    "static_shape_patch",
    "pretrained",
    "torch_version",
    "timm_version",
    "coremltools_version",
    "evidence_label",
]


def _slug(value: str) -> str:
    """把字符串转成适合做文件名的"slug"：小写，连字符和斜杠换成下划线。

    参数：
        value: 原始字符串（如模型变体名 "mobilevit_s"）。
    返回：
        清洗后的文件名安全字符串（如 "mobilevit_s"）。
    """

    return value.lower().replace("-", "_").replace("/", "_")


def _sha256_artifact(path: pathlib.Path) -> str | None:
    """计算导出产物的 SHA-256 校验和，产物可能是文件或目录（.mlpackage 是目录包）。

    用途：为每个导出的模型记录内容指纹，方便日后核对文件是否被篡改。
    参数：
        path: 待计算的文件或目录路径。
    返回：
        十六进制 SHA-256 字符串；路径既不是文件也不是目录时返回 None。
    """

    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    # 目录包：按文件排序逐个读取，路径和内容一起喂进哈希，保证结果稳定可复现
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        # 先记下文件在包内的相对路径（UTF-8 字节 + 分隔符 \0）
        rel = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        # 再按 1 MiB 分块读取文件内容并更新哈希，避免一次性加载大文件
        with child.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _missing_packages() -> list[str]:
    """检查导出 Core ML 所需的第三方包是否齐全。

    用途：在真正动手前先探测依赖，缺包时让脚本优雅地记录 blocked，而不是中途崩溃。
    返回：
        缺失的包名列表；全部就绪时为空列表。
    """

    missing: list[str] = []
    for name in ["torch", "timm", "coremltools"]:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    return missing


def _record_mobilevit_block_shapes(model: Any, x: Any, mobilevit_module: Any, torch: Any) -> dict[int, tuple[int, int, int, int]]:
    """跑一次前向推理，记录每个 MobileViT 块的中间张量形状。

    用途：MobileViT 的 patch 变换要求高/宽能被 patch_size 整除；Core ML 要求输入
          形状静态固定。这里先探明各块实际形状，为后续"静态形状补丁"提供依据。
    参数：
        model: 待分析的 timm 模型。
        x: 输入示例张量。
        mobilevit_module: timm.models.mobilevit 模块（用于判断块类型）。
        torch: torch 模块引用。
    返回：
        dict，键是块的 id(block)，值是 (batch, channels, height, width) 形状元组。
    """

    records: dict[int, tuple[int, int, int, int]] = {}
    hooks = []
    # 遍历模型所有子模块，给每个 MobileVitBlock 的 conv_1x1 注册"前向钩子"
    # 钩子在推理时被调用，顺手记录输出张量形状
    for _name, block in model.named_modules():
        if isinstance(block, mobilevit_module.MobileVitBlock):
            block_id = id(block)

            def hook(_mod: Any, _inp: Any, out: Any, block_id: int = block_id) -> None:
                records[block_id] = tuple(int(dim) for dim in out.shape)

            hooks.append(block.conv_1x1.register_forward_hook(hook))
    # 在无梯度模式下跑一次推理，触发钩子采集形状
    with torch.no_grad():
        model(x)
    # 用完及时移除钩子，避免影响后续推理
    for handle in hooks:
        handle.remove()
    return records


def _patch_mobilevit_blocks_static(model: Any, records: dict[int, tuple[int, int, int, int]], mobilevit_module: Any) -> list[dict[str, Any]]:
    """用"静态形状"版本的前向函数替换每个 MobileViT 块的 forward。

    用途：把运行时动态推断的 reshape 逻辑改成按记录到的固定形状硬编码，
          从而让 Core ML 转换器拿到完全静态的张量形状（转换要求形状必须确定）。
    参数：
        model: 待修补的模型。
        records: _record_mobilevit_block_shapes 采集到的各块形状字典。
        mobilevit_module: timm.models.mobilevit 模块。
    返回：
        每个被修补块的说明 dict 列表（模块名、形状、patch 信息等）。
    """

    import torch
    import torch.nn.functional as F

    patch_rows: list[dict[str, Any]] = []
    for name, block in model.named_modules():
        if not isinstance(block, mobilevit_module.MobileVitBlock):
            continue
        # 取出记录好的形状，并计算 padding 到 patch 整数倍后的尺寸
        batch, channels, height, width = records[id(block)]
        patch_h, patch_w = block.patch_size
        # 向上取整到 patch_size 的整数倍：若原图 33x33、patch 32，则新尺寸为 64x64
        new_h = math.ceil(height / patch_h) * patch_h
        new_w = math.ceil(width / patch_w) * patch_w
        num_patch_h = new_h // patch_h
        num_patch_w = new_w // patch_w
        num_patches = num_patch_h * num_patch_w
        # 若发生了尺寸调整，需要在进 transformer 前插值放大、出来后插值还原
        interpolate_in = bool(new_h != height or new_w != width)
        patch_area = block.patch_area

        def forward_static(
            self: Any,
            x: Any,
            *,
            batch: int = batch,
            channels: int = channels,
            height: int = height,
            width: int = width,
            patch_h: int = patch_h,
            patch_w: int = patch_w,
            new_h: int = new_h,
            new_w: int = new_w,
            num_patch_h: int = num_patch_h,
            num_patch_w: int = num_patch_w,
            num_patches: int = num_patches,
            interpolate_in: bool = interpolate_in,
            patch_area: int = patch_area,
        ) -> Any:
            # 静态版 MobileViT 块前向：所有 reshape 尺寸都是编译期常量
            # 保存捷径（shortcut）分支输入，最后做特征融合
            shortcut = x
            x = self.conv_kxk(x)
            x = self.conv_1x1(x)
            # 必要时先插值放大到 patch 整数倍尺寸
            if interpolate_in:
                x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            # 把特征切成 patch 网格：拆成 h/w 两个 patch 维度再转置，得到 patch 序列
            x = x.reshape(batch * channels * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
            # 展平成 transformer 需要的 (batch*patch_area, num_patches, 通道) 形状
            x = x.reshape(batch, channels, num_patches, patch_area).transpose(1, 3).reshape(
                batch * patch_area, num_patches, -1
            )
            # transformer 全局建模 + 归一化
            x = self.transformer(x)
            x = self.norm(x)
            # 把 patch 序列还原回特征图：形状还原是上面切分的逆操作
            x = x.contiguous().view(batch, patch_area, num_patches, -1)
            x = x.transpose(1, 3).reshape(batch * channels * num_patch_h, num_patch_w, patch_h, patch_w)
            x = x.transpose(1, 2).reshape(batch, channels, num_patch_h * patch_h, num_patch_w * patch_w)
            # 若之前放大过，这里插值缩小回原始尺寸
            if interpolate_in:
                x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
            x = self.conv_proj(x)
            # 有融合卷积时，把捷径与主路拼接后融合
            if self.conv_fusion is not None:
                x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
            return x

        # 把静态版函数绑定为这个块的实例方法，替换掉原来的动态 forward
        block.forward = types.MethodType(forward_static, block)
        patch_rows.append(
            {
                "module": name,
                "conv_1x1_shape": [batch, channels, height, width],
                "patch_size": [patch_h, patch_w],
                "interpolate": interpolate_in,
                "num_patches": num_patches,
            }
        )
    return patch_rows


def run(output_dir: pathlib.Path, config: dict[str, Any], pretrained: bool, minimum_deployment_target: str) -> dict[str, pathlib.Path]:
    """主流程：把配置里每个 MobileViT 变体导出为 Core ML 模型。

    用途：完成从"配置 -> timm 模型 -> 静态形状补丁 -> Core ML 转换 -> 产物记录"
          的一整套导出工作，并把结果写进 CSV 摘要与 JSON 清单。
    参数：
        output_dir: 输出目录（模型、CSV、manifest 都放这里）。
        config: 实验配置 dict（含各变体的 timm_model、input_resolution 等）。
        pretrained: 是否加载 timm 预训练权重。
        minimum_deployment_target: Core ML 最低部署目标，如 "iOS16"。
    返回：
        dict，包含 "summary_csv" 与 "manifest" 两个输出文件路径。
    """

    ensure_dir(output_dir)
    # 模型统一放在 output_dir/models/ 子目录下
    model_dir = ensure_dir(output_dir / "models")
    missing = _missing_packages()
    manifest_path = output_dir / "mobilevit_coreml_export_manifest.json"
    summary_csv = output_dir / "mobilevit_coreml_export_summary.csv"
    # 生成清单基础信息，并声明这些资产的用途边界（只作延迟基线，不冒充 HPAT 部署证据）
    manifest = base_manifest("mobilevit_coreml_export", "Core ML MobileViT baseline assets; latency-only edge context")
    manifest.update(
        {
            "pretrained": pretrained,
            "minimum_deployment_target": minimum_deployment_target,
            "claim_boundary": (
                "Core ML packages are for author-measured MobileViT latency baseline collection only; "
                "they are not HPAT deployment, not HPAT silicon evidence, and not accuracy evidence unless "
                "a separate labeled validation run is supplied."
            ),
        }
    )

    # 缺包时不做实际导出，仅生成一行 blocked 摘要，保证脚本可读地失败
    if missing:
        rows = [
            {
                "variant": spec["variant"],
                "timm_model": spec.get("timm_model", ""),
                "input_shape": "",
                "batch_size": 1,
                "precision": "fp16",
                "minimum_deployment_target": minimum_deployment_target,
                "output_path": "",
                "sha256": "",
                "status": "blocked",
                "blocked_reason": f"Missing Python packages: {', '.join(missing)}",
                "static_shape_patch": "yes",
                "pretrained": pretrained,
                "torch_version": "",
                "timm_version": "",
                "coremltools_version": "",
                "evidence_label": "blocked Core ML export",
            }
            for spec in variants_from_config(config)
        ]
        write_csv(summary_csv, rows, SUMMARY_FIELDS)
        manifest.update({"status": "blocked", "blocked_reason": rows[0]["blocked_reason"], "outputs": [relative(summary_csv)]})
        write_json(manifest_path, manifest)
        return {"summary_csv": summary_csv, "manifest": manifest_path}

    # 依赖齐备后才真正导入重量级库（coremltools/timm/torch），避免缺包时白加载
    import coremltools as ct
    import timm
    import timm.models.mobilevit as mobilevit_module
    import torch

    # 部署目标字符串 -> coremltools 目标枚举的映射
    target_map = {
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
        "iOS18": ct.target.iOS18,
    }
    target = target_map.get(minimum_deployment_target)
    if target is None:
        raise ValueError(f"Unsupported minimum deployment target: {minimum_deployment_target}")

    # 固定随机种子，让导出结果可复现（默认用配置里的 seed，缺省 20260706）
    torch.manual_seed(int(config.get("seed", 20260706)))
    rows: list[dict[str, Any]] = []
    variants_meta: list[dict[str, Any]] = []
    # 逐变体导出：每个变体走"建模型->记录形状->打静态补丁->转换->保存"的完整流程
    for spec in variants_from_config(config):
        variant = spec["variant"]
        model_name = spec.get("timm_model", "")
        resolution = int(spec["input_resolution"])
        # 输入形状固定为 (batch=1, 通道=3, 高, 宽)
        input_shape = (1, 3, resolution, resolution)
        # 输出文件名按变体与分辨率命名，例如 mobilevit_s_coreml_fp16_256.mlpackage
        out_path = model_dir / f"{_slug(variant)}_coreml_fp16_{resolution}.mlpackage"
        try:
            model = timm.create_model(model_name, pretrained=pretrained).eval()
            # 用随机输入跑一次，采集各 MobileViT 块的中间形状
            example = torch.randn(*input_shape)
            records = _record_mobilevit_block_shapes(model, example, mobilevit_module, torch)
            patch_rows = _patch_mobilevit_blocks_static(model, records, mobilevit_module)
            # 记录原始 torch 输出形状，便于日后核对转换是否改变语义
            with torch.no_grad():
                torch_output = model(example)
            # 用 torch.jit 追踪出静态计算图，供 Core ML 转换器消费
            traced = torch.jit.trace(model, example)
            # 转换为 fp16 精度的 mlprogram 格式（iOS 机器学习程序格式）
            mlmodel = ct.convert(
                traced,
                inputs=[ct.TensorType(name="input", shape=input_shape)],
                minimum_deployment_target=target,
                convert_to="mlprogram",
                compute_precision=ct.precision.FLOAT16,
            )
            # 在模型元数据里写入说明与证据标签，防止日后被误当作 HPAT 部署产物
            mlmodel.short_description = (
                f"{variant} exported from timm for fixed-shape iPhone latency benchmarking; not HPAT deployment."
            )
            mlmodel.user_defined_metadata["hpat_evidence_label"] = (
                "author-measured MobileViT mobile baseline asset; not HPAT deployment"
            )
            mlmodel.user_defined_metadata["hpat_input_shape"] = "x".join(str(v) for v in input_shape)
            mlmodel.user_defined_metadata["hpat_pretrained"] = str(pretrained)
            mlmodel.save(str(out_path))
            # 保存成功后才计算校验和并记录成功行
            file_hash = _sha256_artifact(out_path)
            rows.append(
                {
                    "variant": variant,
                    "timm_model": model_name,
                    "input_shape": "x".join(str(v) for v in input_shape),
                    "batch_size": 1,
                    "precision": "fp16",
                    "minimum_deployment_target": minimum_deployment_target,
                    "output_path": relative(out_path),
                    "sha256": file_hash or "",
                    "status": "ok",
                    "blocked_reason": "",
                    "static_shape_patch": "yes",
                    "pretrained": pretrained,
                    "torch_version": torch.__version__,
                    "timm_version": timm.__version__,
                    "coremltools_version": ct.__version__,
                    "evidence_label": "Core ML MobileViT latency baseline asset; not HPAT deployment",
                }
            )
            variants_meta.append(
                {
                    "variant": variant,
                    "timm_model": model_name,
                    "input_shape": list(input_shape),
                    "torch_output_shape": list(torch_output.shape),
                    "output_path": relative(out_path),
                    "sha256": file_hash,
                    "static_shape_patch": patch_rows,
                }
            )
        except Exception as exc:
            # 单个变体失败不影响其他变体：记录 blocked 行后继续下一个
            rows.append(
                {
                    "variant": variant,
                    "timm_model": model_name,
                    "input_shape": "x".join(str(v) for v in input_shape),
                    "batch_size": 1,
                    "precision": "fp16",
                    "minimum_deployment_target": minimum_deployment_target,
                    "output_path": "",
                    "sha256": "",
                    "status": "blocked",
                    "blocked_reason": str(exc),
                    "static_shape_patch": "yes",
                    "pretrained": pretrained,
                    "torch_version": torch.__version__,
                    "timm_version": timm.__version__,
                    "coremltools_version": ct.__version__,
                    "evidence_label": "blocked Core ML export",
                }
            )

    write_csv(summary_csv, rows, SUMMARY_FIELDS)
    manifest.update(
        {
            "status": "ok" if all(row["status"] == "ok" for row in rows) else "partial",
            "torch_version": torch.__version__,
            "timm_version": timm.__version__,
            "coremltools_version": ct.__version__,
            "outputs": [relative(summary_csv)] + [row["output_path"] for row in rows if row["output_path"]],
            "variants": variants_meta,
            "promotion_note": (
                "These assets enable iPhone Core ML latency collection only. Import raw iPhone latency samples "
                "through run_edge_baseline_import.py before making author-measured mobile baseline claims."
            ),
        }
    )
    write_json(manifest_path, manifest)
    return {"summary_csv": summary_csv, "manifest": manifest_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--minimum-deployment-target", default="iOS16", choices=["iOS16", "iOS17", "iOS18"])
    args = parser.parse_args()
    config = load_json(pathlib.Path(args.config)) if args.config else load_json()
    outputs = run(pathlib.Path(args.output_dir), config, args.pretrained, args.minimum_deployment_target)
    print(json.dumps({key: relative(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
