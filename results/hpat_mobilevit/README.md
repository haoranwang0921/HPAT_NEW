# results/hpat_mobilevit — 输入与结果说明

⚠️ **本目录整体是运行产物的默认落盘位置，但其中两个子目录是「实验输入」，必须入库。**
它们缺失时，任何 runner 都会立刻失败（实测：`FileNotFoundError: .../hardware_combo_32_10_25_v1/array32x32_10ghz_sram25/config.json`）。

---

## 1. `trace_v1/` — 实验输入（MobileViT 形状 trace）

| 项 | 内容 |
|---|---|
| 内容 | 三档模型各一份 `operator_trace.jsonl` + `manifest.json`（共 6 文件） |
| 生成 | `python -B -m experiments.hpat_mobilevit.trace --output results/hpat_mobilevit/trace_v1 --variant all` |
| 定义 | `experiments/hpat_mobilevit/trace.py` —— 真实 ATen 执行 + producer DAG；不引入旧的时延/能量代理 |
| 引用者 | **19 个 runner**（`plan_sweep.py`、`run_batch1_sweep.py`、`run_batch2_gated.py`、`run_batch3_m1.py`、`run_batch5_stage.py`、`run_hardware_scan.py`、`run_llmcompass_baseline.py`、`run_adc_audit.py`、`run_d0_adjusted.py`、`run_row_pipeline.py`、`run_lt_compare.py`、`run_lt_official_trace.py`、`run_soc_reference.py`、`run_device_profiles.py`、`analyze_input_broadcast.py`、`pipeline_operator.py`、`pipeline_smoke.py`、`finalize_plan_review.py`、`freeze_h123.py`） |

三档 manifest 元数据（`--variant all` 实测值）：

| 档位 | 模型 | 输入分辨率 | 算子数 | 参数量 | seed | 精度 |
|---|---|---:|---:|---:|---:|---|
| `xxs` | MobileViT-XXS | 192 | 581 | 1.27 M | 20260706 | fp32 捕获 → 仿真 8 bit |
| `xs` | MobileViT-XS | 224 | 582 | 2.32 M | 20260706 | 同上 |
| `s` | MobileViT-S | 256 | 580 | 5.58 M | 20260706 | 同上 |

> 该 trace 仅用于**形状/成本**研究；manifest 内显式声明 `purpose: "shape/cost trace; no accuracy or nonideality claim"`，
> 且 `pretrained: false`（随机权重，只保证形状真实）。

---

## 2. `hardware_combo_32_10_25_v1/` — 实验输入（冻结 P0 主点）

| 项 | 内容 |
|---|---|
| 含义 | 32×32 阵列 / 10 GHz / 25 MiB 权重 SRAM 的冻结配置与扫描产物 |
| 生成 | `python -B -m experiments.hpat_mobilevit.run_hardware_scan --output results/hpat_mobilevit/hardware_combo_32_10_25_v1 --combo-32-10-25` |
| 脚本说明 | `run_hardware_scan.py` 顶部："XXS one-factor hardware sensitivity; fixed electronic cache layout, fresh costs." |
| **配置入口** | `array32x32_10ghz_sram25/config.json` —— 所有 runner 的 `P0` 常量都指向它 |
| 冻结说明 | `experiments/hpat_mobilevit/P0_BASELINE_20260908.md`（冻结日 **2026-09-08**） |
| 引用者 | 7 个（`plan_sweep.py`、`run_adc_audit.py`、`run_d0_adjusted.py`、`run_row_pipeline.py`、`analyze_input_broadcast.py`、`pipeline_operator.py`、`pipeline_smoke.py`） |

目录内 `base/` 与 `array32x32_10ghz_sram25/` 各自的 `events.jsonl.gz` 是该扫描的**原始事件流**，
随目录一并入库（体量小），以保证 P0 冻结点的可追溯性。

---

## 3. 其余子目录：运行产物

除上述两项外，本目录其他内容均为运行产物，**不入库**（原始事件流体积大）。
本次入库的仅是各实验目录下的摘要报告：

- `experiments_stagepipeline_v1/REPORT.md` —— 三段时间并行（串行 vs 阶段并行）
- `experiments_switchall_v1/REPORT.md` —— 五个优化开关全开对照 + 逐开关消融
- `llmcompass_baseline_v1/REPORT.md` —— LLMCompass 电子基线 A/A2/A3/B 对照

---

## 4. 新克隆后如何跑通（已实测）

```bash
# HPAT 侧（需要 trace_v1 + hardware_combo，本目录已包含）
python -B -m experiments.hpat_mobilevit.run_batch5_stage --models xxs --only b5_serial
# 期望输出与冻结结果逐位一致：D0 0.49220925300122625 ms / H1 0.6296566252518044 / H3 0.7995425730519088

# 开关全开
python -B -m experiments.hpat_mobilevit.run_batch5_stage --models xxs --only b5_serial b5_all

# LLMCompass 电子基线
python -B -m experiments.hpat_mobilevit.run_llmcompass_baseline --config a3 --models xxs
```

**环境依赖（不在仓库内，需自行安装）**：

- `scalesim==2.0.2` —— **必须 v2**。LLMCompass 按 SCALE-Sim v2 的配置格式写 cfg；v3 会因新增必需键与强制 layout 文件而失败。
- `torch`、`pandas`、`matplotlib`、`seaborn`、`scipy`（LLMCompass 依赖）
- `LLMCompass/systolic_array_model/temp/` 会被 runner 自动创建，无需手工准备。

**尚未入库的输入**（仅少数脚本需要）：

- `results/hpat_mobilevit/experiments_mobile_v1/`（约 98 MB）—— 仅 `run_batch1_sweep.py` 引用。
- `external/Lightening-Transformer/` —— LT 对比脚本（`run_lt_compare.py`、`run_lt_official_trace.py`）需要；因是第三方嵌套 git 仓库，未入库。
