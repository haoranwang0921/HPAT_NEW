# HPAT / MobileViT 联合仿真工作区

2026-09-11 整理：当前工作区只以 HPAT 为研究目标。目录名 `lpwm-photonic-sim` 保留不变，避免破坏既有脚本、来源指纹和绝对路径；它不表示当前仍运行 PhotonWM 实验。

## 主要入口

- `experiments/hpat_mobilevit/`：HPAT 映射、电子参照、LT 对照、参数扫描和测试。
- `results/hpat_mobilevit/`：HPAT trace、版本化结果、冻结快照和核验记录。
- `joint_sim/`：HPAT 复用的调度、缓存、资源及后端基础设施。
- `SimPhony/`、`LLMCompass/`：光子/电子建模依赖，保留配置、许可证和原始代码。
- `external/Lightening-Transformer/`：LT 对照依赖。
- `hpat-artifact/`：历史 HPAT 模型及来源资料。

共享依赖中的 LPWM 兼容模块、测试样例和 Git 历史未做破坏性裁剪；不能仅按文件名含有 world/photonic 就删除依赖。旧世界模型论文、汇报、实验配置、trace 和结果已移出工作区。

---

## 基线总览（2026-09-11 增补）

工作区内有**四族**基线，代号彼此独立、极易混淆。此处集中登记，并给出每族的权威出处。

> 全部数值为架构级事件模型输出（XXS / warm / P0 点），**无硅实测、无准确率含义**；除特别注明外，均为开关全关的未优化基线。

### 族 1 · 光子侧映射档 H1 / H2 / H3

同一套光子硬件，按"哪些算子映射到光子阵列"分档（`streaming.py::eligible`）。

| 代号 | 映射范围 | XXS warm 时延 | 开关全开 |
|---|---|---:|---:|
| D0 | 全电子（对照） | 0.492209 ms | 0.492209 ms（不受光子开关影响） |
| H1 | 仅静态权重 Linear | 0.629657 ms | 0.578386 ms |
| H2 | H1 + 1×1 Conv | 本轮未测 | — |
| H3 | H2 + attention MatMul | 0.799543 ms | 0.707694 ms |

### 族 2 · 冻结主点 P0 与设备 profile

- **P0（冻结主点）**：32×32 阵列 / 4 核 / 10 GHz / 25 MiB 权重缓存 / 64 GB/s 外存。
  配置入口 `results/hpat_mobilevit/hardware_combo_32_10_25_v1/array32x32_10ghz_sram25/config.json`；
  冻结说明 `experiments/hpat_mobilevit/P0_BASELINE_20260908.md`（冻结日 2026-09-08）。
  所有 runner 的 `P0` 常量都指向它。
- **P0/P1/P2（设备 profile）**：另一套同名代号，指三种设备参数档，见
  `experiments/hpat_mobilevit/DEVICE_PROFILES_20260910.md`。**与上面的冻结主点不是同一概念。**

### 族 3 · 电子基线 A / A2 / A3 / B（LLMCompass 重建）

工具：LLMCompass（Princeton, ISCA'24）+ SCALE-Sim **v2**；
覆盖 XXS 矩阵类算子 97.28% MAC；逐算子串行求和（保守口径）。

| 代号 | 配置 | XXS warm | 占自身峰值 | 定位 |
|---|---|---:|---:|---|
| **A** | 4 核 / 10 GHz / 32×32 INT8 / 25 MiB / 64 GB/s（与 P0 逐项对齐） | **0.186645 ms** | 2.7% | 同规格电子基线 |
| **A2** | 4 核 / **1.4 GHz** / **16×16** INT8 / 25 MiB（片上 256 GB/s） | **0.395301 ms** | 36.1% | 移动级，但片上带宽偏高 |
| **A3** | 同 A2，但按边缘设备修正：片上 4 MB、片上带宽 44.8 GB/s、DRAM 4ch×16bit、8 GB、4nm、SRAM 64 KB；互连去掉 NVLink 遗产 | **0.607594 ms** | 23.5% | **边缘合理基线** |
| **B** | GA100（A100 级）：128 核 / 1.41 GHz / 16×16 / HBM 2048 GB/s | **0.013816 ms** | 32% | 外部参照 |

S 模型（256²，19.64 亿矩阵类 MAC）：A3 = **3.173378 ms**（43.2%）、B = **0.047463 ms**（89.6%）；
对照 HPAT H1 = 3.789632 ms、H3 = 5.339218 ms、D0 = 2.412991 ms。

权威出处：`results/hpat_mobilevit/llmcompass_baseline_v1/REPORT.md`（另含 A2/A3 子报告）、
设计说明 `experiments/hpat_mobilevit/design_llmcompass_baseline.md`、
配置 `experiments/hpat_mobilevit/llmcompass_configs/*.json`。
复现：`python -B -m experiments.hpat_mobilevit.run_llmcompass_baseline --config a3 --models xxs`

### 族 4 · SoC 参考基线（iPhone 12 锚点拟合）

位置：`results/hpat_mobilevit/iphone12_soc_reference_v1/`（代码 `soc_reference.py` / `run_soc_reference.py`）。
锚点取自 MobileViT 论文（arXiv v2，表 11 及 §4.3）：XS 规模、256×256、batch=1、CoreML 全精度转换、100 次迭代平均。

| 图执行目标 | 原论文时延 | 本次锚点拟合 |
|---|---:|---:|
| iPhone 12 CPU | 17.86 ms | 17.86 ms |
| iPhone 12 Neural Engine | 7.28 ms | 7.28 ms |

⚠️ **两条硬限定（原文要求，不得省略）**：
1. **真实 SoC 校准尚未完成，不能替代公平架构对照中的 D0**；
2. 本次 energy 及 power 返回 **null**，不填 0，也不借用原 D0 能耗当 iPhone 数据。

### 四族之间的关系与一条关键结论

- 与手写 D0 的倍数：H3 为 1.62×（慢）／H1 为 1.28×；
- 换成边缘合理基线 A3 后：H3 = 1.32×、H1 = 1.18×（开关全开时）；
- **带宽归因**：HPAT 自身片上带宽 256 → 44.8 GB/s 会使 H1 从 0.5784 涨到 1.0397 ms（+80%）；
  **在同口径 44.8 GB/s 下，HPAT H1（1.0397）反而比 A3（0.6076）慢 1.71×**。
  ⇒ 相对优势来自片上总线宽度，不是光子乘法本身；且 HPAT 对带宽的敏感度（+80%）高于电子基线（+54%）。
- 判决所用数据见 `results/hpat_mobilevit/experiments_switchall_v1/REPORT.md` 与
  `results/hpat_mobilevit/experiments_stagepipeline_v1/REPORT.md`。

### 实验输入（不是产物，必须保留）

`results/hpat_mobilevit/trace_v1/` 与 `results/hpat_mobilevit/hardware_combo_32_10_25_v1/` 是**运行输入**，
缺失时任何 runner 都会立即失败。生成命令、manifest 元数据、引用者清单与复现步骤见
[`results/hpat_mobilevit/README.md`](results/hpat_mobilevit/README.md)。

### 环境依赖

`scalesim==2.0.2`（**必须 v2**，v3 的配置 schema 不兼容）、torch / pandas / matplotlib / seaborn / scipy。
LLMCompass 需要 `systolic_array_model/temp/`（由 runner 自动创建）。

---

## 整理记录

归档位置：`C:/Users/whr/Desktop/ASP-DAC27_nonHPAT_archive_20260911/`。
逐文件来源、目标和 SHA256 见上级目录 `WORKSPACE_MIGRATION_20260911.csv`。
原 LPWM README 保存在归档中，不能再用其中的旧架构参数描述当前 HPAT。

测试入口：

```powershell
python -m pytest experiments/hpat_mobilevit/tests joint_sim/tests -q
```
