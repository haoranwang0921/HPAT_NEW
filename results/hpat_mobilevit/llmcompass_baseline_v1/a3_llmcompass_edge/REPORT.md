# LLMCompass 电子基线 A3（边缘合理配置）

日期：2026-09-10　｜　工具：LLMCompass（ISCA'24）+ SCALE-Sim 2.0.2　｜　XXS warm

## 1. 改动：每一项都有出处，且**改前先做敏感性测试**

| # | 参数 | A2 | A3 | 依据 | 敏感性测试结果 |
|---|---|---|---|---|---|
| 1 | `global_buffer_MB` | 25 | **4** | Coral Edge TPU 片上 SRAM 为 8 MB；移动 NPU 本地缓冲为个位数 MB | **实测 2/4/25 MB 时延完全不变** → 改它零成本 |
| 2 | `global_buffer_bandwidth_per_cycle_byte` | 183（256 GB/s） | **32（44.8 GB/s）** | 16×16 INT8 阵列每周期需两路操作数（权重 16 B + 激活 16 B）= 32 B/cycle | **在关键路径上**：45 GB/s → 28.53 µs vs 256 GB/s → 16.56 µs（stem 形状，−59%） |
| 3 | DRAM 结构 | 1ch × 64pin | **4ch × 16pin**（总量保持 64 GB/s） | Snapdragon 8 Gen 3 与 Apple A18 Pro 均为 4×16-bit LPDDR5X | **等带宽下改结构不影响时延** → 纯赚可信度 |
| 4 | `total_capacity_GB` | 16 | **8** | Apple A18 Pro 为 8 GB | — |
| 5 | `process_node` | 7nm | **4nm** | Snapdragon 8 Gen 3 为 TSMC N4P | — |
| 6 | `SRAM_KB`（每核） | 192 | **64** | 移动 NPU 量级 | **实测 64/128/192 KB 时延完全不变** → 改它零成本 |
| 7 | 互连 | `NVLink3` / 12 links | **中性占位（清零）** | 移动 SoC 无 NVLink | `device_count=1` 时不参与计算 |

**保留用户指定**：4 核 / 1.4 GHz / 16×16 INT8。
峰值 = 4 × 16×16 × 1.4 GHz × 2 = **2.87 TOPS INT8**，属入门边缘级（Coral Edge TPU = 4 TOPS），**远低于旗舰手机 NPU（35–50 TOPS）**。

## 2. 结果（XXS warm）

| 基线 | ms | 占自身峰值 | HPAT H3 / 该值 |
|---|---:|---:|---:|
| **A3 边缘合理（本文件）** | **0.607594** | **23.5%** | **1.32×** |
| A2（256 GB/s 片上） | 0.395301 | 36.1% | 2.02× |
| A1（10 GHz / 32×32） | 0.186645 | 2.7% | 4.28× |
| B（A100 级） | 0.013816 | — | 57.87× |
| 当前 D0（手写 roofline） | 0.492209 | — | 1.62× |
| 当前 D0（仅计算项） | 0.079280 | — | 10.09× |
| HPAT H3（P0 冻结） | 0.799543 | — | 1.00× |

覆盖 97.28% MAC（83 算子 / 204,318,464 MAC）；覆盖外 7 个 depthwise（2.72%）。能量仍不可用。

## 3. 两个必须写下来的结论

**① 唯一实质影响结果的改动是片上带宽。**
A2 → A3 只把片上带宽从 256 GB/s 降到 44.8 GB/s，时延就从 0.3953 涨到 **0.6076 ms（+54%）**。
其余 6 项改动（缓冲大小、DRAM 结构、容量、工艺节点、每核 SRAM、互连）**对时延零影响或近乎零影响**——它们改的是"配置读起来像不像边缘设备"，不是数字。

**② 结论方向被一个"没有公开数据"的参数支配。**
HPAT H3 相对电子基线的差距：

| 电子基线假设 | H3 慢多少 |
|---|---:|
| 片上带宽 256 GB/s（数据中心级） | 4.28× |
| 片上带宽 44.8 GB/s（按阵列喂数推定） | **1.32×** |

即：**从 1.32× 到 4.28×，全看电子基线的片上带宽取多少**——而真实移动 NPU 的片上带宽（Coral、Hexagon、ANE 等）**均未公开**。
⇒ 这不是"HPAT 赢/输"的问题，而是**结论对该参数高度敏感**。任何引用单一数字的说法都不成立，必须给出带宽敏感性区间。

**③ A3 比手写 D0 还慢 1.23×**（0.6076 vs 0.4922）。原因：D0 的 DRAM 计费虽偏严，但其片内不做分块访存建模；A3 的 tiling 受 44.8 GB/s 片上带宽约束后成为访存瓶颈（利用率从 A2 的 36.1% 降到 23.5%）。

## 4. 边界

- A3 是参数覆盖得到的模型，非流片/实测设备；所有单价与结构来源见 `_provenance` 字段。
- 32 B/cycle 是**由阵列尺寸推定的下界**，不是实测值——这正是上述敏感性的来源。
- 逐算子串行求和（保守）；能量不可用；仅 XXS；depthwise 与向量算子不在覆盖内。
- 若要对标旗舰手机 NPU（35–50 TOPS），需把阵列放大到 4 × 64×64 @1.4 GHz（≈45.9 TOPS）；在 32 B/cycle 的片上带宽下会严重访存受限，是另一组实验。

## 5. 复现

```
python -B -m experiments.hpat_mobilevit.run_llmcompass_baseline --config a3 --models xxs \
  --out-root results/hpat_mobilevit/llmcompass_baseline_v1/a3_llmcompass_edge
```
配置：`experiments/hpat_mobilevit/llmcompass_configs/a3_edge_1p4ghz_16x16.json`
