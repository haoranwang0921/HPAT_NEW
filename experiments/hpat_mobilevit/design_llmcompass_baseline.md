# Design: LLMCompass 电子基线 A/B

日期：2026-09-10　状态：**待用户确认后实施**　依据：`results/hpat_mobilevit/experiments_stagepipeline_v1/REPORT.md`、LLMCompass 实测探测

## 1. 目标

用独立第三方仿真器（LLMCompass，ISCA'24，Princeton）重建电子基线：

- **A —— 同规格更真实基线**：自建与 HPAT 同类的电子配置（4 核、10 GHz、32×32 脉动阵列、INT8、64 GB/s、25 MiB 全局缓冲），替换当前手写标量 roofline。动机：当前 D0 是"10 TOPS + 100% 利用率"的标量模型，对电子侧偏乐观且无时序细节；真实脉动阵列含 tiling/L1/L2/DRAM 开销。
- **B —— 外部参照**：`configs/GA100.json`（A100 级：128 核 @1.41 GHz、16×16 SA、HBM2e）。用于论文的"可信外部参照"，替代手抄的 iPhone/A16/RTX 数字。

**对照口径**：A 与 B 都在 **H3 映射的同一算子集合**（Linear + 1×1 conv + attention matmul，XXS 142.36 M MAC）上仿真，使"同样的算子、两种硬件路线"可直接比较。另出全算子集作为附注。

## 2. 已验证的技术前提（本日实测）

| 项 | 结果 |
|---|---|
| trace 兼容 | LLMCompass `TraceReader` 直接读 `trace_v1/*/operator_trace.jsonl` ✅（581 ops） |
| 硬件配置 | `template_to_system(configs/GA100.json)` ✅；`configs/template.json` 可自建 |
| 算子仿真 | `Matmul(data_type)` → 以 `Tensor` 调用建图 → `compile_and_simulate(device, mode)` ✅ |
| 依赖 | `scalesim==2.0.2` 已装（**必须 v2**，v3 的配置 schema 不兼容）；Windows 可跑，无需 Linux |
| 覆盖率 | 离线路径 51.0% MAC；退化（N=1 depthwise 类）2.8%；其余 ~46% 走 scalesim 回退且已通 |

## 3. 待确认的设计选择

### 3.1 配置 A 的参数（需拍板）
| 参数 | 建议值 | 依据 |
|---|---|---|
| core_count | 4 | 与 HPAT 同核数 |
| frequency | 10 GHz | 与 HPAT 同频 |
| systolic array | 32×32 | 与 HPAT 物理阵列同尺寸；`look_up_table_32_32.csv` 存在 |
| data_type | int8 | 与 HPAT 成本模型一致（1 MAC = 2 ops） |
| DRAM 带宽 | 64 GB/s（1 ch × 64 pin × 8 Gbit/s） | 与 P0 冻结口径一致 |
| global buffer | 25 MB | 与 P0 权重 SRAM 一致 |
| vector unit | int32_count 16 / fp32_count 16 | 需定标 |

⚠️ 若 A 的"同规格"参数与 HPAT 逐项对齐，则 A 与 D0 的差异只反映**调度与存储层级建模**，这是最干净的对照。

### 3.2 算子覆盖策略（需拍板）
- **GEMM 类**（Linear / 1×1 Conv / attention matmul）：LLMCompass 原生仿真。
- **3×3 空间卷积**（XXS 61.96 M MAC，29.5%）：需 im2col 成 GEMM 后送入，**或**声明为覆盖外。建议 im2col（形状可精确构造）。
- **depthwise conv**（5.72 M MAC，2.7%）：LLMCompass 无模型，且 N=1 形状返回退化值 → **声明为覆盖外并显式记录**，不静默丢弃。
- **向量类**（softmax/LayerNorm/SiLU/add）：LLMCompass 有 softmax/layernorm/gelu 模块，但当前 HPAT 的 D0 用的是 `VECTOR_WORK` 权重表。建议 A 沿用 HPAT 的向量模型以保证可比，并在报告中注明。

### 3.3 编译模式
`heuristic-GPU`（v2 下已通）作为主口径；`heuristic-TPU-new` 作交叉校验（不需要 scalesim，快）。两者差异需报告。

## 4. 交付物

```
results/hpat_mobilevit/llmcompass_baseline_v1/
  a_matched/            # A：同规格配置
    config.json         # 自建硬件配置（含全部参数与来源）
    per_op.csv          # 逐算子 latency/energy + 覆盖标记
    aggregate.csv       # 按映射集合汇总（H3 集合 / 全算子集）
    coverage.json       # MAC 覆盖率与未覆盖清单
    provenance.json     # LLMCompass 版本、scalesim 版本、源码哈希、mode
    REPORT.md
  b_ga100/              # B：A100 级参照，同结构
  comparison.md         # A / B / 当前 D0 / HPAT H3 四方对照
```

**驱动脚本**：`experiments/hpat_mobilevit/run_llmcompass_baseline.py`（不改 HPAT 既有路径，默认不接入任何报告）。

## 5. 验收标准

1. **确定性与可复现**：同配置两次运行逐位一致；`provenance.json` 记录 LLMCompass/scalesim 版本与源码 SHA-256。
2. **覆盖透明**：`coverage.json` 给出可仿真 / 退化 / 未覆盖三类的 MAC 占比，合计 100%。
3. **交叉校验**：对 ≥5 个 GEMM 形状，与当前 D0 的 roofline 单算子时延对比，差异 >10× 的必须给出原因（tiling / 存储层级 / 编译模式）。
4. **不覆盖既有结果**：写入新目录；不改 `experiments/hpat_mobilevit/` 既有模块行为（新驱动独立）。
5. **诚实报告**：A 无论比当前 D0 快或慢都照实报告；不挑选有利口径。

## 6. 风险与已知限制

| 风险 | 说明 |
|---|---|
| LLMCompass 非为 batch-1 小 M 设计 | MobileViT 的 M∈[1,9216] 且 K/N 小，其启发式可能不最优；需与 D0 对照说明 |
| scalesim 回退慢 | 全 49 形状扫描超 25 min；驱动需缓存结果（按 (M,K,N,b,mode) 缓存） |
| 版本敏感性 | LLMCompass 绑定 SCALE-Sim v2；已在项目约定中记录 |
| conv/depthwise 缺口 | 见 3.2，必须显式声明 |
| energy 字段 | `heuristic-TPU-new` 不设置 energy；需用 `heuristic-GPU` 或独立调用能量模型 |
| 环境安全 | 本环境 pip 必须 `--no-deps`（见 `.workbuddy/memory/MEMORY.md`），已发生过一次损坏事故 |
