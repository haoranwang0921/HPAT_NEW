# MobileViT 光-电混合推理的调度与数据驻留优化策略（创新点设计）

- 状态: 设计文档 v1.0（2026-09-06），基于 events/mobile_reference_10tops 与 A16 对齐批次的事件级证据
- 适用范围: joint_sim + experiments/hpat_mobilevit 的 StreamingPhotonic 调度层
- 基线口径: mobile_reference 10 TOPS（回滚后），10 ps 写环，2 tile × 2 core 16×16，符号行配对

## 1. 动机（事件流证据）

S/H1 光子域 30.0 ms 墙钟中，光学计算本体（DAC+光算+ADC）仅 6.8 ms（事件口径 18%），
其余被三类"可调度开销"占据：

| 开销 | 量级（S） | 现状机制 |
| --- | ---: | --- |
| signed_reduce（电子 ALU 归约） | 11.9 ms | 每 wave 光算全部结束后才串行执行，与光算/下一波零重叠 |
| partial_sum 读改写（sram_noc） | 14.5 ms | 每 K-block 都从 scratchpad 读旧累加值、写新累加值（单 qkv 62 MB） |
| 权重重编程（跨帧） | 27608 事件/帧 | core_weights 缓存不跨帧，warm 帧仍全量重编程 |

三者都是**调度与数据驻留策略的产物，不是光子器件的物理极限**——这就是创新空间。

## 2. 三个机制的设计

### M1 符号归约-光算流水重叠（Reduce-Under-Compute Pipelining）

- 现状：`streaming.py` 中每 wave 先把 `self.cores` 个 tile 的 DAC→光算→ADC 全部排完
  （`tasks` 循环），再按完成时间逐个做 `partial_sum_read → signed_reduce → partial_sum_write`；
  归约是单点串行（`acquire_electronic`），且下一 wave 光算被 `done = max(done, reduced)` 阻塞。
- 设计：
  1. 把每 core 的光算结束事件直接挂入归约就绪队列（per-core reduce slot）；
  2. 归约单元视为独立资源（electronic ALU pool），与各 core 光算并行推进；
  3. 下一 wave 的 DAC 编码只依赖 scratch_bus 输入可用性，不等待前一 wave 归约完成；
  4. 真依赖约束仅保留"K 维相邻 block 的归约顺序"（accumulator 链）。
- 目标函数：min 光子域墙钟 = max(光路关键路径, 归约关键路径, 搬运关键路径) 的调度长度。
- 理论上限：signed_reduce 与光算完全重叠时，光子域 ≈ max(sram_noc, 光路) ≈ 14.5 ms（S），
  即 30.0 → ~18 ms（-40%）。

### M2 K-block 部分和片上驻留（Accumulator Residency）

- 现状：每 K-block 循环 `partial_sum_read`（0.60 ms/qkv）+ `partial_sum_write`（0.32 ms/qkv），
  62.2 MB @ 50 GB/s scratchpad——K 维每前进一步都付两次全量搬运。
- 设计：
  1. 为每个 `(batch, n_index)` 输出块分配驻留 accumulator（int32），K 维连续 block 迭代在
     片上完成，K 扫描结束后一次性写回；
  2. 容量约束 `Σ M×N_live×4B ≤ accumulator_bytes`（当前 scratchpad 8 MB + 权重 SRAM 1 MB
     之外可复用 8 MB 的一部分），超限时按 `(M×N)` 降序 spill 最大块（LRU）；
  3. 与 M1 复用同一 accumulator 键控，归约直接读写驻留值，不再经过 scratch_bus。
- 收益：单 qkv 的 partial_sum 流量 0.92 ms → ~0.05 ms（仅首末两次）；
  S 全模型 sram_noc 14.5 ms → ~8-9 ms（two_sign_input_passes 0.32 ms/qkv 保留）。
- 代价：accumulator SRAM 面积/漏电（8 MB 级在移动 SoC 可接受），不增加 DRAM 流量。

### M3 跨帧 MRR 权重驻留（Cross-Frame Weight Residency）

- 现状：`run()` 每帧重置调度状态，core_weights 缓存随之失效 → warm 帧仍 27608 次编程
  （S），编程能耗 0.68 mJ/帧 + 编程 lane 占用 + 每帧 `static_weight`/DMA 权重重灌。
- 设计：
  1. core_weights 缓存提升为跨帧对象，容量 `mrr_capacity_tiles`（默认 96）做 LRU；
  2. 权重块键 `(layer, tile_idx)` 不变即免编程，仅更新 `tile_last_accessed`；
  3. 驻留期间 hold 功率（HPAT_MRR 0.1 mW/环 × 1024 环 × 驻留时长）计入账本；
  4. 可选刷新模式 `program_on_miss_with_refresh` 对齐漂移模型（周期 τ_refresh 可参数扫描）。
- 权衡模型：驻留收益 = 省下的编程能耗+编程时间；驻留成本 = hold 功率 × 驻留时长增量。
  10 ps 写环下编程很快，收益主要来自编程能耗（×multiplier 100000 的补偿系数）与
  权重重灌 DMA；hold 在光子窗口内本来就计，跨帧驻留只增加帧间隙的 hold。
- 收益：warm 帧 programming 事件 27608 → 0，S 每帧 -0.68 mJ、冷启动只付一次。

## 3. 优化目标与约束（形式化）

```
min  T_wall = 调度长度(events 时间线)
min  E_total = Σ(光路动态 + 编程 + hold + laser_static×T_active + sram/hbm 流量)
s.t. SRAM: accum_bytes + weight_cache ≤ 容量上限
     MRR: 驻留权重块数 ≤ mrr_capacity_tiles
     正确性: DAG 依赖 + accumulator 键控一致 + 刷新周期 ≤ 漂移阈值
```

多目标处理：以 T_wall 为主目标、E 为约束（移动端时延敏感），输出 Pareto 前沿
（消融 M1/M2/M3 的 2³ 组合）。

## 4. 评估方法

1. **仿真协议**：joint_sim 事件流，D0/H1/H2/H3 × MobileViT XXS/XS/S × warm 3 帧；
   每个机制单独开启与组合开启，共 8 个配置；对照基线 = 当前 serial 调度。
2. **指标**：
   - 光子域墙钟、总时延、总能耗（及分账：programming/hold/laser/sram_noc/光路）
   - 编程事件数、sram 流量字节、SRAM 峰值占用、归约-光算重叠率（重叠时长/归约总时长）
3. **有效性检查**：MAC 守恒、能耗分账闭合（verify.py 已有 partial_sum 一致性检查）、
   驻留正确性（权重块键不变断言）。
4. **敏感性**：vector_peak_fraction、scratchpad 带宽、accumulator 容量、τ_refresh 扫描。

## 5. 创新点（与现有工作对照）

| 维度 | 现有 joint_sim 基线 | 已发表光子加速器文献（2014-2026 六类符号方案） | 本设计 |
| --- | --- | --- | --- |
| 符号/负数处理 | 正负行配对+双 pass，成本全进账本 | 普遍按常数或忽略，成本系统性低估 | 符号链成本显式化**且可调度优化**（重叠/驻留） |
| 归约与光算关系 | 串行（零重叠，实测 96% 墙钟） | 少见讨论 | Reduce-Under-Compute 流水，ALU 独立资源池 |
| partial sum | 每 K-block RMW 到 scratchpad | 通常假设无限片上累加 | 驻留+容量约束 spill，流量-面积显式权衡 |
| MRR 权重驻留 | 每帧全量重编程 | 多为单帧视角 | 跨帧 LRU 驻留 × hold 功率 × 刷新周期三方权衡 |
| 账本口径 | 能耗/时延统一账本 | 多为能耗-only | 时延+能耗双目标，事件级可验证 |

差异化的核心主张：**可编程光子阵列的时延瓶颈不在光子器件，而在"符号-负数处理链 +
数据驻留策略"这些调度层决策；把这三类开销纳入统一账本并给出流水化/驻留化方案后，
混合映射的光子域开销可压缩 ~50%，这是文献普遍没有量化过的部分。**

## 6. 预期收益（解析估算，待仿真验证）

| 机制 | S/H1 光子域墙钟 | 能耗影响 |
| --- | ---: | --- |
| 基线 | 30.0 ms | — |
| +M1 重叠 | ~18-20 ms（归约隐藏） | ≈不变 |
| +M2 驻留 | ~14-15 ms（sram 流量减半） | sram 能耗 ↓~50% |
| +M3 跨帧驻留 | ≈不变 | -0.68 mJ/帧（programming）+编程 lane 时间 |
| 组合（M1+M2+M3） | **~12-14 ms（-55%±）** | H1 总能耗 -6~9% |

诚实边界：组合后 H1（≈9.5-10 ms 级）仍劣于 D0（6.87 ms @10 TOPS 回滚口径）。
本创新点的定位是**把混合映射的结构性开销压到与全电子可比的区间**，并在"电子更强/
存储更弱"（真实移动 SoC、带宽地板主导）的参照系下重新评估混合方案的价值，
而不是声称在当前账本下反超 D0。

## 7. 实施切点（代码位置）

- M1: `experiments/hpat_mobilevit/streaming.py` 波循环（~L213-270）的 tasks/reductions 两段
  合并为 per-core 事件队列；归约挂 `acquire_electronic` 资源池。
- M2: `streaming.py` accumulator 键控（`accumulator_key`）扩展为驻留对象 + 容量预算
  （`scratch_required` 检查处拆分 accum 与 input 两池）。
- M3: `streaming.py` `run()` 帧循环把 `core_weights`/`core_ready` 提升为实例级跨帧状态，
  `weight_blocks` 的键与 MRR 容量管理器（`joint_sim/mrr_residency.py`，已有 LRU）对接。
- 验证: `experiments/hpat_mobilevit/verify.py` 已检查 partial_sum 读写时序，扩展驻留断言。
