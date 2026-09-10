# HPAT × MobileViT：A16 手机基线对齐报告

- 实验: `results/hpat_mobilevit/experiments_a16_aligned/mobile_reference_0.15tops/`(2026-09-06,5 轮迭代后终版)
- 规格文件: `experiments/hpat_mobilevit/mobile_a16_alignment.json`
- 对齐图: `fig4_a16_alignment.png`
- 表格取 warm 帧(warm frame #1)单帧值;不是随机重复实验均值
- 数据声明(重要): **A16 目标值为对数轴柱状图柱高估读,不是 iPhone 原始放电数据**

## 1. 任务与口径

论文图 `hpat_latency_energy_efficiency_context.png` 含 iPhone A16(实测)与 HPAT(估)对比列。
本次任务:修改电子基线参数,使 joint_sim 仿真的 **D0(全电子)最终时延与能耗对齐图中 A16 手机实测量级**,
从而给"HPAT vs 手机"对比一个尽量公平的电子参照系。

- 基准规格: `mobile_reference.json`(mobile_reference profile) + `config_10ps_fixed_write_energy.json`(10 ps 写环)
- 调参对象(仅电子侧): `main_dense_int8_tops`、DRAM/LPDDR 带宽、片上 scratchpad 带宽、
  `electronic_static_power_w`、`external_memory_energy_per_byte_j`
- 光侧参数(2 tile × 2 核 16×16、10 ps 写、符号行配对)与之前批次完全一致,不参与拟合

### 目标值(图估读)

| 模型 | A16 时延估读 (ms) | A16 能耗估读 (mJ) |
| --- | ---: | ---: |
| MobileViT-XXS | 11 | 53 |
| MobileViT-XS | 30 | 167 |
| MobileViT-S | 50 | 167 |

估读不确定度:对数轴柱高读数,按 **±30%** 对待,直至拿到原始放电 CSV。

## 2. 调参轨迹(5 轮迭代)

| 轮次 | 手段 | XXS D0 时延 (ms) | 说明 |
| --- | --- | ---: | --- |
| 起点 | mobile_reference @ 10 TOPS | 0.76 | 远低于 11 ms 目标,电子计算峰值+带宽地板太高 |
| 1–5 | 逐步降 `main_dense_int8_tops`(10→2→0.6→0.25→0.18→**0.15**)、LPDDR 带宽降到 12 GB/s、scratchpad 50 GB/s、静态功耗抬到 3.5 W | 7.56 | 每次 ~16 min 后台仿真,共 5 轮 |

### 终版规格(即 `mobile_a16_alignment.json`)

| 参数 | 终值 | 物理注记 |
| --- | --- | --- |
| `main_dense_int8_tops` | **0.15** | 仅为拟合旋钮;真实 A16 ANE 常报 15–17 TOPS(见 §5 局限) |
| `external_memory_bandwidth_bytes_per_s` | 12 GB/s | 接近单 16-bit LPDDR5X 通道量级 |
| `scratchpad_bandwidth_bytes_per_s` | 50 GB/s | — |
| `electronic_static_power_w` | 3.5 W | SoC 级平均静态 |
| `external_memory_energy_per_byte_j` | 3.5e-10 (0.35 nJ/B) | LPDDR5X 类 DRAM 搬运能量,量级合理 |

## 3. 对齐结果(终版,warm)

| 模型 | 目标 (ms/mJ) | **D0 全电子** | H1 +Linear | H2 +1×1Conv | H3 +Attn |
| --- | ---: | ---: | ---: | ---: | ---: |
| XXS | 11 / 53 | **7.56 / 34.6** | 9.20 / 50.2 | 10.09 / 54.5 | 10.62 / 57.1 |
| XS | 30 / 167 | **26.75 / 118.7** | 32.20 / 171.7 | 38.54 / 202.3 | 40.40 / 211.3 |
| S | 50 / 167 | **54.65 / 233.2** | 71.20 / 367.6 | 84.91 / 433.6 | 89.57 / 456.2 |

### 残余误差(D0 vs 目标)

| 模型 | 时延误差 | 能耗误差 | 判断 |
| --- | ---: | ---: | ---: |
| XXS | −31% | −35% | 图估读 ±30% 不确定带边缘 |
| XS | −11% | −29% | 时延已贴近;能耗仍偏低 |
| S | +9% | **+40%** | 时延贴近;能耗**过冲**(静态项主导) |

## 4. 为什么"三档同时精确对齐"做不到(结构原因)

对齐后各档能耗构成里 **电子静态项 ≈ 静态功率 × 时延** 占大头:

- XXS: 3.5 W × 7.56 ms ≈ 26.5 mJ(占总 34.6 的 ~77%)
- XS: 3.5 W × 26.75 ms ≈ 94 mJ(~79%)
- S: 3.5 W × 54.65 ms ≈ 191 mJ(**~82%**)

而图估读目标自身不自洽:XS 与 S 能耗同为 167 mJ,但时延是 30 vs 50 ms → 隐含平均功率 5.6 W vs **3.3 W**,
随模型变大反而下降,与任何"静态+动态"标量模型都矛盾(或 S 档估读偏低)。
因此 3 个标量旋钮无法同时满足 6 个目标:把 XXS/XS 能耗抬上去的旋钮会让已过冲的 S 更过冲。
在 ±30% 估读带内,XS 时延(−11%)已达标,XXS(−31%)与 S(+9%)时延贴近,能耗整体在"量级对齐"而非"精确对齐"。

## 5. 局限与诚实性声明(发表前必须处理)

1. **A16 目标是柱状图估读**:来源是对数轴柱高读数,±30% 不确定;工作区无 iPhone 原始放电 CSV。
2. **0.15 TOPS 是拟合旋钮,不是器件事实**:真实 A16 ANE 常报 15–17 TOPS dense INT8。差 ~100×,
   说明 joint_sim 的 roofline 电子后端**缺存储受限地板与真实功耗管理建模**——真实手机 11–50 ms/图 的延迟
   由 DRAM 带宽与 DVFS/调度主导,不是 MAC 峰值;用 0.15 TOPS 只是把"带宽/静态地板"代理到正确量级。
   这是 roofline 后端的已知局限(峰算力×峰值带宽公式不含 cache/NoC/调度),不应被解读为对 A16 的器件级标定。
3. **单 profile 三档不自洽**:见 §4;若要发表级对齐,需逐模型实测(每档一个目标)或引入功耗-负载曲线。
4. **结构性结论不变**:即便 D0 已对齐到 A16 量级,H1–H3 混合映射仍全面劣于 D0
   (XXS H1 +22% 时延/+45% 能耗;S H3 +64%/+96%),光-电混合在"以 A16 为电子基线"下仍不占优。
   HPAT 相对优势场景仍是**桌面 CPU/GPU 基线**(E2E 51 ms / 2.7 J,30-step rollout 光-电混合 2.96 ms beat 3.12 ms full-electronic 的 LPWM 线),以及能耗效率量纲的体系对比。

## 6. 下一步(按优先级)

1. **拿真实数据**:向论文共同作者索取 A16 三档原始放电 CSV(替代图估读),对齐才有发表级可信度。
2. **修后端模型**:给电子 roofline 补存储受限地板(显式 DRAM 流量已在账本内,但时延公式未体现排队/带宽竞争),
   而不是继续用 TOPS 旋钮拟合。
3. 若只用于论文中的"量级参照",当前 D0(XS −11% 时延、整体能耗偏差 <±40%)可作上限估计引用,但须标注图估读来源。
