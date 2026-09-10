# HPAT × MobileViT 联合仿真实验

本目录是在远端 `lpwm-photonic-sim` 克隆上新增的 MobileViT 工作负载集成。
复用 `hpat-artifact` 的模型配置、加载和分类辅助代码以及旧算子表；不复用旧延迟、
能耗占比、Amdahl 倍率或论文结论。来源、版本和许可证见 `PROVENANCE.md`。

## 当前可用产物

仓库根目录为 `C:/Users/whr/Desktop/ASP-DAC27/lpwm-photonic-sim`。
中文结果解读和主实验表见同目录 `RESULTS_20260905.md`。

2026-09-06 更新：用户确认整块阵列写入并稳定为 **10 ps**，每次写入能耗不变。
新配置为 `config_10ps_fixed_write_energy.json`；新主实验与前后对比见
`RESULTS_10PS_20260906.md`。旧 `config.json` 保留 1 μs 供复现，运行 10 ps
实验必须显式传入新配置，不能混用旧扫描或旧图注。

- `results/hpat_mobilevit/experiments_10ps_fixed_energy_v1/`：最新 10 ps 主实验，12 组、36 帧。
- `results/hpat_mobilevit/figures_10ps_fixed_energy_v1/`：冷启动与预热的 1 μs/10 ps 对比图。
- `results/hpat_mobilevit/test_report_10ps_v1.xml`：95 项测试全部通过的报告。

以下为已归档的 1 μs 完整实验产物，后文建模说明和旧复现命令也对应此版本：

- `results/hpat_mobilevit/trace_v1/`：三模型真实 ATen 生产者 DAG。
- `results/hpat_mobilevit/hardware_combo_32_10_25_v1/`：冻结 P0 主点（32×32 / 10 GHz / 25 MiB）的配置与扫描产物。

> ⚠️ 上述**两项是「实验输入」而非运行产物**，已入库；缺失时任何 runner 都会立即失败。
> 其生成命令、manifest 元数据、引用者清单与复现步骤见
> [`results/hpat_mobilevit/README.md`](../../results/hpat_mobilevit/README.md)。

- `results/hpat_mobilevit/experiments_v2/`：240 组完整配置、720 个推理帧；
  主结果、所有扫描、原始参数、成本缓存和来源指纹。
- `results/hpat_mobilevit/experiments_v2/verification.json`：独立核验报告。
- `results/hpat_mobilevit/figures_v3/`：最终 7 组图的 PDF、SVG、PNG 和绘图来源清单。
- `results/hpat_mobilevit/test_report_v3.xml`：最终 92 项测试全部通过的报告。

`figures_v2` 是较早的绘图预览；最终图使用 `figures_v3`，包含黑白可辨的纹理编码。

`pilot_v1` 是早期单模型调试；`experiments_v1` 未完成且已被后续实现替代。
两者保留排查记录，**不得混入正式汇总或作图**。

## 已确认的选择与额外假设

用户确认：2 个 tile、每 tile 2 个 16×16 物理 PDPU；三档映射消融；
1 μs 名义写环及其扫描；XXS/XS/S 分别输入 192/224/256、batch=1；
名义电子模型和存储扫描；固定面积正负权重行配对及数字差分。

参数全部保存在 `config.json`。以下是显式建模假设，不是用户提供的器件实测值：

- 8 bit 输入/权重/输出成本，32 bit 部分和。FP32 随机权重只用于真实形状轨迹，
  **没有据此证明量化准确率或非理想性鲁棒性**。
- 沿用 5 GHz 光子工作频率、312 TFLOPS 名义电子峰值和 10% 向量吞吐系数。
- 主点：1 MiB 权重缓存、512 GB/s HBM、100 ns 每次 DMA 固定开销、
  2 TB/s 片上缓冲总线、4 条写环通道。
- 另有独立的 8 MiB 激活/部分和工作缓冲；因此不能把“1 MiB 权重缓存”
  描述成整机只有 1 MiB 存储。所有数字和混合配置使用相同预算。
  算法检查实际峰值，不会在不足时静默扩容。
- 静态功率按所有存在硬件域的整个推理时长积分；没有假定电子侧等待期间完全掉电。
- 以算子顺序派发，依赖来自真实 DAG；单算子内部最多 4 块并行，分批写入和计算。
  这不是逐周期、跨算子乱序或细粒度 ping-pong 最优流水模型。
- 为了明确系统计费边界，实体化算子之间的激活读写显式经过 HBM。
  reshape/view 等元数据操作为零成本；clone/cat/插值等真实数据操作计入搬运。
- 权重缓存使用 128-byte 填充块，与 16×8 逻辑权重布局对应；数字基线采用相同
  缓存布局。DMA 固定开销按块计，没有隐藏突发合并或无限带宽。
- 固定模型能耗没有统计随机性；三张连续输入用于区分冷启动与预热状态，不能作为
  三次独立随机测量计算置信区间。“warm”在图中明确是第 3 张图，不自动等于渐近稳态。

## 三模型工作量核对

| 模型 | 参数量 | 捕获算子 | MAC |
| --- | ---: | ---: | ---: |
| XXS | 1,272,024 | 581 | 210,041,600 |
| XS | 2,317,848 | 582 | 772,340,288 |
| S | 5,578,632 | 580 | 2,000,831,488 |

逐层 MAC 与旧算子表一致。采集模式前后的 FP32 输出逐元素一致。
参数量来自实际实例化模型，旧配置中的概括性参数量字段没有用作计算依据。

## 实现及上游适配

`trace.py` 用 TorchDispatchMode 捕获实际 ATen 操作，保留模块名、形状和生产者。
Attention 在相同模型中关闭融合执行，以显式捕获 QKᵀ、Softmax 和 AV。
`backends.py` 复用克隆中的 SimPhony 设备初始化、三阶段时延和 LLMCompass 能耗参数。
`streaming.py` 复用 joint_sim 的 ResourcePool、SramManager 和 LRU 实现。

HPAT 适配层处理了以下不能直接照搬的语义：

1. 单个 kernel 查询只表示一个物理核；四核面积/器件数/静态功率单独查询。
2. MVM 输入向量逐次流入，`M` 不映射到物理阵列宽度，避免多算一维并行。
3. 静态权重可以在处理多个输入向量时复用，但输入 DAC/EOM 活动不能除以向量数。
4. 每个逻辑权重映射到两个正负物理行，16 物理输出行仅提供 8 个逻辑输出。
5. 正负输入分两次；四个非负乘积通过三次差分恢复，并对 K 分块累加。
   部分和读取等待上次写入完成，读、算、写分别占用资源。
6. 物理核从写入到转换完成不得被其他权重覆盖。一次只接纳最多 4 个权重块，
   不能先写完整个大矩阵再假定全部权重仍驻留。
7. 总动态能耗由 SimPhony 初始化后的每器件单位能耗乘显式活动数得出。
   MRR 编程和偏置 DAC 从核动态账本剔除，在实际写环事件中收费；DAC/ADC 仅为子项。
8. 激光按整个四核输出行扇出查询；账本使用 architecture_cost 的激光功率，
   同时保留上游另一初始化路径的功率参考值，避免两种不同消光比口径混加。

上述修改只在新增 HPAT 适配层，不改动原世界模型实验路径。它们修正了本工作负载
的实现/记账语义，**不意味着器件参数已校准**。面积是 SimPhony 光电数据通路估计，
不包含此处完整 SRAM、电子计算平台及控制布线，不用于整机面积优势结论。

## 实验矩阵

- D0：全电子。
- H1：静态 Linear 映射到光子侧。
- H2：H1 + 单组 1×1 Conv。
- H3：H2 + QKᵀ/AV；动态权重标识包含输入帧和 batch/head，不能跨图复用。

每个点运行三种模型和四种映射：主点 1 个，写环时间 5 个点、写环能耗倍率 3 个点、
权重 SRAM 4 个点、HBM 带宽 4 个点、编程通道 3 个点，共 240 组。
主点在扫描中出现的重复值有意保留，用作重复执行的一致性核对。
每组仿真连续三张图，共 720 帧。

写环时间扫 0.1/0.3/1/3/10 μs 时保持名义调谐功率不变，能耗随持续时间变化；
独立能耗倍率扫描仅改变调谐事件能耗。快写点是探索假设，不表示已存在相应器件。

## 复现命令

在仓库根目录运行，使用新的输出目录名，防止覆盖历史数据：

```powershell
python -B -m pytest joint_sim/tests experiments/hpat_mobilevit/tests -q -p no:cacheprovider
python -B -m experiments.hpat_mobilevit.trace --output results/hpat_mobilevit/trace_NEW
python -B -m experiments.hpat_mobilevit.run --traces results/hpat_mobilevit/trace_NEW --output results/hpat_mobilevit/experiments_NEW --suite all --timeline
python -B -m experiments.hpat_mobilevit.verify results/hpat_mobilevit/experiments_NEW
python -B -m experiments.hpat_mobilevit.figures.gen_fig_evaluation --results results/hpat_mobilevit/experiments_NEW --output results/hpat_mobilevit/figures_NEW
```

已使用的环境：Python 3.11.5、torch 2.10.0+cpu、timm 1.0.25、torchvision 0.25.0+cpu、
matplotlib 3.7.2。未安装新包、未下载模型权重或数据集。

每组目录包含完整有效配置、原 trace manifest、分帧统计、逐算子结果和能耗分项。
12 组主实验还保存压缩原始事件；扫描使用相同运行时守恒检查，但不保存全部事件。
独立验证器重新读取主实验事件，检查资源冲突、权重就绪、读写顺序和能耗。
核验不是单纯读取 `checks=true` 字段。

## 结果应如何解释

当前名义参数下，新增四核小型 HPAT 并未优于同一电子模型的全电子执行。
本实验的用途是暴露编程、分块、数据搬运和常开功耗的成本，而不是凑出旧稿加速比。
不能把这些结果称为真实边缘 NPU、真实 GPU 测量或已实现芯片性能。
进入论文前需要进一步确定目标电子平台、DMA/突发和融合策略、掉电策略、
真实快写机制及器件功耗依据；不能仅凭本批内部一致性检查提升证据等级。
