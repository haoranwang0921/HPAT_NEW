# LT / MobileViT 轻量接入报告

## 已完成与证据边界

本轮完成三件事：官方代码运行验证、官方 Linear 预测器接收 MobileViT trace、SimPhony LT 核接入统一电子/存储账本的端到端受控比较。
这是架构探索，不是 LT 原论文完整系统复现、流片测量或分类准确率验证。

官方来源：

- 论文：https://arxiv.org/abs/2305.19533 （HPCA 2024，v3）
- 官方代码：https://github.com/zhuhanqing/Lightening-Transformer
- 本地固定提交：`7c3d4ad8839a61e42a5475c7f89cef515e104f87`
- 本地目录：`external/Lightening-Transformer`，保留上游 GPL-3.0 许可证，未修改上游代码。

LT 通过相干场的相位表达符号，使用干涉与平衡探测，不使用 HPAT 的正负行配对。固定相移器、路由微环与动态权重编码不是同一种器件活动，不能把路由环当作 HPAT 权重环编程。

## 官方代码如何运行

官方仓库分为 software_model（模型/准确率）、hardware_simulator（行为级成本预测）和 profile（GPU 测量）。本次只需硬件仿真，不需要下载 ImageNet、训练模型或安装旧版 timm。

在 `external/Lightening-Transformer/hardware_simulator` 执行：

```powershell
python -B entry_energy_latency_workload.py -e codex_smoke_20260906 --tokens 197 --model_name deit-t --config ./params/device_params/Dota_B_8bit.yaml
```

已成功产生 `results/codex_smoke_20260906/deit-t_197_8bit/dota_arch_opt_4t_2c/total.csv`。
官方示例输出为 1.205474 mJ、0.0194372 ms。它属于官方 DeiT 工作负载成本模型，不是本次 MobileViT 对照值；官方入口还按标准 Transformer 层数放大模块成本，因此不能直接将 MobileViT 名称替换进去。

从本项目根目录运行官方 Linear trace 接口：

```powershell
python -B -m experiments.hpat_mobilevit.run_lt_official_trace --output results/hpat_mobilevit/lt_official_trace_smoke_v1 --variant xxs
```

本轮接入 XXS 的 37 个实际 Linear，75,522,560 MAC；逐算子映射 `in_features=K, out_features=N, bs=M*batch_repetitions`，不使用官方入口的固定 12 层倍乘。
官方原生 8 核、12×12、8bit、开启共享和时间累加的 Linear-only 总和为 0.0918093 mJ、0.001666 ms。**不含 MobileViT 的卷积、其他算子和完整端到端执行，不能与 HPAT 整网结果直接相除。**

## 统一账本的受控比较

主入口：`run_lt_compare.py`。LT 器件适配位于 `lt_backend.py`，LT 专用分块调度位于 `lt_streaming.py`。
未更改 generic SimPhonyBackend 的 HPAT-only 限制；LT 适配直接调用底层支持 output_stationary 的 GEMM API，同时将 sub_arch.dataflow 设为同一值，避免周期与能耗数据流不一致。

共同设置：

- MobileViT-XXS，192×192，batch=1，完整 581 算子 trace；FP32 捕获、INT8/INT32 成本表示，不含量化准确率证据。
- 2 tile × 2 core；5 GHz；8bit。
- 电子 10 TOPS，外部 DRAM 64 GB/s，片上总线 256 GB/s；1 MiB 权重缓存、8 MiB 激活 scratchpad。
- 使用相同 128B 权重缓存布局、LRU 策略、单位能耗和非光子算子执行路径。
- 关闭理想化输入重叠、DMA 摊销、门控和驻留优化；逐传输保留 100ns 固定时延。此为保守串行测试口径，不代表优化编译器或真实移动 SoC。
- 所有存在的静态功耗域按完整推理时长积分；动态能耗只计一次。
- HPAT 保持 16×16 物理权重阵列、16×8 有符号逻辑容量、双输入遍历、10ps 写环且每次写入能耗不变。
- LT 为独立 DPTC，M/N/K 容量分别为 16/16/16，全范围单次计算；两个操作数保守串行编码；保留跨 K 的数字累加。
- LT 暂停跨核光广播、ADC 共享、模拟时间累加；复制单核成本得到四核系统，不暗中享用原始八核系统的共享器件折扣。

**同核数并非同面积或同峰值 MAC 数。** HPAT 的 16×16 是权重阵列；LT 的 16×16 是并行点积单元阵列，每个点积单元还有 16 波长。因此这只是受控接入测试，后续正式比较需等面积/资源预算。

## 结果（有效版本 v2）

数据：`results/hpat_mobilevit/lt_compare_smoke_v2/aggregate.csv`。
每案运行冷启动和预热各 1 帧，下表取预热帧，不是多随机重复均值。

| 架构 / 映射 | 时延 ms | 能耗 mJ | 平均功率 W |
|---|---:|---:|---:|
| D0 全电子 | 0.764859 | 2.661252 | 3.479401 |
| HPAT H1：Linear | 1.173502 | 4.623327 | 3.939768 |
| LT H1：Linear | 0.959459 | 22.610505 | 23.565899 |
| HPAT H3：Linear + 1×1 Conv + QKᵀ/AV | 1.550758 | 5.659935 | 3.649787 |
| LT H3：同映射范围 | 1.106641 | 25.922441 | 23.424438 |

本情景 LT 相比 HPAT，H1 延迟低约 18.2%，H3 低约 28.6%；能耗分别约为 HPAT 的 4.89×、4.58×。两者均未超过本情景的 D0。

不能由此宣称 HPAT 在真实器件上优于原始 LT：LT 关键共享/积分优化被关闭，面积没有匹配，静态功耗全程开启，存储模型未校准。

## 激光功率与面积重点检查

- HPAT 四核架构面积估计 3.8222 mm²；LT 独立四核估计 77.0718 mm²；不含统一电子侧完整布局证明。
- HPAT 链路预算激光功率 0.889503 W；LT 四核 19.287122 W。
- LT 较高的功率来自 8bit 的 `2^bits` 因子、更多输出点积单元、链路损耗及 PD 灵敏度；此为本地 SimPhony 模型估计，不是 LT 实测。
- 官方 LT 的 `PhotonicCrossbar.cal_laser_power` 已按 H×W 扇出计算整个核的激光预算；统一适配沿用 HPAT 的 architecture-cost 口径，按每核预算乘独立核数，不再乘光源器件个数。
- `lt_compare_smoke_v1` 在适配初期重复乘了光源数，**作废，仅保留诊断记录**；v2 修正后已重跑全部五案，不应引用 v1 的能耗。

## 验证

- 46 项测试通过，3 个依赖弃用警告；测试报告在 v2/tests.xml。
- 五案、十帧、1,538,942 条事件已检查；结果有完整压缩时间线、有效配置、源码指纹、trace 哈希、核成本缓存和校验报告。
- HPAT/D0 使用原独立时间线校验；LT 增加编码→计算→转换顺序、资源互斥、部分和先后关系、光子 MAC 数守恒、无权重写环检查。
- 能耗动态分账闭合、静态功率×时长校验通过；新 D0/HPAT 结果复现原串行口径。
- 这些校验不证明光学数值准确率，也不证明未建模的真实硬件队列和驱动行为。

## 复现

在项目根目录执行，输出目录必须是新的，防止覆盖旧结果：

```powershell
python -B -m pytest experiments/hpat_mobilevit/tests -q -p no:cacheprovider
python -B -m experiments.hpat_mobilevit.run_lt_compare --output results/hpat_mobilevit/lt_compare_next --variant xxs --frames 2
```

`--variant xs` / `--variant s` 可切换 trace；`--modes` 支持 linear、linear_pointwise、linear_pointwise_attention。

下一阶段应先校准 LT 器件/激光口径，加入真实的跨核共享和时间累加，再做等面积比较；不要只扩展模型规模而把本轮探索值当作论文主结论。
