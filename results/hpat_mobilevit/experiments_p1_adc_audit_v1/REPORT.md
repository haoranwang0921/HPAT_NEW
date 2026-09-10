# P1 ADC/DAC 账本验收（2026-09-08）

## 结论

记账实现验收通过：4 cases / 8 frames 复现 P0，133 tests passed / 3 deprecation warnings。
本次没有改变器件参数、调度、位宽、写环成本或 P0 文件。仅拆分 HPAT 动态能量的事件归属；LT 路径不在本次修改范围。
这不是器件物理可实现性、测量精度或完整系统建模的认证。

## 冻结文档检查

`experiments/hpat_mobilevit/P0_BASELINE_20260908.md` 的主点、结果和不应外推范围与保存产物一致。
P0 aggregate SHA-256 经重新计算仍为 `9d0d09234bb2a587d66cdcf21a4f744e4424164404df74cbfe25cab777f8cf92`。
以下文字/来源问题应在下一版冻结说明中澄清；为保持冻结产物不可变，本次未改写 P0：

1. 当前阵列是 32×32，但 config 的 `user_confirmed.program_response_scope` 仍写 `entire_16x16_physical_array_written_and_settled`。执行器实际按当前阵列块调度一次 10 ps 写入；这是旧标签残留，不是执行器退回 16×16。32×32 的 10 ps 稳定仍需物理依据。
2. 匹配 D0 采用利用率 1.0、电子静态 0.3 W、MAC 0.4 pJ、权重缓存 25 MiB；不是此前另跑的 80% / 0.5 W / 0.9 pJ / 2 MiB D0。D0 不使用光子阵列或光子频率。
3. 原 provenance.json 顶层只有 Python 版本，没有宣称的 Torch/timm 版本字段；trace 相关环境应查对应 manifest，不应笼统声称全部在该 provenance 文件。
4. 对照表中的 base 16×16/5 GHz/1 MiB 实际使用同一移动参考模型的 64 GB/s DRAM，不能与文档所述更早的 512 GB/s 历史配置混称。
5. verification 是现有实现的记账/时间线检查，不是对所有真实依赖或物理机制的完整验证。

## 事件归属

| 事件/账目 | 内容 |
|---|---|
| dac_encode | 输入 DAC + MZM 动态能量 |
| adc_convert | ADC + TIA + PD 动态能量 |
| optical_compute | 光学路径剩余动态项；当前模型这些项为零 |
| programming | MRR 调谐事件能量 + 权重偏置 DAC 样本能量 |
| mrr_hold | 全推理时间 × MRR 保持功率 |
| laser_static | 全推理时间 × architecture_cost 的系统激光墙插功率 |

光学动态项为零不代表光计算无能耗：激光、MRR hold 和 programming 均单独保留，不移入 optical_compute 以免双计。
不应把 dac_encode 当成“仅 DAC”，或把 adc_convert 当成“仅 ADC”。每个 kernel 的 `stage_components_j` 可进一步区分器件。
偏置 DAC（i8）只在编程事件收费，不在输入编码再次收费。

普通 streaming、完整算子流水线、行块流水线均使用同一 stage ledger。未知器件或误放入动态账本的 MRR/偏置 DAC 会显式报错。
旧合成测试后端只有 aggregate 时保留兼容；真实 PhysicalCoreCosts 始终输出显式分组。

## 单位与器件来源核对

| 器件 | 本地源配置 | P0 10 GHz 初始化后的单器件动态功率 | 每样本能量 |
|---|---|---:|---:|
| ADC_SAR_1 | adc.yml：14 nm / 8 bit / 10 GSPS / 14.8 mW | 14.8 mW | 1.48 pJ |
| DAC_2 | dac.yml：16 nm / 8 bit / 14 GSPS / 50 mW | 35.7142857 mW | 3.5714286 pJ |
| TIA_1 | tia.yml：3 mW dynamic | 3 mW | 0.30 pJ |
| LT_PD | photodetector.yml：1.1 mW dynamic | 1.1 mW | 0.11 pJ |
| LT_MZM | 后端初始化值 | 4.5 mW | 0.45 pJ |

ADC/DAC 在 SimPhony `energy.py` 根据工作频率及位宽缩放功率；DAC 的 50 mW 是 14 GSPS 源点，不是 10 GHz 主点功率。
换算关系：mW / GHz = pJ；pJ × 1e-12 = J；mW × 1e-3 = W；W × s = J。
每核动态能量 = 两个输入符号 pass × M 个输入向量 × 已初始化物理器件数量 × 每样本 J。
当前按物理实例数收费，未新增尾块通道门控优化。

本地数据库文献链接：

- ADC：IEEE 9731625，`https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9731625`。
- DAC：IEEE 9162776，`https://ieeexplore.ieee.org/document/9162776`。
- TIA：IEEE 8510668，`https://ieeexplore.ieee.org/document/8510668`。
- LT_PD 的 `ref` 为空，1.1 mW 目前不能称为已经独立核验的原论文实测参数。
- HPAT_MRR 明确是 mixed literature-calibrated model，`exp: False`，不是一颗器件的联合实测配置。

本次已核对本地配置和运行时初始化值。外部检索连接失败，因此未独立复核上述原文是否支持全部参数，不能把“数据库填了参考链接”当作“原论文全部参数已核验”。
kernel_costs.json 保存 initialized_devices 供追溯；其中 upstream total_* 是原始参考值，实际账本使用 dynamic_energy 单样本值和显式活动次数，不采用 upstream reuse-discounted 总量。

## MRR 覆盖关系

`mrr_weight.yml: HPAT_MRR.cfgs.response_time = 1000 ns = 1e-6 s`
→ SimPhony 原始参考 programming latency 仍为 1 us
→ `config.user_confirmed.program_response_time_s = 1e-11 s`
→ HPAT 调度器的实际整块 programming 事件为 10 ps。

这是调度层覆盖，不是修改器件数据库。各硬件点的 override_audit.json 保存声明、实际阵列、单位、覆盖字段与能量规则。
当前 tuning 能量公式：reference_per_ring_J × physical_ring_count × (ptime/1e-6) × program_energy_multiplier。
当前 multiplier = 100000，因此 10 ps 主点保留旧 1 us 的每环调谐能量；偏置 DAC 能量另外添加一次。
限制：今后扫描 ptime 而不同时调整 multiplier，写环能量仍会随时间变化；本次只是忠实复现固定主点，没有把该公式改成独立扫描参数。

## 当前 P0 H3 warm 结果

| 项目 | 新账本 |
|---|---:|
| 时延 | 0.7995425730519088 ms |
| 总能量 | 5.831783351774694 mJ |
| DAC + MZM | 0.092046832457128 mJ |
| ADC + TIA + PD | 0.043260376320000 mJ |
| optical_compute 动态 | 0 mJ |
| MRR programming + bias DAC | 0.287668516571394 mJ |
| MRR hold | 0.327492637922062 mJ |
| Laser | 2.055980651788702 mJ |
| DRAM | 2.270772000000000 mJ |

上表是选定分量，完整闭合分解见对应 energy.csv。编码和转换合计仍为旧 optical_compute 的 0.135307208777 mJ。
DAC+MZM 占总量 1.578%，ADC+TIA+PD 占 0.742%；两者合计 2.320%。就当前能量模型而言，主要能量项仍是 DRAM（38.938%）与激光（35.255%），不是 ADC。
这只能解释当前模型，不能外推为真实芯片 ADC 不重要。

## 验收证据及命令

```
python -B -m pytest joint_sim/tests experiments/hpat_mobilevit/tests -q -p no:cacheprovider
133 passed, 3 warnings in 9.48s

python -B -m experiments.hpat_mobilevit.run_adc_audit --output results/hpat_mobilevit/experiments_p1_adc_audit_v1
```

复现涵盖 base/current × D0/H3 × cold/warm。每个 frame 的时延、事件数量、非三阶段能量项与旧值完全一致；总能量及三阶段总和按浮点容差检查。
当前 H3 warm 总量变化为 -8.673617379884035e-18 J，仅求和舍入；能量分解闭合误差为 0 J。
新 aggregate SHA-256：`d917c2911e6031e4bf537a7e2412193f1d112068a345ef3f6dd0723815ee8ebf`。
参数标签是冻结主点 Nominal；10 ps 可实现性单列为 Aggressive hypothetical，不代表文献标称器件。
provenance、完整 config、trace SHA-256、kernel 初始化、override_audit、verification、completion 已保存。
