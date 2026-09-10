# 修订实验计划执行记录

依据：EXPERIMENT_PLAN_REVISED_20260908.md（v2）。本轮已执行到 Phase 4，停止扩展；未加载官方权重、未扩展 XS/S、未开展准确率实验。

| 阶段 | 状态 | 产物 |
|---|---|---|
| P0 | 保留冻结 | results/hpat_mobilevit/hardware_combo_32_10_25_v1 |
| P1 | 账本验收及事件审计补齐 | experiments_p1_adc_audit_v1；P4/p1_event_audit_supplement.csv |
| P2 | 50点，99次运行，198帧 | results/hpat_mobilevit/experiments_p2_sensitivity_v1 |
| P3 | 6候选，12次运行，24帧 | results/hpat_mobilevit/experiments_p3_optimization_v1 |
| P4 | 排序、推荐、6组PDF/PNG及最终复核 | results/hpat_mobilevit/experiments_p4_final_summary_v1 |

P2比计划额外把联合静态功率拆成激光、MRR hold、电子静态三个独立轴，避免归因混淆。全部比较保持电子主点100%利用率/0.3W/0.4pJ，未换成历史调整版D0。

写环时间扫描固定每环能量；组织扫描固定编程并行度4。C5因编程时间/并行度影响小于5%被删除，理由已记录；新增C6=H1+2MiB，依据P2映射和缓存结果。C4是转换器能量/时间均减半的条件性组合，不是实测器件。

最终111个案例配置、源码、trace和时间线哈希复核通过；136 tests passed，3条依赖弃用警告。P0与P1未覆盖。P1完整表补在P4而非回写冻结目录。

主要结果：2MiB和25MiB的XXS cold/warm时延能耗一致；32×16降低能量但增加时延；C6相对H3/P0 warm时延-21.25%、能耗-16.50%，平均功率增加约6.03%。所有候选仍慢于且耗能高于匹配D0，不能报告HPAT胜出。

图表样式来自scientific-visualization的publication模板；确定性单次cold/warm扫描不伪造置信区间。经人工查看后统一breakdown图颜色/纹理和事件条。复现脚本为plan_sweep.py、summarize_plan.py及finalize_plan_review.py。

下一轮如果改10ps物理假设、增加SRAM面积/漏电、真实器件或官方权重，应另开实验版本。当前排序仅针对列出的扫描范围，不是范围无关的固有重要性。
