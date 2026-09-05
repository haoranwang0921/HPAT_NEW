"""
Joint simulation framework for LLMCompass + SimPhony LPWM co-simulation.

Core modules:
  - schema.py          : OperatorRecord, TileRecord, TileAccess
  - tile_mapper.py     : 64x64 weight tile decomposition and trace annotation
  - sram_manager.py    : Tile-SRAM residency manager with pluggable policies
  - mrr_residency.py   : MRR tile-residency state (tuning/refresh)
  - scheduler.py       : ResourcePool and ResourceScheduler with DAG scheduling
  - op_classifier.py   : White-list operator classification
  - electronic_backend.py : Roofline model for electronic ops
  - simphony_backend.py   : SimPhony GemmCostSimulator wrapper
  - cost_cache.py      : SQLite cache for photonic kernel costs
  - report.py          : Report + SweepReport generation
  - lpwm_trace_exporter.py : PyTorch hook-based LPWM trace export
"""
# =============================================================================
# 中文阅读说明（本包的总体分工）
# =============================================================================
# 本项目用光子微环谐振器（MRR）阵列加速 LPWM 世界模型（视频预测模型）的推理。
# 本包 joint_sim 是核心仿真器，整体数据流是：
#   1. lpwm_trace_exporter 在 PyTorch 模型上挂 hook，导出算子执行轨迹（trace），
#      即"每个算子叫什么、输入输出多大、谁依赖谁"的列表；
#   2. op_classifier 判断每个算子该走光子还是电子后端；
#   3. tile_mapper 把大的权重矩阵切成 64x64 小块（tile），映射到光子阵列；
#   4. scheduler（调度器）按依赖关系 DAG 把算子排到共享资源上执行；
#   5. 光子算子用 simphony_backend 算成本，电子算子用 electronic_backend 算成本，
#      cost_cache 负责缓存光子成本避免重复仿真；
#   6. sram_manager/mrr_residency 管理权重在片上 SRAM 和 MRR 的驻留与复用；
#   7. report 汇总所有成本，输出时延与能耗的最终结果。
# 时间单位一律为秒（s），能量单位一律为焦耳（J）。
# =============================================================================
