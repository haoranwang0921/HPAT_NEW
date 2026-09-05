# -*- coding: utf-8 -*-
"""joint_sim 的命令行入口（CLI）包。

本包是「26世界模型光子加速」研究项目仿真器（joint_sim）的命令行工具集，
每个脚本对应一类实验或一个工具。它们共同的工作方式是：
读入 LPWM 视频预测大模型的算子执行轨迹（trace）→ 交给 joint_sim 仿真器调度
→ 统计"光子后端 + 电子后端"混合推理的总时延与总能耗。

常用入口速查：
- run_joint_sim.py            主入口：跑一次完整的混合光子/电子联合仿真，输出时延与能耗。
- run_ablation.py             消融实验（实验2）：改变"映射范围"和"流水并行度"看影响。
- run_experiment4.py 系列     实验4：研究 SRAM 容量 / 权重驻留策略 / MRR（微环）复用。
- run_signed_weight_ablation.py 负权重符号处理（能否去掉符号拆分）的消融实验。
- power_breakdown.py          输出功耗构成分解（谁耗电最多）。
- hybrid_area.py              估算混合架构的芯片面积。
- export_lpwm_trace.py        把 LPWM 的算子轨迹导出/转成仿真器可用的数据。
- simphony_cli.py             面向 SimPhony 风格 MRR 阵列的命令行工具。

注：本文件本身不含逻辑，仅作为包的说明文档。
"""
