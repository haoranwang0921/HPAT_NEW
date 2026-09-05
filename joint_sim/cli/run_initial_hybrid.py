#!/usr/bin/env python
"""Compatibility entry point for the original hybrid diagnostic.

The original implementation used whole-weight residency and an inconsistent
energy ledger. It now delegates to the maintained tile-level joint simulator
so old commands cannot silently produce results with obsolete semantics.

【中文说明】
一个"兼容入口"（已废弃的旧脚本壳）。早期版本的混合架构诊断脚本用的是
"整块权重驻留"假设和一套不一致的能量账本；为了让旧命令不会悄悄产出过时
语义的结果，这个脚本现在不再自行仿真，而是把执行转发给维护中的主入口
run_joint_sim.py（tile 级联合仿真器）。
运行方式：python joint_sim/cli/run_initial_hybrid.py [run_joint_sim 的参数]
效果：打印一条 DeprecationWarning 弃用警告，然后等价于运行 run_joint_sim.py。
"""

import warnings

# 优先从包路径导入主入口；包不可用时回退到同目录导入（兼容老用法）
try:
    from joint_sim.cli.run_joint_sim import main
except ImportError:
    from run_joint_sim import main


if __name__ == "__main__":
    # 提醒用户这是旧入口，应改用 run_joint_sim.py
    warnings.warn(
        "run_initial_hybrid.py is deprecated; delegating to run_joint_sim.py",
        DeprecationWarning,
        stacklevel=1,
    )
    # 委托给主仿真入口执行
    main()
