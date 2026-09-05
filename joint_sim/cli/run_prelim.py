#!/usr/bin/env python
"""Deprecated compatibility wrapper for the maintained joint simulator.

【中文说明】
同样是"兼容壳"脚本（已废弃）。早期用于跑初步（preliminary）仿真的入口，
现在不再维护，直接转发给主入口 run_joint_sim.py，防止旧命令在过时语义下
运行。运行后会先打印一条弃用警告，再等价于运行 run_joint_sim.py。
"""

import warnings

# 优先从包路径导入主入口；包不可用时回退到同目录导入
try:
    from joint_sim.cli.run_joint_sim import main
except ImportError:
    from run_joint_sim import main


if __name__ == "__main__":
    # 提醒用户改用新入口
    warnings.warn(
        "run_prelim.py is deprecated; delegating to run_joint_sim.py",
        DeprecationWarning,
        stacklevel=1,
    )
    # 委托给主仿真入口执行
    main()
