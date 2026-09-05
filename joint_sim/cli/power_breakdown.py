#!/usr/bin/env python
"""Print the audited energy ledger produced by ``run_joint_sim.py``.

【中文说明】
本脚本是一个"查看器"：把 run_joint_sim.py 运行后写出的能量总账
（results/joint_sim/energy_total.csv）读取出来，格式化成表格打印到屏幕。
它自己不仿真、不计算，只负责把"能量账本"展示给人看，方便分析
"光子动态能耗 / 调度编程事件能耗 / 激光功率 / MRR 保持功率 / 电子后端能耗"
中哪一项占比最大。

运行方式：python joint_sim/cli/power_breakdown.py [--results 结果目录]
  --results 默认是 results/joint_sim（该目录下需要有 energy_total.csv，
  即需要先运行过 run_joint_sim.py 才会存在）。
产出：屏幕打印的能量账本表格（无结果文件）。
"""

import argparse
import csv
from pathlib import Path


def main() -> None:
    """命令行入口：读取能量总账 CSV 并打印成表格。

    流程：解析 --results 参数 → 拼出 energy_total.csv 的完整路径 →
    检查文件是否存在（不存在则报错并提示先运行 run_joint_sim.py）→
    用 csv.DictReader 读成字典列表 → 逐行打印"类别/组件/能量/功率"。
    每行的 energy_j（能量，焦耳）与 power_w（功率，瓦特）如果读得出来
    就格式化成科学计数法/浮点数，读不出则原样输出（容错空值）。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", default="results/joint_sim",
        help="Result directory containing energy_total.csv",
    )
    args = parser.parse_args()
    # 把结果目录转成绝对路径，并定位到 energy_total.csv
    path = Path(args.results).resolve() / "energy_total.csv"
    if not path.is_file():
        # 能量账本不存在，多半是还没跑过主仿真
        raise FileNotFoundError(
            f"Audited energy ledger not found: {path}. Run run_joint_sim.py first."
        )

    # 用 DictReader 读 CSV：表头（category/component/energy_j/power_w 等）作为字典键
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    # 打印表头：类别 | 组件 | 能量（焦耳） | 功率（瓦特）
    print(f"Energy ledger: {path}")
    print(f"{'category':<12} {'component':<28} {'energy (J)':>14} {'power (W)':>12}")
    print("-" * 72)
    for row in rows:
        energy = row.get("energy_j", "")
        power = row.get("power_w", "")
        # 能量用科学计数法（数值可能很大/很小），转失败就保留原字符串
        try:
            energy = f"{float(energy):.6e}"
        except (TypeError, ValueError):
            pass
        # 功率用普通浮点数，转失败同样保留原字符串
        try:
            power = f"{float(power):.6f}"
        except (TypeError, ValueError):
            pass
        print(
            f"{row.get('category', ''):<12} {row.get('component', ''):<28} "
            f"{energy:>14} {power:>12}"
        )


if __name__ == "__main__":
    main()
