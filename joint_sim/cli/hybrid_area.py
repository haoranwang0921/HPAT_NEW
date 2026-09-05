#!/usr/bin/env python
"""Hybrid area breakdown: LLMCompass electronic + SimPhony photonic.

Two scenarios:
  1. Naive:  replace all 128 SM systolic arrays with photonic (upper bound)
  2. Scaled: ~16 lightweight vector units + 2ch HBM (actual need estimate)

【中文说明】
本脚本估算"光子+电子混合架构"的芯片面积（单位 mm²），是一个纯分析计算
脚本，不改任何文件，只把结果打印到屏幕。面积主要来自两个部分：
- 电子部分（LLMCompass 项目）：以 A100 GPU（128 个 SM，每个 SM 含脉动阵列
  SA、ALU、控制、寄存器堆、本地缓冲等）为基线。
- 光子部分（SimPhony 项目）：PIC（光子计算芯片）+ RF EIC（射频电子接口，
  含 DAC/ADC/TIA 等驱动电路）。

对比两个方案：
  [2] Naive（粗暴上限）：保留 128 个 SM，只是把每个 SM 里的脉动阵列（SA）
      换成光子阵列——面积不一定变小，仅作上界参考。
  [3] Scaled（实际需求估计）：因为光子阵列已经替代了大部分矩阵乘，电子端
      只需要约 16 个"轻量向量单元"（只有 ALU + 精简版控制/寄存器/缓冲），
      IO 也从 6 通道 HBM 减到 2 通道——这才是贴近实际的方案。

运行方式：python joint_sim/cli/hybrid_area.py
产出：屏幕打印的分项面积表和汇总表（无结果文件）。
"""
import sys, os, json

# 把项目根目录及其它子项目目录加入搜索路径，以便 import LLMCompass/SimPhony 的库
_PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT, "joint_sim"))
sys.path.insert(0, os.path.join(_PROJECT, "SimPhony"))
sys.path.insert(0, os.path.join(_PROJECT, "LLMCompass"))

# 从 SimPhony 后端拿光子面积；从 LLMCompass 成本模型拿电子面积
from simphony_backend import SimPhonyBackend
from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2

# ---------------------------------------------------------------------------
# SimPhony photonic area (measured from HPAT config)
# ---------------------------------------------------------------------------
# 光子部分面积：直接查 SimPhony 后端的架构成本数据（HPAT 配置里测量得到）
sim = SimPhonyBackend()
arch = sim.architecture_cost()

pic_mm2 = arch["pic_area_um2"] * 1e-6  # PIC 光子芯片面积（微米² → 毫米²，1e-6 换算）
rf_eic_mm2 = arch["rf_eic_area_um2"] * 1e-6  # 射频电子接口面积
photonic_total = pic_mm2 + rf_eic_mm2  # 光子部分总面积

# ---------------------------------------------------------------------------
# LLMCompass A100 electronic area baseline
# ---------------------------------------------------------------------------
# 电子部分：读 A100（GA100）的架构配置，用 LLMCompass 的成本模型算计算芯粒和 IO die 面积
config_path = os.path.join(_PROJECT, "LLMCompass", "configs", "GA100.json")
with open(config_path) as f:
    config = json.load(f)

compute_area, core_bd, compute_bd = calc_compute_chiplet_area_mm2(config, verbose=True)
io_area, io_bd = calc_io_die_area_mm2(config, verbose=True)
elec_total = compute_area + io_area  # 纯电子 A100 总面积的基准值

n_cores = config["device"]["compute_chiplet"]["physical_core_count"]  # 128  # A100 的 SM 数量（128）

# Per-core breakdown (core_bd values are per-SM before N-core scaling)
# 单个 SM 内部各组件面积分解（core_bd 是"单个核"的数据，未乘核数）
per_core_total = core_bd["total_core_area"]        # 4.10
per_core_sa = core_bd["sa_area"]                    # 0.27  脉动阵列（矩阵乘单元）
per_core_alu = core_bd["alu_area"]                  # 0.43  标量/向量运算单元
per_core_control = core_bd["control_area"]          # 1.32  控制逻辑
per_core_regfile = core_bd["regfile_area"]          # 1.57  寄存器堆
per_core_localbuf = core_bd["local_buffer_area"]    # 0.51  本地缓冲

# Totals across 128 SMs
# 128 个 SM 的累计面积
total_sa = per_core_sa * n_cores                     # 34.56  全部脉动阵列面积
total_cores_no_sa = (per_core_total - per_core_sa) * n_cores  # 490.24  去掉脉动阵列后的全部核心面积
crossbar_full = compute_bd.get("crossbar_area", 0.0) # 58.88  交叉开关（核间互连）面积

# ==================================================================
# 以下开始打印各分项面积（纯输出，无计算逻辑之外的动作）
print("=" * 72)
print("  Hybrid Area Model — A100 baseline + SimPhony HPAT photonic")
print("=" * 72)

# ---- Baseline ----
# [0] A100 纯电子基线面积
print(f"\n  [0] A100 electronic baseline (7 nm)")
print(f"      Compute chiplet:  {compute_area:>10.2f} mm2  "
      f"({n_cores} SMs x {per_core_total:.2f} mm2 + crossbar {crossbar_full:.2f})")
print(f"      IO die:           {io_area:>10.2f} mm2  (6ch HBM2e + 12x NVLink)")
print(f"      Electronic total: {elec_total:>10.2f} mm2")

# ---- Per-SM breakdown ----
# [1] 单个 SM 的面积分解，"混合架构需要什么"列给出每个组件的取舍
print(f"\n  [1] Per-SM breakdown (1 of {n_cores} cores)")
print(f"      {'Component':<20} {'Area (mm2)':>10}  {'Hybrid need':>15}")
# 各组件在混合架构里的处置：脉动阵列被光子替代，其余保留或按比例精简
items = [
    ("SA (16x16 systolic)", per_core_sa,     "removed (photon)"),
    ("ALU (scalar/vector)", per_core_alu,    "keep"),
    ("Control",             per_core_control,"keep, reduced 40%"),
    ("Register file",       per_core_regfile,"keep, reduced 30%"),
    ("Local buffer",        per_core_localbuf,"keep, reduced 30%"),
    ("Total per SM",        per_core_total,  ""),
]
for name, area, note in items:
    print(f"      {name:<20} {area:>10.2f}  {note}")

# ---- Naive: just swap SA for photonic (keep all 128 SMs) ----
# [2] 朴素方案：128 个 SM 全部保留，只是把脉动阵列换成光子阵列
naive_digital = total_cores_no_sa + crossbar_full + io_area  # 去掉脉动阵列后的电子部分
naive_total = naive_digital + photonic_total  # 加上光子面积
print(f"\n  [2] Naive: 128 SMs, SA -> photonic")
print(f"      Cores w/o SA:     {total_cores_no_sa:>10.2f} mm2")
print(f"      Crossbar:         {crossbar_full:>10.2f} mm2")
print(f"      IO die:           {io_area:>10.2f} mm2")
print(f"      Photonic (PIC+RF):{photonic_total:>10.2f} mm2")
print(f"      {'--':>20}  {'----------':>10}")
# 计算相对 A100 纯电子面积的百分比
print(f"      Total:            {naive_total:>10.2f} mm2  ({naive_total/elec_total*100:.0f}% of A100)")

# ---- Scaled: actual need (~16 lightweight units, 2ch HBM) ----
# [3] 精简方案（更贴近实际）：矩阵乘交给光子后，电子端只要约 16 个轻量向量单元
# Lightweight unit: ALU + reduced control + reduced regfile + reduced localbuf
# 单个轻量单元 = ALU 全保留 + 控制逻辑缩 40% + 寄存器堆缩 30% + 本地缓冲缩 30%
lightweight_unit = (
    per_core_alu
    + per_core_control * 0.40
    + per_core_regfile * 0.30
    + per_core_localbuf * 0.30
)
num_lightweight = 16  # 轻量单元数量（实际需求估计值）
digital_cores = lightweight_unit * num_lightweight

# IO: 2ch HBM (vs A100 6ch)
# IO：从 6 通道 HBM 减到 2 通道，按单通道面积线性缩放
io_per_channel = io_area / 6
digital_io = io_per_channel * 2

# Crossbar: scale with core count + photonic weight bank overhead
# 交叉开关按核数缩放，再乘 1.5 计光子权重库的额外开销
crossbar_per_sm = crossbar_full / n_cores
digital_crossbar = crossbar_per_sm * num_lightweight * 1.5  # photonic weight bank overhead

scaled_digital = digital_cores + digital_crossbar + digital_io
scaled_total = scaled_digital + photonic_total  # 精简方案总面积

print(f"\n  [3] Scaled: ~{num_lightweight} lightweight vector units + 2ch HBM + photonic")
print(f"      Lightweight unit: {lightweight_unit:>10.2f} mm2")
print(f"        = ALU({per_core_alu:.2f}) + ctrl*0.4({per_core_control*0.4:.2f}) "
      f"+ regfile*0.3({per_core_regfile*0.3:.2f}) + localbuf*0.3({per_core_localbuf*0.3:.2f})")
print(f"      Digital cores:    {digital_cores:>10.2f} mm2  "
      f"({num_lightweight} units x {lightweight_unit:.2f})")
print(f"      Crossbar:         {digital_crossbar:>10.2f} mm2  "
      f"(= A100 CB/SM {crossbar_per_sm:.2f} x {num_lightweight} x 1.5 photon overhead)")
print(f"      IO die (2ch HBM): {digital_io:>10.2f} mm2  "
      f"(= {io_per_channel:.0f} mm2/ch x 2)")
print(f"      Photonic (PIC+RF):{photonic_total:>10.2f} mm2")
print(f"      {'--':>20}  {'----------':>10}")
print(f"      Total:            {scaled_total:>10.2f} mm2  ({scaled_total/elec_total*100:.0f}% of A100)")

# ---- Comparison table ----
# [4] 汇总对比表
print(f"\n  [4] Summary")
print(f"      {'Scenario':<30} {'Area (mm2)':>12} {'vs A100':>10} {'Notes':>20}")
print(f"      {'-'*30} {'-'*12} {'-'*10} {'-'*20}")
print(f"      {'A100 electronic':<30} {elec_total:>12.2f} {'100%':>10} {'6ch HBM+12xNVLink':>20}")
print(f"      {'Naive (128SM-photon)':<30} {naive_total:>12.2f} {naive_total/elec_total*100:>9.0f}% "
      f"{'keep 128 SMs w/o SA':>20}")
print(f"      {'Scaled (16LU+2ch+photon)':<30} {scaled_total:>12.2f} {scaled_total/elec_total*100:>9.0f}% "
      f"{'~16 light units':>20}")

# ---- Die composition of scaled chip ----
# [5] 精简方案的芯片各"die"组成占比
print(f"\n  [5] Scaled chip die composition")
print(f"      {'Die':<24} {'Area (mm2)':>12} {'Share':>10}")
print(f"      {'-'*24} {'-'*12} {'-'*10}")
for name, area in [
    ("PIC (photonic compute)", pic_mm2),
    ("RF EIC (DAC/ADC/TIA)",  rf_eic_mm2),
    (f"{num_lightweight} light vector units", digital_cores),
    ("Crossbar + photon ctrl", digital_crossbar),
    ("IO die (2ch HBM)",       digital_io),
]:
    # 每一项占精简方案总面积的比例
    print(f"      {name:<24} {area:>12.2f} {area/scaled_total*100:>9.1f}%")
print(f"      {'-'*24} {'-'*12} {'-'*10}")
print(f"      {'TOTAL':<24} {scaled_total:>12.2f} {100.0:>9.1f}%")

print()
