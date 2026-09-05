#!/usr/bin/env python
"""
JSON-based CLI for SimPhony queries (subprocess mode).

Reads a JSON request from stdin and writes a JSON response to stdout.
SimPhony debug output is redirected to stderr so stdout stays clean.

Usage:
    echo '{"command":"kernel_cost","M":257,"K":512,"N":512}' | python simphony_cli.py

【中文说明】
SimPhony（光子矩阵乘仿真器）的"命令行查询接口"（子进程模式）。
为什么需要它？SimPhony 的 onnarchsim 库调试时会往标准输出打印大量日志，
如果直接在主进程里调用，那些日志会污染我们自己的输出。所以本脚本把 SimPhony
包在一个独立进程里跑：从 stdin 读入一条 JSON 请求（命令+参数），把 SimPhony
的调试输出重定向到 stderr，再把干净的 JSON 结果写到 stdout。

支持的命令（JSON 里的 "command" 字段）：
- architecture_cost  查询光子阵列的架构成本（面积、激光器功率、器件数量等）
- kernel_cost        查询一次矩阵乘内核的时延/能耗（需给 M/K/N 和位宽）
- programming_cost   查询权重编程开销（把权重"写"进微环的成本）

用法示例：
    echo '{"command":"kernel_cost","M":257,"K":512,"N":512}' | python joint_sim/cli/simphony_cli.py
"""

import json
import sys
import os
import io

# 把 SimPhony 项目根目录加入搜索路径，以便 import onnarchsim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "SimPhony")))


def handle_request(request: dict) -> dict:
    """根据 JSON 请求执行一条 SimPhony 查询，返回结果字典。

    参数：request 是从 stdin 解析出来的请求字典，必须含 "command" 键。
    支持的配置键（均有默认值）：arch_cfg（架构配置 yml）、device_root /
    device_cfg_globs（器件配置）、arch_version、sub_arch_name（子架构名）。
    返回值：结果字典，成功时含 "status":"ok" 和对应字段，失败时含
    "status":"error" 与 message。未知命令返回 error。
    """
    cmd = request.get("command", "")
    # 读取可选参数（都有默认值，对应 SimPhony HPAT 异构架构的默认配置）
    arch_cfg = request.get("arch_cfg", "configs/design/architectures/HPAT_hetero.yml")
    device_root = request.get("device_root", "configs/devices")
    device_cfg_globs = request.get("device_cfg_globs", "*/*.yml")
    arch_version = request.get("arch_version", "v1")
    sub_arch_name = request.get("sub_arch_name", "HPAT")

    # 创建矩阵乘成本仿真器（封装了架构配置解析与成本计算）
    from onnarchsim.gemm_simulator import GemmCostSimulator
    sim = GemmCostSimulator(
        arch_cfg=arch_cfg,
        device_root=device_root,
        device_cfg_globs=device_cfg_globs,
        arch_version=arch_version,
        sub_arch_name=sub_arch_name,
    )

    if cmd == "architecture_cost":
        # 查询架构成本：面积、激光器功率、各类器件数量
        arch = sim.get_architecture_cost()
        return {
            "status": "ok",
            "pic_area_um2": arch.pic_area_um2,  # 光子计算芯片面积（平方微米）
            "rf_eic_area_um2": arch.rf_eic_area_um2,  # 射频电子接口面积
            "total_area_um2": arch.total_area_um2,
            "core_insertion_loss_db": arch.core_insertion_loss_db,  # 插入损耗（分贝）
            "laser_wall_plug_power_w": arch.laser_wall_plug_power_w,  # 激光器墙插功率
            "mrr_count": arch.mrr_count,  # 微环数量
            "pd_count": arch.pd_count,  # 光电探测器数量
            "dac_count": arch.dac_count,  # 数模转换器数量
            "adc_count": arch.adc_count,  # 模数转换器数量
        }

    elif cmd == "kernel_cost":
        # 查询一次矩阵乘内核的成本：M/K/N 是矩阵乘三维尺寸
        M = request["M"]
        K = request["K"]
        N = request["N"]
        cost = sim.simulate_gemm(
            M=M, K=K, N=N,
            input_bits=request.get("input_bits", 8),  # 输入位宽（默认 8 位）
            weight_bits=request.get("weight_bits", 8),  # 权重位宽
            output_bits=request.get("output_bits", 8),  # 输出位宽
            dataflow=request.get("dataflow", "weight_stationary"),  # 数据流方式
        )
        return {
            "status": "ok",
            "compute_latency_s": cost.compute_latency_s,  # 计算时延
            "operand_encoding_latency_s": cost.operand_encoding_latency_s,  # 操作数编码（电→光）时延
            "conversion_latency_s": cost.conversion_latency_s,  # 光→电转换时延
            "programming_latency_s": cost.programming_latency_s,  # 权重编程时延
            "dynamic_energy_j": cost.dynamic_energy_j,  # 动态能耗
            "dac_energy_j": cost.dac_energy_j,  # DAC 能耗
            "adc_energy_j": cost.adc_energy_j,  # ADC 能耗
            "laser_energy_j": cost.laser_energy_j,  # 激光能耗
            "mrr_tuning_energy_j": cost.mrr_tuning_energy_j,  # 微环调谐能耗
            "mrr_hold_energy_j": cost.mrr_hold_energy_j,  # 微环保持能耗
            "iter_M": cost.iter_M,  # M 维的分块迭代次数
            "iter_K": cost.iter_K,
            "iter_N": cost.iter_N,
            "switching_cycles": cost.switching_cycles,  # 切换周期数
            "max_cycles": cost.max_cycles,  # 最大周期数
            "utilization": cost.utilization,  # 阵列利用率
        }

    elif cmd == "programming_cost":
        # 查询权重编程开销（把权重写进微环阵列的花费）
        wp = sim.get_weight_programming_cost()
        return {
            "status": "ok",
            "tile_count": wp.tile_count,  # 权重分块数
            "programmed_mrr_count": wp.programmed_mrr_count,  # 已编程微环数
            "programming_latency_s": wp.programming_latency_s,
            "programming_energy_j": wp.programming_energy_j,
            "hold_power_w": wp.hold_power_w,  # 保持功率（维持已编程状态）
        }

    else:
        # 未知命令
        return {"status": "error", "message": f"Unknown command: {cmd}"}


if __name__ == "__main__":
    raw = sys.stdin.read()  # 从标准输入读取整段 JSON
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as e:
        # 输入不是合法 JSON：直接输出错误响应并退出
        json.dump({"status": "error", "message": str(e)}, sys.stdout)
        sys.exit(1)

    # chdir to SimPhony root so relative config paths resolve
    # 切换到 SimPhony 根目录，让相对配置文件路径能正确解析
    _simphony_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "SimPhony"))
    _cwd = os.getcwd()
    os.chdir(_simphony_root)

    # Capture all SimPhony debug output; only JSON goes to stdout
    # 关键技巧：临时把 stdout 换成 StringIO，接住 SimPhony 的调试日志，
    # 处理完再还原——保证 stdout 上只出现我们要输出的 JSON
    real_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        resp = handle_request(req)  # 执行查询
    except Exception as e:
        # 任何异常都转成错误响应，不让堆栈把 stdout 弄脏
        resp = {"status": "error", "message": str(e)}
    finally:
        debug_output = sys.stdout.getvalue()  # 取出被接住的调试输出
        sys.stdout = real_stdout  # 还原 stdout
        os.chdir(_cwd)  # 还原工作目录
        # 把调试输出改写到 stderr（用户仍能看到，但不污染 JSON）
        if debug_output.strip():
            print(debug_output.strip(), file=sys.stderr)

    json.dump(resp, sys.stdout, indent=2)  # 输出干净的 JSON 响应
