"""
Report generation for joint simulation results (Plan Section 14).

Produces:
  - summary.json        : top-level metrics
  - operator_trace.jsonl: annotated trace with costs
  - kernel_costs.csv    : per-kernel photonic cost table
  - event_timeline.csv  : scheduled event timeline (from scheduler)
  - energy_breakdown.csv: per-component energy accounting
  - layer_breakdown.csv : per-operator latency/energy

中文阅读提示：报告层只整理已有成本和事件，不重新计算硬件。读结果时优先看
summary.json、event_timeline.csv 与 energy_breakdown.csv 三者是否相互一致。
"""
# =============================================================================
# 本文件角色一句话：把仿真跑完后的结果整理成一份份可审计的文件。
# 输出文件：
#   summary.json          顶层指标（端到端时延、能耗合计、驻留/调度摘要）
#   operator_trace.jsonl  带成本的算子级 trace
#   kernel_costs.csv      每个光子 GEMM 内核的成本明细
#   event_timeline.csv    调度器的事件时间线（DAC/计算/ADC/编程/DMA 事件）
#   energy_breakdown.csv  按静态/动态分类的能耗账本
#   layer_breakdown.csv   每个算子的时延/能耗
#   energy_total.csv      系统级能耗总账（静态 + 动态，含平均功率）
# 能耗账本的核心公式（见 _write_energy_totals）：
#   总能耗 = 激光持续功率 x 端到端时延         （静态，连续）
#          + MRR 保持功率 x 端到端时延         （静态，连续）
#          + 光子动态能耗（各调用求和，不含内嵌的一次性调谐）
#          + 调度编程事件能耗之和              （动态，逐事件）
#          + 电子后端能耗                      （动态）
# 关键提醒：静态项是"功率 x 总时间"，绝不能按内核重复相加；
# DAC/ADC 子项已包含在光子动态能耗里，也不能重复加。
# =============================================================================

import json
import csv
import os
from typing import Dict, List, Optional


class JointSimReport:
    # 一个实例对应一次实验运行；数据先累计在内存中，再集中写出多个可审计文件。
    """Collects results and writes structured output files.

    Parameters
    ----------
    output_dir : str
        Directory to write report files.
    run_id : str
        Unique identifier for this simulation run.
    """
    # 中文说明：一次实验运行的"报告对象"。外部代码把各类结果
    # （架构成本、内核成本、算子成本、时间线、驻留/调度摘要、守恒性检查、
    # 能耗合计）通过 set_* / add_* 方法喂进来，最后调 write_all() 一次性
    # 写出所有文件。数据先存内存再集中落盘，方便审计与复现。

    def __init__(self, output_dir: str = "results", run_id: str = "run_001"):
        self.output_dir = output_dir
        self.run_id = run_id
        os.makedirs(output_dir, exist_ok=True)

        # 各类结果的容器（初始为空，set_*/add_* 填充）
        self._architecture_cost: Optional[dict] = None
        self._kernel_costs: List[dict] = []
        self._operator_costs: List[dict] = []
        self._timeline_csv: str = ""
        self._residency_summary: Optional[dict] = None
        self._scheduler_summary: Optional[dict] = None
        self._conservation_checks: List[dict] = []
        self._energy_totals: dict = {}

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def set_architecture(self, cost: dict):
        """记录一次性架构成本（面积、器件数、激光功率等）。"""
        self._architecture_cost = cost

    def add_kernel_cost(self, M: int, K: int, N: int, op_id: str, cost: dict):
        # 记录一个光子内核的成本；只保留数值/字符串字段（避免把对象写进 CSV）
        self._kernel_costs.append({
            "op_id": op_id,
            "M": M, "K": K, "N": N,
            **{k: v for k, v in cost.items()
               if isinstance(v, (int, float, str))},
        })

    def add_operator_cost(self, rec: dict, cost: dict, resource: str):
        # 记录一个算子（含其原始字段与成本、所在后端）
        self._operator_costs.append({
            "op_id": rec.get("op_id", ""),
            "op_type": rec.get("op_type", ""),
            "op_role": rec.get("op_role", ""),
            "phase": rec.get("phase", ""),
            "M": rec.get("M"), "K": rec.get("K"), "N": rec.get("N"),
            "resource": resource,
            **cost,
        })

    def set_timeline(self, csv_str: str):
        """记录调度器的事件时间线 CSV 字符串。"""
        self._timeline_csv = csv_str

    def set_residency_summary(self, summary: dict):
        """记录驻留管理器的摘要（命中率、缺失数等）。"""
        self._residency_summary = summary

    def set_scheduler_summary(self, summary: dict):
        """记录调度器的摘要（端到端时延、事件数等）。"""
        self._scheduler_summary = summary

    def add_conservation_check(self, name: str, passed: bool, detail: str = ""):
        """追加一条守恒性检查结果。"""
        self._conservation_checks.append({
            "check": name, "passed": passed, "detail": detail,
        })

    def set_energy_totals(self, totals: dict):
        """记录系统级能耗总账（静态+动态，见 _write_energy_totals 的说明）。"""
        self._energy_totals = totals

    # ------------------------------------------------------------------
    # Output generation
    # ------------------------------------------------------------------

    def write_all(self):
        """写出本次运行的全部结果文件；文件分开便于复现和审计。"""
        self._write_summary_json()
        if self._operator_costs:
            self._write_operator_costs()
        if self._kernel_costs:
            self._write_kernel_costs()
        if self._timeline_csv:
            self._write_timeline()
        self._write_energy_breakdown()
        self._write_layer_breakdown()
        if hasattr(self, '_energy_totals') and self._energy_totals:
            self._write_energy_totals()

    def _write_summary_json(self):
        # 汇总：光子动态能耗、编程能耗。若调用方传了 _energy_totals
        # （系统级账本），优先用它里面的值；否则退回对内核成本做简单求和
        # （注意：简单求和会低估，因为相同签名的内核只算了一次）。
        total_photonic_energy = sum(
            k.get("dynamic_energy_j", 0) for k in self._kernel_costs
        )
        total_programming_energy = sum(
            k.get("programming_energy_j", 0) for k in self._kernel_costs
        )
        if hasattr(self, "_energy_totals") and self._energy_totals:
            total_photonic_energy = self._energy_totals.get(
                "dynamic_energy_j", total_photonic_energy
            )
            total_programming_energy = self._energy_totals.get(
                "programming_energy_j", total_programming_energy
            )
        e2e_latency = (
            self._scheduler_summary.get("end_to_end_latency_s", 0.0)
            if self._scheduler_summary else 0.0
        )

        summary = {
            "run_id": self.run_id,
            "architecture": self._architecture_cost,
            "end_to_end_latency_s": e2e_latency,
            "total_photonic_energy_j": total_photonic_energy,
            "total_programming_energy_j": total_programming_energy,
            "photonic_kernel_count": len(self._kernel_costs),
            "operator_count": len(self._operator_costs),
            "residency": self._residency_summary,
            "scheduler": self._scheduler_summary,
            "conservation_checks": self._conservation_checks,
        }
        # Include invocation-weighted energy totals if available
        if hasattr(self, '_energy_totals') and self._energy_totals:
            summary["energy_totals"] = self._energy_totals
        with open(
            os.path.join(self.output_dir, "summary.json"),
            "w", encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2, default=str)

    def _write_operator_costs(self):
        # 算子级 trace（JSONL，每行一个算子及其成本）
        path = os.path.join(self.output_dir, "operator_trace.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rec in self._operator_costs:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def _write_kernel_costs(self):
        # 光子内核成本表（CSV）：一行一个内核，字段固定便于分析
        if not self._kernel_costs:
            return
        path = os.path.join(self.output_dir, "kernel_costs.csv")
        keys = ["op_id", "M", "K", "N", "compute_latency_s",
                "operand_encoding_latency_s", "conversion_latency_s",
                "programming_latency_s", "dynamic_energy_j",
                "dac_energy_j", "adc_energy_j", "laser_energy_j",
                "mrr_tuning_energy_j", "mrr_hold_energy_j",
                "iter_M", "iter_K", "iter_N", "utilization"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(self._kernel_costs)

    def _write_timeline(self):
        # 事件时间线：把调度器生成的 CSV 字符串原样落盘
        path = os.path.join(self.output_dir, "event_timeline.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._timeline_csv)

    def _write_energy_breakdown(self):
        """按静态/动态来源写能耗分解，避免将持续功耗按 kernel 重复相加。"""
        # 中文说明：能耗分解账本（含"参考项"）。
        # 核心原则：静态功耗（激光、MRR 保持）是"功率 x 总时长"，属于连续
        # 消耗，只在整个系统级别记一笔；dynamic_energy_j / programming_energy_j
        # 是逐事件累加的动态能耗，可以安全相加。
        # dac/adc 是 dynamic_energy_j 内部子项，标注为"reference"提醒读者
        # 不要再加一遍（防重复计算）。
        path = os.path.join(self.output_dir, "energy_breakdown.csv")
        if hasattr(self, '_energy_totals') and self._energy_totals:
            comps = [
                ("static", "laser_total_j", "Laser: continuous P_laser x E2E_time"),
                ("static", "mrr_hold_total_j", "MRR hold: configured hold power x E2E time"),
                ("dynamic", "dynamic_energy_j", "SimPhony invocation energy excluding embedded one-shot MRR tuning"),
                ("dynamic", "programming_energy_j", "Weight program: scheduled tile-program events"),
                ("dynamic", "electronic_energy_j", "Electronic ops: LLMCompass ElectronicEnergyModel"),
                ("ref", "dac_energy_j", "REFERENCE SUBCOMPONENT: already included in dynamic_energy_j"),
                ("ref", "adc_energy_j", "REFERENCE SUBCOMPONENT: already included in dynamic_energy_j"),
                ("ref", "embedded_mrr_tuning_excluded_j", "REFERENCE ONLY: removed to avoid double-counting scheduled programming"),
                ("ref", "laser_per_kernel_sum_j", "REFERENCE ONLY: per-kernel laser sum (overlaps)"),
                ("ref", "mrr_hold_per_kernel_sum_j", "REFERENCE ONLY: per-kernel hold sum (overlaps)"),
            ]
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["category", "component", "total_energy_j", "notes"])
                for cat, key, notes in comps:
                    v = self._energy_totals.get(key, 0.0)
                    w.writerow([cat, key, v, notes])
        elif self._kernel_costs:
            # Fallback: unique-signature sums (undercounts)
            # 兜底：没有系统级账本时，按内核"唯一签名"求和（会低估重复调用）
            comps = ["dac_energy_j", "adc_energy_j", "laser_energy_j",
                     "mrr_tuning_energy_j", "mrr_hold_energy_j"]
            totals = {c: sum(k.get(c, 0) for k in self._kernel_costs) for c in comps}
            totals["total_dynamic_energy_j"] = sum(totals.values())
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["component", "total_energy_j"])
                for c, v in totals.items():
                    w.writerow([c, v])

    def _write_energy_totals(self):
        """Write system-level energy totals with proper static + dynamic accounting."""
        # 中文说明：系统级能耗总账。这是能量账本的"汇总表"，公式如下：
        #   静态项（连续功耗，整个系统只记一笔）：
        #     激光 = 激光壁插功率 P_laser x 端到端时延 E2E
        #     MRR 保持 = 每微环保持功率(0.1 mW) x 微环总数 x E2E
        #   动态项（离散事件，可以逐次累加）：
        #     光子动态能耗（各次调用之和，已剔除"内嵌的一次性调谐"——那部分
        #     改由调度器按编程事件逐次记账，避免重复计算）
        #     编程事件能耗之和
        #     电子后端能耗
        #   总计 = 静态 + 动态；平均功率 = 总计 / E2E；每步能耗 = 总计 / 步数
        path = os.path.join(self.output_dir, "energy_total.csv")
        et = self._energy_totals
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["category", "component", "energy_j", "power_w", "notes"])
            # Static (P x T — continuous, not per-kernel sum)
            w.writerow(["static", "laser_wall_plug", et.get("laser_total_j", 0),
                        et.get("laser_power_w", 0), "continuous: laser_w x E2E_time"])
            w.writerow(["static", "mrr_hold", et.get("mrr_hold_total_j", 0),
                        et.get("hold_power_w", 0), "continuous: 0.1 mW/MRR x 393216 MRRs"])
            # Dynamic (discrete events — safe to sum across invocations)
            w.writerow(["dynamic", "photonic_compute", et.get("dynamic_energy_j", 0),
                        "", "invocation sum excluding embedded one-shot tuning"])
            w.writerow(["dynamic", "weight_programming", et.get("programming_energy_j", 0),
                        "", "sum of scheduled tile-program event energies"])
            w.writerow(["dynamic", "electronic_ops", et.get("electronic_energy_j", 0),
                        "", "LLMCompass ElectronicEnergyModel (dynamic + static)"])
            # 参考项：已包含在上面，仅供核对，不要再加
            w.writerow(["reference", "dac_subcomponent", et.get("dac_energy_j", 0),
                        "", "included in photonic_compute; do not add again"])
            w.writerow(["reference", "adc_subcomponent", et.get("adc_energy_j", 0),
                        "", "included in photonic_compute; do not add again"])
            w.writerow(["reference", "embedded_tuning_excluded",
                        et.get("embedded_mrr_tuning_excluded_j", 0), "",
                        "replaced by scheduled weight_programming"])
            # Total
            # 总计公式（与 energy_breakdown 一致）：
            # 激光 + MRR保持 + 光子动态 + 编程 + 电子
            total = et.get("total_energy_j", (
                et.get("laser_total_j", 0) + et.get("mrr_hold_total_j", 0)
                + et.get("dynamic_energy_j", 0)
                + et.get("programming_energy_j", 0)
                + et.get("electronic_energy_j", 0)
            ))
            avg_w = total / max(et.get("e2e_latency_s", 1.0), 1e-12)
            num_steps = max(int(et.get("num_steps", 1)), 1)
            w.writerow(["total", "", total, avg_w,
                        f"E2E={et.get('e2e_latency_s',0)*1e3:.1f}ms, "
                        f"{total/num_steps*1e3:.3f}mJ/step"])

    def _write_layer_breakdown(self):
        # 层（算子）级明细表：每个算子的时延/能耗
        if not self._operator_costs:
            return
        path = os.path.join(self.output_dir, "layer_breakdown.csv")
        keys = ["op_id", "op_type", "op_role", "phase", "M", "K", "N",
                "resource", "latency_s", "dynamic_energy_j",
                "memory_energy_j", "programming_energy_j"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(self._operator_costs)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def print_summary(self):
        # 控制台摘要：打印关键指标（MRR/PD 数量、面积、端到端时延、
        # 分步时延、驻留命中率、分项能耗、总量与平均功率、守恒性检查）
        print("=" * 60)
        print(f"  Joint Simulation Report: {self.run_id}")
        print("=" * 60)
        if self._architecture_cost:
            a = self._architecture_cost
            print(f"  MRR: {a.get('mrr_count', 0)}  PD: {a.get('pd_count', 0)}")
            print(f"  Area: {a.get('total_area_um2', 0):.1f} um^2  "
                  f"IL: {a.get('core_insertion_loss_db', 0):.2f} dB")
        if self._scheduler_summary:
            s = self._scheduler_summary
            print(f"  E2E latency: {s.get('end_to_end_latency_s', 0):.6e} s")
            print(f"  Total events: {s.get('total_events', 0)}")
            per_step = s.get("per_step_latency_s", {})
            if per_step:
                print(f"  Per-step latency (s):")
                for step, lat in sorted(per_step.items()):
                    print(f"    step {step:2d}: {lat:.6e}")
        if self._residency_summary:
            b = self._residency_summary
            print(f"  Residency hit rate: {b.get('hit_rate', 0):.1%}  "
                  f"misses: {b.get('misses', 0)}")
        if hasattr(self, '_energy_totals') and self._energy_totals:
            et = self._energy_totals
            print(f"  --- Energy (invocation-weighted) ---")
            print(f"  Static:")
            print(f"    Laser total:      {et.get('laser_total_j', 0)*1e3:.1f} mJ")
            print(f"    MRR hold total:   {et.get('mrr_hold_total_j', 0)*1e3:.1f} mJ")
            print(f"  Dynamic:")
            print(f"    Photonic compute: {et.get('dynamic_energy_j', 0)*1e3:.2f} mJ")
            print(f"    Weight program:   {et.get('programming_energy_j', 0)*1e3:.1f} mJ")
            print(f"    Electronic:       {et.get('electronic_energy_j', 0)*1e3:.2f} mJ")
            total = et.get('total_energy_j', (
                et.get('laser_total_j', 0) + et.get('mrr_hold_total_j', 0)
                + et.get('dynamic_energy_j', 0)
                + et.get('programming_energy_j', 0)
                + et.get('electronic_energy_j', 0)
            ))
            print(f"  TOTAL:              {total*1e3:.1f} mJ  "
                  f"({total/max(et.get('e2e_latency_s',1), 1e-12):.1f} W avg)")
        print(f"  Photonic kernels: {len(self._kernel_costs)}")
        print(f"  Total operators:  {len(self._operator_costs)}")
        all_ok = all(c["passed"] for c in self._conservation_checks)
        print(f"  Conservation: {'ALL PASS' if all_ok else 'FAILURES'}")


# ---------------------------------------------------------------------------
# Sweep report — Experiment 4 capacity/parallelism scans (Section 7.6)
# ---------------------------------------------------------------------------

class SweepReport:
    """Aggregates results across a parameter sweep.

    Used for Experiment 4a (capacity sweep) and 4b (parallelism sweep).
    """
    # 中文说明：扫描实验（sweep）的结果汇总器。
    # 实验 4a 扫描 SRAM 容量（total_tile_slots），实验 4b 扫描编程并行度
    # （program_parallelism）。每个参数组合跑一次仿真，add_run() 记一条，
    # 最后 write_all() 输出一张对比表 + JSON 汇总，方便画曲线。

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._runs: List[dict] = []

    def add_run(self, config: dict, scheduler_summary: dict,
                residency_summary: dict = None):
        """Record one sweep point.

        Parameters
        ----------
        config : dict
            {strategy, total_slots, program_parallelism, pipeline_mode, ...}
        scheduler_summary : dict
            Output of ResourceScheduler.summary().
        residency_summary : dict, optional
            SRAM residency summary (SramManager.summary()). Defaults to the
            sram_manager_summary embedded in scheduler_summary.
        """
        # 把"该点的配置 + 调度结果 + 驻留统计"拼成一行记录；
        # 优先用调用方传入的驻留摘要，否则取调度摘要里内嵌的 SRAM 摘要
        sram = residency_summary or scheduler_summary.get("sram_manager_summary") or {}
        streaming = scheduler_summary.get("streaming_stats") or {}
        self._runs.append({
            **config,
            "e2e_latency_s": scheduler_summary.get("end_to_end_latency_s", 0.0),
            "total_events": scheduler_summary.get("total_events", 0),
            "hit_rate": sram.get("hit_rate", 0.0),
            "hits": sram.get("hits", 0),
            "misses": sram.get("misses", 0),
            "evictions": sram.get("evictions", 0),
            "programs": streaming.get("programs", 0),
            "programmed_weight_tiles": streaming.get("programs", 0),
            "programming_latency_s": streaming.get("programming_latency_s", 0.0),
            "programming_energy_j": streaming.get("programming_energy_j", 0.0),
            "dma_latency_s": sram.get("dma_latency_s", 0.0),
            "dma_bytes": sram.get("dma_bytes", 0),
            "prefetch_hits": 0,
            "prefetch_total": 0,
            "per_step_latency_s": scheduler_summary.get("per_step_latency_s", {}),
            "conservation_checks": scheduler_summary.get("conservation_checks", []),
        })

    def write_all(self):
        """写出扫描结果表（CSV）与汇总（JSON）。"""
        self._write_sweep_csv()
        self._write_sweep_summary_json()

    def _write_sweep_csv(self):
        """Write the sweep results table (weight_bank_policy_sweep.csv)."""
        path = os.path.join(self.output_dir, "weight_bank_policy_sweep.csv")
        keys = [
            "strategy", "total_tile_slots", "program_parallelism", "pipeline_mode",
            "e2e_latency_s", "hit_rate", "hits", "misses", "evictions",
            "programs", "programmed_tiles", "programming_latency_s",
            "programming_energy_j", "dma_latency_s", "dma_bytes",
            "prefetch_hits", "prefetch_total",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(self._runs)

    def _write_sweep_summary_json(self):
        # 与 CSV 同内容，但保持完整 dict（含 per_step_latency 等嵌套字段）
        path = os.path.join(self.output_dir, "sweep_summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._runs, f, indent=2, default=str)

    def to_dataframe(self):
        """Return runs as a list of dicts suitable for pandas."""
        # 返回记录列表，可直接喂给 pandas.DataFrame()
        return self._runs

    def print_sweep_table(self):
        """Print a compact sweep comparison table."""
        # 控制台打印一张对比表：策略/容量/并行度/流水线/命中率/E2E/编程时延
        header = f"{'Strategy':<18} {'Slots':>6} {'P_prog':>7} {'Pipeline':<22} {'HitRate':>8} {'E2E(s)':>12} {'Program(s)':>12}"
        print(header)
        print("-" * len(header))
        for r in sorted(self._runs, key=lambda x: (x.get("strategy", ""), x.get("total_tile_slots", 0))):
            print(
                f"{r.get('strategy', ''):<18} "
                f"{r.get('total_tile_slots', 0):>6} "
                f"{r.get('program_parallelism', 0):>7} "
                f"{r.get('pipeline_mode', ''):<22} "
                f"{r.get('hit_rate', 0):>7.1%} "
                f"{r.get('e2e_latency_s', 0):>12.6e} "
                f"{r.get('programming_latency_s', 0):>12.6e}"
            )
