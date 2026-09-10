"""P1 档证据管线冒烟测试（smoke test：只快速粗测主干能否跑通）。

本测试做什么：
    用最小配置依次调用 P1 证据管线的 6 个实验脚本（QKV 流量校准、能量单位
    成本校准、架构消融、可扩展性物理代理、P1 就绪汇总、实验图渲染），逐项
    检查它们的 manifest（清单）状态、CSV 表头列与关键数值是否符合约定。

数据从哪来：
    - proxy 模式：由脚本内置配置直接生成（离线代理估计，无真实硬件）；
    - trace 模式（真实硬件实测相关）：使用 fixtures/ 下的
      trace_operator_activity.csv 作为算子活动迹（trace，模拟采集到的真实执行记录）。

产出什么：
    测试全部通过 = 证据管线主干在冒烟层面可用；任何失败 = 某环节断裂。

怎么运行：
    python -m unittest experiments/tests/test_p1_pipeline_smoke.py
    注意：个别测试会临时改写仓库 tables/mobilevit_operator_activity.csv，
    测完会自动恢复或删除。
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest


# REPO_ROOT：仓库根目录（本文件位于 experiments/tests/ 下，往上 2 级即仓库根）
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
# FIXTURES：固定样例数据（fixture）目录，存放模拟真实采集的算子活动迹 CSV
FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def run_script(args: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    """以子进程方式运行一个实验脚本，并断言其退出码符合预期。

    参数：
        args: 传给脚本的命令行参数列表（不含 Python 解释器本身）。
        expected: 期望的退出码（0 表示成功），默认 0。
    返回：
        subprocess.CompletedProcess：子进程运行结果，含标准输出/标准错误。
    抛异常：
        实际退出码与 expected 不符时抛出 AssertionError，并附带完整输出便于定位。
    """
    env = dict(os.environ)
    # 默认禁止脚本写入仓库 tables/ 目录，防止冒烟测试污染正式数据
    env.setdefault("HPAT_WRITE_PROJECT_TABLES", "0")
    # 用当前 Python 解释器启动脚本，工作目录切到仓库根，同时捕获 stdout/stderr
    proc = subprocess.run([sys.executable, *args], cwd=REPO_ROOT, text=True, capture_output=True, env=env)
    if proc.returncode != expected:
        # 退出码不对 = 脚本跑挂了，把两路输出都亮出来方便排查
        raise AssertionError(
            f"command returned {proc.returncode}, expected {expected}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    """读取 UTF-8 编码的 CSV 文件，返回字典列表。

    参数：
        path: CSV 文件路径。
    返回：
        列表，每个元素是一个字典（列名 -> 单元格值），顺序与文件行一致。
    """
    # DictReader 以首行为列名；newline="" 用于正确处理字段内嵌换行的情况
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class P1PipelineSmokeTests(unittest.TestCase):
    """P1 证据管线冒烟测试集合。

    每个测试都把产出写入临时目录（测完自动清理），不触碰仓库正式产物。
    """

    def test_qkv_traffic_calibration_outputs_p1_proxy_rows(self) -> None:
        """冒烟点 1：QKV 流量校准在 config 模式下能输出 P1 代理行。

        检查 manifest 状态为 proxy（离线代理估计）、产出 CSV 非空，
        且每行都带齐论文所需的证据标签字段（evidence_label、claim_boundary 等）。
        """
        with tempfile.TemporaryDirectory() as tmp:
            # 在临时目录下新建 qkv 子目录作为输出地
            out = pathlib.Path(tmp) / "qkv"
            run_script(
                [
                    "experiments/scripts/run_qkv_traffic_calibration.py",
                    "--output-dir",
                    str(out),
                    "--trace-mode",
                    "config",
                ]
            )
            manifest = json.loads((out / "qkv_traffic_calibration_manifest.json").read_text(encoding="utf-8"))
            rows = read_csv(out / "qkv_traffic_calibrated.csv")
            self.assertEqual(manifest["status"], "proxy")
            self.assertGreater(len(rows), 0)
            required = {
                "variant",
                "n_tokens",
                "embedding_dim",
                "bit_width",
                "weight_mode",
                "activation_input_bytes",
                "qkv_projection_output_bytes",
                "attention_score_bytes",
                "value_product_bytes",
                "electronic_remainder_bytes",
                "programming_bytes",
                "bus_bytes",
                "total_bytes",
                "traceability_source",
                "evidence_label",
                "claim_boundary",
            }
            self.assertTrue(required.issubset(rows[0].keys()))
            self.assertIn("modelled P1 proxy", rows[0]["evidence_label"])
            self.assertIn("Does not support measured HPAT speedup", rows[0]["claim_boundary"])

    def test_qkv_traffic_calibration_can_use_trace_rows(self) -> None:
        """冒烟点 2：QKV 校准在 trace 模式下能消费真实的算子活动迹。

        用 fixtures 提供的 trace CSV 作为输入，验证 manifest 状态变为
        trace-backed（基于真实迹），且 MobileViT-XXS 变体的行引用了
        torch_hooks 采集到的源层数据（带 SHA256 摘要，可溯源）。
        """
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "qkv_trace"
            run_script(
                [
                    "experiments/scripts/run_qkv_traffic_calibration.py",
                    "--output-dir",
                    str(out),
                    "--operator-activity-csv",
                    str(FIXTURES / "trace_operator_activity.csv"),
                    "--trace-mode",
                    "trace",
                ]
            )
            manifest = json.loads((out / "qkv_traffic_calibration_manifest.json").read_text(encoding="utf-8"))
            rows = read_csv(out / "qkv_traffic_calibrated.csv")
            trace_rows = [row for row in rows if row["variant"] == "MobileViT-XXS" and row["trace_source"] == "torch_hooks"]
            self.assertEqual(manifest["status"], "trace-backed")
            self.assertGreater(len(trace_rows), 0)
            self.assertEqual(trace_rows[0]["activity_source_status"], "trace-backed local")
            self.assertGreater(int(trace_rows[0]["trace_row_count"]), 0)
            self.assertTrue(trace_rows[0]["source_rows_sha256"])
            self.assertIn("blocks.0.attn.qkv", trace_rows[0]["source_layer_names"])

    def test_energy_unit_cost_calibration_outputs_explicit_ranges(self) -> None:
        """冒烟点 3：能量单位成本校准必须给每个成本项显式的低/高区间。

        能量成本（如每 bit 能耗）只报单点值缺乏说服力，必须带 low/high
        区间、来源角色（literature_context 文献背景 / 内部模型假设），
        并明确声明该数据不支撑"实测 HPAT 加速比"的结论。
        """
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "unit_costs"
            run_script(["experiments/scripts/run_energy_unit_cost_calibration.py", "--output-dir", str(out)])
            manifest = json.loads((out / "energy_unit_cost_calibration_manifest.json").read_text(encoding="utf-8"))
            rows = read_csv(out / "tables" / "energy_unit_costs.csv")
            self.assertEqual(manifest["status"], "proxy")
            self.assertGreater(len(rows), 0)
            required = {
                "cost_key",
                "value",
                "low",
                "high",
                "unit",
                "source",
                "technology_assumption",
                "precision",
                "scope_note",
                "evidence_status",
                "low_multiplier",
                "high_multiplier",
                "claim_boundary",
            }
            self.assertTrue(required.issubset(rows[0].keys()))
            self.assertEqual(rows[0]["evidence_status"], "modelled_proxy")
            self.assertIn("source_id", rows[0])
            self.assertIn("numeric_basis", rows[0])
            self.assertIn("Does not support measured HPAT speedup", rows[0]["claim_boundary"])
            ledger = read_csv(out / "tables" / "energy_unit_cost_source_ledger.csv")
            roles = {row["source_role"] for row in ledger}
            self.assertIn("literature_context", roles)
            self.assertIn("internal_model_assumption", roles)

    def test_architecture_ablation_covers_required_design_axes(self) -> None:
        """冒烟点 4：架构消融扫描必须覆盖论文要论证的设计轴。

        逐一检查 baseline（基准）以及去掉广播、权重重编程、WDM 通道数、
        处理单元组数、4bit 精度、仅光学乐观模式、不同校准间隔等消融变体
        是否都出现在产出表里，并验证 baseline 行参数符合预期。
        """
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "ablation"
            run_script(["experiments/scripts/run_architecture_ablation_sweep.py", "--output-dir", str(out)])
            manifest = json.loads((out / "architecture_ablation_manifest.json").read_text(encoding="utf-8"))
            rows = read_csv(out / "tables" / "hpat_architecture_ablation.csv")
            # 收集所有消融变体名（去重），供下方逐个断言是否齐全
            ablations = {row["ablation"] for row in rows}
            self.assertEqual(manifest["status"], "proxy")
            for required in [
                "baseline",
                "no_broadcast",
                "reprogrammed_weights",
                "wdm_channels_8",
                "pdpu_banks_1",
                "precision_4bit",
                "optimistic_optical_only",
                "calibration_interval_100",
            ]:
                self.assertIn(required, ablations)
            baseline = [row for row in rows if row["variant"] == "MobileViT-XXS" and row["ablation"] == "baseline"][0]
            self.assertEqual(baseline["weight_mode"], "resident")
            self.assertEqual(baseline["broadcast_enabled"], "true")
            self.assertEqual(baseline["wavelengths"], "16")
            self.assertIn("modelled P1 proxy", baseline["evidence_label"])

    def test_scalability_physical_proxy_exposes_resource_proxy_columns(self) -> None:
        """冒烟点 5：可扩展性物理代理必须暴露全部资源代理列。

        HPAT 用 39 万 MRR 微环谐振器做光矩阵乘法，这里用 mrr_count_proxy 等
        物理量代理列（微环数、波长预算、ADC/DAC 数、内存带宽、热调谐、
        芯片面积、激光功率等）估算规模扩展后的资源占用，并明确标注只是
        物理代理、未做版图收敛（physical_proxy_not_layout_closure）。
        """
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "scale"
            run_script(["experiments/scripts/run_scalability_physical_proxy.py", "--output-dir", str(out)])
            manifest = json.loads((out / "scalability_physical_proxy_manifest.json").read_text(encoding="utf-8"))
            rows = read_csv(out / "tables" / "scalability_physical_proxy.csv")
            self.assertEqual(manifest["status"], "proxy")
            self.assertGreater(len(rows), 0)
            required = {
                "mrr_count_proxy",
                "wavelength_budget",
                "wavelength_budget_status",
                "adc_count_proxy",
                "dac_count_proxy",
                "memory_bandwidth_demand_gbps",
                "memory_bandwidth_budget_gbps",
                "thermal_tuning_proxy",
                "mrr_area_proxy_mm2",
                "converter_area_proxy_mm2",
                "path_loss_db_proxy",
                "laser_power_proxy_mw",
                "thermal_density_proxy",
                "resource_limit_status",
                "latency_estimate_ns",
                "energy_estimate_mj",
                "proxy_status",
                "evidence_label",
                "claim_boundary",
            }
            self.assertTrue(required.issubset(rows[0].keys()))
            self.assertEqual(rows[0]["proxy_status"], "physical_proxy_not_layout_closure")
            self.assertIn("not physical layout closure", rows[0]["evidence_label"])

    def test_p1_readiness_summary_schema(self) -> None:
        """冒烟点 6：P1 就绪汇总 JSON 的 schema（数据结构约定）必须稳定。

        先把前面 4 个实验脚本的产出都生成到同一个临时目录，再跑就绪汇总，
        校验 schema 版本号、整体结论"可用但有局限"、以及 P1-E5/E6/E8 三个
        关键实验各自的证据等级与版图就绪度标记是否符合预期。
        """
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "readiness"
            run_script(
                [
                    "experiments/scripts/run_qkv_traffic_calibration.py",
                    "--output-dir",
                    str(out),
                    "--operator-activity-csv",
                    str(FIXTURES / "trace_operator_activity.csv"),
                    "--trace-mode",
                    "trace",
                ]
            )
            run_script(["experiments/scripts/run_energy_unit_cost_calibration.py", "--output-dir", str(out)])
            run_script(["experiments/scripts/run_architecture_ablation_sweep.py", "--output-dir", str(out)])
            run_script(["experiments/scripts/run_scalability_physical_proxy.py", "--output-dir", str(out)])
            run_script(["experiments/scripts/run_p1_readiness_summary.py", "--output-dir", str(out)])
            summary = json.loads((out / "tables" / "p1_readiness_summary.json").read_text(encoding="utf-8"))
            statuses = {row["experiment"]: row for row in summary["experiments"]}
            self.assertEqual(summary["schema_version"], "p1-readiness-v1")
            self.assertTrue(summary["overall_ready_with_limitations"])
            self.assertEqual(statuses["P1-E5"]["evidence_tier"], "trace-backed local")
            self.assertEqual(statuses["P1-E6"]["evidence_tier"], "literature-context/modelled")
            self.assertEqual(statuses["P1-E8"]["layout_readiness"], "proxy only; no layout closure")

    def test_proxy_activity_trace_does_not_clobber_project_trace_table(self) -> None:
        """冒烟点 7：跑代理 trace 不允许覆盖仓库里正式的算子活动迹表。

        用一份 fixture 迹先填进仓库正式表，再让 trace 脚本去读一个明显
        错误的配置（不存在的 timm 模型名），验证失败场景下正式表内容
        仍然保留、不被清空或改写；无论成败，finally 都会还原正式表。
        """
        project_csv = REPO_ROOT / "tables" / "mobilevit_operator_activity.csv"
        # 备份正式表原文以便测试后恢复；原本不存在则记 None，之后删除
        original = project_csv.read_text(encoding="utf-8") if project_csv.exists() else None
        try:
            project_csv.write_text((FIXTURES / "trace_operator_activity.csv").read_text(encoding="utf-8"), encoding="utf-8")
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = pathlib.Path(tmp)
                config = tmp_path / "bad_config.json"
                config.write_text(
                    json.dumps(
                        {
                            "mobilevit_variants": [
                                {
                                    "name": "MobileViT-XXS",
                                    "timm_model": "definitely_not_a_real_timm_model",
                                    "d": 128,
                                    "input_resolution": 192,
                                    "token_count_sweep": [196],
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                run_script(
                    [
                        "experiments/scripts/run_mobilevit_activity_trace.py",
                        "--output-dir",
                        str(tmp_path / "trace"),
                        "--config",
                        str(config),
                    ]
                )
                after = project_csv.read_text(encoding="utf-8")
                self.assertIn("torch_hooks", after)
                self.assertIn("blocks.0.attn.qkv", after)
        finally:
            if original is None:
                project_csv.unlink(missing_ok=True)
            else:
                project_csv.write_text(original, encoding="utf-8")

    def test_qkv_auto_prefers_project_trace_over_run_local_proxy(self) -> None:
        """冒烟点 8：QKV 校准默认模式优先用仓库正式 trace，而非本地代理数据。

        在仓库正式表放一份 torch_hooks 采集的迹，同时在本轮输出目录里放
        一份"降级"为 config_proxy 的同源数据，验证脚本最终选择正式表，
        manifest 状态为 trace-backed 且包含 torch_hooks 来源。
        """
        project_csv = REPO_ROOT / "tables" / "mobilevit_operator_activity.csv"
        original = project_csv.read_text(encoding="utf-8") if project_csv.exists() else None
        try:
            fixture_text = (FIXTURES / "trace_operator_activity.csv").read_text(encoding="utf-8")
            project_csv.write_text(fixture_text, encoding="utf-8")
            with tempfile.TemporaryDirectory() as tmp:
                out = pathlib.Path(tmp) / "qkv_auto"
                run_tables = out / "tables"
                run_tables.mkdir(parents=True)
                (run_tables / "mobilevit_operator_activity.csv").write_text(
                    fixture_text.replace(",torch_hooks,", ",config_proxy,"),
                    encoding="utf-8",
                )
                run_script(["experiments/scripts/run_qkv_traffic_calibration.py", "--output-dir", str(out)])
                manifest = json.loads((out / "qkv_traffic_calibration_manifest.json").read_text(encoding="utf-8"))
                rows = read_csv(out / "qkv_traffic_calibrated.csv")
                self.assertEqual(manifest["status"], "trace-backed")
                self.assertEqual(manifest["operator_activity_csv"], "tables/mobilevit_operator_activity.csv")
                self.assertIn("torch_hooks", {row["trace_source"] for row in rows})
        finally:
            if original is None:
                project_csv.unlink(missing_ok=True)
            else:
                project_csv.write_text(original, encoding="utf-8")

    # 机器上没装 PIL（图像处理库）时跳过此用例，因为它要实际渲染 PNG 图
    @unittest.skipUnless(importlib.util.find_spec("PIL") is not None, "PIL is required for experiment figure sidecar smoke")
    def test_p1_figure_sidecars_include_readiness_sources(self) -> None:
        """冒烟点 9：实验图的元数据（sidecar，伴生 JSON）要能溯源到就绪汇总。

        渲染架构消融图后，检查其 .meta.json 里记录的来源文件列表包含
        p1_readiness_summary.json，且图注明确写着"非实测硬件证据"，
        防止图表被误当成真实测量结果。
        """
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "figures"
            run_script(["experiments/scripts/run_qkv_traffic_calibration.py", "--output-dir", str(out)])
            run_script(["experiments/scripts/run_energy_unit_cost_calibration.py", "--output-dir", str(out)])
            run_script(["experiments/scripts/run_architecture_ablation_sweep.py", "--output-dir", str(out)])
            run_script(["experiments/scripts/run_scalability_physical_proxy.py", "--output-dir", str(out)])
            run_script(["experiments/scripts/run_p1_readiness_summary.py", "--output-dir", str(out)])
            run_script(["experiments/scripts/render_experiment_figures.py", "--output-dir", str(out)])
            sidecar = json.loads(
                (out / "figures" / "fig_architecture_ablation.png.meta.json").read_text(encoding="utf-8")
            )
            source_paths = [source["path"] for source in sidecar["sources"]]
            self.assertIn("tables/p1_readiness_summary.json", source_paths)
            self.assertIn("not measured hardware evidence", sidecar["caption"])


if __name__ == "__main__":
    # 直接执行本文件时启动 unittest 测试运行器
    unittest.main()
