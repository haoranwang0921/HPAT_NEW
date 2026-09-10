# 模块 docstring：
# 本测试文件属于 hpat-artifact（《HPAT：光子张量处理器》论文可复现实验库）的 P2 证据等级管线。
# 它是一组"冒烟测试"（smoke test，只快速验证各脚本能跑通、能产出符合 schema 的结果，不追求完整精度），
# 覆盖 5 个 P2 证据脚本：版图面积可行性代理、热调谐压力、公开边缘上下文刷新、
# 附加模型家族检查、P2 就绪度汇总，以及实验配图 sidecar（旁挂元数据文件）的生成。
# 说明：这些脚本的产物都带有"P2 proxy"或明确 claim_boundary（证据边界），
#       即只做代理/估算、不做流片级（fabricated-silicon）验证，测试同时守住了这条边界不越界。
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


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def run_script(args: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    # 以子进程方式运行实验脚本：设好环境变量、捕获输出，若退出码不符则抛出带完整日志的异常。
    env = dict(os.environ)
    env.setdefault("HPAT_WRITE_PROJECT_TABLES", "0")  # 冒烟测试禁止写项目正式表格，只能写临时输出目录
    proc = subprocess.run([sys.executable, *args], cwd=REPO_ROOT, text=True, capture_output=True, env=env)
    if proc.returncode != expected:
        raise AssertionError(
            f"command returned {proc.returncode}, expected {expected}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    # 读 CSV 表格并转成"行字典"列表（第一行作为列名），方便测试里按列名取值断言。
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def minimal_p2_config(path: pathlib.Path) -> pathlib.Path:
    # 基于仓库自带配置生成一份"最小化 P2 配置"：只保留一个附加模型家族（EfficientFormer-L1），
    # 让附加模型家族检查脚本能真正跑出追踪（trace）结果，而不是空跑。生成后的配置写入 path 并返回。
    config = json.loads((REPO_ROOT / "experiments" / "config" / "hpat_experiment_config.json").read_text(encoding="utf-8"))
    config["p2"]["additional_model_families"] = [
        {
            "family": "EfficientFormer",  # 家族名
            "name": "EfficientFormer-L1",  # 具体模型名
            "timm_model": "efficientformer_l1",  # timm 库里的模型标识
            "parameter_count_m": 12.3,  # 参数量（百万），用于规模估算
            "d": 448,  # 隐藏维度
            "input_resolution": 224,  # 输入分辨率
            "token_count_sweep": [196],  # 只扫一个 token 数，缩小冒烟范围
        }
    ]
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


class P2PipelineSmokeTests(unittest.TestCase):
    # 测试类：P2 证据等级管线的冒烟测试（快速验证每个脚本能运行并产出结构正确的产物）。
    def test_layout_area_proxy_outputs_p2_boundary_rows(self) -> None:
        # 测什么：版图面积可行性代理脚本能跑通，CSV 里有数据，且每行带齐 P2 边界相关列。
        # 怎么测：在临时目录运行脚本，读取 manifest 与 CSV，检查状态、行数、必需列及边界声明。
        # 预期结果：状态为 "proxy"；CSV 非空；首行含 9 个必需列；evidence_label 含 "P2 proxy"；
        #          claim_boundary 声明"不支持流片验证"。
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "layout"
            run_script(["experiments/scripts/run_layout_area_feasibility_proxy.py", "--output-dir", str(out)])
            manifest = json.loads((out / "layout_area_feasibility_proxy_manifest.json").read_text(encoding="utf-8"))
            rows = read_csv(out / "tables" / "layout_area_feasibility_proxy.csv")
            self.assertEqual(manifest["status"], "proxy")  # 脚本状态必须是"代理估算"，不能宣称实测
            self.assertGreater(len(rows), 0)
            required = {
                "mrr_count_proxy",  # MRR 数量代理值
                "mrr_area_proxy_mm2",  # MRR 面积代理值（平方毫米）
                "converter_area_proxy_mm2",  # 转换器面积代理值
                "interconnect_area_proxy_mm2",  # 互连面积代理值
                "total_area_proxy_mm2",  # 总面积代理值
                "area_budget_status",  # 面积预算状态
                "proxy_status",  # 代理状态
                "evidence_label",  # 证据标签
                "claim_boundary",  # 证据边界声明
            }
            self.assertTrue(required.issubset(rows[0].keys()))  # 首行必须覆盖全部必需列
            self.assertIn("P2 proxy", rows[0]["evidence_label"])  # 证据标签要标注"P2 代理"
            self.assertIn("Does not support fabricated-silicon", rows[0]["claim_boundary"])  # 边界：不声称支持流片验证

    def test_thermal_tuning_stress_outputs_proxy_rows(self) -> None:
        # 测什么：热调谐压力（热漂移下反复重调谐）脚本能跑通，并产出带明确代理状态的 CSV 行。
        # 怎么测：在临时目录运行脚本，读取 manifest 与 CSV，检查状态、行数与代理状态声明。
        # 预期结果：状态为 "proxy"；CSV 非空；首行含 8 个必需列；
        #          proxy_status 明确写着"未做封装设备实测验证"。
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "thermal"
            run_script(["experiments/scripts/run_thermal_tuning_stress.py", "--output-dir", str(out)])
            manifest = json.loads((out / "thermal_tuning_stress_manifest.json").read_text(encoding="utf-8"))
            rows = read_csv(out / "tables" / "thermal_tuning_stress.csv")
            self.assertEqual(manifest["status"], "proxy")
            self.assertGreater(len(rows), 0)
            required = {
                "thermal_drift_c",  # 温度漂移量（摄氏度）
                "thermal_calibration_multiplier",  # 热校准乘子
                "retune_interval_inferences",  # 两次重调谐之间的推理次数
                "latency_overhead_ns_proxy",  # 时延开销代理值（纳秒）
                "energy_overhead_mj_proxy",  # 能量开销代理值（毫焦）
                "stress_status",  # 压力测试状态
                "proxy_status",  # 代理状态
                "claim_boundary",  # 证据边界
            }
            self.assertTrue(required.issubset(rows[0].keys()))
            self.assertEqual(rows[0]["proxy_status"], "thermal_tuning_proxy_not_packaged_device_validation")  # 明确：只是代理，未做封装器件实测

    def test_public_edge_context_refresh_is_context_only(self) -> None:
        # 测什么：公开边缘上下文刷新脚本只是收集"外部公开上下文"（外部数据的边界信息），
        #         且不会用它来算 HPAT 加速比——防止外部数据被误当成实测依据。
        # 怎么测：运行脚本后检查 manifest、来源表数量、是否含 mlperf_mobile 来源，以及边界文本。
        # 预期结果：状态为 "context"；来源至少 6 个且含 mlperf_mobile；
        #          markdown 边界文本必须包含"不得用于计算 HPAT 加速比"的声明。
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "public"
            run_script(["experiments/scripts/run_public_edge_context_refresh.py", "--output-dir", str(out)])
            manifest = json.loads((out / "public_edge_context_refresh_manifest.json").read_text(encoding="utf-8"))
            source_rows = read_csv(out / "tables" / "public_edge_context_sources.csv")
            text = (out / "tables" / "public_edge_context.md").read_text(encoding="utf-8")
            self.assertEqual(manifest["status"], "context")  # 只产出上下文，不是实测证据
            self.assertGreaterEqual(len(source_rows), 6)  # 至少收集到 6 个公开来源
            self.assertIn("mlperf_mobile", {row["source_id"] for row in source_rows})  # 必须含 MLPerf 移动端来源
            self.assertIn("must not be used to compute HPAT speedup", text)  # 边界声明：不得用于算加速比

    def test_additional_model_family_check_traces_or_blocks_visibly(self) -> None:
        # 测什么：附加模型家族检查脚本对给定模型（EfficientFormer-L1）要么成功追踪出算子并写明边界，
        #         要么被阻塞且给出可见的 blocked_reason，不能静默失败。
        # 怎么测：用最小配置运行脚本，读取 manifest 与算子汇总 CSV，再按状态分支断言。
        # 预期结果：只产出 1 行；status 在 {ok, partial, blocked} 之一；
        #          ok 时 row_count>0、trace_source 为 torch_hooks、证据标签含 "P2 proxy"；
        #          非 ok 时必须能看到 blocked_reason。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            config = minimal_p2_config(tmp_path / "config.json")
            out = tmp_path / "family"
            run_script(
                [
                    "experiments/scripts/run_additional_model_family_check.py",
                    "--output-dir",
                    str(out),
                    "--config",
                    str(config),
                ]
            )
            manifest = json.loads((out / "additional_model_family_check_manifest.json").read_text(encoding="utf-8"))
            rows = read_csv(out / "tables" / "additional_model_family_operator_summary.csv")
            self.assertEqual(len(rows), 1)  # 最小配置只扫一个模型，汇总应恰好一行
            self.assertIn(manifest["status"], {"ok", "partial", "blocked"})  # 三种结果都算"可见反馈"
            if manifest["status"] == "ok":
                self.assertGreater(int(rows[0]["row_count"]), 0)  # 追踪成功时算子行数必须大于 0
                self.assertEqual(rows[0]["trace_source"], "torch_hooks")  # 追踪来源必须是 PyTorch 钩子
                self.assertIn("P2 proxy", rows[0]["evidence_label"])  # 同样标注 P2 代理证据
            else:
                self.assertTrue(manifest.get("blocked_reason") or rows[0]["blocked_reason"])  # 非 ok 必须给出阻塞原因

    def test_p2_readiness_summary_schema(self) -> None:
        # 测什么：P2 就绪度汇总脚本能聚合上面 4 个脚本的产物，输出符合固定 schema 的汇总 JSON。
        # 怎么测：依次运行 4 个上游脚本（同一输出目录），再运行汇总脚本，解析汇总 JSON 并断言。
        # 预期结果：schema_version 为 "p2-readiness-v1"；四个实验（版图、热调谐、公开上下文、附加家族）
        #          都出现在 experiments 里；claim_boundary 提到"物理设计收敛"这一上限。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            config = minimal_p2_config(tmp_path / "config.json")
            out = tmp_path / "run"
            run_script(["experiments/scripts/run_layout_area_feasibility_proxy.py", "--output-dir", str(out), "--config", str(config)])
            run_script(["experiments/scripts/run_thermal_tuning_stress.py", "--output-dir", str(out), "--config", str(config)])
            run_script(["experiments/scripts/run_public_edge_context_refresh.py", "--output-dir", str(out), "--config", str(config)])
            run_script(["experiments/scripts/run_additional_model_family_check.py", "--output-dir", str(out), "--config", str(config)])
            run_script(["experiments/scripts/run_p2_readiness_summary.py", "--output-dir", str(out)])
            summary = json.loads((out / "tables" / "p2_readiness_summary.json").read_text(encoding="utf-8"))
            statuses = {row["experiment"]: row for row in summary["experiments"]}
            self.assertEqual(summary["schema_version"], "p2-readiness-v1")  # 汇总 JSON 必须符合 v1 schema
            self.assertIn("P2-layout-area", statuses)  # 四个实验结果必须齐全
            self.assertIn("P2-thermal-tuning", statuses)
            self.assertIn("P2-public-edge-context", statuses)
            self.assertIn("P2-additional-model-family", statuses)
            self.assertIn("physical-design closure", summary["claim_boundary"])  # 证据边界上限为"物理设计收敛"

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "PIL is required for experiment figure sidecar smoke")
    def test_p2_figure_sidecars_include_readiness_sources(self) -> None:
        # 测什么：实验配图渲染脚本生成的 PNG 必须带 sidecar 元数据（.meta.json），
        #         且元数据里要写明它引用了哪些数据源，防止配图脱离数据来源。
        # 怎么测：先跑完 5 个上游脚本，再渲染配图，读取版图图的 sidecar，检查来源路径与图注。
        # 预期结果：sidecar 的 sources 里含 "tables/p2_readiness_summary.json"；
        #          图注声明"不是物理设计收敛"，避免配图被误读为流片级结论。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            config = minimal_p2_config(tmp_path / "config.json")
            out = tmp_path / "figures"
            run_script(["experiments/scripts/run_layout_area_feasibility_proxy.py", "--output-dir", str(out), "--config", str(config)])
            run_script(["experiments/scripts/run_thermal_tuning_stress.py", "--output-dir", str(out), "--config", str(config)])
            run_script(["experiments/scripts/run_public_edge_context_refresh.py", "--output-dir", str(out), "--config", str(config)])
            run_script(["experiments/scripts/run_additional_model_family_check.py", "--output-dir", str(out), "--config", str(config)])
            run_script(["experiments/scripts/run_p2_readiness_summary.py", "--output-dir", str(out)])
            run_script(["experiments/scripts/render_experiment_figures.py", "--output-dir", str(out)])
            sidecar = json.loads(
                (out / "figures" / "fig_layout_area_feasibility_proxy.png.meta.json").read_text(
                    encoding="utf-8"
                )
            )
            source_paths = {source["path"] for source in sidecar["sources"]}  # 收集配图引用的所有数据源路径
            self.assertIn("tables/p2_readiness_summary.json", source_paths)  # 版图配图必须引用就绪度汇总表
            self.assertIn("not physical-design closure", sidecar["caption"])  # 图注需澄清：非物理设计收敛结论


if __name__ == "__main__":
    unittest.main()
