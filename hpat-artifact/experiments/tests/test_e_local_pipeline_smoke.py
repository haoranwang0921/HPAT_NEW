# 模块 docstring：
# 本测试文件属于 hpat-artifact（《HPAT：光子张量处理器》论文可复现实验库）的 e-local 管线。
# e-local 指"作者本机可复现的证据等级"（evidence-local）：不依赖外部流片设备，用本地脚本直接产出证据。
# 本文件是冒烟测试（只验证管线能跑通、产物结构正确，不追求数值精度），覆盖：
#   - 就绪度汇总的"写保护"（冒烟测试不能覆盖项目正式表格）；
#   - 基于轨迹（trace）的能耗计算（自底向上、不做归一化）；
#   - 能耗盈亏平衡与外部边缘上下文（严格限定证据边界）；
#   - 固定子集（imagenette 小样本）缺失数据时的阻塞与就绪度汇总；
#   - 固定子集解析器（从本地压缩包解析、不联网）；
#   - 基于 topk 裕度（置信度差值）fixture 的鲁棒性统计。
from __future__ import annotations

import csv
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"  # 测试固定数据目录（预置的假数据文件）
EXPERIMENT_SRC = REPO_ROOT / "experiments" / "src"
if str(EXPERIMENT_SRC) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_SRC))  # 加入实验源码目录，便于 import 实验模块
EXPERIMENT_SCRIPTS = REPO_ROOT / "experiments" / "scripts"
if str(EXPERIMENT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_SCRIPTS))  # 加入脚本目录，便于动态导入脚本


def run_script(args: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    # 以子进程运行实验脚本：捕获输出，退出码不符则抛异常；与 test_p2_pipeline_smoke.py 中同名工具保持一致。
    env = dict(os.environ)
    env.setdefault("HPAT_WRITE_PROJECT_TABLES", "0")  # 冒烟测试禁止写项目正式表格，只写临时目录
    proc = subprocess.run([sys.executable, *args], cwd=REPO_ROOT, text=True, capture_output=True, env=env)
    if proc.returncode != expected:
        raise AssertionError(
            f"command returned {proc.returncode}, expected {expected}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


class ELocalPipelineSmokeTests(unittest.TestCase):
    # 测试类：e-local（作者本机可复现证据）管线的冒烟测试。
    def test_project_table_write_guard_prevents_smoke_overwrite(self) -> None:
        # 测什么：运行就绪度汇总脚本时，只能写临时输出目录，绝不能覆盖仓库里的项目正式表格
        #         tables/e_local_readiness_summary.json（写保护机制）。
        # 怎么测：先记录正式表格的前后内容，在临时目录里运行脚本，再对比前后内容。
        # 预期结果：临时目录里生成了表格文件；仓库正式表格在冒烟测试前后内容完全一致。
        project_json = REPO_ROOT / "tables" / "e_local_readiness_summary.json"
        before = project_json.read_text(encoding="utf-8") if project_json.exists() else None  # 记录运行前内容
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "readiness"
            run_script(["experiments/scripts/run_e_local_readiness_summary.py", "--output-dir", str(out)])
            self.assertTrue((out / "tables" / "e_local_readiness_summary.json").exists())  # 产物写在临时目录里
        after = project_json.read_text(encoding="utf-8") if project_json.exists() else None  # 记录运行后内容
        self.assertEqual(before, after)  # 正式表格前后一致 => 写保护生效，冒烟测试未污染仓库

    def test_trace_driven_activity_uses_bottom_up_energy_without_normalization(self) -> None:
        # 测什么：基于轨迹的能耗计算应使用"自底向上"（按算子逐项累计）的算法，且不做归一化
        #         （normalize，即不把总能耗强行对齐到某个能量包络）。
        # 怎么测：用 fixtures 里的算子活动 CSV 与单位成本 JSON 运行脚本，读取 manifest 与能耗 CSV。
        # 预期结果：状态为 "ready_with_limitations"；未开启归一化；来源是轨迹驱动；
        #          每行 normalization_factor 都为 1.0、normalization_target_mj 为空字符串。
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "trace"
            run_script(
                [
                    "experiments/scripts/run_e_local_trace_driven_activity.py",
                    "--output-dir",
                    str(out),
                    "--operator-activity-csv",
                    str(FIXTURES / "trace_operator_activity.csv"),
                    "--unit-costs",
                    str(FIXTURES / "unit_costs_complete.json"),
                ]
            )
            manifest = json.loads((out / "e_local_trace_driven_activity_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "ready_with_limitations")  # 可出结果但带限制说明
            self.assertFalse(manifest["normalize_to_energy_envelope"])  # 未做能量包络归一化
            self.assertEqual(manifest["activity_source"], "trace_driven_operator_activity")  # 活动来源确认为轨迹驱动
            with (out / "tables" / "hpat_energy_by_component_trace_driven.csv").open(
                "r", encoding="utf-8", newline=""
            ) as f:
                rows = list(csv.DictReader(f))
            self.assertGreater(len(rows), 0)
            self.assertTrue(all(float(row["normalization_factor"]) == 1.0 for row in rows))  # 归一化因子恒为 1 => 未缩放
            self.assertTrue(all(row["normalization_target_mj"] == "" for row in rows))  # 归一化目标为空 => 未设置

    def test_energy_break_even_and_external_context_are_context_bounded(self) -> None:
        # 测什么：能耗盈亏平衡（HPAT 需要跑多少次推理才能抵消启动开销）与外部边缘上下文，
        #         都严格限定证据边界——只能引用外部公开数据，不得声称测得 HPAT 真实能耗/加速比。
        # 怎么测：先用轨迹能耗产出作为输入跑盈亏平衡脚本，再跑外部上下文刷新脚本，分别断言。
        # 预期结果：盈亏平衡状态为 "ready_with_limitations" 且证据标签说明"非实测能耗"；
        #          外部上下文状态为 "context_ready"、边界文本禁止用于算加速比、
        #          作者实测通道被阻塞（无作者设备日志）、外部公开通道标记不可用于加速比。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            trace_out = tmp_path / "trace"
            run_script(
                [
                    "experiments/scripts/run_e_local_trace_driven_activity.py",
                    "--output-dir",
                    str(trace_out),
                    "--operator-activity-csv",
                    str(FIXTURES / "trace_operator_activity.csv"),
                    "--unit-costs",
                    str(FIXTURES / "unit_costs_complete.json"),
                ]
            )
            break_out = tmp_path / "break_even"
            run_script(
                [
                    "experiments/scripts/run_energy_break_even.py",
                    "--output-dir",
                    str(break_out),
                    "--component-csv",
                    str(trace_out / "tables" / "hpat_energy_by_component_trace_driven.csv"),
                ]
            )
            break_manifest = json.loads((break_out / "energy_break_even_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(break_manifest["status"], "ready_with_limitations")
            with (break_out / "tables" / "energy_break_even_sweep.csv").open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertGreater(len(rows), 0)
            self.assertIn("not measured HPAT deployment energy", rows[0]["evidence_label"])  # 明确：不是实测部署能耗

            context_out = tmp_path / "context"
            run_script(["experiments/scripts/refresh_external_edge_context.py", "--output-dir", str(context_out)])
            context_manifest = json.loads((context_out / "external_edge_context_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(context_manifest["status"], "context_ready")  # 外部上下文已就绪
            text = (context_out / "tables" / "external_edge_context_claim_boundary.md").read_text(encoding="utf-8")
            self.assertIn("must not be used to compute HPAT speedup", text)  # 边界声明：不得用于计算加速比
            with (context_out / "tables" / "edge_mobile_baseline_status.csv").open(
                "r", encoding="utf-8", newline=""
            ) as f:
                status_rows = list(csv.DictReader(f))
            statuses = {row["lane"]: row for row in status_rows}
            self.assertEqual(
                statuses["author_measured_edge_mobile_baseline"]["status"],
                "blocked_no_author_device_logs",  # 作者实测通道：因无作者设备日志而阻塞
            )
            self.assertEqual(
                statuses["external_public_edge_mobile_context"]["external_data_label"].split(";")[0],
                "external_public_context",  # 外部通道数据来源标签：外部公开上下文
            )
            self.assertEqual(statuses["external_public_edge_mobile_context"]["claim_eligible_for_speedup"], "false")  # 外部数据不可用于算加速比
            self.assertTrue((context_out / "tables" / "external_edge_context_verified.csv").exists())  # 已验证来源表也须产出

    def test_fixed_subset_missing_data_is_blocked_but_nonfatal_and_readiness_summarizes(self) -> None:
        # 测什么：固定子集（imagenette 小样本）脚本在缺数据时应"被阻塞"但进程不崩（非致命），
        #         且就绪度汇总里要能看到这条通道的状态。
        # 怎么测：直接运行固定子集脚本（仓库没有对应实测数据），再运行就绪度汇总脚本。
        # 预期结果：固定子集 manifest 状态为 "blocked" 但仍产出状态 CSV；就绪度 schema 为
        #          "e-local-readiness-v1" 且 lanes 里包含 fixed_subset_robustness 通道。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            fixed_out = tmp_path / "fixed"
            run_script(["experiments/scripts/run_e_local_fixed_subset.py", "--output-dir", str(fixed_out)])
            fixed_manifest = json.loads((fixed_out / "e_local_fixed_subset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(fixed_manifest["status"], "blocked")  # 缺数据 => 显式标记阻塞
            self.assertTrue((fixed_out / "tables" / "nonideality_accuracy_fixed_subset_status.csv").exists())  # 但仍产出状态表，便于追踪

            readiness_out = tmp_path / "readiness"
            run_script(["experiments/scripts/run_e_local_readiness_summary.py", "--output-dir", str(readiness_out)])
            readiness = json.loads((readiness_out / "tables" / "e_local_readiness_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(readiness["schema_version"], "e-local-readiness-v1")  # 就绪度汇总须符合 v1 schema
            statuses = {row["lane"]: row["status"] for row in readiness["lanes"]}
            self.assertIn("fixed_subset_robustness", statuses)  # 固定子集鲁棒性通道必须出现在汇总里

    def test_imagenette_fixed_subset_parser_uses_archive_and_label_ledger_without_network(self) -> None:
        # 测什么：固定子集解析器应能完全离线工作——从本地压缩包解析 imagenette 验证集，
        #         用标签账本（label ledger）映射 synset 类别号，并给每个样本记录 SHA-256 与来源边界。
        # 怎么测：先用 PIL 造 2 张假图片并打包成 tar.gz，通过 --archive-url 指向本地 file:// URL，
        #         运行解析脚本后检查 manifest、fixed_subset.csv 与 source_ledger.csv。
        # 预期结果：状态 ok、来源为 imagenette160-valid、共 2 张图；
        #          synset->label 映射正确（n01440764->0，n02102040->217）；每行都有 archive_sha256，
        #          且 external_data_label 注明"非完整 ImageNet 验证集"；账本 2 行且都有 sha256。
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - dependency diagnostic
            self.skipTest(f"PIL unavailable: {exc}")  # 未装 PIL 就跳过，不把环境问题当成功能失败
        synsets = [
            "n01440764",  # tench（丁鱥）
            "n02102040",  # English springer（英国跳猎犬）
            "n02979186",
            "n03000684",
            "n03028079",
            "n03394916",
            "n03417042",
            "n03425413",
            "n03445777",
            "n03888257",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            source_root = tmp_path / "src" / "imagenette2-160" / "val"
            for synset in synsets:
                (source_root / synset).mkdir(parents=True, exist_ok=True)  # 为每个类别建目录
            Image.new("RGB", (12, 12), (10, 20, 30)).save(source_root / "n01440764" / "tench_fixture.JPEG")  # 造一张假 tench 图
            Image.new("RGB", (12, 12), (40, 50, 60)).save(source_root / "n02102040" / "springer_fixture.JPEG")  # 造一张假 springer 图
            archive = tmp_path / "tiny_imagenette2-160.tgz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(tmp_path / "src" / "imagenette2-160", arcname="imagenette2-160")  # 打成 tar.gz 模拟发布包
            out = tmp_path / "prepare"
            dataset = tmp_path / "dataset"
            run_script(
                [
                    "experiments/scripts/prepare_public_fixed_subset.py",
                    "--output-dir",
                    str(out),
                    "--dataset-dir",
                    str(dataset),
                    "--source",
                    "imagenette160-valid",
                    "--archive-url",
                    archive.resolve().as_uri(),  # 用 file:// URL 指向本地包，全程无网络
                ]
            )
            manifest = json.loads((out / "public_fixed_subset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "ok")
            self.assertEqual(manifest["source"], "imagenette160-valid")
            self.assertEqual(manifest["image_count"], 2)  # 恰好解析出 2 张假图
            with (dataset / "fixed_subset.csv").open("r", encoding="utf-8", newline="") as f:
                subset_rows = list(csv.DictReader(f))
            labels = {row["synset"]: int(row["label"]) for row in subset_rows}
            self.assertEqual(labels["n01440764"], 0)  # tench 的类别号是 0
            self.assertEqual(labels["n02102040"], 217)  # English springer 的类别号是 217
            self.assertTrue(all(row["archive_sha256"] for row in subset_rows))  # 每个样本都要记录来源包校验值
            self.assertTrue(all("not full ImageNet validation" in row["external_data_label"] for row in subset_rows))  # 注明不是完整 ImageNet
            with (dataset / "fixed_subset_source_ledger.csv").open("r", encoding="utf-8", newline="") as f:
                ledger_rows = list(csv.DictReader(f))
            self.assertEqual(len(ledger_rows), 2)  # 账本也要 2 条来源记录
            self.assertTrue(all(row["sha256"] for row in ledger_rows))  # 每条来源都要有文件哈希

    def test_robustness_statistics_from_topk_margin_fixture(self) -> None:
        # 测什么：鲁棒性统计脚本能从"topk 裕度"（正确类别与 top1 置信度差值）原始 CSV 出发，
        #         输出 5 张统计表，且最差类别识别正确。
        # 怎么测：手工构造 2 条 fixture 记录（一条扰动后仍正确、一条扰动后变错，margin 变负），
        #         以运行目录的方式交给统计脚本（含 bootstrap 重采样参数），检查输出表。
        # 预期结果：5 个表格文件都存在；最差类别表首行是 "English springer"（margin 掉得最狠的那个）。
        from run_nonideality_accuracy_sweep import PREDICTION_TOPK_MARGIN_FIELDS

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            run_dir = tmp_path / "run_fixture"
            raw = run_dir / "raw"
            raw.mkdir(parents=True)
            rows = []
            # 手工构造两条"预测 topk 裕度"记录：
            #   - 样本 0：扰动后仍预测正确（perturbed_correct=1），margin 只掉 0.01；
            #   - 样本 1：扰动后预测错误（perturbed_correct=0），margin 掉 0.08，最差。
            for sample_index, label, class_name, perturbed_correct, changed, margin_delta in [
                (0, 0, "tench", 1, 0, -0.01),
                (1, 217, "English springer", 0, 1, -0.08),
            ]:
                rows.append(
                    {
                        "variant": "MobileViT-XXS",
                        "trial_index": 0,
                        "trial_seed": 20260706,
                        "sample_index": sample_index,
                        "image_path": f"val/example_{sample_index}.JPEG",
                        "label": label,
                        "class_name": class_name,
                        "synset": "n01440764" if label == 0 else "n02102040",
                        "source_id": f"sample_{sample_index}",
                        "effect": "gaussian_pd_tia_noise",  # 注入效应：高斯 PD/TIA（光电二极管/跨阻放大器）噪声
                        "sweep_variable": "noise_lsb",
                        "sweep_value": "0.5",
                        "clean_top1": label,  # 干净输入的 top1 就是真标签
                        "perturbed_top1": label if perturbed_correct else 999,  # 扰动后错误时 top1 记为 999
                        "changed": changed,
                        "clean_top5": f"{label};1;2;3;4",
                        "perturbed_top5": f"{label};1;2;3;4" if perturbed_correct else "999;1;2;3;4",
                        "clean_label_in_top5": 1,
                        "perturbed_label_in_top5": perturbed_correct,
                        "clean_correct_top1": 1,
                        "perturbed_correct_top1": perturbed_correct,
                        "clean_correct_top5": 1,
                        "perturbed_correct_top5": perturbed_correct,
                        "clean_top1_confidence": "0.70000000",
                        "perturbed_top1_confidence": "0.60000000",
                        "clean_top1_margin": "0.20000000",
                        "perturbed_top1_margin": f"{0.2 + margin_delta:.8f}",  # 扰动后裕度 = 0.2 + 增量
                        "margin_delta": f"{margin_delta:.8f}",
                        "top5_jaccard": "1.00000000" if perturbed_correct else "0.66666667",
                        "injection_boundary": "hpat-mapping",  # 注入边界：HPAT 映射层
                        "evidence_label": "fixture fixed subset",
                        "claim_boundary": "fixture boundary",
                    }
                )
            with (raw / "nonideality_prediction_topk_margin.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=PREDICTION_TOPK_MARGIN_FIELDS)  # 按脚本要求的列名写 CSV
                writer.writeheader()
                writer.writerows(rows)
            out = tmp_path / "stats"
            run_script(
                [
                    "experiments/scripts/run_e_local_robustness_statistics.py",
                    "--output-dir",
                    str(out),
                    "--run-dir",
                    str(run_dir),
                    "--bootstrap-repeats",
                    "8",  # bootstrap 重采样 8 次（冒烟用最小次数）
                    "--bootstrap-sample-cap",
                    "2",  # 每次重采样最多抽 2 个样本
                ]
            )
            # 统计脚本应产出 5 张核心表格，逐个确认存在。
            for name in [
                "imagenette_clean_accuracy_by_model.csv",  # 各模型干净精度
                "nonideality_accuracy_classwise.csv",  # 按类别分组的精度
                "nonideality_accuracy_bootstrap_ci.csv",  # bootstrap 置信区间
                "nonideality_accuracy_worst_class.csv",  # 最差类别
                "nonideality_margin_drift_by_severity.csv",  # 裕度漂移随严重度变化
            ]:
                self.assertTrue((out / "tables" / name).exists(), name)
            with (out / "tables" / "nonideality_accuracy_worst_class.csv").open("r", encoding="utf-8", newline="") as f:
                worst_rows = list(csv.DictReader(f))
            self.assertEqual(worst_rows[0]["worst_class_name"], "English springer")  # margin 掉最多的是 springer

if __name__ == "__main__":
    unittest.main()
