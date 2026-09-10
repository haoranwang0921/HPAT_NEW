"""
P0 管线冒烟测试（smoke test：只验证核心流程能否端到端跑通的最小测试集）。

项目背景：hpat-artifact 是《HPAT：光子张量处理器》（ASP-DAC 2027 投稿）的可复现实验库。
HPAT 是光子计算架构：96 个计算核心（Core），约 39 万个 MRR 微环谐振器
（micro-ring resonator，光子矩阵乘法的核心器件）用于执行光矩阵乘法。

证据等级（evidence tier）说明：
- P0：纯仿真契约（simulation-only contract），本文件验证的就是 P0 管线的各个环节能否
  端到端跑通，并产出规范化的"清单"（manifest：JSON 格式的实验结果元数据文件）。
- P1：有真实硬件实测数据支撑。
- P2：物理代理/仿真证据。

本文件覆盖的 P0 流程：
1) hpat 活动导出（activity export）：把算子活动表 + 单位成本换算成按组件划分的能量表；
2) 边缘设备基线导入（edge baseline import）：导入桌面/边缘设备实测的延迟与功耗；
3) 非理想性精度扫描（nonideality accuracy sweep）：模拟 MRR 噪声/串扰/热漂移等对精度的影响；
4) 基线溯源检查（baseline provenance check）：校验论文 Fig.6 桌面实测参考数据的来源可追溯；
5) 证据强度汇总（evidence strength summary）：汇总成机器可读的 P0 就绪度 JSON；
6) P0 证据输入校验器（validator）：校验证据包字段完整性并做哈希校验（防篡改）。

所有测试都通过子进程运行 experiments/scripts/ 下的真实脚本，再读取脚本产出的
manifest 断言关键字段，做到"测试即契约"。
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
import hashlib
import csv


# 仓库根目录 = 本文件（experiments/tests/ 下）向上两级；fixtures 目录存放测试固定样例数据
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"
# 把 experiments/src 加入模块搜索路径，便于测试直接 import 实验源码（如 hpat_eval 包）
EXPERIMENT_SRC = REPO_ROOT / "experiments" / "src"
if str(EXPERIMENT_SRC) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_SRC))


def run_script(args: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    """以子进程方式运行实验脚本，并断言其退出码符合预期。

    args: 传给脚本的命令行参数（相对仓库根目录的脚本路径 + 选项）；
    expected: 期望的退出码，默认 0（成功）。
    若实际退出码不符，抛出 AssertionError 并附上脚本的 stdout/stderr，方便定位问题。
    这是所有测试与真实脚本交互的统一入口。
    """
    env = dict(os.environ)
    # 默认关闭"写工程汇总表"，避免测试过程在仓库根 tables/ 目录留下污染产物
    env.setdefault("HPAT_WRITE_PROJECT_TABLES", "0")
    proc = subprocess.run([sys.executable, *args], cwd=REPO_ROOT, text=True, capture_output=True, env=env)
    if proc.returncode != expected:
        raise AssertionError(
            f"command returned {proc.returncode}, expected {expected}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


class P0PipelineSmokeTests(unittest.TestCase):
    """P0 管线冒烟测试组。

    每个测试都以真实子进程运行实验脚本（而非 mock），随后读取脚本产出的
    manifest（JSON 元数据清单）并断言关键字段，从而验证 P0 证据链路端到端可跑通。
    """

    def tearDown(self) -> None:
        """每个测试结束后的清理钩子。

        删除校准能量表 hpat_energy_by_component_calibrated.csv，
        避免上一次测试的产物残留影响后续断言或仓库状态。
        """
        calibrated = REPO_ROOT / "tables" / "hpat_energy_by_component_calibrated.csv"
        if calibrated.exists():
            calibrated.unlink()

    def test_hpat_activity_export_requires_complete_unit_cost_metadata(self) -> None:
        """验证 hpat 活动导出的"单位成本元数据门禁"：数据不完整时必须拦截。

        测什么（共 4 种场景）：
        1) 无任何输入 → 脚本应产出 status="proxy"（代理）结果：P0 是纯仿真契约，
           没有实测时用仿真代替，但绝不生成声称"已校准"的能量表；
        2) 完整活动表 + 完整单位成本 → status="ok"，正常产出校准能量表；
        3) 单位成本缺少 source 字段 → status="blocked"（拦截），提示缺少单位成本元数据；
        4) 活动表缺少必要列 → status="blocked"，提示缺少必要列。

        关键断言：manifest 里的 schema_version/gate_version/status/blocked_reason/outputs 字段。
        背景：单位成本（unit cost）是把"算子活动"换算成"能量"的单价，
        缺了来源信息就说明证据链不完整，必须拦截以防过度声明。
        """
        with tempfile.TemporaryDirectory() as tmp:
            proxy_out = pathlib.Path(tmp) / "proxy"
            run_script(["experiments/scripts/run_hpat_activity_export.py", "--output-dir", str(proxy_out)])
            proxy_manifest = json.loads((proxy_out / "hpat_energy_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(proxy_manifest["schema_version"], "hpat-manifest-v2")  # 清单协议版本固定为 v2
            self.assertEqual(proxy_manifest["gate_version"], "p0-gate-v2")  # 证据门禁（gate）版本：P0 关卡 v2
            self.assertEqual(proxy_manifest["status"], "proxy")  # 无外部输入 → 仿真代理状态
            # 代理模式下产物列表不得出现"已校准"能量表——没有实测单位成本就无权声称校准
            self.assertFalse(any("hpat_energy_by_component_calibrated.csv" in path for path in proxy_manifest["outputs"]))

            valid_out = pathlib.Path(tmp) / "valid"
            run_script(
                [
                    "experiments/scripts/run_hpat_activity_export.py",
                    "--output-dir",
                    str(valid_out),
                    "--activity-csv",
                    str(FIXTURES / "valid_hpat_activity.csv"),
                    "--unit-costs-json",
                    str(FIXTURES / "unit_costs_complete.json"),
                ]
            )
            valid_manifest = json.loads((valid_out / "hpat_energy_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(valid_manifest["status"], "ok")
            self.assertEqual(valid_manifest["activity_source"], "simulator_export")  # 活动数据来自仿真导出
            self.assertFalse(valid_manifest["normalize_to_energy_envelope"])  # 未强制归一化到能量包络
            # 完整输入下应真正产出"已校准"能量表，与代理模式形成对照
            self.assertTrue(any("hpat_energy_by_component_calibrated.csv" in path for path in valid_manifest["outputs"]))

            missing_source_out = pathlib.Path(tmp) / "missing_source"
            run_script(
                [
                    "experiments/scripts/run_hpat_activity_export.py",
                    "--output-dir",
                    str(missing_source_out),
                    "--activity-csv",
                    str(FIXTURES / "valid_hpat_activity.csv"),
                    "--unit-costs-json",
                    str(FIXTURES / "unit_costs_missing_source.json"),
                ],
                expected=1,  # 期望脚本以非零码退出：数据不完整，流程被门禁拦下
            )
            missing_manifest = json.loads((missing_source_out / "hpat_energy_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(missing_manifest["status"], "blocked")
            self.assertIn("missing unit-cost metadata", missing_manifest["blocked_reason"])

            invalid_out = pathlib.Path(tmp) / "invalid"
            run_script(
                [
                    "experiments/scripts/run_hpat_activity_export.py",
                    "--output-dir",
                    str(invalid_out),
                    "--activity-csv",
                    str(FIXTURES / "invalid_hpat_activity.csv"),
                ],
                expected=1,
            )
            invalid_manifest = json.loads((invalid_out / "hpat_energy_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(invalid_manifest["status"], "blocked")
            self.assertIn("missing required columns", invalid_manifest["blocked_reason"])

    def test_edge_latency_only_import(self) -> None:
        """验证边缘设备基线导入支持"仅延迟"（latency-only）导入。

        测什么：只提供延迟 CSV（不提供功耗 CSV）时，导入应成功并标记为
        latency-only（只收集了延迟、未收集功耗的采集状态），同时正确统计延迟
        样本数，并声明该数据"可申请证据"（claim_eligible，即证据链达标）。

        预期：manifest.status == "ok"；power_collection_status == "latency-only"；
        latency_sample_count == 4（样例文件有 4 条延迟样本）；
        coverage.claim_eligible 为真。
        """
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "edge"
            run_script(
                [
                    "experiments/scripts/run_edge_baseline_import.py",
                    "--output-dir",
                    str(out),
                    "--latency-csv",
                    str(FIXTURES / "edge_latency_samples_full.csv"),
                    "--device-config",
                    str(FIXTURES / "edge_device.yaml"),
                ]
            )
            manifest = json.loads((out / "edge_baseline_import_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "ok")
            self.assertEqual(manifest["power_collection_status"], "latency-only")  # 未给功耗 CSV → 仅延迟采集
            self.assertEqual(manifest["latency_sample_count"], 4)  # 样例延迟文件恰好 4 条样本
            self.assertTrue(manifest["coverage"]["claim_eligible"])  # 覆盖度达标，可申请证据

    def test_edge_missing_variant_is_partial_and_invalid_power_blocks(self) -> None:
        """验证边缘导入的两种异常路径：缺变体=部分成功（partial）、非法功耗=拦截（blocked）。

        测什么：
        1) 延迟 CSV 缺少 MobileViT-S 变体时，导入整体记为"部分成功"（partial，
           部分数据可用但覆盖不完整），并在 coverage.missing_variants 中列出缺失变体；
        2) 功耗 CSV 缺少必要值时，导入被拦截为 blocked（阻止使用不完整数据做声明）。

        预期：partial_manifest.status == "partial" 且 missing_variants 含 "MobileViT-S"；
        blocked_manifest.status == "blocked" 且 blocked_reason 含 "missing required values"。
        """
        with tempfile.TemporaryDirectory() as tmp:
            partial_out = pathlib.Path(tmp) / "edge_partial"
            run_script(
                [
                    "experiments/scripts/run_edge_baseline_import.py",
                    "--output-dir",
                    str(partial_out),
                    "--latency-csv",
                    str(FIXTURES / "edge_latency_samples.csv"),
                    "--device-config",
                    str(FIXTURES / "edge_device.yaml"),
                ]
            )
            partial_manifest = json.loads((partial_out / "edge_baseline_import_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(partial_manifest["status"], "partial")  # 数据不全但未到"不可用"程度 → 部分成功
            self.assertIn("MobileViT-S", partial_manifest["coverage"]["missing_variants"])  # 缺失变体被点名

            blocked_out = pathlib.Path(tmp) / "edge_power_blocked"
            run_script(
                [
                    "experiments/scripts/run_edge_baseline_import.py",
                    "--output-dir",
                    str(blocked_out),
                    "--latency-csv",
                    str(FIXTURES / "edge_latency_samples_full.csv"),
                    "--power-csv",
                    str(FIXTURES / "edge_power_invalid.csv"),
                    "--device-config",
                    str(FIXTURES / "edge_device.yaml"),
                ],
                expected=1,  # 功耗数据非法 → 期望非零退出码
            )
            blocked_manifest = json.loads((blocked_out / "edge_baseline_import_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(blocked_manifest["status"], "blocked")
            self.assertIn("missing required values", blocked_manifest["blocked_reason"])

    def test_edge_import_without_input_is_blocked_but_nonfatal(self) -> None:
        """验证"无任何输入参数"的边界情况：应标记为 blocked 但不导致进程崩溃。

        测什么：完全不传延迟/功耗 CSV 直接运行导入脚本。脚本应产出清单，
        status == "blocked"，blocked_reason 提示缺少外部边缘延迟 CSV；
        同时退出码为 0（非致命），保证 CI 等环境中无输入运行不会直接崩溃。
        """
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "edge_blocked"
            run_script(["experiments/scripts/run_edge_baseline_import.py", "--output-dir", str(out)])
            manifest = json.loads((out / "edge_baseline_import_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "blocked")
            self.assertIn("No external edge latency CSV", manifest["blocked_reason"])  # 拦截原因明确可读

    def test_nonideality_proxy_mode(self) -> None:
        """验证非理想性精度扫描在无输入时进入"代理"（proxy）模式。

        背景：非理想性扫描模拟 MRR 微环谐振器的噪声、串扰、热漂移等真实物理
        缺陷对模型精度的影响。P0 是纯仿真契约，没有真实测量数据时用仿真代替，
        产物状态记为 proxy（代理/替身证据）。
        预期：status == "proxy"，且扫描结果行数 > 0（仿真确实产出了数据）。
        """
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "nonideality_proxy"
            run_script(["experiments/scripts/run_nonideality_accuracy_sweep.py", "--output-dir", str(out)])
            manifest = json.loads((out / "nonideality_accuracy_sweep_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "proxy")  # 无输入 → 仿真代理状态
            self.assertGreater(manifest["row_count"], 0)  # 扫描结果非空，仿真流程真正产出过数据

    def test_evidence_summary_writes_machine_readable_p0_readiness(self) -> None:
        """验证证据强度汇总脚本能产出机器可读的 P0 就绪度 JSON。

        测什么：依次运行 P0-E1（活动导出）、P0-E2（边缘导入）、P0-E3（非理想性
        扫描）、P0-E4（基线溯源）四个实验（均无外部输入，故 E1/E3 为 proxy，
        E2/E4 为 blocked），再运行汇总脚本，检查：
        1) 就绪度 JSON 的协议/门禁版本正确（schema p0-readiness-v1、gate p0-gate-v2）；
        2) 四个实验的状态与预期一致；
        3) 没有 P1 实测证据时 overall_g1_ready（G1 关卡就绪标志）必须为假——
           防止系统误报"可以申请更高证据等级"。

        意义：保证 P0 门禁报告是机器可读、可被 CI 解析的，是论文证据链的"仪表盘"。
        """
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "run"
            run_script(["experiments/scripts/run_hpat_activity_export.py", "--output-dir", str(out)])
            run_script(["experiments/scripts/run_edge_baseline_import.py", "--output-dir", str(out)])
            run_script(["experiments/scripts/run_nonideality_accuracy_sweep.py", "--output-dir", str(out)])
            run_script(["experiments/scripts/run_baseline_provenance_check.py", "--output-dir", str(out)])
            run_script(["experiments/scripts/run_evidence_strength_summary.py", "--output-dir", str(out)])
            readiness = json.loads((out / "tables" / "p0_readiness_summary.json").read_text(encoding="utf-8"))
            # 把每个实验的 status 汇总成 {实验名: 状态} 字典，便于逐项断言
            statuses = {row["experiment"]: row["status"] for row in readiness["experiments"]}
            self.assertEqual(readiness["schema_version"], "p0-readiness-v1")  # 就绪度清单协议版本
            self.assertEqual(readiness["gate_version"], "p0-gate-v2")  # 就绪度所属门禁版本
            # 关键断言：无实测证据时绝不允许 G1 关卡就绪，防止越级声明
            self.assertFalse(readiness["overall_g1_ready"])
            self.assertEqual(statuses["P0-E1"], "proxy")  # 活动导出：无输入 → 仿真代理
            self.assertEqual(statuses["P0-E2"], "blocked")  # 边缘导入：无输入 → 拦截
            self.assertEqual(statuses["P0-E3"], "proxy")  # 非理想性扫描：无输入 → 仿真代理
            self.assertEqual(statuses["P0-E4"], "blocked")  # 基线溯源：无输入 → 拦截
            manifest = json.loads((out / "evidence_strength_summary_manifest.json").read_text(encoding="utf-8"))
            # 汇总清单必须指向真实的就绪度 JSON 文件路径
            self.assertTrue(manifest["p0_readiness_json"].endswith("tables/p0_readiness_summary.json"))
            self.assertFalse(manifest["overall_g1_ready"])  # manifest 与就绪度结论一致

    def test_p0_evidence_input_validator_blocks_incomplete_package_and_accepts_hashes(self) -> None:
        """验证 P0 证据输入校验器：不完整证据包被拦截、带哈希的完整包被接受。

        背景：证据包（evidence package）是 P0 阶段的标准数据容器，用于把实验采集
        结果（如延迟 CSV）连同元数据打包，作为后续 P1/P2 证据升级的交接物；
        哈希校验（对数据文件做 sha256 摘要）用于防止包内文件被篡改。

        测什么：
        1) 只写了 package_kind 的"半成品"清单 → 校验结果 status == "blocked"，
           claim_eligible（可否申请证据）为假，blocked_reason 提示缺少必需字段；
        2) 补齐全部字段（文件清单、来源描述、采集方式、文件哈希、声明范围、
           已知局限）→ 校验通过，status == "ok" 且 claim_eligible 为真。

        关键断言：两端的 status/claim_eligible/blocked_reason 是否符合预期。
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            incomplete = tmp_path / "incomplete.json"
            incomplete.write_text(json.dumps({"package_kind": "edge-baseline"}), encoding="utf-8")
            blocked_out = tmp_path / "blocked"
            run_script(
                [
                    "experiments/scripts/validate_p0_evidence_inputs.py",
                    "--kind",
                    "edge-baseline",
                    "--input-manifest",
                    str(incomplete),
                    "--output-dir",
                    str(blocked_out),
                ]
            )
            blocked = json.loads((blocked_out / "p0_evidence_input_validation_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(blocked["status"], "blocked")  # 字段不全 → 拦截
            self.assertFalse(blocked["claim_eligible"])  # 不完整的包无权申请证据
            self.assertIn("missing required field", blocked["blocked_reason"])  # 拦截原因：缺必需字段

            data_file = tmp_path / "edge_latency_samples.csv"
            data_file.write_text("model_variant,latency_ms\nMobileViT-XXS,1.0\n", encoding="utf-8")
            # 哈希校验：对数据文件计算 sha256 摘要，放进证据包以证明文件未被改动
            digest = hashlib.sha256(data_file.read_bytes()).hexdigest()
            complete = tmp_path / "complete.json"
            complete.write_text(
                json.dumps(
                    {
                        "package_kind": "edge-baseline",
                        "schema_version": "p0-evidence-package-v1",
                        "files": ["edge_latency_samples.csv"],  # 包内文件清单
                        "source_description": "unit-test edge package",  # 数据来源描述
                        "collection_command_or_method": "unit-test",  # 采集命令/方法
                        "hashes": {"edge_latency_samples.csv": digest},  # 文件 sha256 哈希
                        "claim_scope": "MobileViT edge baseline context only",  # 声明范围（只限边缘基线）
                        "known_limitations": ["not HPAT deployment"],  # 已知局限（非 HPAT 部署）
                    }
                ),
                encoding="utf-8",
            )
            ok_out = tmp_path / "ok"
            run_script(
                [
                    "experiments/scripts/validate_p0_evidence_inputs.py",
                    "--kind",
                    "edge-baseline",
                    "--input-manifest",
                    str(complete),
                    "--output-dir",
                    str(ok_out),
                ]
            )
            ok = json.loads((ok_out / "p0_evidence_input_validation_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(ok["status"], "ok")  # 完整包 → 校验通过
            self.assertTrue(ok["claim_eligible"])  # 完整包可申请证据

    def test_hpat_mapping_boundary_parser_records_coverage_and_exclusions(self) -> None:
        """验证 HPAT 算子映射边界解析器：正确记录匹配层、缺失层与覆盖率。

        背景：HPAT 光子计算只把部分算子（如注意力 QKV 投影、FFN 线性层）映射到
        光域（由 MRR 微环谐振器做光矩阵乘法），其余算子留在电子域。边界解析器
        （injection boundary）据此划分"光域/电子域"的算子边界。

        测什么：给 3 条算子活动记录（候选标记分别为 yes / partial / no），并限定
        实际可用的模块名（available_module_names），验证边界结果：
        - mode == "hpat-mapping"（映射模式）；
        - matched_layers（匹配到光域的层）只保留确实存在的层；
        - missing_layers（声称候选但实际缺失的层）被单独记录；
        - coverage（光域覆盖比例）为 0.5（1 个匹配层 / 2 个候选层）；
        - 纯电子域算子组（归一化/softmax/激活）被列入排除清单。
        """
        from hpat_eval.injection_boundary import boundary_from_operator_rows

        # 3 条样例算子活动：qkv 投影=yes、FFN 线性=partial、归一化激活=no
        rows = [
            {
                "model": "MobileViT-XXS",
                "pdpu_candidate": "yes",
                "op_group": "qkv_projection",
                "layer_name": "blocks.0.attn.qkv",
                "execution_domain": "optical candidate",
            },
            {
                "model": "MobileViT-XXS",
                "pdpu_candidate": "partial",
                "op_group": "ffn_linear",
                "layer_name": "blocks.0.mlp.fc1",
                "execution_domain": "optical candidate",
            },
            {
                "model": "MobileViT-XXS",
                "pdpu_candidate": "no",
                "op_group": "normalization_softmax_activation",
                "layer_name": "blocks.0.norm",
                "execution_domain": "electronic remainder",
            },
        ]
        boundary = boundary_from_operator_rows(
            rows,
            model_variant="MobileViT-XXS",
            available_module_names={"blocks.0.attn.qkv", "blocks.0.norm"},  # 只有这两个模块真实存在
        )
        self.assertEqual(boundary["mode"], "hpat-mapping")  # 模式：hpat 算子映射
        self.assertEqual(boundary["matched_layers"], ["blocks.0.attn.qkv"])  # 只有真实存在的层被匹配
        self.assertEqual(boundary["missing_layers"], ["blocks.0.mlp.fc1"])  # partial 但模块缺失的层被记录
        self.assertAlmostEqual(boundary["coverage"], 0.5)  # 覆盖比例 = 匹配层 1 / 候选层 2
        # 纯电子域算子组被排除在光域映射之外
        self.assertIn("normalization_softmax_activation", boundary["excluded_electronic_groups"])

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) is not None for name in ["torch", "timm", "PIL", "numpy"]),
        "dataset-mode smoke requires torch, timm, Pillow, and numpy in the active Python",
    )
    def test_nonideality_dataset_mode_smoke(self) -> None:
        """数据集模式冒烟测试：用真实模型跑一遍非理想性扫描（需 torch/timm/PIL/numpy 全可用）。

        测什么（两个阶段）：
        1) "干净"运行：用 PIL 生成一张 32x32 占位图片 + 单样本子集文件，把所有非理想性
           参数（噪声、串扰、量化位宽、插损、失谐、MRR 工艺偏差、热漂移）都设为无害值，
           在 CPU 上跑 1 个样本，验证流程端到端跑通 → 状态 "smoke"；
        2) "映射"运行：额外提供一张算子活动表（声明某个 Linear 层映射到光域），驱动第二次
           扫描，验证注入边界清单被产出、可申请证据，且匹配层与实际映射层一致、电子域
           算子组被正确排除 → 状态 "ok"。

        关键断言：clean 清单 status=="smoke" 且样本数==1；mapped 清单 status=="ok"、
        injection_boundary_manifest.claim_eligible 为真。
        """
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            dataset_root = tmp_path / "dataset"
            dataset_root.mkdir()
            # 生成一张 32x32 的纯色占位图片作为"数据集"
            Image.new("RGB", (32, 32), (80, 120, 160)).save(dataset_root / "sample.png")
            subset = tmp_path / "subset.txt"
            subset.write_text("sample.png 0\n", encoding="utf-8")  # 子集文件：一行 = 图片名 + 标签
            config = json.loads((REPO_ROOT / "experiments/config/hpat_experiment_config.json").read_text(encoding="utf-8"))
            config["mobilevit_variants"] = [config["mobilevit_variants"][0]]  # 只保留第一个变体，加速
            # 所有非理想性参数设为"无影响"值，保证干净运行只测流程、不测物理缺陷
            config["nonideality"] = {
                "noise_lsb": [0.0],  # 噪声（最低有效位）
                "crosstalk_alpha": [0.0],  # 串扰系数
                "quant_bits": [8],  # 量化位宽
                "insertion_loss_db": [0.0],  # 插入损耗（dB）
                "detuning_pm": [0.0],  # 失谐（pm）
                "mrr_variation_percent": [0.0],  # MRR 微环工艺偏差（%）
                "thermal_drift_c": [0.0],  # 热漂移（℃）
            }
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            out = tmp_path / "dataset_run"
            # —— 阶段 1：干净运行（干净运行 = 非理想性全关闭的基准仿真）——
            run_script(
                [
                    "experiments/scripts/run_nonideality_accuracy_sweep.py",
                    "--output-dir",
                    str(out),
                    "--config",
                    str(config_path),
                    "--dataset-root",
                    str(dataset_root),
                    "--subset-file",
                    str(subset),
                    "--max-samples",
                    "1",  # 只跑 1 个样本，冒烟测试控制耗时
                    "--device",
                    "cpu",
                    "--model-variant",
                    "MobileViT-XXS",
                    "--injection-boundary",
                    "all-linear-smoke",  # 注入边界（光/电划分）：本阶段用内置的冒烟边界
                ]
            )
            manifest = json.loads((out / "nonideality_accuracy_sweep_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "smoke")  # 冒烟运行成功
            self.assertEqual(manifest["dataset"]["sample_count"], 1)  # 确实只处理了 1 个样本
            self.assertTrue((out / "raw" / "nonideality_logits_clean.npz").exists())  # 干净仿真的 logits 已落盘
            self.assertTrue((out / "nonideality_injection_boundary_manifest.json").exists())  # 边界清单已产出

            import timm  # type: ignore
            import torch  # type: ignore

            # 加载 MobileViT 模型，找出第一个 Linear 层作为"映射到光域"的真实层名
            model = timm.create_model(config["mobilevit_variants"][0]["timm_model"], pretrained=False)
            mapped_layer = next(name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear))
            operator_activity = tmp_path / "operator_activity.csv"
            operator_activity.write_text(
                "\n".join(
                    [
                        "model,pdpu_candidate,op_group,layer_name,execution_domain",
                        f"MobileViT-XXS,yes,qkv_projection,{mapped_layer},optical candidate",
                        "MobileViT-XXS,no,normalization_softmax_activation,stem.bn.act,electronic remainder",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            # —— 阶段 2：映射运行（映射运行 = 按算子活动表把部分层注入光域的仿真）——
            mapped_out = tmp_path / "dataset_mapped_run"
            run_script(
                [
                    "experiments/scripts/run_nonideality_accuracy_sweep.py",
                    "--output-dir",
                    str(mapped_out),
                    "--config",
                    str(config_path),
                    "--dataset-root",
                    str(dataset_root),
                    "--subset-file",
                    str(subset),
                    "--max-samples",
                    "1",
                    "--device",
                    "cpu",
                    "--model-variant",
                    "MobileViT-XXS",
                    "--operator-activity-csv",
                    str(operator_activity),
                    "--min-boundary-coverage",
                    "1.0",  # 要求光域覆盖比例达到 100%，否则拒绝
                ]
            )
            mapped_manifest = json.loads(
                (mapped_out / "nonideality_accuracy_sweep_manifest.json").read_text(encoding="utf-8")
            )
            mapped_boundary = json.loads(
                (mapped_out / "nonideality_injection_boundary_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(mapped_manifest["status"], "ok")  # 映射运行正常完成
            self.assertTrue(mapped_boundary["claim_eligible"])  # 覆盖达标，可申请证据
            self.assertEqual(mapped_boundary["records"][0]["matched_layers"], [mapped_layer])  # 匹配层与声明层一致
            # 纯电子域算子组（归一化/softmax/激活）应出现在排除清单中
            self.assertIn("normalization_softmax_activation", mapped_boundary["records"][0]["excluded_electronic_groups"])

    def test_baseline_provenance_repair_smoke(self) -> None:
        """验证基线溯源检查/修复脚本的冒烟流程。

        背景：该脚本负责校验论文 Fig.6 中桌面实测参考数据的来源是否可追溯
        （provenance，数据血缘），并把修复后的证据表写到 tables/ 目录。

        测什么：无参数运行 run_baseline_provenance_check.py，脚本应产出溯源清单
        （status 为 "ok" 或 "blocked" 均合法，取决于基线表是否存在），
        且必须生成桌面基线测量表 mobilevit_desktop_baseline_measured.csv。
        """
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "provenance"
            run_script(["experiments/scripts/run_baseline_provenance_check.py", "--output-dir", str(out)])
            manifest = json.loads((out / "baseline_provenance_manifest.json").read_text(encoding="utf-8"))
            # 两种状态都接受：跑通即可，具体结论取决于仓库内基线数据的有无
            self.assertIn(manifest["status"], {"ok", "blocked"})
            # 无论结论如何，修复后的桌面基线测量表都必须产出
            self.assertTrue((out / "tables" / "mobilevit_desktop_baseline_measured.csv").exists())

    def test_fig6_energy_power_rows_remain_desktop_measured_references(self) -> None:
        """验证 Fig.6 的 energy/power 行始终被标记为"桌面实测参考"、禁止过度解读。

        背景：论文 Fig.6 中的 GPU/CPU 桌面实测数据只是"参考上下文"，
        不能被解读成 HPAT 光子芯片或移动/边缘设备的直接成绩，因此证据表必须
        附带"安全解读"（safe_interpretation）与"禁止暗示"（must_not_imply）两栏。

        测什么：喂入 2 行 CSV（energy 和 power 各 1 行），运行溯源检查后读取
        修复表，断言：
        - energy/power 且平台为 CPU/GPU 的行恰好 2 行；
        - 每行 evidence_tier 保持 "measured desktop reference"（桌面实测参考）；
        - safe_interpretation 注明"桌面实测参考上下文"；
        - must_not_imply 明确排除"移动/边缘"场景。
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            fig6 = tmp_path / "fig6.csv"
            fig6.write_text(
                "\n".join(
                    [
                        "model_variant,metric,platform_label,value,unit,evidence_tier,source,safe_interpretation,must_not_imply",
                        "MobileViT-XXS,energy,GPU measured desktop reference,1.0,mJ,measured desktop reference,Extracted from current HPAT draft Fig. 6 text in tmp/hpat_text.txt,unsafe,unsafe",
                        "MobileViT-XXS,power,CPU desktop reference,1.0,W,measured desktop reference,Extracted from current HPAT draft Fig. 6 text in tmp/hpat_text.txt,unsafe,unsafe",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            out = tmp_path / "provenance"
            run_script(
                [
                    "experiments/scripts/run_baseline_provenance_check.py",
                    "--output-dir",
                    str(out),
                    "--fig6-csv",
                    str(fig6),
                ]
            )
            with (out / "tables" / "fig6_evidence_repaired.csv").open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            # 筛选出 energy/power 且平台标注为 CPU 或 GPU 的行（即桌面实测参考行）
            guarded = [
                row
                for row in rows
                if row["metric"] in {"energy", "power"} and ("CPU" in row["platform_label"] or "GPU" in row["platform_label"])
            ]
            self.assertEqual(len(guarded), 2)  # 恰好 2 行（energy + power）被保护性标注
            self.assertTrue(all(row["evidence_tier"] == "measured desktop reference" for row in guarded))  # 证据等级不被改写
            self.assertTrue(all("Desktop measured-reference context" in row["safe_interpretation"] for row in guarded))  # 安全解读已补充
            self.assertTrue(all("mobile/edge" in row["must_not_imply"] for row in guarded))  # 禁止暗示移动/边缘场景


if __name__ == "__main__":
    unittest.main()
