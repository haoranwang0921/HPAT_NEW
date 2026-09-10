"""
可复现性打包（reproducibility package）测试：验证论文实验的"可复现运行包"机制。

项目背景：hpat-artifact 是《HPAT：光子张量处理器》（ASP-DAC 2027 投稿）的可复现
实验库。HPAT 是光子计算架构（96 Core、约 39 万个 MRR 微环谐振器做光矩阵乘法）。
P0 是纯仿真契约：论文的一切结论都要求"可复现"——同一输入、同一配置、同一代码
必须产生逐字节一致的结果。

本文件围绕 hpat_eval/reproducibility 模块验证以下几个核心承诺：
1) 规范输出（canonical outputs）：只有被"声明为规范"的产物才计入复现哈希，
   浮点诊断文件（如 topk 置信度边界）会被排除，避免位数抖动破坏可复现；
2) 冻结发布（freeze_run）与归档校验：一次运行可冻结成 tar.gz 发布归档，
   归档内容确定、可逐字节复现；
3) 哈希校验（对文件算 sha256 摘要）贯穿始终：篡改任何产物都会被 verify_run 发现；
4) 设备选择（resolve_device）：请求 MPS 时绝不允许静默回退到 CPU（会改变数值结果）；
5) 分层采样（stratified_sample）：1024 样本在 10 类 ImageNet 上按 103/102 分布，
   采样是确定性的；
6) 工程保护：写"工程汇总表"必须显式开关（HPAT_WRITE_PROJECT_TABLES=1）才允许，
   防止测试意外污染仓库 tables/ 目录。

测试大量使用 mock（模拟对象），如 _FakeTorch 伪造 torch 的 mps/cuda 状态，
从而在无真实 GPU 的机器上也能验证设备回退逻辑。
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock


# 仓库根目录 / 实验源码目录 / 实验脚本目录，全部加入 sys.path 供测试直接导入
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXPERIMENT_SRC = REPO_ROOT / "experiments" / "src"
if str(EXPERIMENT_SRC) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_SRC))
EXPERIMENT_SCRIPTS = REPO_ROOT / "experiments" / "scripts"
if str(EXPERIMENT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_SCRIPTS))

# noqa: E402：这些导入依赖上面的 sys.path 调整，故跳过"导入未放顶部"的 lint 告警
from _common import project_writes_enabled  # noqa: E402  # 判断"工程表写入"是否被显式开启

# 可复现性模块的核心常量与函数（详见各测试函数的说明）
from hpat_eval.reproducibility import (  # noqa: E402
    CLAIM_BOUNDARY,  # 声明边界：说明数据只能用于本地/建模诊断
    PAPER_ARTIFACT_STATUS,  # 论文产物状态（paper_eligible_with_limitations 等）
    PAPER_CONFIG,  # 论文规范配置文件的路径
    PAPER_EVIDENCE_TIER,  # 论文证据等级
    PAPER_REPEAT_SEEDS,  # 论文固定随机种子列表
    REPRO_MANIFEST_SCHEMA_VERSION,  # 可复现清单（manifest）协议版本
    REFERENCE_COMPARISON_SCHEMA_VERSION,  # 参考比对（reference comparison）协议版本
    ReproducibilityError,  # 可复现性相关自定义异常
    _output_records,  # 扫描运行目录生成输出记录（相对路径 + sha256 + 字节数）
    compare_run_to_reference,  # 把一次运行与参考归档（reference archive）比对
    freeze_run,  # 冻结运行：把运行目录固化成发布归档
    load_paper_config,  # 加载论文规范配置 JSON
    prepare_data,  # 准备数据集（构造合成样例并逐图哈希）
    resolve_device,  # 解析计算设备（mps/cuda/cpu），决定是否允许回退
    run_pipeline,  # 一键运行完整 smoke 管线
    sha256_file,  # 对文件计算 sha256 摘要（哈希校验）
    stratified_sample,  # 按类别分层采样（保证每类数量均衡）
    verify_run,  # 校验一次运行是否完整、产物是否被篡改
    write_json,  # 把对象写为 JSON 文件
    write_subset_csv,  # 把子集清单写为 CSV
)


class _FakeMPS:
    """伪造 torch.backends.mps 的假对象，用于在无真实 MPS 设备的机器上测试设备回退逻辑。"""

    def __init__(self, built: bool, available: bool) -> None:
        self._built = built  # MPS 是否已编译进 torch
        self._available = available  # MPS 当前是否可用

    def is_built(self) -> bool:
        return self._built

    def is_available(self) -> bool:
        return self._available


class _FakeCuda:
    """伪造 torch.cuda 的假对象，模拟 CUDA 是否可用。"""

    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorch:
    """整体伪造的 torch 模块：组合 _FakeMPS 与 _FakeCuda，供 resolve_device 测试使用。"""

    def __init__(self, *, mps_built: bool, mps_available: bool, cuda_available: bool) -> None:
        self.backends = type("Backends", (), {"mps": _FakeMPS(mps_built, mps_available)})()
        self.cuda = _FakeCuda(cuda_available)


class ReproducibilityPackageTests(unittest.TestCase):
    """可复现性打包测试组：验证规范输出、采样、设备选择、运行校验与冻结发布。"""

    def test_canonical_digest_excludes_only_declared_float_diagnostic(self) -> None:
        """验证"规范输出摘要"只排除已声明的浮点诊断文件，其余产物保留。

        背景：可复现性的最大敌人是浮点数在不同机器/库版本上的尾数抖动。
        非理想性预测的"topk 置信度边界"（浮点诊断文件）不该进入复现哈希，
        否则 0.12345678 这种值一抖动，整个摘要就变。因此这类文件必须被排除；
        而"预测变化"（0/1 整数结果）是稳定的，应保留。

        测什么：在 raw/ 下造两个文件，调用 _canonical_output_record：
        - nonideality_prediction_changes.csv 保留在 canonical_paths 中；
        - nonideality_prediction_topk_margin.csv 进入 excluded_diagnostics（排除清单）。
        """
        from hpat_eval.reproducibility import _canonical_output_record

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = pathlib.Path(temporary)
            raw = run_dir / "raw"
            raw.mkdir()
            # 整数型诊断：0/1 变化的稳定结果，应计入规范输出
            (raw / "nonideality_prediction_changes.csv").write_text(
                "changed\n0\n", encoding="utf-8", newline="\n"
            )
            # 浮点型诊断：top1 置信度边界（尾数会抖动），应排除出规范输出
            (raw / "nonideality_prediction_topk_margin.csv").write_text(
                "perturbed_top1_confidence\n0.12345678\n", encoding="utf-8", newline="\n"
            )
            record = _canonical_output_record(run_dir, _output_records(run_dir))
            canonical_paths = {row["path"] for row in record["files"]}
            excluded_paths = {row["path"] for row in record["excluded_diagnostics"]}
            self.assertIn("raw/nonideality_prediction_changes.csv", canonical_paths)  # 整数诊断保留
            self.assertNotIn("raw/nonideality_prediction_topk_margin.csv", canonical_paths)  # 浮点诊断不保留
            self.assertEqual(excluded_paths, {"raw/nonideality_prediction_topk_margin.csv"})  # 恰好只有它被排除

    def test_project_table_promotion_requires_exact_explicit_opt_in(self) -> None:
        """验证"工程汇总表写入"必须是精确的显式开关（HPAT_WRITE_PROJECT_TABLES=1）才开启。

        背景：部分实验脚本会把结果"晋升"（promotion）到仓库根 tables/ 目录。
        这是有副作用的操作，必须显式 opt-in，防止测试或误操作污染仓库。

        测什么：
        - 环境变量完全清空 → project_writes_enabled() 为 False；
        - 设为 "0"、"false"、"true"、"yes" 等常见取值 → 仍然 False（只认精确的 "1"）；
        - 只有设为 "1" → 为 True。
        """
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(project_writes_enabled())  # 无环境变量 → 关闭
        for value in ["0", "false", "true", "yes"]:
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {"HPAT_WRITE_PROJECT_TABLES": value}, clear=True
            ):
                # 只认精确字符串 "1"，其它值一律视为关闭（不搞"真值表"混淆）
                self.assertFalse(project_writes_enabled())
        with mock.patch.dict(os.environ, {"HPAT_WRITE_PROJECT_TABLES": "1"}, clear=True):
            self.assertTrue(project_writes_enabled())  # 精确 "1" 才开启

    def test_paired_summary_rejects_project_table_target_without_opt_in(self) -> None:
        """验证"配对预测迁移汇总"脚本未显式 opt-in 时，拒绝写入工程汇总表。

        测什么：把 _common.PROJECT_TABLES_DIR mock 成临时目录，以
        --output 指向该目录下的 summary.csv 调用脚本 main()，但环境变量为空
        （未开启工程表写入）。预期：
        - 脚本以 SystemExit 退出，退出码 2（参数/权限校验失败）；
        - target 文件没有被创建（写表被拒绝，不留脏文件）。
        意义：没有显式允许，任何脚本都不得往仓库 tables/ 晋升结果。
        """
        import _common
        import summarize_paired_prediction_transitions

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            project_tables = root / "tables"
            project_tables.mkdir()
            target = project_tables / "summary.csv"
            with mock.patch.object(_common, "PROJECT_TABLES_DIR", project_tables), mock.patch.dict(
                os.environ, {}, clear=True  # 不设置 HPAT_WRITE_PROJECT_TABLES → 未 opt-in
            ), mock.patch.object(
                sys,
                "argv",  # 伪造命令行参数，指向工程表目录下的输出
                [
                    "summarize_paired_prediction_transitions.py",
                    "--input",
                    str(root / "unused.csv"),
                    "--output",
                    str(target),
                ],
            ), self.assertRaises(SystemExit) as raised:
                summarize_paired_prediction_transitions.main()
            self.assertEqual(raised.exception.code, 2)  # 退出码 2：拒绝写入
            self.assertFalse(target.exists())  # 关键断言：目标文件未被创建

    def test_evidence_report_respects_current_p0_boundaries(self) -> None:
        """验证证据强度报告（Markdown）如实反映当前 P0 门禁边界，不夸大。

        背景：P0-E1 边缘基线导入、P0-E4 活动溯源目前都被门禁拦截（blocked），
        只有 P0-E3 非理想性扫描达到 paper_eligible_with_limitations（论文可用带局限）。
        报告必须忠实呈现这一点，且措辞不得出现"Apple Silicon MPS 实测"这类越级说法
        （HPAT 光子芯片与桌面 MPS 实测无关）。

        测什么：喂入一个状态为 paper_eligible_with_limitations 的扫描清单，
        断言生成的 Markdown 报告：
        - 三行实验状态与门禁结论正确（blocked→false，eligible→true）；
        - 明确写着"unnormalized explicit-unit-cost local model"（未归一化的
          显式单位成本本地模型，如实说明能量口径）；
        - 不包含"timing is measured on Apple Silicon MPS"（不得暗示实测）。
        """
        from run_evidence_strength_summary import _md_report

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = pathlib.Path(temporary)
            (run_dir / "nonideality_accuracy_sweep_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "paper_eligible_with_limitations",
                        "injection_boundary": "reasonable-max",  # 注入边界：合理上限场景
                        "dataset": {"label_count": 1024},
                    }
                ),
                encoding="utf-8",
            )
            report = _md_report(run_dir, [], [], [])
            self.assertIn("| P0-E1 | blocked | false |", report)  # 边缘导入：拦截、不可声明
            self.assertIn("| P0-E3 | paper_eligible_with_limitations | true |", report)  # 非理想性扫描：可用带局限
            self.assertIn("| P0-E4 | blocked | false |", report)  # 活动溯源：拦截、不可声明
            self.assertIn("unnormalized explicit-unit-cost local model", report)  # 能量口径如实说明
            self.assertNotIn("timing is measured on Apple Silicon MPS", report)  # 不得暗示实测

    def test_canonical_profile_locks_experiment_protocol(self) -> None:
        """验证"规范配置"（canonical profile）把实验协议的所有关键参数钉死。

        背景：可复现实验的核心是"协议锁定"——论文配置里不允许出现可被随意改动、
        会影响数值结果的参数（如非理想性扫描的 trials/dimensions 等），
        并固定种子、精度、后端、采样数等一切要素。这个测试相当于一把"锁"：
        任何人改配置导致这些值变化，测试立刻失败。

        测什么：加载论文配置后逐项断言：
        - 全局种子 20260706、重复种子列表 = PAPER_REPEAT_SEEDS、精度 fp32；
        - 能量单位成本代理为 "8-bit nominal"（8 位标称），跨精度耦合"未建模"；
        - 配置里不允许出现 energy_envelope_mj（能量包络），也不允许非理想性配置
          含 trials/dimensions 等可调项；
        - 规范后端 = mps；映射场景/注入边界固定；样本数 1024、10 类；
        - 3 个权重文件的 git 修订号（40 位）与 sha256（64 位）长度正确；
        - 非理想性效应恰好是 4 种：高斯 PD/TIA 噪声、WDM 相邻串扰、MRR 工艺偏差、热漂移。
        """
        config = load_paper_config()
        repro = config["reproducibility"]
        self.assertEqual(config["seed"], 20260706)  # 全局随机种子被锁定
        self.assertEqual(config["repeat_seeds"], PAPER_REPEAT_SEEDS)  # 重复种子列表被锁定
        self.assertEqual(config["precision"], "fp32")  # 数值精度被锁定
        self.assertEqual(config["precision_scope"]["energy_unit_cost_proxy"], "8-bit nominal")  # 能量口径
        self.assertEqual(
            config["precision_scope"]["cross_precision_accuracy_energy_coupling"],
            "not modelled",  # 跨精度"精度-能量耦合"明确声明未建模
        )
        self.assertNotIn("energy_envelope_mj", config)  # 不允许出现能量包络字段
        # 非理想性配置不允许出现这些"可调项"——防止跑出非规范结果
        self.assertFalse(
            {"trials", "dimensions", "quant_bits", "insertion_loss_db", "detuning_pm"}
            & set(config["nonideality"])
        )
        self.assertEqual(repro["canonical_backend"], "mps")  # 规范后端锁死 MPS
        self.assertEqual(repro["mapping"]["scenario"], "reasonable_max")  # 映射场景锁死
        self.assertEqual(repro["mapping"]["injection_boundary"], "reasonable-max")  # 注入边界锁死
        self.assertEqual(repro["dataset"]["sample_count"], 1024)  # 样本数锁死
        self.assertEqual(len(repro["dataset"]["classes"]), 10)  # 类别数锁死
        self.assertEqual(len(repro["weights"]), 3)  # 权重文件数量
        self.assertTrue(all(len(row["revision"]) == 40 for row in repro["weights"]))  # git 修订号为 40 位哈希
        self.assertTrue(all(len(row["sha256"]) == 64 for row in repro["weights"]))  # sha256 为 64 位十六进制
        self.assertEqual(
            set(repro["nonideality_run"]["effects"]),  # 非理想性效应集合被锁死
            {
                "gaussian_pd_tia_noise",  # 高斯 PD/TIA 噪声
                "wdm_adjacent_crosstalk",  # WDM 相邻通道串扰
                "mrr_variation",  # MRR 微环谐振器工艺偏差
                "thermal_drift",  # 热漂移
            },
        )

    def test_stratified_1024_is_deterministic_and_uses_first_four_remainder(self) -> None:
        """验证 1024 样本分层采样：结果确定，且余数规则是"前 4 类各多 1 个"。

        背景：论文用 ImageNet 10 类各 120 张（共 1200 张）抽 1024 张。
        要保证"谁被抽中"只取决于类别与顺序，不取决于机器/库——这就是确定性。
        具体分配：每类基础 102 张 = 1020，剩余 4 张给前 4 类 → [103]*4 + [102]*6。

        测什么：
        - 正序输入与逆序输入两次采样结果完全相同（确定性，与输入顺序无关）；
        - 总数 1024、且路径无重复（一张图不会被抽两次）；
        - 每类数量 = [103,103,103,103,102,102,102,102,102,102]。
        """
        rows = []
        synsets = [f"n{index:08d}" for index in range(10)]
        for class_index, synset in enumerate(synsets):
            for sample_index in range(120):  # 每类 120 张候选
                rows.append(
                    {
                        "path": f"val/{synset}/image_{sample_index:03d}.JPEG",
                        "label": class_index,
                        "synset": synset,
                        "class_name": synset,
                        "image_sha256": f"{class_index:02d}{sample_index:04d}".ljust(64, "0"),
                    }
                )
        first = stratified_sample(rows, expected_synsets=synsets)
        second = stratified_sample(list(reversed(rows)), expected_synsets=list(reversed(synsets)))
        self.assertEqual(first, second)  # 关键断言：采样与输入顺序无关，确定性成立
        self.assertEqual(len(first), 1024)
        counts = {synset: sum(row["synset"] == synset for row in first) for synset in synsets}
        self.assertEqual([counts[synset] for synset in synsets], [103] * 4 + [102] * 6)  # 余数给前 4 类
        self.assertEqual(len({row["path"] for row in first}), 1024)  # 路径无重复

    def test_subset_csv_is_lf_only_and_byte_stable(self) -> None:
        """验证子集 CSV 用 LF 换行且字节稳定（同一数据两次写出完全一致）。

        背景：CSV 若在 Windows 上写成 CRLF、在 Linux 上写成 LF，哈希校验值就会不同，
        破坏跨平台可复现。因此子集 CSV 必须强制 LF，且"同数据→同字节"。

        测什么：用同一行数据写两次 CSV，断言：
        - 不含 CRLF 换行符（只允许 LF）；
        - 两次写入的字节完全相同；
        - sha256 摘要相同（哈希校验的跨平台稳定性由此而来）。
        """
        rows = [
            {
                "path": "val/n00000001/image.JPEG",
                "label": 1,
                "synset": "n00000001",
                "class_name": "fixture",
                "image_sha256": "a" * 64,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            first = root / "first.csv"
            second = root / "second.csv"
            write_subset_csv(first, rows)
            write_subset_csv(second, rows)
            self.assertNotIn(b"\r\n", first.read_bytes())  # 关键断言：无 CRLF，字节稳定
            self.assertEqual(first.read_bytes(), second.read_bytes())  # 同数据 → 同字节
            self.assertEqual(sha256_file(first), sha256_file(second))  # 哈希也一致

    def test_explicit_mps_never_silently_falls_back(self) -> None:
        """验证设备选择逻辑：显式请求 MPS 时绝不允许"静默回退"到别的设备。

        背景：可复现实验的数值结果与设备强相关（MPS 与 CPU 的浮点运算顺序不同）。
        若 MPS 不可用却偷偷用 CPU 跑，"结果可复现"就成了一句空话，
        所以 resolve_device 必须对"回退"报错，而不是悄悄降级。

        测什么（用 _FakeTorch 伪造各种设备状态）：
        - MPS 不可用 + 显式请求 "mps" → 抛 ReproducibilityError，提示 "Requested MPS"；
        - MPS 不可用 + 请求 "auto"（自动）→ 同样抛错，提示 "will not silently fall back"；
        - 请求 "cpu" → 正常返回 cpu，且 canonical_backend 为 False（非规范后端，不能用于论文结论）；
        - MPS 可用 + 请求 "auto" → 选中 mps，canonical_backend 为 True（满足规范要求）。
        """
        unavailable = _FakeTorch(mps_built=True, mps_available=False, cuda_available=False)  # MPS 不可用
        with self.assertRaisesRegex(ReproducibilityError, "Requested MPS"):
            resolve_device("mps", profile="paper", torch_module=unavailable)  # 显式请求 → 必须报错
        with self.assertRaisesRegex(ReproducibilityError, "will not silently fall back"):
            resolve_device("auto", profile="paper", torch_module=unavailable)  # 自动模式也不许静默回退
        comparison = resolve_device("cpu", profile="paper", torch_module=unavailable)  # CPU 是合法显式请求
        self.assertEqual(comparison.selected, "cpu")
        self.assertFalse(comparison.fallback_used)  # 没有发生回退
        self.assertFalse(comparison.canonical_backend)  # CPU 不是规范后端
        available = _FakeTorch(mps_built=True, mps_available=True, cuda_available=False)  # MPS 可用
        canonical = resolve_device("auto", profile="paper", torch_module=available)
        self.assertEqual(canonical.selected, "mps")  # 自动模式优先选中 MPS
        self.assertTrue(canonical.canonical_backend)  # 且满足规范后端要求

    def test_smoke_prepare_data_is_isolated_and_hashes_each_image(self) -> None:
        """验证 smoke 数据准备：完全隔离（不下载权重、不碰真实 ImageNet）且逐图哈希。

        测什么：调用 prepare_data(profile="smoke", download_weights=False)：
        - manifest 的 profile 是 "smoke"，样本数 3、类别数 3；
        - 图片标记为可再分发（images_redistributable，合成图无版权问题）；
        - subset.csv 表头含 image_sha256（每张图都有哈希校验值）；
        - 生成的合成图真实落盘（data/synthetic/class_0/sample_0.ppm）；
        - 不依赖公共 ImageNet 固定子集（public_fixed_imagenet_subset 未出现）；
        - 不创建 hf_home（未下载 HuggingFace 缓存/权重）。
        意义：保证 smoke 实验在任何干净环境都能就地取材、可复现。
        """
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "prepared"
            manifest = prepare_data(profile="smoke", output_dir=output, download_weights=False)
            self.assertEqual(manifest["profile"], "smoke")
            self.assertEqual(manifest["dataset"]["sample_count"], 3)  # smoke 只有 3 个样本
            self.assertEqual(manifest["dataset"]["class_count"], 3)  # 3 个类别
            self.assertTrue(manifest["dataset"]["images_redistributable"])  # 合成图可再分发
            subset = (output / "subset.csv").read_text(encoding="utf-8")
            self.assertIn("image_sha256", subset.splitlines()[0])  # 表头含图片哈希列
            self.assertTrue((output / "data" / "synthetic" / "class_0" / "sample_0.ppm").is_file())  # 合成图已落盘
            self.assertNotIn("public_fixed_imagenet_subset", subset)  # 不引用真实 ImageNet 子集
            self.assertFalse((output / "hf_home").exists())  # 未创建权重缓存目录

    def test_smoke_pipeline_uses_existing_scripts_without_project_writes(self) -> None:
        """验证 smoke 管线复用现有脚本且绝不写工程汇总表（隔离性）。

        背景：run_pipeline 统一入口即使父环境允许写工程表（HPAT_WRITE_PROJECT_TABLES=1），
        也必须在自己的运行目录内完成一切，工程表晋升只能走独立的 freeze/promote 边界。

        测什么：
        - 记录三个"可能被污染"的仓库级表文件运行前后的字节（before/after）并断言不变；
        - 在允许写工程表的环境变量下跑 run_pipeline（smoke、cpu）：
          * 清单 schema 正确、run_status=completed、后端=cpu、命令数=3；
          * 输出目录内生成 operator_mapping_closure.csv；
          * 每个阶段清单的 project_write_performed 都为 False，且 outputs 不以 tables/ 开头
            （说明结果全部落在运行目录内，没有晋升到仓库）；
        - verify_run 报告 valid=True 但 release_ready=False（smoke 级运行不能直接发布）。
        """
        watched = [
            REPO_ROOT / "tables" / "operator_mapping_closure.csv",
            REPO_ROOT / "tables" / "hpat_energy_by_component_reasonable_max_mapping.csv",
            REPO_ROOT / "tables" / "nonideality_accuracy_sweep.csv",
        ]
        before = {path: path.read_bytes() if path.exists() else None for path in watched}
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "run"
            # The unified run command is always isolated; promotion happens via
            # the separate freeze/promote boundary, even if the parent shell is
            # permissive.
            # 统一运行命令始终隔离；即使父 shell 允许，晋升也只能走独立的冻结/晋升边界
            with mock.patch.dict(os.environ, {"HPAT_WRITE_PROJECT_TABLES": "1"}):
                manifest = run_pipeline(
                    profile="smoke",
                    device_request="cpu",
                    output_dir=output,
                    use_caffeinate=False,
                )
            self.assertEqual(manifest["schema_version"], REPRO_MANIFEST_SCHEMA_VERSION)
            self.assertEqual(manifest["run_status"], "completed")  # 运行正常完成
            self.assertEqual(manifest["backend"]["selected"], "cpu")
            self.assertEqual(len(manifest["commands"]), 3)  # 三个子脚本（映射/活动/非理想性）
            self.assertTrue((output / "tables" / "operator_mapping_closure.csv").is_file())  # 产物在运行目录内
            for manifest_name in [
                "operator_mapping_closure_manifest.json",
                "e_local_trace_driven_activity_manifest.json",
                "nonideality_accuracy_sweep_manifest.json",
            ]:
                stage_manifest = json.loads((output / manifest_name).read_text(encoding="utf-8"))
                # 关键断言：阶段脚本没有执行工程表写入，产物也不指向仓库 tables/
                self.assertFalse(stage_manifest["project_write_performed"])
                self.assertTrue(
                    all(not str(path).startswith("tables/") for path in stage_manifest["outputs"]),
                    stage_manifest["outputs"],
                )
            report = verify_run(output)
            self.assertTrue(report["valid"], report["errors"])  # 运行本身有效
            self.assertFalse(report["release_ready"])  # smoke 级结果不能直接发布
        after = {path: path.read_bytes() if path.exists() else None for path in watched}
        self.assertEqual(before, after)  # 关键断言：仓库级表文件前后字节完全一致，未被污染

    def test_verify_detects_output_tampering(self) -> None:
        """验证校验器能发现"产物被篡改"：哈希校验（sha256）不匹配即判无效。

        测什么：跑一次 smoke 管线后，往 operator_mapping_closure.csv 末尾追加一行
        "tampered\n"（模拟有人/程序改动产物），再 verify_run：
        - report["valid"] 必须为 False；
        - 错误信息中必须出现 "SHA-256 mismatch"（哈希不匹配）。
        意义：可复现包必须能证明"没被改过"，篡改检测是防伪的底线。
        """
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "run"
            run_pipeline(
                profile="smoke",
                device_request="cpu",
                output_dir=output,
                use_caffeinate=False,
            )
            target = output / "tables" / "operator_mapping_closure.csv"
            target.write_text(target.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")  # 模拟篡改
            report = verify_run(output)
            self.assertFalse(report["valid"])  # 关键断言：校验失败
            self.assertTrue(any("SHA-256 mismatch" in error for error in report["errors"]))  # 报错是哈希不匹配

    def test_canonical_verify_and_freeze_archive_are_deterministic(self) -> None:
        """验证规范校验、参考比对与冻结归档都是确定性的（端到端综合测试）。

        测什么（一次性覆盖完整可复现链）：
        1) 手工构造一个规范的运行目录（含配置快照、诊断表、被排除的 logits、
           带规范字段的扫描清单），补全 repro_manifest.json 并写 canonical_outputs；
        2) verify_run → valid 且 release_ready=True（该运行具备发布资格）；
        3) 构造 reference_manifest（参考清单）并 compare_run_to_reference → match=True；
           把参考清单里的规范输出 sha256 篡改成全 0 → match=False 且 status="Red"；
        4) freeze_run 冻结两次（first/second）→ 两份发布归档的 sha256 完全相同
           （字节级确定性）；归档内含 SHA256SUMS（校验和清单）；
           归档路径为 hpat-artifact-v1.0.1.tar.gz；
           归档成员里不得出现图片/权重/原始 logits/论文材料
           （jpg/jpeg/png/safetensors/npz/pdf/tex 后缀全禁止）。
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            run_dir = root / "run"
            (run_dir / "tables").mkdir(parents=True)
            (run_dir / "raw").mkdir()
            shutil.copyfile(PAPER_CONFIG, run_dir / "config_snapshot.json")  # 配置快照 = 论文配置
            (run_dir / "tables" / "diagnostic.csv").write_text("metric,value\nfixture,1\n", encoding="utf-8")
            (run_dir / "raw" / "diagnostic_logits.npz").write_bytes(b"excluded")  # 应被排除的中间产物
            write_json(
                run_dir / "nonideality_accuracy_sweep_manifest.json",
                {
                    "status": PAPER_ARTIFACT_STATUS,
                    "evidence_tier": PAPER_EVIDENCE_TIER,
                    "claim_boundary": CLAIM_BOUNDARY,
                },
            )
            distribution = {  # 10 类的样本分布：前 4 类 103、后 6 类 102
                f"n{index:08d}": 103 if index < 4 else 102 for index in range(10)
            }
            manifest = {
                "schema_version": REPRO_MANIFEST_SCHEMA_VERSION,
                "created_utc": "2026-07-17T00:00:00+00:00",  # 固定时间戳，保证确定性
                "profile": "paper",
                "run_status": "completed",
                "error": "",
                "artifact_status": PAPER_ARTIFACT_STATUS,
                "git": {"commit": "0" * 40},  # 占位 git 提交号
                "config": {
                    "path": "config_snapshot.json",
                    "sha256": sha256_file(run_dir / "config_snapshot.json"),  # 配置快照哈希
                    "source_path": "experiments/config/paper_v1.json",
                    "source_sha256": sha256_file(PAPER_CONFIG),  # 源配置哈希
                    "schema_version": "hpat-paper-config-v1",
                },
                "data": {
                    "sample_count": 1024,
                    "class_count": 10,
                    "class_distribution": distribution,
                },
                "weights": [],
                "backend": {
                    "requested": "mps",
                    "selected": "mps",
                    "reason": "fixture",
                    "fallback_used": False,  # 没有回退
                    "canonical_backend": True,  # 规范后端
                },
                "runtime": {"python": "3.12.13", "packages": {}},
                "randomness": {"repeat_seeds": PAPER_REPEAT_SEEDS},
                "numeric": {"precision": "fp32", "synchronization": "torch.mps.synchronize"},
                "mapping": {"scenario": "reasonable_max", "injection_boundary": "reasonable-max"},
                "commands": [{"script": "fixture", "returncode": 0}],
                "outputs": _output_records(run_dir),
                "canonical_outputs": {},  # 占位，稍后用 _canonical_output_record 填充
                "evidence": {"tier": PAPER_EVIDENCE_TIER, "claim_boundary": CLAIM_BOUNDARY},
            }
            from hpat_eval.reproducibility import _canonical_output_record

            manifest["canonical_outputs"] = _canonical_output_record(run_dir, manifest["outputs"])
            write_json(run_dir / "repro_manifest.json", manifest)
            report = verify_run(run_dir)
            self.assertTrue(report["valid"], report["errors"])  # 运行完整有效
            self.assertTrue(report["release_ready"], report["errors"])  # 具备发布资格

            reference_manifest = json.loads(json.dumps(manifest))  # 深拷贝，避免改到原清单
            reference_manifest["reference_comparison"] = {
                "schema_version": REFERENCE_COMPARISON_SCHEMA_VERSION,
                "canonical_outputs": manifest["canonical_outputs"],  # 参考归档应含同样的规范输出
            }
            reference_path = root / "reference_manifest.json"
            write_json(reference_path, reference_manifest)
            comparison = compare_run_to_reference(run_dir, reference_path)
            self.assertTrue(comparison["match"], comparison["errors"])  # 与参考归档一致

            # 篡改参考归档的规范输出哈希 → 比对必须失败并标红
            reference_manifest["reference_comparison"]["canonical_outputs"]["sha256"] = "0" * 64
            write_json(reference_path, reference_manifest)
            mismatch = compare_run_to_reference(run_dir, reference_path)
            self.assertFalse(mismatch["match"])  # 哈希不一致 → 不匹配
            self.assertEqual(mismatch["status"], "Red")

            # 冻结两次：两份归档哈希必须完全相同（字节级确定性）
            first = root / "freeze-first"
            second = root / "freeze-second"
            first_manifest = freeze_run(run_dir=run_dir, output_dir=first)
            second_manifest = freeze_run(run_dir=run_dir, output_dir=second)
            self.assertEqual(
                first_manifest["release_archive"]["sha256"],  # 关键断言：两次冻结哈希一致
                second_manifest["release_archive"]["sha256"],
            )
            self.assertTrue((first / "SHA256SUMS").is_file())  # 归档含校验和清单
            self.assertEqual(first_manifest["release_archive"]["path"], "hpat-artifact-v1.0.1.tar.gz")
            # 发布归档成员不允许出现图片/权重/原始 logits/论文材料
            self.assertFalse(
                any(
                    path.lower().endswith((".jpg", ".jpeg", ".png", ".safetensors", ".npz", ".pdf", ".tex"))
                    for path in first_manifest["release_archive"]["members"]
                )
            )


if __name__ == "__main__":
    unittest.main()
