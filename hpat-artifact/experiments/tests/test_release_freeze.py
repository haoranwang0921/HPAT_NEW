"""
发布冻结包（release freeze）测试：验证论文发布用"冻结包"是否完整且可复现。

项目背景：hpat-artifact 是《HPAT：光子张量处理器》（ASP-DAC 2027 投稿）的可复现
实验库。HPAT 是光子计算架构（96 Core、约 39 万个 MRR 微环谐振器做光矩阵乘法）。
论文发布时会打一个"冻结包"（freeze release）：把实验产物（清单 manifest、
数据表、配置快照）固化成一版不可篡改的发布归档，供审稿人/读者复现。

本文件围绕两个核心机制：
1) freeze_release：冻结发布。把一次实验运行固化成发布目录，产出：
   - 各类报告（REPORT_NAMES：清单、校验和 checksums、表格审计等）；
   - 发布归档 tar.gz（内含 repro_manifest.json 可复现清单）；
   - 六道门禁（gate）：canonical_run（规范运行）、reference_package（参考包）、
     table_schema（表结构）、claim_lint（声明规范）、anonymity_and_size
     （匿名与体积）、repository_binding（仓库绑定）。
   任一关键门禁不过 → 发布为 Red（红灯，禁止发布）。
2) audit_public_release：公开发布审计。发布物必须匿名（不能泄露作者主目录路径、
   邮箱等身份信息）、无论文正文/图表（tex/pdf/png 等"发表材料"不允许混入）、
   文件大小与清单（manifest）受控，防止把审稿人看不到的私有内容发出去。

所有测试都用临时目录构造最小"论文运行现场"（fixture），再断言冻结/审计结果，
确保测试可在任何干净机器上复现。
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import sys
import tarfile
import tempfile
import unittest


# 仓库根目录 = 本文件向上两级；SCRIPTS 是实验脚本目录，加入 sys.path 以便导入冻结/审计脚本
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "experiments" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# noqa: E402：这些导入必须放在 sys.path 调整之后，故跳过"导入未放顶部"的 lint 告警
from audit_public_release import audit_file_set, audit_public_release  # noqa: E402
from freeze_repro_release import REPORT_NAMES, freeze_release  # noqa: E402
# 可复现性模块：清单协议版本、规范输出记录、sha256 文件摘要、参考归档校验
from hpat_eval.reproducibility import (  # noqa: E402
    REPRO_MANIFEST_SCHEMA_VERSION,
    canonical_output_record,
    sha256_file,
    verify_reference_archive,
)


def _write_json(path: pathlib.Path, payload: object) -> None:
    """把对象以 JSON 写入文件（自动建父目录、缩进排序、UTF-8）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: pathlib.Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    """把表头 + 若干行字典以 CSV 格式写入文件（自动建父目录、统一换行符）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class ReleaseFreezeTests(unittest.TestCase):
    """发布冻结与公开审计测试组。

    每个测试先在临时目录构造一个最小但完整的"论文运行现场"（fixture），
    再调用真实的 freeze_release / audit_public_release，断言发布状态
    （Green=可发布 / Red=不可发布）与报告内容。
    """

    def _fixture(self, root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
        """构造最小"论文运行现场"：返回 (run_dir 运行目录, spec 发布规格文件)。

        现场内容包括：
        - runs/paper_v1/：一次实验运行的产物，如输入子集 subset.csv、配置快照
          config_snapshot.json、P0 就绪度汇总、可复现清单 repro_manifest.json 等；
        - tables/mapping.csv：模型与"声明边界"（claim boundary，说明本数据只能
          做本地/建模诊断，不能当作硬件实测）的映射表；
        - experiments/config/paper_v1.json + release_spec_v1.json：发布规格，
          声明所需产物分组、表结构、证据等级与封禁证据通道等。

        注意 raw/ 下特意放一个 nonideality_logits_clean.npz（原始 logits），
        用于验证"原始大文件/中间产物不得进入发布包"。
        """
        run_dir = root / "runs" / "paper_v1"
        run_dir.mkdir(parents=True)
        table = root / "tables" / "mapping.csv"
        _write_csv(
            table,
            ["model", "claim_boundary"],
            [{"model": "MobileViT-XXS", "claim_boundary": "Modelled bound; not measured hardware speedup."}],
        )

        # 构造 10 个 ImageNet 类别的子集清单：前 4 类各 103 张、其余各 102 张（共 1024 张）
        sample_rows: list[dict[str, object]] = []
        synsets = [f"n{index:08d}" for index in range(10)]  # ImageNet synset 编号，如 n00000000
        for class_index, synset in enumerate(synsets):
            count = 103 if class_index < 4 else 102
            for index in range(count):
                sample_rows.append(
                    {
                        "path": f"val/{synset}/{index:04d}.jpg",
                        "label": class_index,
                        "synset": synset,
                        "class_name": f"class-{class_index}",
                        # 每张图片用哈希伪造一个"图片摘要"，模拟真实数据集的防篡改标识
                        "image_sha256": hashlib.sha256(f"{synset}-{index}".encode()).hexdigest(),
                    }
                )
        _write_csv(
            run_dir / "inputs" / "subset.csv",
            ["path", "label", "synset", "class_name", "image_sha256"],
            sample_rows,
        )

        # 配置快照：记录本次运行启用的非理想性效应（MRR 噪声、串扰、工艺偏差、热漂移）
        effects = ["noise", "crosstalk", "mrr", "thermal"]
        _write_json(
            run_dir / "config_snapshot.json",
            {"reproducibility": {"nonideality": {"effects": effects}}},
        )
        _write_json(
            run_dir / "tables" / "p0_readiness_summary.json",
            {
                "schema_version": "p0-readiness-v1",
                "overall_g1_ready": False,  # G1 关卡未就绪（无 P1 实测证据）
                # 四个 P0 实验的状态：E1/E2/E4 被拦截，只有 E3 达到"论文可用"等级
                "experiments": [
                    {"experiment": "P0-E1", "status": "blocked"},
                    {"experiment": "P0-E2", "status": "blocked"},
                    {"experiment": "P0-E3", "status": "paper_eligible_with_limitations"},
                    {"experiment": "P0-E4", "status": "blocked"},
                ],
            },
        )
        _write_json(
            root / "experiments" / "config" / "paper_v1.json",
            {"schema_version": "fixture-experiment-v1"},
        )

        spec = root / "experiments" / "config" / "release_spec_v1.json"
        _write_json(
            spec,
            {
                "release_version": "v1.0.0",
                "require_git_commit": False,  # 测试现场无 git 提交，故不强制
                "canonical_profile": "paper_v1",  # 规范配置（canonical profile）名
                "canonical_manifest_names": ["repro_manifest.json"],  # 权威可复现清单名
                "canonical": {
                    "device": "mps",
                    "allow_device_fallback": False,  # 不允许设备回退（保证可复现）
                    "sample_count": 1024,
                    "repeat_seeds": [1, 2, 3, 4, 5],  # 多次随机种子，平均结果
                    "effects": effects,
                    "evidence_tier": "paper_eligible_with_limitations",
                },
                # 必须提供的产物分组及其校验模式（all=全要，any=有其一即可）
                "required_artifact_groups": {
                    "mapping": ["mapping.csv"],
                    "sample_manifest": ["subset.csv"],
                    "p0_gate": ["p0_readiness_summary.json"],
                },
                "required_group_modes": {
                    "mapping": "all",
                    "sample_manifest": "any",
                    "p0_gate": "all",
                },
                # 每张表的期望列结构（table schema），冻结时会逐表校验
                "table_schemas": {
                    "mapping.csv": ["model", "claim_boundary"],
                    "subset.csv": ["path", "label", "synset", "class_name", "image_sha256"],
                },
                # 声明边界：本数据只能做本地/建模诊断，不是实测硅片或部署证据
                "claim_boundary": "Local/modelled diagnostic only; not measured silicon or deployment evidence.",
                # 封禁的证据通道：P0-E1/E2 不可用于支撑更高等级声明
                "blocked_evidence_lanes": ["P0-E1", "P0-E2"],
            },
        )

        # 原始运行产物：故意放一个 logits 文件，验证它会被排除出发布包
        raw = run_dir / "raw"
        raw.mkdir()
        (raw / "nonideality_logits_clean.npz").write_bytes(b"excluded-logits")
        (raw / "nonideality_prediction_changes.csv").write_text(
            "changed\n0\n", encoding="utf-8", newline="\n"
        )
        # 汇总运行目录内所有文件的相对路径 + sha256 摘要 + 字节数，作为清单的 outputs
        outputs = [
            {
                "path": path.relative_to(run_dir).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(run_dir.rglob("*"))
            if path.is_file() and path.name != "repro_manifest.json"  # 清单自身不引用自己
        ]
        _write_json(
            run_dir / "repro_manifest.json",
            {
                "schema_version": REPRO_MANIFEST_SCHEMA_VERSION,
                "profile": "paper_v1",
                "run_status": "completed",
                "artifact_status": "paper_eligible_with_limitations",
                "git": {"commit": "0" * 40},  # 占位 commit（40 个 0）
                "config": {
                    "path": "config_snapshot.json",
                    "sha256": sha256_file(run_dir / "config_snapshot.json"),  # 配置快照的哈希
                },
                "data": {"sample_count": 1024},
                "weights": [],
                "backend": {
                    "requested": "mps",
                    "selected": "mps",
                    "reason": "fixture",
                    "fallback_used": False,  # 没有发生设备回退
                },
                "runtime": {"python": "3.12.13"},
                "randomness": {"repeat_seeds": [1, 2, 3, 4, 5]},
                "numeric": {"precision": "fp32"},
                "mapping": {"scenario": "reasonable_max"},
                "commands": [{"script": "fixture", "returncode": 0}],
                "outputs": outputs,
                "canonical_outputs": canonical_output_record(outputs),  # 规范输出记录（用于与参考归档比对）
                "evidence": {"tier": "local-modelled diagnostic"},
            },
        )
        return run_dir, spec

    def test_green_freeze_emits_experiment_only_release(self) -> None:
        """验证"全绿"冻结发布：产出仅含实验产物的发布包（不含论文材料/原始 logits）。

        测什么（这是最核心的端到端用例）：
        1) 用合法 fixture 跑 freeze_release，状态应为 Green（绿灯，可发布）；
        2) 所有 REPORT_NAMES 报告文件（清单、校验和、表格审计等）都已生成；
        3) 生成参考归档 tar.gz，且 verify_reference_archive 校验为 Green（归档完整）；
        4) 发布目录里不得出现论文材料（pdf/tex/png/npz）与任何"logits"中间产物
           ——发布包只含实验产物（experiment-only）；
        5) 归档内的 repro_manifest.json 中，raw/nonideality_logits_clean.npz 必须被
           移到 reference_comparison.excluded_outputs（排除清单），不能出现在 outputs；
        6) 校验和文件 checksums 用 LF 换行（跨平台字节一致），并包含 release_manifest.json；
        7) 门禁集合恰好为六道（canonical_run/reference_package/table_schema/claim_lint/
           anonymity_and_size/repository_binding）。

        背景：哈希校验（对每个文件计算 sha256）保证发布包内容可逐字节复现与核对。
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            run_dir, spec = self._fixture(root)
            output = root / "artifacts" / "reference" / "v1.0.0"
            result = freeze_release(run_dir, output, spec_path=spec, repo_root=root)
            self.assertEqual(result["status"], "Green")  # 冻结结果：绿灯可发布
            for name in REPORT_NAMES.values():
                self.assertTrue((output / name).is_file(), name)  # 每份报告都真实落盘

            archive = output / "release_assets" / "hpat-artifact-v1.0.0.tar.gz"
            self.assertTrue(archive.is_file())
            # 参考归档（reference archive）也要通过可复现性校验
            self.assertEqual(verify_reference_archive(archive)["status"], "Green")
            # 发布包不得包含论文材料（pdf/tex/png/npz）
            self.assertFalse(any(path.suffix in {".pdf", ".tex", ".npz", ".png"} for path in output.rglob("*")))
            # 发布包不得包含任何 logits 中间产物
            self.assertFalse(any("logits" in path.name for path in output.rglob("*")))

            with tarfile.open(archive, "r:gz") as handle:
                packaged_manifest = json.load(handle.extractfile("hpat-artifact-v1.0.0/repro_manifest.json"))
            # 清单 outputs 中不应再出现被排除的原始 logits 文件
            packaged_paths = {record["path"] for record in packaged_manifest["outputs"]}
            self.assertNotIn("raw/nonideality_logits_clean.npz", packaged_paths)
            # 它应被记录进"参考比对时排除的输出"清单
            excluded_paths = {
                record["path"]
                for record in packaged_manifest["reference_comparison"]["excluded_outputs"]
            }
            self.assertIn("raw/nonideality_logits_clean.npz", excluded_paths)

            manifest = json.loads((output / REPORT_NAMES["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "Green")
            self.assertEqual(manifest["canonical_profile"], "paper_v1")  # 规范配置与现场一致
            self.assertEqual(manifest["blocked_evidence_lanes"], ["P0-E1", "P0-E2"])  # 封禁通道被如实记录
            self.assertEqual(
                set(result["gate_summary"]["gates"]),  # 六道门禁一个都不能少
                {
                    "canonical_run",  # 规范运行门禁
                    "reference_package",  # 参考归档门禁
                    "table_schema",  # 表结构门禁
                    "claim_lint",  # 声明规范门禁
                    "anonymity_and_size",  # 匿名与体积门禁
                    "repository_binding",  # 仓库绑定门禁
                },
            )
            # 校验和文件必须用 LF 换行（不用 CRLF），保证任意平台字节一致；且包含清单自身
            checksums = (output / REPORT_NAMES["checksums"]).read_bytes()
            self.assertNotIn(b"\r\n", checksums)
            self.assertIn(b"release_manifest.json", checksums)

    def test_freeze_is_byte_deterministic(self) -> None:
        """验证冻结发布是"字节级确定"的：同一现场冻结两次，产物逐字节相同。

        背景：可复现性要求"同一输入 → 同一输出"。若两次冻结产出不同字节，
        说明有文件顺序、时间戳或随机性泄漏，破坏可复现承诺。

        测什么：对同一 fixture 分别冻结到 first/ 和 second/，把两个目录中每个
        文件的相对路径 → 字节内容 收集成字典，断言两个字典完全相等。
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            run_dir, spec = self._fixture(root)
            first = root / "first"
            second = root / "second"
            freeze_release(run_dir, first, spec_path=spec, repo_root=root)
            freeze_release(run_dir, second, spec_path=spec, repo_root=root)
            # 收集 {相对路径: 原始字节}，便于整体比对
            first_files = {
                path.relative_to(first).as_posix(): path.read_bytes()
                for path in first.rglob("*")
                if path.is_file()
            }
            second_files = {
                path.relative_to(second).as_posix(): path.read_bytes()
                for path in second.rglob("*")
                if path.is_file()
            }
            self.assertEqual(first_files, second_files)  # 关键断言：两次发布逐字节一致

    def test_p0_json_requires_current_canonical_lane_statuses(self) -> None:
        """验证 P0 就绪度 JSON 中的实验状态必须符合"规范通道状态"：不符则红灯。

        测什么：把 fixture 里 P0-E3 的状态从"论文可用"改成"smoke"（冒烟），
        再冻结。表结构门禁应把该表判为有问题，发布状态为 Red（红灯）。
        关键断言：audit 表中该行的 issues 含 "p0_status_mismatch"（P0 状态不匹配）。
        意义：防止论文发布时引用了过期/降级的证据状态。
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            run_dir, spec = self._fixture(root)
            p0_path = run_dir / "tables" / "p0_readiness_summary.json"
            payload = json.loads(p0_path.read_text(encoding="utf-8"))
            payload["experiments"][2]["status"] = "smoke"  # 篡改状态：模拟"过期证据"
            _write_json(p0_path, payload)

            output = root / "freeze"
            result = freeze_release(run_dir, output, spec_path=spec, repo_root=root)
            self.assertEqual(result["status"], "Red")  # 证据状态不符 → 发布红灯
            audit = json.loads((output / REPORT_NAMES["tables"]).read_text(encoding="utf-8"))
            p0_row = next(row for row in audit["tables"] if row["table"] == p0_path.name)
            self.assertIn("p0_status_mismatch", p0_row["issues"])  # 审计明确标出状态不匹配

    def test_invalid_p0_json_keeps_table_gate_red(self) -> None:
        """验证 P0 就绪度 JSON 损坏（非法 JSON）时表结构门禁保持红灯。

        测什么：把 p0_readiness_summary.json 写成非 JSON 文本，再冻结。
        预期：状态 Red，审计表中该行 issues 含 "invalid_p0_readiness_json"。
        意义：确保"读不了的文件"不会被静默跳过，防止带病发布。
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            run_dir, spec = self._fixture(root)
            p0_path = run_dir / "tables" / "p0_readiness_summary.json"
            p0_path.write_text("not-json\n", encoding="utf-8")  # 故意写入非法 JSON

            output = root / "freeze"
            result = freeze_release(run_dir, output, spec_path=spec, repo_root=root)
            self.assertEqual(result["status"], "Red")
            audit = json.loads((output / REPORT_NAMES["tables"]).read_text(encoding="utf-8"))
            p0_row = next(row for row in audit["tables"] if row["table"] == p0_path.name)
            self.assertIn("invalid_p0_readiness_json", p0_row["issues"])  # 明确标记"非法 JSON"

    def test_missing_required_schema_keeps_gate_red(self) -> None:
        """验证发布规格里要求了"不存在的表列"时，表结构门禁保持红灯。

        测什么：在 spec 的 table_schemas.mapping.csv 里多加一个"不存在"的列
        missing_column，再冻结。mapping.csv 实际不含该列，表结构校验应失败。
        预期：发布状态 Red。
        意义：发布规格与真实数据不一致时必须拦下，不能带病发布。
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            run_dir, spec = self._fixture(root)
            payload = json.loads(spec.read_text(encoding="utf-8"))
            payload["table_schemas"]["mapping.csv"].append("missing_column")  # 要求一个不存在的列
            _write_json(spec, payload)
            output = root / "freeze"
            result = freeze_release(run_dir, output, spec_path=spec, repo_root=root)
            self.assertEqual(result["status"], "Red")  # 表结构不匹配 → 红灯

    def test_public_allowlist_size_and_identity_are_blocking(self) -> None:
        """验证公开审计的三类阻断问题：不在允许清单、文件超限、身份泄露。

        测什么：构造一个含 README.md（8 字节）和 secret.txt（含 /home/alice/private
        这样的绝对路径）的目录，调用 audit_public_release：
        - allowed_patterns 只放行 README.md → secret.txt 触发 not_in_public_allowlist；
        - max_file_bytes=8 → README.md 有 8 字节，恰好触发 file_exceeds_limit
          （文件超过大小上限，防止发布包过大）；
        - secret.txt 里出现 /home/alice/private 绝对路径 → 触发
          identity_leak:absolute_user_path（绝对用户路径泄露作者身份）。

        预期：report.status == "Red"，三种 code 全部出现在 blockers 中。
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / "README.md").write_text("anonymous artifact\n", encoding="utf-8")
            fixture_home = "/" + "home" + "/alice/private\n"  # 伪造一个作者主目录绝对路径
            (root / "secret.txt").write_text(fixture_home, encoding="utf-8")
            report = audit_public_release(root, max_file_bytes=8, allowed_patterns=["README.md"])
            codes = {row["code"] for row in report["blockers"]}  # 收集所有阻断原因编码
            self.assertEqual(report["status"], "Red")
            self.assertIn("not_in_public_allowlist", codes)  # 文件不在公开发布允许清单
            self.assertIn("file_exceeds_limit", codes)  # 文件超过大小上限
            self.assertIn("identity_leak:absolute_user_path", codes)  # 绝对路径泄露身份

    def test_public_audit_rejects_publication_material_independently_of_allowlist(self) -> None:
        """验证"发表材料"（论文源文件/PDF/成图）即使被加白名单也会被拒绝。

        背景：公开发布只允许放"实验产物"，论文本身（submission.tex、artifact.pdf、
        figures/final/figure.png）属于 publication material（发表材料），
        即便 allowed_patterns 显式允许这些路径，也必须被拦下。

        测什么：构造 tex/pdf/png 三个文件并全部加入 allowed_patterns，
        仍断言三个阻断码同时出现：forbidden_directory（禁止目录）、
        forbidden_publication_file（禁止的发表文件）、forbidden_publication_path
        （禁止的发表路径）。
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            tex = root / "paper_src" / "submission.tex"
            pdf = root / "artifact.pdf"
            figure = root / "figures" / "final" / "figure.png"
            tex.parent.mkdir(parents=True)
            figure.parent.mkdir(parents=True)
            tex.write_text("fixture\n", encoding="utf-8")
            pdf.write_bytes(b"%PDF-fixture")
            figure.write_bytes(b"fixture")
            report = audit_public_release(
                root,
                allowed_patterns=["paper_src/*", "*.pdf", "figures/final/*"],  # 即使允许也拒绝
            )
            codes = {row["code"] for row in report["blockers"]}
            self.assertIn("forbidden_directory", codes)  # 论文源码目录被禁止
            self.assertIn("forbidden_publication_file", codes)  # PDF 被禁止
            self.assertIn("forbidden_publication_path", codes)  # 成图路径被禁止

    def test_public_audit_requires_declared_files(self) -> None:
        """验证公开审计要求"声明过的必需文件"必须真实存在。

        测什么：allowed_patterns 与 required_files 都声明 README.md 和
        gate_summary.json，但目录里只有 README.md。审计应报
        missing_required_public_file（缺少必需的公开文件），状态 Red。
        意义：保证公开发布永远包含门禁摘要等核心文件，不会缺东少西。
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / "README.md").write_text("anonymous artifact\n", encoding="utf-8")
            report = audit_public_release(
                root,
                allowed_patterns=["README.md", "gate_summary.json"],
                required_files=["README.md", "gate_summary.json"],  # 声明必需文件
            )
            codes = {row["code"] for row in report["blockers"]}
            self.assertIn("missing_required_public_file", codes)  # 缺文件 → 阻断

    def test_public_audit_rejects_binary_provenance_and_email_metadata(self) -> None:
        """验证二进制文件里的"来源元数据"与"邮箱地址"也会触发身份泄露阻断。

        测什么：造一个伪装成 PNG 的二进制文件，里面嵌入 C2PA 签名来源标记
        （urn:c2pa:... 是数字内容来源元数据标准）和 example.org 邮箱。
        审计应同时报 embedded_provenance_metadata（嵌入来源元数据）与
        identity_leak:email_address（邮箱地址泄露）。
        意义：审稿人可能从二进制元数据里挖出作者身份，必须提前拦截。
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            image = root / "figure.png"
            # PNG 魔数 + 伪造的 C2PA 来源标记 + 邮箱，模拟真实图片里残留的作者元数据
            image.write_bytes(
                b"\x89PNG\r\n\x1a\nurn:c2pa:fixture signer" + b"@" + b"example.org"
            )
            report = audit_public_release(root, allowed_patterns=["figure.png"])
            codes = {row["code"] for row in report["blockers"]}
            self.assertIn("embedded_provenance_metadata", codes)
            self.assertIn("identity_leak:email_address", codes)

    def test_public_auditor_source_does_not_flag_redaction_literals(self) -> None:
        """验证审计器自己的源码不会误报"脱敏字面量"为身份泄露。

        背景：audit_public_release.py 的源码里必然含有正则/示例邮箱、路径等
        "脱敏字面量"（redaction literal，用于检测的占位文本），
        不应被当作真实泄露误报。

        测什么：把审计器源码本身作为被审计文件跑 audit_file_set，断言所有
        identity_leak:* 阻断码为空列表。
        意义：保证审计器不会"自己举报自己"，避免误报打断 CI。
        """
        source = SCRIPTS / "audit_public_release.py"
        report = audit_file_set(
            [(source, "experiments/scripts/audit_public_release.py")],
            allowed_patterns=["experiments/scripts/*.py"],
        )
        identity_codes = [
            row["code"] for row in report["blockers"] if row["code"].startswith("identity_leak:")
        ]
        self.assertEqual(identity_codes, [])  # 没有身份泄露类误报


if __name__ == "__main__":
    unittest.main()
