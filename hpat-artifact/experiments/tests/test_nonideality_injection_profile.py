# 模块 docstring：
# 本测试文件属于 hpat-artifact（《HPAT：光子张量处理器》论文的可复现实验库）。
# 它验证"非理想性注入"（nonideality injection，即模拟光子硬件不完美带来的误差）脚本
# run_nonideality_accuracy_sweep.py 的内部工具函数是否符合预期，包括：
#   - 随机种子（seed）如何分配，保证同一严重度点可复现；
#   - 如何挑选注入效应（effect）组合、如何把嵌套模块名折叠到最外层；
#   - 三种典型误差注入函数（WDM 串扰、MRR 工艺偏差、热漂移）的数学行为。
# 说明：这些是研究光子误差如何影响精度的基础工具，测试它们能保证后续精度扫描实验可信。
from __future__ import annotations

import importlib.util
import pathlib
import random
import sys
import unittest


# 定位仓库根目录与脚本目录，并把 scripts 目录加进模块搜索路径，方便下方动态加载脚本。
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "experiments" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# 动态加载非理想性精度扫描脚本（同 test_export_cli.py 的方式），测试直接复用脚本里的内部函数。
SPEC = importlib.util.spec_from_file_location(
    "run_nonideality_accuracy_sweep",
    SCRIPTS / "run_nonideality_accuracy_sweep.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NonidealityInjectionProfileTests(unittest.TestCase):
    # 测试类：对非理想性注入脚本内部工具的单元测试（不跑完整实验，只测核心函数）。
    def test_severity_points_share_one_trial_seed(self) -> None:
        # 测什么：同一个"严重度点 + 同一个注入效应"应共用同一个随机种子（保证可复现），
        #         而不同注入效应之间不应共用种子（避免互相干扰）。
        # 怎么测：用相同参数（trial=17, MobileViT-XXS, mrr_variation）调用两次 _trial_torch_seed，
        #         再用相同 trial 但不同效应（thermal_drift）调用一次，比较结果。
        # 预期结果：前两次相等，与第三次不等。
        first = MODULE._trial_torch_seed(17, "MobileViT-XXS", "mrr_variation")
        second = MODULE._trial_torch_seed(17, "MobileViT-XXS", "mrr_variation")
        other_effect = MODULE._trial_torch_seed(17, "MobileViT-XXS", "thermal_drift")
        self.assertEqual(first, second)  # 同效应重复调用必须得到相同种子，确保实验可复现
        self.assertNotEqual(first, other_effect)  # 不同效应应使用不同种子，避免效应之间种子串扰

    def test_effect_filter_keeps_requested_panels_only(self) -> None:
        # 测什么：扫描计划 _selected_sweep_plan 应只保留请求的注入效应（面板）。
        # 怎么测：用空配置但显式要求 {"mrr_variation", "thermal_drift"} 两个效应，
        #         取出计划里每个条目的第一个元素（效应名）做成集合。
        # 预期结果：计划里的效应集合恰好等于请求的两个效应，不多不少。
        plan = MODULE._selected_sweep_plan({}, {"mrr_variation", "thermal_drift"})
        self.assertEqual({entry[0] for entry in plan}, {"mrr_variation", "thermal_drift"})

    def test_nested_sites_are_collapsed_to_outermost_module(self) -> None:
        # 测什么：嵌套的模块注入点（site）应折叠到最外层模块名，避免同一模块的子层被重复注入。
        # 怎么测：给出一组含嵌套层级（attn 下还有 qkv/proj，mlp 下还有 fc1/fc2）的模块名，
        #         调用 _collapse_nested_module_names。
        # 预期结果：block.attn.qkv 与 block.attn.proj 折叠为 block.attn（以 block.attn 为前缀的仅保留最外层）；
        #          而 fc1/fc2 是并列子模块、互不为前缀，两个都保留。
        names = {
            "block.attn",
            "block.attn.qkv",
            "block.attn.proj",
            "block.mlp.fc1",
            "block.mlp.fc2",
        }
        self.assertEqual(
            MODULE._collapse_nested_module_names(names),
            {"block.attn", "block.mlp.fc1", "block.mlp.fc2"},
        )

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch required")
    def test_wdm_crosstalk_uses_conv_channel_axis(self) -> None:
        # 测什么：WDM（波分复用）相邻信道串扰注入应沿"通道维"（channel 轴）做邻居扰动。
        # 怎么测：构造 1 个样本、3 个通道、1x1 张量，生成串扰扰动函数并作用到 conv 层输出；
        #         与公式"0.8*自身 + 0.1*上移一位 + 0.1*下移一位"（torch.roll 在通道维上滚动）对比。
        # 预期结果：扰动输出与公式结果逐元素一致（torch.allclose），说明串扰方向/比例正确。
        import torch

        output = torch.tensor([[[[1.0]], [[2.0]], [[4.0]]]])  # 3 个通道各一个数值，便于手算验证
        perturb = MODULE._make_perturb_fn(
            "wdm_adjacent_crosstalk", 0.1, torch, random.Random(1), {"_injection_site_count": 1}
        )
        actual = perturb(output, "conv")
        expected = 0.8 * output + 0.1 * torch.roll(output, 1, 1) + 0.1 * torch.roll(output, -1, 1)  # 80% 自留 + 各 10% 给相邻通道
        self.assertTrue(torch.allclose(actual, expected))

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch required")
    def test_mrr_variation_is_static_per_site(self) -> None:
        # 测什么：MRR（微环谐振器）工艺偏差注入应是"静态"的——对同一注入点多次扰动结果必须完全一样，
        #         因为工艺偏差是制造时固化下来的，不会随时间变化。
        # 怎么测：固定随机种子后，对同一全 1 张量用同一个扰动函数连续调用两次。
        # 预期结果：两次输出逐元素完全相等（torch.equal）。
        import torch

        torch.manual_seed(7)
        output = torch.ones(2, 8)
        perturb = MODULE._make_perturb_fn("mrr_variation", 3.0, torch, random.Random(1), {})
        first = perturb(output, "linear")
        second = perturb(output, "linear")
        self.assertTrue(torch.equal(first, second))  # 静态注入：同点两次扰动必须比特级一致

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch required")
    def test_compensated_thermal_proxy_avoids_global_collapse(self) -> None:
        # 测什么：带热补偿（thermal compensation）代理的"热漂移"注入在大多数情况下应保持温和，
        #         不会让整个张量发生全局性崩溃（整体数值被大幅改变）。
        # 怎么测：构造热补偿配置（补偿效率 0.98、残余增益 0.02、每度共同损耗 0.002），
        #         对全 1 张量施加 10 度热漂移注入，计算平均绝对偏差。
        # 预期结果：平均绝对偏差 < 0.02，说明 98% 的补偿让绝大多数元素几乎不变。
        import torch

        torch.manual_seed(11)
        output = torch.ones(2, 64)
        cfg = {
            "thermal_compensation_efficiency": 0.98,  # 98% 热效应被补偿掉
            "thermal_residual_gain_sigma_per_c": 0.02,  # 每摄氏度残余增益的波动标准差
            "thermal_common_loss_per_residual_c": 0.002,  # 每摄氏度残余共同损耗
        }
        perturb = MODULE._make_perturb_fn("thermal_drift", 10.0, torch, random.Random(1), cfg)
        actual = perturb(output, "linear")
        self.assertLess(float(torch.mean(torch.abs(actual - output))), 0.02)  # 平均偏差很小 => 未全局崩溃


if __name__ == "__main__":
    unittest.main()
