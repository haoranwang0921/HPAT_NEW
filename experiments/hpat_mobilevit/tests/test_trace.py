import torch
from experiments.hpat_mobilevit.trace import ProducerTrace, compare_reference, export_variant


def test_real_dependencies_and_signed_values():
    class Branch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3, 4)
        def forward(self, x):
            y = self.fc(x)
            return y + y.sin()
    model = Branch().eval()
    x = torch.randn(2, 3)
    with torch.no_grad():
        expected = model(x)
        capture = ProducerTrace(model)
        with capture:
            actual = model(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    linear = next(r for r in capture.records if r["op_type"] == "Linear")
    sine = next(r for r in capture.records if r["op_type"] == "sin")
    add = capture.records[-1]
    assert linear["macs"] == 24
    assert set(add["dependencies"]) == {linear["op_id"], sine["op_id"]}
    assert linear["weight_static"]


def test_grouped_conv_dimensions():
    model = torch.nn.Conv2d(4, 8, 3, groups=4, bias=False).eval()
    with torch.no_grad():
        capture = ProducerTrace(model)
        with capture:
            model(torch.ones(1, 4, 6, 6))
    conv = next(r for r in capture.records if r["op_type"] == "Conv2d")
    assert (conv["M"], conv["K"], conv["N"], conv["batch_repetitions"]) == (16, 9, 2, 4)
    assert conv["macs"] == 16*9*8


def test_reference_comparison_rejects_missing_macs():
    result = compare_reference([], [{"estimated_macs": "42", "layer_name": "fc"}])
    assert not result["matching"]
    assert result["differences"][0]["reference"] == 42


def test_mobilevit_trace_preserves_outputs(tmp_path):
    torch.set_num_threads(2)
    result = export_variant({"variant": "MobileViT-XXS", "timm_model": "mobilevit_xxs",
                             "input_resolution": 32}, tmp_path / "xxs")
    assert result["capture_output_max_abs_error"] == 0
    assert result["parameter_count"] == 1272024
    assert result["aten_operator_counts"]["aten.bmm.default"] == 18
