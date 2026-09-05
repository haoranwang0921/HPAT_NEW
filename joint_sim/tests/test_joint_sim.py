"""
Tests for the joint_sim glue layer.

Covers:
  1. Operator classification (op_classifier)
  2. Backend connectivity (simphony_backend)
  3. Cache hit/miss (cost_cache)
  4. Schema round-trip
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schema import OperatorRecord, ClassificationResult
from op_classifier import classify_operator, build_weight_id
from cost_cache import CostCache


# ---------------------------------------------------------------------------
# Operator classification tests
# ---------------------------------------------------------------------------

def test_classify_static_weight_linear():
    result = classify_operator("Linear", "Q", weight_static=True)
    assert result["eligible"] is True
    assert result["reason"] == "static_weight_linear"


def test_classify_dynamic_matmul():
    result = classify_operator("MatMul", "QK_T", weight_static=False)
    assert result["eligible"] is False
    assert result["reason"] == "both_operands_dynamic"


def test_classify_nonlinear():
    for op in ["LayerNorm", "Softmax", "GELU", "Conv2d"]:
        result = classify_operator(op, "unknown", weight_static=False)
        assert result["eligible"] is False, f"{op} should be ineligible"
        assert result["reason"] == "nonlinear", f"{op} reason: {result['reason']}"


def test_classify_dynamic_weight_linear():
    result = classify_operator("Linear", "Q", weight_static=False)
    assert result["eligible"] is False
    assert result["reason"] == "dynamic_weight"


def test_classify_unrecognized_role():
    # All static Linear are eligible (Plan: 所有固定Linear映射到MRR)
    result = classify_operator("Linear", "custom_layer", weight_static=True)
    assert result["eligible"] is True
    assert result["reason"] == "static_weight_linear"


def test_build_weight_id_stability():
    """weight_id must be stable (same input → same output)."""
    id1 = build_weight_id("backbone.layer1.linear", (512, 256))
    id2 = build_weight_id("backbone.layer1.linear", (512, 256))
    id3 = build_weight_id("backbone.layer1.linear", (256, 512))
    assert id1 == id2
    assert id1 != id3
    assert len(id1) == 16


# ---------------------------------------------------------------------------
# SQLite cache tests
# ---------------------------------------------------------------------------

def test_cache_kernel_cost():
    with CostCache(tempfile.mktemp(suffix=".db")) as cache:
        cost = {
            "compute_latency_s": 1e-6,
            "operand_encoding_latency_s": 2e-7,
            "conversion_latency_s": 3e-7,
            "programming_latency_s": 4e-6,
            "dynamic_energy_j": 1e-9,
            "dac_energy_j": 2e-10,
            "adc_energy_j": 3e-10,
            "laser_energy_j": 4e-10,
            "mrr_tuning_energy_j": 1e-10,
            "mrr_hold_energy_j": 5e-10,
            "iter_M": 1, "iter_K": 2, "iter_N": 3,
            "switching_cycles": 10, "max_cycles": 100,
            "utilization": 0.5,
        }
        # Should miss first
        miss = cache.get_kernel_cost(
            257, 512, 512, 8, 8, 8, "weight_stationary",
            "aa", "bb", "1.0.0",
        )
        assert miss is None

        # Put and get
        cache.put_kernel_cost(
            257, 512, 512, 8, 8, 8, "weight_stationary",
            "aa", "bb", "1.0.0", cost,
        )
        hit = cache.get_kernel_cost(
            257, 512, 512, 8, 8, 8, "weight_stationary",
            "aa", "bb", "1.0.0",
        )
        assert hit is not None
        assert hit["compute_latency_s"] == 1e-6

        # Different schema version → miss
        miss2 = cache.get_kernel_cost(
            257, 512, 512, 8, 8, 8, "weight_stationary",
            "aa", "bb", "2.0.0",
        )
        assert miss2 is None


def test_cache_architecture_cost():
    with CostCache(tempfile.mktemp(suffix=".db")) as cache:
        cost = {
            "pic_area_um2": 1e6,
            "rf_eic_area_um2": 2e6,
            "total_area_um2": 3e6,
            "core_insertion_loss_db": 4.5,
            "laser_wall_plug_power_w": 0.5,
            "mrr_count": 16384,
            "pd_count": 1024,
            "dac_count": 32,
            "adc_count": 16,
        }
        cache.put_architecture_cost("aa", "bb", cost)
        hit = cache.get_architecture_cost("aa", "bb")
        assert hit is not None
        assert hit["mrr_count"] == 16384

        # Different config → miss
        miss = cache.get_architecture_cost("aa", "cc")
        assert miss is None


def test_cache_programming_cost():
    with CostCache(tempfile.mktemp(suffix=".db")) as cache:
        cost = {
            "tile_count": 4,
            "programmed_mrr_count": 16384,
            "programming_latency_s": 1e-6,
            "programming_energy_j": 1.5e-6,
            "hold_power_w": 1.6384,
        }
        cache.put_programming_cost("aa", "bb", cost)
        hit = cache.get_programming_cost("aa", "bb")
        assert hit is not None
        assert hit["hold_power_w"] == 1.6384


def test_cache_run_manifest():
    with CostCache(tempfile.mktemp(suffix=".db")) as cache:
        cache.record_run("run001", "abc123", "def456", "ghi789", "1.0.0")
        # No error = pass


def test_hash_helpers():
    d1 = {"a": 1, "b": 2}
    d2 = {"b": 2, "a": 1}  # key order differs
    assert CostCache.hash_dict(d1) == CostCache.hash_dict(d2)


# ---------------------------------------------------------------------------
# Backend smoke test (requires SimPhony import)
# ---------------------------------------------------------------------------

def test_backend_import():
    """SimPhonyBackend can be imported and initialised."""
    from simphony_backend import SimPhonyBackend
    backend = SimPhonyBackend()
    assert backend.schema_version == "1.0.0"


def test_backend_architecture_cost():
    """Backend returns architecture cost with expected key device counts."""
    from simphony_backend import SimPhonyBackend
    backend = SimPhonyBackend()
    arch = backend.architecture_cost()
    assert arch["mrr_count"] == 393216  # 6144 nodes × 64 MRRs/node = H*R*C*N
    assert arch["pd_count"] == 6144     # 6144 nodes × 1 PD/node = H*R*C
    assert arch["dac_count"] == 12288   # (N + W) × R × C = 128 per core × 96 cores
    assert arch["adc_count"] == 6144    # H × R × C = 64 per core × 96 cores


def test_electronic_backend_energy_model():
    """ElectronicBackend loads LLMCompass's energy model by default (energy > 0)."""
    from electronic_backend import ElectronicBackend
    backend = ElectronicBackend()
    assert backend.energy_model is not None, "Expected LLMCompass energy model wired by default"
    rec = {"op_type": "MatMul", "M": 64, "K": 512, "N": 512,
           "input_bits": 16, "output_bits": 16}
    energy = backend.operator_energy_j(rec, latency_s=1e-5)
    assert energy > 0


def test_llmcompass_matmul_import_without_scalesim():
    """LLMCompass Matmul imports even when scalesim is not installed.

    Runs in a subprocess with only LLMCompass on sys.path: the joint_sim
    test process already has SimPhony on the path, and SimPhony + LLMCompass
    both export a top-level `utils` module, so a same-process import would
    resolve to the wrong one.
    """
    import subprocess
    import os
    llmc = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "LLMCompass"))
    code = (
        f"import sys; sys.path.insert(0, {llmc!r}); "
        "from software_model.matmul import Matmul, BatchedMatmul; "
        "print('OK')"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, f"import failed: {r.stderr}"
    assert "OK" in r.stdout


def test_backend_kernel_cost():
    """Backend returns kernel cost with correct iteration counts."""
    from simphony_backend import SimPhonyBackend
    backend = SimPhonyBackend()
    cost = backend.kernel_cost(M=257, K=512, N=512)
    assert cost["iter_M"] == 1
    assert cost["iter_K"] == 8
    assert cost["iter_N"] == 1
    assert cost["compute_latency_s"] > 0
    assert cost["programming_latency_s"] > 0
    assert cost["utilization"] > 0


def test_backend_programming_cost():
    """Programming cost returns correct ring count."""
    from simphony_backend import SimPhonyBackend
    backend = SimPhonyBackend()
    pc = backend.programming_cost()
    assert pc["programmed_mrr_count"] == 393216
    assert pc["tile_count"] == 96
    assert pc["hold_power_w"] > 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [
        ("classify_static_weight_linear", test_classify_static_weight_linear),
        ("classify_dynamic_matmul", test_classify_dynamic_matmul),
        ("classify_nonlinear", test_classify_nonlinear),
        ("classify_dynamic_weight_linear", test_classify_dynamic_weight_linear),
        ("classify_unrecognized_role", test_classify_unrecognized_role),
        ("build_weight_id_stability", test_build_weight_id_stability),
        ("cache_kernel_cost", test_cache_kernel_cost),
        ("cache_architecture_cost", test_cache_architecture_cost),
        ("cache_programming_cost", test_cache_programming_cost),
        ("cache_run_manifest", test_cache_run_manifest),
        ("hash_helpers", test_hash_helpers),
        ("backend_import", test_backend_import),
        ("backend_architecture_cost", test_backend_architecture_cost),
        ("backend_kernel_cost", test_backend_kernel_cost),
        ("backend_programming_cost", test_backend_programming_cost),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1

    print(f"\n  {passed} passed, {failed} failed, {len(tests)} total")
    if failed > 0:
        sys.exit(1)
