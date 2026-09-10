"""
Test the direct GEMM cost API and generate the LPWM six-case cost table.

Covers:
  1. Basic GEMM query
  2. Architecture cost query
  3. Weight programming cost query
  4. LPWM six-case Linear cost table (implementation plan B5 / Section 6)
  5. Consistency checks (MRR count, iteration integer match)
"""

import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from onnarchsim.gemm_simulator import GemmCostSimulator
from onnarchsim.schema import SCHEMA_VERSION

# Architecture constants for hand-calculation verification
R, C, H, W, NW = 12, 8, 64, 64, 64
ARCH_HEIGHT = H * R   # 768
ARCH_WIDTH  = W * C   # 512
ARCH_WL = NW          # 64
EXPECTED_MRR = H * R * C * NW  # 393216 (6144 nodes × 64 MRRs/node)
EXPECTED_PD = H * R * C         # 6144   (6144 nodes × 1 PD/node)


def expected_iter(M, K, N):
    return (
        math.ceil(N / ARCH_HEIGHT),
        math.ceil(K / ARCH_WL),
        math.ceil(M / ARCH_WIDTH),
    )


def main():
    print("=" * 70)
    print("  GemmCostSimulator API Test + LPWM Linear Cost Table")
    print(f"  Schema version: {SCHEMA_VERSION}")
    print("=" * 70)

    # --- Initialisation ---
    sim = GemmCostSimulator()
    print("\n[1] GemmCostSimulator initialised OK")

    # --- Architecture cost ---
    arch = sim.get_architecture_cost()
    print(f"\n[2] Architecture Cost:")
    print(f"    MRR count:     {arch.mrr_count} (expected {EXPECTED_MRR})")
    print(f"    PD count:      {arch.pd_count}")
    print(f"    DAC count:     {arch.dac_count}")
    print(f"    ADC count:     {arch.adc_count}")
    print(f"    Total area:    {arch.total_area_um2:.2f} um^2")
    print(f"    Core IL:       {arch.core_insertion_loss_db:.2f} dB")
    print(f"    Laser power:   {arch.laser_wall_plug_power_w:.4f} W")

    assert arch.mrr_count == EXPECTED_MRR, \
        f"MRR count mismatch: {arch.mrr_count} != {EXPECTED_MRR}"
    assert arch.pd_count == EXPECTED_PD, \
        f"PD count mismatch: {arch.pd_count} != {EXPECTED_PD}"

    # --- Weight programming cost ---
    wp = sim.get_weight_programming_cost()
    print(f"\n[3] Weight Programming Cost (all MRRs):")
    print(f"    MRR count:        {wp.programmed_mrr_count}")
    print(f"    Program latency:   {wp.programming_latency_s:.6e} s")
    print(f"    Program energy:    {wp.programming_energy_j:.6e} J")
    print(f"    Hold power:        {wp.hold_power_w:.4f} W")

    # --- LPWM six-case Linear cost table ---
    cases = [
        (1,   16,   16,   "single-core minimal"),
        (17,  16,   16,   "M overflow"),
        (257, 512,  512,  "LPWM Q/O projection"),
        (257, 512,  2048, "FFN up-projection"),
        (257, 2048, 512,  "FFN down-projection"),
        (1,   512,  3072, "c_proj"),
    ]

    print(f"\n[4] LPWM Six-Case Linear Cost Table")
    print("-" * 85)
    print(f"{'M':>5} {'K':>5} {'N':>5} {'i_M':>4} {'i_K':>4} {'i_N':>4} "
          f"{'comp_lat(s)':>13s} {'dyn_E(J)':>12s} {'util':>6s}")
    print("-" * 85)

    failures = []
    for M, K, N, desc in cases:
        cost = sim.simulate_gemm(M=M, K=K, N=N)
        e_iN, e_iK, e_iM = expected_iter(M, K, N)

        ok = True
        if cost.iter_M != e_iM:
            print(f"  FAIL {desc}: iter_M {cost.iter_M} != expected {e_iM}")
            ok = False
        if cost.iter_K != e_iK:
            print(f"  FAIL {desc}: iter_K {cost.iter_K} != expected {e_iK}")
            ok = False
        if cost.iter_N != e_iN:
            print(f"  FAIL {desc}: iter_N {cost.iter_N} != expected {e_iN}")
            ok = False

        status = "OK" if ok else "FAIL"
        print(f"{M:>5} {K:>5} {N:>5} {cost.iter_M:>4} {cost.iter_K:>4} {cost.iter_N:>4} "
              f"{cost.compute_latency_s:>13.6e} {cost.dynamic_energy_j:>12.6e} "
              f"{cost.utilization:>6.4f}  {status}  [{desc}]")

        if not ok:
            failures.append(desc)

    # --- Consistency checks ---
    print(f"\n[5] Consistency Checks")
    all_ok = True

    # Check: total energy >= sum of components (with reasonable tolerance)
    for M, K, N, desc in cases:
        cost = sim.simulate_gemm(M=M, K=K, N=N)
        components_sum = (
            cost.dac_energy_j + cost.adc_energy_j
            + cost.laser_energy_j + cost.mrr_tuning_energy_j
            + cost.mrr_hold_energy_j
        )
        if cost.dynamic_energy_j > 0 and components_sum > cost.dynamic_energy_j * 2:
            print(f"  WARN {desc}: component sum ({components_sum:.6e} J) "
                  f">> dynamic_energy ({cost.dynamic_energy_j:.6e} J)")

    # Check: utilisation in [0, 1] or close to it
    for M, K, N, desc in cases:
        cost = sim.simulate_gemm(M=M, K=K, N=N)
        if not (0.0 <= cost.utilization <= 1.01):
            print(f"  FAIL {desc}: utilisation {cost.utilization} out of [0,1]")
            all_ok = False

    # Check: MRR energy scales with N*D (more weights = more rings to tune)
    cost_small = sim.simulate_gemm(M=1, K=16, N=16)
    cost_large = sim.simulate_gemm(M=257, K=512, N=2048)
    if cost_large.mrr_tuning_energy_j <= cost_small.mrr_tuning_energy_j:
        print(f"  WARN: large GEMM MRR energy not > small GEMM MRR energy")
    else:
        print(f"  MRR energy scaling OK: {cost_small.mrr_tuning_energy_j:.6e} -> "
              f"{cost_large.mrr_tuning_energy_j:.6e} J")

    if failures:
        print(f"\n  {len(failures)} iteration-count FAILURES detected:")
        for f in failures:
            print(f"    - {f}")
        sys.exit(1)
    else:
        print(f"\n  All {len(cases)} cases iteration counts match hand calculation.")

    print("\nDone.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
