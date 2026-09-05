# Source provenance

- Joint simulator: https://github.com/haoranwang0921/lpwm-photonic-sim
- Cloned commit: `9f7836eec68bc2eb5339e5c51212f963346e52a7`.
- HPAT source: local `C:/Users/whr/Desktop/HPAT测试/hpat-artifact`, commit
  `e34fcaf78c5f4cfc96c25aba6b42e5c20d404da9`.
- `vendor/hpat_eval/mobilevit_loader.py` and `activity_trace.py` are verbatim
  copies, Apache-2.0. Notices and licenses are retained under `vendor/licenses`.
- `reference/mobilevit_operator_activity.csv` is the existing activity table
  (CC BY 4.0). `reference/paper_v1.json` is the old experiment configuration.
- The source trees were not changed. The generated manifests hash these copied
  inputs to distinguish working-tree contents from the commit identifiers.

The old activity table is reused for shape/MAC cross-checks, not as a producer
DAG. It contains module hooks and shape-inferred Attention operations; its
latency/energy-share columns are proxies and are never used for new costs.
New DAGs are captured by executing the same timm model families. Random seeded
weights suffice for static shape/cost traces, but do not establish accuracy,
pretrained provenance, quantization validity, or nonideality robustness.

The user confirmed the four 16x16 physical cores, three mapping ablations,
1-us nominal write model with sweeps, original resolutions, nominal electronic
model with memory sweeps, and fixed-area signed row-pair/digital subtraction on
2026-09-05. Other explicit modelling assumptions are in `config.json`.
No result is approved for manuscript use merely because tests pass.
