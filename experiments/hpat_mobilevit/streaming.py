"""Physical-tile streaming for small HPAT, reusing joint_sim resources/cache.

The upstream aggregate scheduler prepares all weight tiles before a GEMM.
Here each physical core is locked from programming through conversion. Larger
weights therefore stream through a bounded set of cores without overwritten
weights being treated as resident. Electronic residuals and digital subtraction
share one resource; HBM and scratchpad traffic have separate bounded channels.
"""
from __future__ import annotations

import math
from itertools import islice
from collections import defaultdict
from dataclasses import dataclass

from joint_sim.scheduler import ResourcePool
from joint_sim.sram_manager import SramManager, LruSramPolicy
from .energy_ledger import stage_energies


VIEWS = {"view", "_unsafe_view", "transpose", "permute", "t", "unbind", "expand",
         "detach", "alias", "slice", "select", "as_strided", "empty"}
COPIES = {"clone", "cat", "copy_", "_to_copy", "upsample_bilinear2d"}
VECTOR_WORK = {"native_batch_norm": 6, "native_layer_norm": 6, "_softmax": 5,
               "silu": 8, "silu_": 8, "add": 1, "mul": 1, "mean": 1,
               "gelu": 8, "relu": 1, "relu_": 1}
MODES = {"digital", "linear", "linear_pointwise", "linear_pointwise_attention"}


def eligible(rec, mode):
    if mode not in MODES:
        raise ValueError(mode)
    return mode != "digital" and (
        rec["op_type"] == "Linear" or
        (mode in {"linear_pointwise", "linear_pointwise_attention"}
         and rec["op_type"] == "Conv2d" and rec.get("kernel_size") == [1, 1]
         and rec.get("groups") == 1) or
        (mode == "linear_pointwise_attention" and rec["op_type"] == "MatMul"
         and rec["op_role"] in {"attention_score", "attention_value"}))


def io_sizes(rec):
    if rec["op_type"] in {"Linear", "MatMul"}:
        M, K, N, b = (rec[k] for k in ["M", "K", "N", "batch_repetitions"])
        read = M*K*b*rec["input_bits"]/8
        if not rec["weight_static"]:
            read += K*N*b*rec["weight_bits"]/8
        read += rec.get("bias_elements", 0)*rec["input_bits"]/8
        return math.ceil(read), math.ceil(M*N*b*rec["output_bits"]/8)
    if rec["op_type"] == "Conv2d":
        return math.ceil(math.prod(rec["input_shapes"][0])*rec["input_bits"]/8), rec["output_bytes"]
    return rec["input_bytes"], rec["output_bytes"]


def vector_work(rec):
    kind = rec["op_type"]
    if kind in VIEWS | COPIES:
        return 0
    if kind not in VECTOR_WORK:
        raise ValueError(f"Unsupported operator {rec['aten_op']} at {rec['module_path']}")
    elements = rec["output_elements"]
    if kind == "mean":
        elements = math.prod(rec["input_shapes"][0])
    return elements * VECTOR_WORK[kind]


@dataclass(frozen=True)
class Block:
    key: str
    K: int
    N: int
    k_index: int
    n_index: int
    batch_index: int


def weight_blocks(rec, physical_k, logical_n, frame=0):
    """Same padded weight-cache layout for digital and photonic baselines."""
    b = rec["batch_repetitions"]
    for batch in range(b):
        for n in range(0, rec["N"], logical_n):
            for k in range(0, rec["K"], physical_k):
                stem = rec["weight_id"] if rec["weight_static"] else f"dynamic:f{frame}:{rec['op_id']}"
                yield Block(f"{stem}:b{batch}:k{k}:n{n}", min(physical_k, rec["K"]-k),
                            min(logical_n, rec["N"]-n), k//physical_k, n//logical_n, batch)


class TileStreamSimulation:
    def __init__(self, config, costs, mode, *, event_sink=None, keep_events=False):
        self.config, self.costs, self.mode = config, costs, mode
        self.a, self.u = config["model_assumptions"], config["user_confirmed"]
        self.pk, pn = self.u["physical_array"]
        if pn % 2:
            raise ValueError("Signed row pairing requires an even physical row count")
        self.ln = pn//2
        self.cores = self.u["tiles"]*self.u["cores_per_tile"]
        self.cache_k, self.cache_n = self.a.get("weight_cache_block_shape", [self.pk, self.ln])
        if self.pk % self.cache_k or self.ln % self.cache_n:
            raise ValueError("Physical weight block must contain whole cache lines")
        self.line_bytes = math.ceil(self.cache_k*self.cache_n*self.a["bits"]/8)
        self.physical_weight_bytes = math.ceil(self.pk*self.ln*self.a["bits"]/8)
        self.pool = ResourcePool(num_photonic_cores=self.cores, num_dac_channels=self.cores,
                                 num_adc_channels=self.cores, num_program_lanes=self.a["program_parallelism"],
                                 hbm_bandwidth_bytes_per_s=self.a["hbm_bandwidth_bytes_per_s"],
                                 dma_fixed_latency_s=self.a["dma_fixed_latency_s"],
                                 weight_tile_bytes=self.line_bytes)
        self.sram = SramManager(total_bytes=self.a["sram_capacity_bytes"], policy=LruSramPolicy(),
                                tile_bytes=self.line_bytes,
                                hbm_bandwidth_bytes_per_s=self.a["hbm_bandwidth_bytes_per_s"],
                                dma_fixed_latency_s=self.a["dma_fixed_latency_s"])
        self.core_ready = [0.0]*self.cores
        self.core_weights = [None]*self.cores
        self.bus_ready = 0.0
        self.hbm_ready = 0.0
        self.time = 0.0
        self.events = []
        self.keep_events = keep_events
        self.event_sink = event_sink
        self.resource_end = defaultdict(float)
        self.energy = defaultdict(float)
        self.event_counts = defaultdict(int)
        self.layer_rows = []
        self.frame_rows = []
        self.current_frame = 0
        self.last_event_end = 0.0
        self.scratch_peak = 0
        self.em = costs.electronic_energy
        count = costs.programming["programmed_mrr_count"]
        self.program_nominal_per_ring = costs.programming["programming_energy_j"] / count if count else 0.0
        # Batch-2 accounting switches (see mobile_profile.resolve_profile).
        self.io_overlap = bool(self.a.get("electronic_io_compute_overlap", False))
        self.gated_static = bool(self.a.get("activity_gated_static", False))
        self.idle_fraction = float(self.a.get("electronic_idle_power_fraction", 0.1))
        self.laser_standby = float(self.a.get("laser_standby_power_fraction", 0.0))
        # Batch-3 M1: reduce-under-compute pipelining. When on, a photonic
        # core's next wave is released as soon as the core itself is free
        # (core_ready), rather than waiting for the prior wave's signed
        # reduction to finish. The accumulator chain (prior) still serializes
        # the reductions that share one output block.
        self.reduce_overlap = bool(self.a.get("reduce_compute_overlap", False))
        # Batch-3 M2: accumulator residency. The int32 partial sum stays
        # beside the ALU across K blocks (no read-modify-write on the scratch
        # bus per block); it is written back once, at the last K block, so the
        # photonic_output HBM turn can read the finished sum.
        self.accum_resident = bool(self.a.get("accumulator_residency", False))
        # Batch-4 DMA coalescing: weight fetches are treated as one DRAM
        # session per operator. dma_fixed_latency_s is charged once per burst
        # of dma_burst_bytes, NOT once per 128B tile (per-tile accounting
        # inflates HBM time ~3x) and NOT zero (zero would silently delete the
        # fixed overhead). The lookup/evict order is unchanged, so LRU
        # residency semantics stay intact.
        self.dma_batch = bool(self.a.get("dma_batch_fetch", False))
        # Stage-level timing: when on, DAC encoding / optical pass-through /
        # ADC conversion occupy three distinct device pools and stream
        # successive input vectors concurrently, so only the bottleneck stage
        # (plus one-iteration fill for each upstream stage) is on the critical
        # path. Default off keeps the historical serialized chaining.
        self.stage_pipeline = bool(self.a.get("photonic_stage_pipeline", False))
        self.dma_burst_bytes = max(float(self.a.get("dma_burst_bytes", 32768.0)),
                                   float(self.line_bytes))
        self._dma_session_bytes = 0.0  # weight DMA bytes in the current session
        self.e_busy = []   # (start, end) of electronic-domain events this frame
        self.p_busy = []   # (start, end) of photonic-domain events this frame

    def event(self, kind, op, start, duration, resource, energy=0.0, **meta):
        if not all(math.isfinite(v) and v >= 0 for v in [start, duration, energy]):
            raise AssertionError((kind, start, duration, energy))
        if duration and start + 1e-15 < self.resource_end[resource]:
            raise AssertionError(("resource overlap", resource, start, self.resource_end[resource]))
        self.resource_end[resource] = max(self.resource_end[resource], start+duration)
        self.last_event_end = max(self.last_event_end, start+duration)
        self.energy[kind] += energy
        self.event_counts[kind] += 1
        if self.gated_static and duration > 0:
            if kind in {"electronic_dynamic", "signed_reduce", "hbm", "sram_noc"}:
                self.e_busy.append((start, start+duration))
            elif kind in {"dac_encode", "optical_compute", "adc_convert", "programming"}:
                self.p_busy.append((start, start+duration))
        row = dict(frame=self.current_frame, event_type=kind, op_id=op, start_s=start,
                   duration_s=duration, end_s=start+duration, resource=resource,
                   energy_j=energy, **meta)
        if self.keep_events:
            self.events.append(row)
        if self.event_sink:
            self.event_sink(row)
        return start+duration

    def hbm(self, op, earliest, count, label, amortize=False):
        if count <= 0:
            return earliest
        start = max(earliest, self.hbm_ready)
        if amortize:
            # Weight-session transfer: bill flow time now, defer the fixed
            # startup cost to _settle_dma_session (once per burst).
            fixed = 0.0
        else:
            fixed = self.a["dma_fixed_latency_s"]
        duration = count/self.a["hbm_bandwidth_bytes_per_s"] + fixed
        self.hbm_ready = self.event("hbm", op, start, duration, "hbm",
                                    self.em.hbm_read_energy(count), bytes=count, purpose=label)
        return self.hbm_ready

    def _settle_dma_session(self, op, earliest):
        """Bill the deferred weight-DMA fixed overhead: once per dma_burst_bytes."""
        if self._dma_session_bytes <= 0:
            return earliest
        bursts = max(1, math.ceil(self._dma_session_bytes / self.dma_burst_bytes))
        overhead = bursts * self.a["dma_fixed_latency_s"]
        self._dma_session_bytes = 0.0
        if overhead <= 0:
            return earliest
        start = max(earliest, self.hbm_ready)
        self.hbm_ready = self.event("hbm", op, start, overhead, "hbm", 0.0,
                                    bytes=0, purpose="dma_burst_overhead")
        return self.hbm_ready

    def scratch(self, op, earliest, count, label, **metadata):
        if count <= 0:
            return earliest
        start = max(earliest, self.bus_ready)
        self.bus_ready = self.event("sram_noc", op, start, count/self.a["sram_bandwidth_bytes_per_s"],
                                    "scratch_bus", self.em.sram_read_energy(count)+self.em.noc_energy(count),
                                    bytes=count, purpose=label, **metadata)
        return self.bus_ready

    def static_weight(self, op, block, earliest):
        # Cache layout is independent of the optical array in hardware scans.
        stem = block.key.rsplit(":k", 1)[0]
        k0, n0 = block.k_index*self.pk, block.n_index*self.ln
        now = earliest
        for n in range(n0, n0+block.N, self.cache_n):
            for k in range(k0, k0+block.K, self.cache_k):
                sub = Block(f"{stem}:k{k}:n{n}", min(self.cache_k, k0+block.K-k),
                            min(self.cache_n, n0+block.N-n), k//self.cache_k, n//self.cache_n, block.batch_index)
                now = self._cache_line(op, sub, now)
        return now

    def _cache_line(self, op, block, earliest):
        now = max(earliest, self.hbm_ready, self.bus_ready)
        result = self.sram.lookup_tile(block.key, now)
        if not result["hit"]:
            if self.dma_batch:
                self._dma_session_bytes += self.line_bytes
            now = self.hbm(op, now, self.line_bytes, "static_weight", amortize=self.dma_batch)
            self.sram.mark_tile_ready(block.key, now)
            now = self.scratch(op, now, self.line_bytes, "weight_cache_fill")
        else:
            now = max(now, result["ready_time_s"])
        if self.sram.occupied > self.sram.capacity_tiles:
            raise AssertionError("SRAM capacity exceeded")
        return self.scratch(op, now, self.line_bytes, "weight_cache_read")

    def digital_compute(self, rec, ready):
        op = rec["op_id"]
        if rec["op_type"] in VIEWS:
            self.event("view_metadata", op, ready, 0, "metadata")
            return ready
        macs = rec.get("macs", 0)
        work = 0 if macs else vector_work(rec)
        ib, ob = io_sizes(rec)
        if rec["weight_static"] and macs:
            for block in weight_blocks(rec, self.cache_k, self.cache_n, self.current_frame):
                ready = self._cache_line(op, block, ready)
            if self.dma_batch:
                ready = self._settle_dma_session(op, ready)
        if macs:
            latency = 2*macs/self.a["electronic_peak_flops"]
            energy = self.em.matmul_energy(macs)
        else:
            latency = work/(self.a["electronic_peak_flops"]*self.a["vector_peak_fraction"])
            energy = self.em.vector_energy(work)
        if self.io_overlap:
            # IDEALIZED LATENCY LOWER BOUND, NOT A PIPELINE IMPLEMENTATION.
            # Compute starts at `ready` concurrently with the input fetch and
            # the operator finishes when both complete (max). This assumes the
            # input is already fully prefetched/available; there is NO
            # first-block-ready, block-by-block consumption, or pipeline
            # fill/drain cost. Use for latency-floor exploration only; do not
            # claim event-level data dependencies are fully verified under
            # this switch. Output writeback stays a serial HBM turn.
            fetch_end = self.hbm(op, ready, ib, "electronic_input")
            start = self.pool.acquire_electronic(ready, latency)
            end = self.event("electronic_dynamic", op, start, latency, "electronic", energy)
            done = max(fetch_end, end)
        else:
            ready = self.hbm(op, ready, ib, "electronic_input")
            start = self.pool.acquire_electronic(ready, latency)
            end = self.event("electronic_dynamic", op, start, latency, "electronic", energy)
            done = end
        return self.hbm(op, done, ob, "electronic_output")

    def photonic_compute(self, rec, ready):
        op, M = rec["op_id"], rec["M"]
        ib, ob = io_sizes(rec)
        # N-major/K-inner traversal finishes each output-column block before
        # advancing. At most one block per physical core is live in a wave;
        # full MxN int32 accumulation is neither needed nor silently allocated.
        accumulation_bytes = M*min(rec["N"]*rec["batch_repetitions"], self.ln*self.cores)*self.a["accumulator_bits"]//8
        scratch_required = ib+ob+accumulation_bytes
        self.scratch_peak = max(self.scratch_peak, scratch_required)
        if scratch_required > self.a["activation_scratchpad_bytes"]:
            raise ValueError(f"{op} needs {scratch_required} scratchpad bytes; adjust explicit capacity or implement spill")
        ready = self.hbm(op, ready, ib, "photonic_input")
        ready = self.scratch(op, ready, ib, "input_scratch_fill")
        done = ready
        accumulator_ready = {}
        max_k_index = (rec["K"] - 1)//self.pk
        blocks = iter(weight_blocks(rec, self.pk, self.ln, self.current_frame))
        while wave := list(islice(blocks, self.cores)):
            tasks, used = [], set()
            # Admit no more tiles than physical cores. Prepare the whole wave
            # before reserving future readout traffic, so independent writing
            # lanes are actually able to operate in parallel.
            for block in wave:
                available = [i for i in range(self.cores) if i not in used]
                cached = [i for i in available if self.core_weights[i] == block.key]
                core = min(cached or available, key=lambda i: self.core_ready[i])
                used.add(core)
                begin = self.core_ready[core] if self.reduce_overlap else max(done, self.core_ready[core])
                kernel = self.costs.kernel(M, block.K, block.N)
                if rec["weight_static"]:
                    begin = self.static_weight(op, block, begin)
                else:
                    begin = self.scratch(op, begin, self.physical_weight_bytes, "dynamic_weight_read")
                if self.core_weights[core] != block.key:
                    ptime = self.u["program_response_time_s"]
                    pstart, lane = self.pool.acquire_program_lane(begin, ptime)
                    multiplier = self.a.get("program_energy_multiplier", 1.0)
                    tuning = self.program_nominal_per_ring*self.pk*(2*self.ln)*(ptime/1e-6)*multiplier
                    bias = self.pk*(2*self.ln)*kernel["bias_dac_sample_j"]
                    begin = self.event("programming", op, pstart, ptime, f"program_{lane}",
                                       tuning+bias, core=core, weight_block=block.key)
                    self.core_weights[core] = block.key
                tasks.append((block, core, begin, kernel))
            reductions = []
            for block, core, begin, kernel in tasks:
                begin = self.scratch(op, begin, 2*M*block.K*self.a["bits"]//8, "two_sign_input_passes")
                end = self.execute_core_block(op, begin, kernel, core, block)
                self.core_ready[core] = end
                reductions.append((end, core, block))
            for end, core, block in sorted(reductions, key=lambda t: t[0]):
                # Four non-negative products -> three differences; one extra
                # accumulation for all but the first K block.
                alu_ops = M*block.N*(3 + int(block.k_index > 0))
                latency = alu_ops/(self.a["electronic_peak_flops"]*self.a["vector_peak_fraction"])
                accumulator_key = (block.batch_index, block.n_index)
                prior = accumulator_ready.get(accumulator_key, ready)
                if self.accum_resident:
                    # Partial product only; the int32 accumulator stays on-chip.
                    read_bytes = M*block.N*(4*self.a["bits"])//8
                else:
                    read_bytes = M*block.N*(4*self.a["bits"] +
                                 (self.a["accumulator_bits"] if block.k_index else 0))//8
                group = f"b{block.batch_index}:n{block.n_index}"
                reduce_ready = self.scratch(op, max(end, prior), read_bytes, "partial_sum_read",
                                            accumulator=group, k_index=block.k_index)
                start = self.pool.acquire_electronic(reduce_ready, latency)
                reduced = self.event("signed_reduce", op, start, latency, "electronic",
                                      self.em.vector_energy(alu_ops), core=core,
                                      accumulator=group, k_index=block.k_index)
                traffic = M*block.N*self.a["accumulator_bits"]//8
                if self.accum_resident:
                    # Write back once, at the last K block of this output block.
                    if block.k_index == max_k_index:
                        reduced = self.scratch(op, reduced, traffic, "partial_sum_write",
                                               accumulator=group, k_index=block.k_index)
                else:
                    reduced = self.scratch(op, reduced, traffic, "partial_sum_write",
                                           accumulator=group, k_index=block.k_index)
                accumulator_ready[accumulator_key] = reduced
                done = max(done, reduced)
        if self.dma_batch and rec["weight_static"]:
            done = self._settle_dma_session(op, done)
        return self.hbm(op, done, ob, "photonic_output")

    def execute_core_block(self, op, begin, kernel, core, block):
        energy = stage_energies(kernel)
        if not self.stage_pipeline:
            encode_end = self.event("dac_encode", op, begin, kernel["encode_s"], f"core_{core}",
                                    energy["dac_encode"], core=core, weight_block=block.key)
            compute_end = self.event("optical_compute", op, encode_end, kernel["compute_s"], f"core_{core}",
                                     energy["optical_compute"], core=core, weight_block=block.key)
            return self.event("adc_convert", op, compute_end, kernel["convert_s"], f"core_{core}",
                              energy["adc_convert"], core=core, weight_block=block.key)
        # Stage-parallel path. The architecture carries distinct device pools
        # for encoding (dac), the optical pass (mrr array) and conversion
        # (adc), so the three stages stream successive input vectors
        # concurrently instead of chaining on one core timeline. Over
        # N = 2*M vectors (two signed passes) the completion time is
        #   max(N*t_e, t_e + N*t_c, t_e + t_c + N*t_v)
        # with t_* the per-vector stage times: each stage starts one stage-time
        # after its upstream stage has emitted its first vector. Per-stage
        # energy is unchanged; only overlap changes. For N = 1 this reduces to
        # the serialized sum.
        iterations = 2*kernel.get("M", 0)
        if iterations <= 0:
            raise ValueError("photonic_stage_pipeline requires the kernel's M")
        t_enc = kernel["encode_s"]/iterations
        t_cmp = kernel["compute_s"]/iterations
        t_cnv = kernel["convert_s"]/iterations
        enc_end = self.event("dac_encode", op, begin, kernel["encode_s"], f"dac_{core}",
                             energy["dac_encode"], core=core, weight_block=block.key)
        cmp_end = self.event("optical_compute", op,
                             max(begin + t_enc, self.resource_end[f"optic_{core}"]),
                             kernel["compute_s"], f"optic_{core}",
                             energy["optical_compute"], core=core, weight_block=block.key)
        cnv_end = self.event("adc_convert", op,
                             max(begin + t_enc + t_cmp, self.resource_end[f"adc_{core}"]),
                             kernel["convert_s"], f"adc_{core}",
                             energy["adc_convert"], core=core, weight_block=block.key)
        t_pipe = max(kernel["encode_s"],
                     t_enc + kernel["compute_s"],
                     t_enc + t_cmp + kernel["convert_s"])
        return max(begin + t_pipe, enc_end, cmp_end, cnv_end)

    @staticmethod
    def _union(intervals):
        if not intervals:
            return 0.0
        ordered = sorted(intervals)
        total = 0.0
        cur_s, cur_e = ordered[0]
        for s, e in ordered[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                total += cur_e-cur_s
                cur_s, cur_e = s, e
        return total + cur_e-cur_s

    def run(self, records, inferences=3):
        for frame in range(inferences):
            self.current_frame = frame
            self.e_busy, self.p_busy = [], []
            start = self.time
            before = dict(self.energy)
            before_stats = dict(self.sram.stats)
            before_counts = dict(self.event_counts)
            complete = {}
            for rec in records:
                deps = rec["dependencies"]
                if any(d not in complete for d in deps):
                    raise ValueError("Trace is not in producer-before-consumer order")
                ready = max(self.time, max((complete[d] for d in deps), default=start))
                is_photonic = eligible(rec, self.mode)
                ebefore = sum(self.energy.values())
                end = self.photonic_compute(rec, ready) if is_photonic else self.digital_compute(rec, ready)
                complete[rec["op_id"]] = end
                self.time = end
                self.layer_rows.append(dict(frame=frame, op_id=rec["op_id"], module_path=rec["module_path"],
                    op_type=rec["op_type"], op_role=rec["op_role"], domain="photonic" if is_photonic else "electronic",
                    latency_s=end-ready, dynamic_energy_j=sum(self.energy.values())-ebefore, macs=rec["macs"]))
            duration = self.time-start
            breakdown = {k: v-before.get(k, 0) for k, v in self.energy.items()}
            if self.gated_static:
                e_active = self._union(self.e_busy)
                p_active = self._union(self.p_busy)
                e_idle = max(duration-e_active, 0.0)
                breakdown["electronic_static"] = (self.em.static_energy(e_active)
                                                  + self.em.static_energy(e_idle)*self.idle_fraction)
                if self.mode != "digital":
                    for k, p in self.costs.static_power.items():
                        if k == "laser_static":
                            breakdown[k] = p*(p_active + max(duration-p_active, 0.0)*self.laser_standby)
                        else:
                            breakdown[k] = p*duration
            else:
                breakdown["electronic_static"] = self.em.static_energy(duration)
                if self.mode != "digital":
                    breakdown.update({k: p*duration for k, p in self.costs.static_power.items()})
            count_delta = {k: v-before_counts.get(k, 0) for k, v in self.event_counts.items()}
            hit = self.sram.stats["hits"]-before_stats["hits"]
            miss = self.sram.stats["misses"]-before_stats["misses"]
            gated_extra = (dict(electronic_active_s=e_active, photonic_active_s=p_active)
                           if self.gated_static else {})
            self.frame_rows.append(dict(frame=frame, state="cold" if frame == 0 else "warm",
                latency_s=duration, energy_j=sum(breakdown.values()), energy_breakdown_j=breakdown,
                **gated_extra,
                event_counts=count_delta, sram_hits=hit, sram_misses=miss,
                sram_hit_rate=hit/max(hit+miss, 1)))
        if not math.isclose(self.time, self.last_event_end, abs_tol=1e-14):
            raise AssertionError("E2E does not match final event completion")
        return dict(mode=self.mode, frames=self.frame_rows, scratchpad_peak_bytes=self.scratch_peak,
                    sram=self.sram.summary(), total_events=sum(self.event_counts.values()),
                    checks={"resource_nonoverlap": True, "dag_dependencies": True,
                            "physical_core_write_compute_exclusion": True, "sram_capacity": True,
                            "scratchpad_capacity": True, "end_time_matches_events": True},
                    model_scope="nominal event-level architecture model; no silicon or quantized accuracy claim")
