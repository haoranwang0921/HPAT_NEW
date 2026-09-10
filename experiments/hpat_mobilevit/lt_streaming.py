"""Explicit output-block/K-inner DPTC schedule over the shared electronic ledger."""
import math
from .streaming import TileStreamSimulation, Block, io_sizes


class LTStreamSimulation(TileStreamSimulation):
    def photonic_compute(self, rec, ready):
        op = rec["op_id"]
        wm, hn = self.u["physical_array"]
        ib, ob = io_sizes(rec)
        required = ib + ob + self.cores*wm*hn*self.a["accumulator_bits"]//8
        self.scratch_peak = max(self.scratch_peak, required)
        if required > self.a["activation_scratchpad_bytes"]:
            raise ValueError("LT scratch capacity exceeded; spill is not implemented")
        ready = self.hbm(op, ready, ib, "photonic_input")
        ready = self.scratch(op, ready, ib, "input_scratch_fill")
        done = ready
        covered = 0
        for batch in range(rec["batch_repetitions"]):
            for n in range(0, rec["N"], hn):
                nn = min(hn, rec["N"]-n)
                for m in range(0, rec["M"], wm):
                    mm = min(wm, rec["M"]-m)
                    core = min(range(self.cores), key=lambda i: self.core_ready[i])
                    prior = max(ready, self.core_ready[core])
                    for k in range(0, rec["K"], self.pk):
                        kk = min(self.pk, rec["K"]-k)
                        begin = prior
                        if rec["weight_static"]:
                            # Same 128B cache layout as D0/HPAT: assemble the
                            # DPTC's wider logical output from paired cache lines.
                            for nc in range(n, n+nn, self.ln):
                                key = f"{rec['weight_id']}:b{batch}:k{k}:n{nc}"
                                block = Block(key, kk, min(self.ln, rec["N"]-nc),
                                              k//self.pk, nc//self.ln, batch)
                                begin = self.static_weight(op, block, begin)
                        else:
                            begin = self.scratch(op, begin, math.ceil(kk*nn*self.a["bits"]/8), "lt_dynamic_operand")
                        begin = self.scratch(op, begin, math.ceil(mm*kk*self.a["bits"]/8), "lt_input_operand")
                        cost = self.costs.kernel(mm, kk, nn)
                        end = self.event("dac_encode", op, begin, cost["encode_s"], f"core_{core}", core=core)
                        end = self.event("optical_compute", op, end, cost["compute_s"], f"core_{core}",
                                         cost["dynamic_energy_j"], core=core, macs=mm*kk*nn)
                        end = self.event("adc_convert", op, end, cost["convert_s"], f"core_{core}", core=core)
                        self.core_ready[core] = end
                        group = f"b{batch}:m{m}:n{n}"
                        # Digitize every K block (temporal accumulation disabled).
                        read_bytes = mm*nn*(self.a["bits"]+(self.a["accumulator_bits"] if k else 0))//8
                        end = self.scratch(op, end, read_bytes, "partial_sum_read", accumulator=group, k_index=k//self.pk)
                        alu_ops = mm*nn if k else 0
                        latency = alu_ops/(self.a["electronic_peak_flops"]*self.a["vector_peak_fraction"])
                        start = self.pool.acquire_electronic(end, latency)
                        end = self.event("lt_accumulate", op, start, latency, "electronic", self.em.vector_energy(alu_ops))
                        prior = self.scratch(op, end, mm*nn*self.a["accumulator_bits"]//8,
                                             "partial_sum_write", accumulator=group, k_index=k//self.pk)
                        covered += mm*kk*nn
                    done = max(done, prior)
        if covered != rec["macs"]:
            raise AssertionError(("LT MAC coverage", covered, rec["macs"]))
        return self.hbm(op, done, ob, "photonic_output")

    def run(self, records, inferences=3):
        if self.dma_batch or self.io_overlap or self.gated_static or self.accum_resident or self.reduce_overlap:
            raise ValueError("LT smoke uses explicit serial accounting; optimizations require separate validation")
        result = super().run(records, inferences)
        result["checks"].pop("physical_core_write_compute_exclusion", None)
        result["checks"]["lt_no_weight_programming"] = not any(f["event_counts"].get("programming", 0) for f in result["frames"])
        result["model_scope"] = "LT independent-core controlled architecture model; no accuracy or original-LT system replication claim"
        return result
