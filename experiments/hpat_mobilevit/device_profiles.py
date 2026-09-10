"""Versioned device/system profiles; frozen simulation files remain untouched.

This first supported subset does not invent a faster/lower-energy ADC or optical
loss improvement. Burst packing and stage buffers are explicit system hypotheses.
"""
import copy
from .streaming import TileStreamSimulation, weight_blocks, io_sizes
from .pipeline_smoke import schedule
from .energy_ledger import stage_energies


SPECS = {
    'P0': dict(label='Conservative_requested', sram_mib=1, cores_per_tile=2,
               programming_lanes=4, burst_bytes=128, pipeline_tokens=0),
    'P1': dict(label='Balanced_supported_subset', sram_mib=1, cores_per_tile=2,
               programming_lanes=8, burst_bytes=512, pipeline_tokens=16),
    'P2': dict(label='Optimistic_supported_subset', sram_mib=2, cores_per_tile=4,
               programming_lanes=16, burst_bytes=2048, pipeline_tokens=32),
}


def configure_profile(base, name):
    c=copy.deepcopy(base); s=SPECS[name]; a=c['model_assumptions']; u=c['user_confirmed']
    u.update(tiles=2, cores_per_tile=s['cores_per_tile'], program_response_time_s=1e-11,
             program_response_scope='entire_32x32_array_written_and_settled_hypothesis')
    a.update(sram_capacity_bytes=s['sram_mib']*1024**2,
             hbm_bandwidth_bytes_per_s=512e9, external_memory_type='HBM_class_abstract_not_phone',
             dma_fixed_latency_s=100e-9, program_parallelism=s['programming_lanes'],
             program_energy_multiplier=100000., profile_burst_bytes=s['burst_bytes'],
             profile_pipeline_tokens=s['pipeline_tokens'], inferences_per_run=2)
    for k in ['electronic_io_compute_overlap','activity_gated_static','reduce_compute_overlap',
              'accumulator_residency','dma_batch_fetch']:
        a[k]=False
    c['device_profile']=dict(name=name, **s, evidence='supported_subset_not_complete_literature_profile',
        adc_pj_per_sample=1.48, dac_pj_per_sample=50/14,
        adc_gsps=10, dac_reference_gsps=14, converter_energy_changed=False,
        loss_changed=False, extra_driver_buffer_area_and_leakage_calibrated=False,
        whole_chip_equal_area_claim=False,
        deferred=['higher_bandwidth_lower_energy_ADC','lower_optical_loss',
                  'extra_converter_parallelism','physical_driver_and_buffer_validation'])
    c['scope']='XXS192 batch1 cost-only exploration; shared ideal electronic model, not iPhone. Requested P0 differs from frozen P0. 10ps is hypothetical. No accuracy claim.'
    return c


class ProfileSimulation(TileStreamSimulation):
    """Demand-triggered packed-weight bursts and bounded per-core stage pipeline.

Compiler layout assumption: cache lines of one operator are stored in the
weight_blocks(cache_k,cache_n) order. Full aligned bursts (including resident
lines) are transferred; no sparse gather discount or deferred startup charge.
"""
    def _prepare(self,r):
        self.profile_lines=list(weight_blocks(r,self.cache_k,self.cache_n,self.current_frame)) if r['weight_static'] and r.get('macs',0) else []
        self.profile_index={b.key:i for i,b in enumerate(self.profile_lines)}

    def digital_compute(self,r,ready):
        self._prepare(r)
        return super().digital_compute(r,ready)

    def photonic_compute(self,r,ready):
        self._prepare(r)
        count=self.a['profile_pipeline_tokens']
        if count:
            ib,ob=io_sizes(r)
            acc=r['M']*min(r['N']*r['batch_repetitions'],self.ln*self.cores)*self.a['accumulator_bits']//8
            extra=self.cores*count*(2*self.pk+4*self.ln)*self.a['bits']//8
            required=ib+ob+acc+extra
            if required>self.a['activation_scratchpad_bytes']: raise ValueError('Pipeline buffers exceed capacity')
            self.scratch_peak=max(self.scratch_peak,required)
        return super().photonic_compute(r,ready)

    def _cache_line(self,op,block,earliest):
        burst=self.a['profile_burst_bytes']//self.line_bytes
        if burst<=1: return super()._cache_line(op,block,earliest)
        now=max(earliest,self.hbm_ready,self.bus_ready)
        hit=self.sram.lookup_tile(block.key,now)
        if hit['hit']:
            return self.scratch(op,max(now,hit['ready_time_s']),self.line_bytes,'weight_cache_read')
        i=self.profile_index[block.key]; start=i//burst*burst
        group=self.profile_lines[start:start+burst]
        assert len(group)<=self.sram.capacity_tiles
        # Atomic burst reservation: prefetch accesses are not part of the old
        # trace's future-order metadata. Protect all lines of this burst while
        # reserving its slots, otherwise an earlier reservation can be evicted.
        protected={b.key for b in group}
        policy=self.sram.policy; select=policy.select_victim
        def victim(resident,capacity,time,future_info=None):
            candidates=[k for k in resident if k not in protected]
            return min(candidates,key=lambda k: resident[k].last_used_s) if candidates else None
        policy.select_victim=victim
        try:
            for b in group:
                if b.key!=block.key: self.sram.lookup_tile(b.key,now)
        finally:
            policy.select_victim=select
        # Full startup precedes readiness of EVERY line. Burst tail never crosses tensor boundary.
        now=self.hbm(op,now,len(group)*self.line_bytes,'static_weight')
        now=self.scratch(op,now,len(group)*self.line_bytes,'weight_cache_fill')
        for b in group: self.sram.mark_tile_ready(b.key,now)
        assert self.sram.occupied<=self.sram.capacity_tiles
        return self.scratch(op,now,self.line_bytes,'weight_cache_read')

    def execute_core_block(self,op,begin,kernel,core,block):
        chunk=self.a['profile_pipeline_tokens']
        if not chunk: return super().execute_core_block(op,begin,kernel,core,block)
        M=kernel['M']; ds=[kernel[k]/(2*M) for k in ['encode_s','compute_s','convert_s']]
        energy=stage_energies(kernel); now=begin
        for sign in ('positive','negative'):
            for first in range(0,M,chunk):
                elapsed,events=schedule(min(chunk,M-first),ds,ready=now)
                for e in events:
                    kind=['dac_encode','optical_compute','adc_convert'][e['stage']]
                    self.event(kind,op,e['start_s'],e['end_s']-e['start_s'],
                        f'core_{core}_stage_{e["stage"]}',energy[kind]/(2*M),
                        core=core,weight_block=block.key,token=first+e['token'],sign=sign,
                        input_ready_s=begin,sign_ready_s=now)
                now+=elapsed
        return now
