"""Opt-in row-chunk scheduling with finite credit and FIFO shared-bus arbitration."""
import heapq
from collections import defaultdict, deque
from itertools import islice
from .streaming import TileStreamSimulation, io_sizes, weight_blocks
from .pipeline_smoke import schedule
from .energy_ledger import stage_energies


def dispatch_dag(nodes, start, emit):
    """Dispatch only ready tasks on idle resources; never reserve future bus slots."""
    waiting={k:len(n['deps']) for k,n in nodes.items()}
    followers=defaultdict(list);queues=defaultdict(deque)
    for k,n in nodes.items():
        for d in n['deps']:followers[d].append(k)
        if not n['deps']:queues[n['resource']].append(k)
    running=[];busy=set();finished={};t=start;order=0
    while len(finished)<len(nodes):
        for resource,q in queues.items():
            if q and resource not in busy:
                key=q.popleft();n=nodes[key]
                assert all(d in finished and finished[d]<=t+1e-14 for d in n['deps'])
                end=emit(key,n,t)
                assert end>=t
                busy.add(resource);heapq.heappush(running,(end,order,resource,key));order+=1
        if not running:raise AssertionError('DAG deadlock')
        t=running[0][0]
        while running and running[0][0]<=t:
            end,_,resource,key=heapq.heappop(running)
            busy.remove(resource);finished[key]=end
            for child in followers[key]:
                waiting[child]-=1
                if waiting[child]==0:queues[nodes[child]['resource']].append(child)
    return t


class RowPipeline(TileStreamSimulation):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.row_chunk=int(self.a.get('pipeline_row_chunk',16))
        if self.row_chunk<=0:raise ValueError('row chunk must be positive')
        self.node_records=[];self.pipeline_buffer_bytes=0

    def photonic_compute(self,r,ready):
        if any([self.io_overlap,self.reduce_overlap,self.dma_batch,self.accum_resident,self.gated_static]):
            raise ValueError('Disable other experimental optimizations')
        op,M=r['op_id'],r['M'];ib,ob=io_sizes(r)
        acc=M*min(r['N']*r['batch_repetitions'],self.ln*self.cores)*self.a['accumulator_bits']//8
        buffer=self.cores*min(M,self.row_chunk)*(2*self.pk+4*self.ln)*self.a['bits']//8
        self.pipeline_buffer_bytes=max(self.pipeline_buffer_bytes,buffer)
        required=ib+ob+acc+buffer
        if required>self.a['activation_scratchpad_bytes']:raise ValueError('Row buffers exceed capacity')
        self.scratch_peak=max(self.scratch_peak,required)
        ready=self.hbm(op,ready,ib,'photonic_input')
        ready=self.scratch(op,ready,ib,'input_scratch_fill')
        done=ready;blocks=iter(weight_blocks(r,self.pk,self.ln,self.current_frame));wave_id=0
        while wave:=list(islice(blocks,self.cores)):
            prepared=[]
            for core,b in enumerate(wave):
                begin=max(done,self.core_ready[core]);kernel=self.costs.kernel(M,b.K,b.N)
                if r['weight_static']:begin=self.static_weight(op,b,begin)
                else:begin=self.scratch(op,begin,self.physical_weight_bytes,'dynamic_weight_read')
                if self.core_weights[core]!=b.key:
                    pt=self.u['program_response_time_s'];ps,lane=self.pool.acquire_program_lane(begin,pt)
                    energy=self.program_nominal_per_ring*self.pk*2*self.ln*(pt/1e-6)*self.a.get('program_energy_multiplier',1)
                    energy+=self.pk*2*self.ln*kernel['bias_dac_sample_j']
                    begin=self.event('programming',op,ps,pt,f'program_{lane}',energy,core=core,weight_block=b.key)
                    self.core_weights[core]=b.key
                prepared.append((core,b,kernel,begin))
            # No weight writes while the wave is active; existing N/K wave order retained.
            start=max(x[3] for x in prepared);nodes={};acc_nodes={}
            for core,b,kernel,_ in prepared:
                prior_chunk=None
                for row in range(0,M,self.row_chunk):
                    count=min(self.row_chunk,M-row);base=f'w{wave_id}:c{core}:r{row}'
                    group=f'b{b.batch_index}:n{b.n_index}:r{row}'
                    meta=dict(core=core,block=b,kernel=kernel,row=row,count=count,accumulator=group)
                    def add(suffix,resource,deps,kind,bytes=0):
                        key=base+':'+suffix
                        nodes[key]=dict(resource=resource,deps=deps,kind=kind,bytes=bytes,**meta)
                        return key
                    inp=add('in','scratch_bus',[prior_chunk] if prior_chunk else [],'two_sign_input_passes',2*count*b.K*self.a['bits']//8)
                    optical=add('core',f'core_{core}',[inp],'pipeline_core')
                    deps=[optical]
                    # Same output row and column: K partials cannot overtake predecessors.
                    if group in acc_nodes:deps.append(acc_nodes[group])
                    read=add('read','scratch_bus',deps,'partial_sum_read',count*b.N*(4*self.a['bits']+(self.a['accumulator_bits'] if b.k_index else 0))//8)
                    reduce=add('reduce','electronic',[read],'signed_reduce')
                    write=add('write','scratch_bus',[reduce],'partial_sum_write',count*b.N*self.a['accumulator_bits']//8)
                    prior_chunk=write;acc_nodes[group]=write
            def emit(key,n,t):
                b=n['block'];core=n['core'];count=n['count'];row=n['row'];kernel=n['kernel'];kind=n['kind']
                meta=dict(core=core,weight_block=b.key,row_start=row,row_count=count,
                          accumulator=n['accumulator'],k_index=b.k_index,dag_node=key)
                if kind=='pipeline_core':
                    ds=[kernel[k]/(2*M) for k in ['encode_s','compute_s','convert_s']]
                    end=t
                    for sign in ['positive','negative']:
                        elapsed,events=schedule(count,ds,
                            pipelined=bool(self.a.get('row_core_pipeline',True)),ready=end)
                        for e in events:
                            stage=e['stage'];event=['dac_encode','optical_compute','adc_convert'][stage]
                            self.event(event,op,e['start_s'],e['end_s']-e['start_s'],f'core_{core}_stage_{stage}',
                                stage_energies(kernel)[event]/(2*M),
                                token=row+e['token'],sign=sign,input_ready_s=t,**meta)
                        end+=elapsed
                    self.core_ready[core]=end
                elif kind=='signed_reduce':
                    ops=count*b.N*(3+int(b.k_index>0));duration=ops/(self.a['electronic_peak_flops']*self.a['vector_peak_fraction'])
                    actual=self.pool.acquire_electronic(t,duration)
                    assert abs(actual-t)<1e-14
                    end=self.event(kind,op,t,duration,'electronic',self.em.vector_energy(ops),**meta)
                else:
                    assert self.bus_ready<=t+1e-14
                    end=self.scratch(op,t,n['bytes'],kind,**meta)
                self.node_records.append(dict(frame=self.current_frame,op_id=op,node=key,
                    deps=n['deps'],resource=n['resource'],start_s=t,end_s=end))
                return end
            done=dispatch_dag(nodes,start,emit);wave_id+=1
        return self.hbm(op,done,ob,'photonic_output')


def verify_nodes(records):
    ends={};resources=defaultdict(float)
    for n in records:
        scope=(n['frame'],n['op_id'])
        assert all((*scope,d) in ends and ends[(*scope,d)]<=n['start_s']+1e-14 for d in n['deps'])
        assert resources[n['resource']]<=n['start_s']+1e-14
        resources[n['resource']]=n['end_s'];ends[(*scope,n['node'])]=n['end_s']
    return len(records)
