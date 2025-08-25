# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_Adithya/q5/cq5j55awppjrhzhx3oizg7kjibjv7hdsbcpokdjgx5sb4nrtcleo.py
# Topologically Sorted Source Nodes: [pred_probs, mul, intersection, sum_2, sum_3, union, mul_1, add_1, add_2, dice], Original ATen: [aten.sigmoid, aten.mul, aten.sum, aten.add, aten.div]
# Source node to ATen node mapping:
#   add_1 => add_1
#   add_2 => add_2
#   dice => div
#   intersection => sum_1
#   mul => mul
#   mul_1 => mul_1
#   pred_probs => sigmoid
#   sum_2 => sum_2
#   sum_3 => sum_3
#   union => add
# Graph fragment:
#   %sigmoid : [num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%primals_2,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, %primals_1), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [2, 3]), kwargs = {dtype: torch.float32})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sigmoid, [2, 3]), kwargs = {dtype: torch.float32})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%primals_1, [2, 3]), kwargs = {dtype: torch.float32})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_2, %sum_3), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_1, 2.0), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, 1e-06), kwargs = {})
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, 1e-06), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_1, %add_2), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div, %add_2), kwargs = {})
triton_per_fused_add_div_mul_sigmoid_sum_0 = async_compile.triton('triton_per_fused_add_div_mul_sigmoid_sum_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_sigmoid_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 3, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 216, 'r0_': 147456}}
)
@triton.jit
def triton_per_fused_add_div_mul_sigmoid_sum_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr3, xnumel, r0_numel):
    xnumel = 9
    XBLOCK: tl.constexpr = 1
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None)
    tmp2 = tl.load(in_ptr1 + (r0_1 + 1024*x0), None)
    tmp1 = tl.sigmoid(tmp0)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp7 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.broadcast_to(tmp2, [R0_BLOCK])
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = tmp9 + tmp12
    tmp14 = 1e-06
    tmp15 = tmp13 + tmp14
    tmp16 = 2.0
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 + tmp14
    tmp19 = (tmp18 / tmp15)
    tmp20 = (tmp19 / tmp15)
    tl.store(out_ptr0 + (r0_1 + 1024*x0), tmp1, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp15, None)
    tl.store(out_ptr3 + (x0), tmp20, None)
    tl.store(out_ptr1 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/7t/c7tpio2nkmho5bnj7xlugrp34c2pjldrt5smpwt3c3kv2i74ky7a.py
# Topologically Sorted Source Nodes: [mul_1, add_1, dice, mean, dice_val], Original ATen: [aten.mul, aten.add, aten.div, aten.mean, aten.rsub]
# Source node to ATen node mapping:
#   add_1 => add_1
#   dice => div
#   dice_val => sub
#   mean => mean
#   mul_1 => mul_1
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_1, 2.0), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, 1e-06), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_1, %add_2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%div,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mean), kwargs = {})
triton_per_fused_add_div_mean_mul_rsub_1 = async_compile.triton('triton_per_fused_add_div_mean_mul_rsub_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mean_mul_rsub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 72}}
)
@triton.jit
def triton_per_fused_add_div_mean_mul_rsub_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 9
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r0_0), r0_mask, other=0.0)
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tmp3 = 1e-06
    tmp4 = tmp2 + tmp3
    tmp6 = (tmp4 / tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = 9.0
    tmp12 = (tmp10 / tmp11)
    tmp13 = 1.0
    tmp14 = tmp13 - tmp12
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp14, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (9, 1, 32, 32), (1024, 1024, 32, 1))
    assert_size_stride(primals_2, (9, 1, 32, 32), (1024, 1024, 32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((9, 1, 32, 32), (1024, 1024, 32, 1), torch.float32)
        buf1 = empty_strided_cuda((9, 1), (1, 9), torch.float32)
        buf2 = empty_strided_cuda((9, 1), (1, 9), torch.float32)
        buf4 = reinterpret_tensor(buf2, (9, 1), (1, 1), 0); del buf2  # reuse
        buf6 = empty_strided_cuda((9, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pred_probs, mul, intersection, sum_2, sum_3, union, mul_1, add_1, add_2, dice], Original ATen: [aten.sigmoid, aten.mul, aten.sum, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_mul_sigmoid_sum_0.run(buf4, primals_2, primals_1, buf0, buf1, buf6, 9, 1024, stream=stream0)
        del primals_2
        buf5 = empty_strided_cuda((), (), torch.float32)
        buf7 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [mul_1, add_1, dice, mean, dice_val], Original ATen: [aten.mul, aten.add, aten.div, aten.mean, aten.rsub]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_mean_mul_rsub_1.run(buf7, buf1, buf4, 1, 9, stream=stream0)
        del buf1
    return (buf0, primals_1, buf7, primals_1, buf0, buf4, buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((9, 1, 32, 32), (1024, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((9, 1, 32, 32), (1024, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
