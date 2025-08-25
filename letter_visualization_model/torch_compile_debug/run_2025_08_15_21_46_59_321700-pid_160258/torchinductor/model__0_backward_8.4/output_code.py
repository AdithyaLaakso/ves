# AOT ID: ['0_backward']
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


# kernel path: /tmp/torchinductor_Adithya/66/c66p4wdtbg45jbbnpcjn4kbofnl5kmiw5vky7m3lrfnzdgot6g5b.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.neg, aten.expand, aten.div, aten.mul, aten.unsqueeze, aten.add, aten.sigmoid_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%tangents_2,), kwargs = {})
#   %expand : [num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%neg, [20, 1]), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, 20), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div_1,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_1, %div_3), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_1, %add_2), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, 2.0), kwargs = {})
#   %unsqueeze : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_2, 2), kwargs = {})
#   %unsqueeze_1 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 3), kwargs = {})
#   %expand_1 : [num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_1, [20, 1, 32, 32]), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_1, %expand_1), kwargs = {})
#   %unsqueeze_2 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_3, 2), kwargs = {})
#   %unsqueeze_3 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 3), kwargs = {})
#   %expand_2 : [num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_3, [20, 1, 32, 32]), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %primals_1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %mul_4), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, %sub_1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, %mul_5), kwargs = {})
triton_poi_fused_add_div_expand_mul_neg_sigmoid_backward_unsqueeze_0 = async_compile.triton('triton_poi_fused_add_div_expand_mul_neg_sigmoid_backward_unsqueeze_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_expand_mul_neg_sigmoid_backward_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 409600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_expand_mul_neg_sigmoid_backward_unsqueeze_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x2), None)
    tmp3 = -tmp2
    tmp4 = 0.05
    tmp5 = tmp3 * tmp4
    tmp6 = -tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp0 + tmp8
    tmp11 = (tmp5 / tmp10)
    tmp12 = 2.0
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tmp9 + tmp15
    tmp18 = 1.0
    tmp19 = tmp18 - tmp17
    tmp20 = tmp17 * tmp19
    tmp21 = tmp16 * tmp20
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, sigmoid, add_2, div_3, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_1, (20, 1, 32, 32), (1024, 1024, 32, 1))
    assert_size_stride(sigmoid, (20, 1, 32, 32), (1024, 1024, 32, 1))
    assert_size_stride(add_2, (20, 1), (1, 1))
    assert_size_stride(div_3, (20, 1), (1, 1))
    assert_size_stride(tangents_1, (20, 1, 32, 32), (1024, 1024, 32, 1))
    assert_size_stride(tangents_2, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((20, 1, 32, 32), (1024, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.neg, aten.expand, aten.div, aten.mul, aten.unsqueeze, aten.add, aten.sigmoid_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_expand_mul_neg_sigmoid_backward_unsqueeze_0.run(tangents_1, tangents_2, div_3, add_2, primals_1, sigmoid, buf0, 20480, stream=stream0)
        del add_2
        del div_3
        del primals_1
        del sigmoid
        del tangents_1
        del tangents_2
    return (None, buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((20, 1, 32, 32), (1024, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    sigmoid = rand_strided((20, 1, 32, 32), (1024, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    add_2 = rand_strided((20, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((20, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((20, 1, 32, 32), (1024, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, sigmoid, add_2, div_3, tangents_1, tangents_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
