# AOT ID: ['25_backward']
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


# kernel path: /tmp/torchinductor_Adithya/l5/cl5ijthntobiwtedgzftbaqoorfaq5k74xso5j5wl5qthlzeqmve.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.neg, aten.add, aten._unsafe_index_put]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_300 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tangents_1, torch.float32), kwargs = {})
#   %mul_2407 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_300, %clamp_max_3), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_2407,), kwargs = {})
#   %add_2667 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_300, %neg), kwargs = {})
#   %mul_2408 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2407, %clamp_max_2), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_2408,), kwargs = {})
#   %add_2668 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2407, %neg_1), kwargs = {})
#   %mul_2409 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2667, %clamp_max_2), kwargs = {})
#   %neg_2 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_2409,), kwargs = {})
#   %add_2669 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2667, %neg_2), kwargs = {})
#   %index_put_1 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default, [None, None, %clamp_max, %clamp_max_1], %mul_2408, True), kwargs = {})
#   %index_put_2 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default, [None, None, %clamp_max, %convert_element_type_298], %add_2668, True), kwargs = {})
#   %index_put_3 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default, [None, None, %convert_element_type_296, %clamp_max_1], %mul_2409, True), kwargs = {})
#   %index_put_4 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default, [None, None, %convert_element_type_296, %convert_element_type_298], %add_2669, True), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_put_add_mul_neg_0 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_put_add_mul_neg_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_put_add_mul_neg_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_put_add_mul_neg_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
    tl.store(out_ptr2 + (x0), tmp0, None)
    tl.store(out_ptr3 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/zk/czkwf5jwf73icdme4ctg2mdp27n6rbx3wv34zplinso2yijvnvfc.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.neg, aten.add, aten._unsafe_index_put]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_300 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tangents_1, torch.float32), kwargs = {})
#   %mul_2407 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_300, %clamp_max_3), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_2407,), kwargs = {})
#   %add_2667 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_300, %neg), kwargs = {})
#   %mul_2408 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2407, %clamp_max_2), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_2408,), kwargs = {})
#   %add_2668 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2407, %neg_1), kwargs = {})
#   %mul_2409 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2667, %clamp_max_2), kwargs = {})
#   %neg_2 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_2409,), kwargs = {})
#   %add_2669 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2667, %neg_2), kwargs = {})
#   %index_put_1 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default, [None, None, %clamp_max, %clamp_max_1], %mul_2408, True), kwargs = {})
#   %index_put_2 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default, [None, None, %clamp_max, %convert_element_type_298], %add_2668, True), kwargs = {})
#   %index_put_3 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default, [None, None, %convert_element_type_296, %clamp_max_1], %mul_2409, True), kwargs = {})
#   %index_put_4 : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default, [None, None, %convert_element_type_296, %convert_element_type_298], %add_2669, True), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_put_add_mul_neg_1 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_put_add_mul_neg_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_put_add_mul_neg_1', 'mutated_arg_names': ['out_ptr0', 'out_ptr1', 'out_ptr2', 'out_ptr3'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_put_add_mul_neg_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x3 = xindex
    x2 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x3), xmask).to(tl.float32)
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 128, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 128)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 128")
    tmp7 = tmp6 + tmp1
    tmp8 = tmp6 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp6)
    tl.device_assert(((0 <= tmp9) & (tmp9 < 128)) | ~(xmask), "index out of bounds: 0 <= tmp9 < 128")
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 * tmp15
    tmp18 = tmp17 + tmp1
    tmp19 = tmp17 < 0
    tmp20 = tl.where(tmp19, tmp18, tmp17)
    tl.device_assert(((0 <= tmp20) & (tmp20 < 128)) | ~(xmask), "index out of bounds: 0 <= tmp20 < 128")
    tmp22 = -tmp16
    tmp23 = tmp14 + tmp22
    tmp25 = tmp24 + tmp1
    tmp26 = tmp24 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp24)
    tl.device_assert(((0 <= tmp27) & (tmp27 < 128)) | ~(xmask), "index out of bounds: 0 <= tmp27 < 128")
    tmp29 = -tmp14
    tmp30 = tmp12 + tmp29
    tmp31 = tmp30 * tmp15
    tmp32 = -tmp31
    tmp33 = tmp30 + tmp32
    tl.atomic_add(out_ptr0 + (tmp9 + 128*tmp4 + 16384*x2), tmp16, xmask, sem='relaxed')
    tl.atomic_add(out_ptr1 + (tmp20 + 128*tmp4 + 16384*x2), tmp23, xmask, sem='relaxed')
    tl.atomic_add(out_ptr2 + (tmp9 + 128*tmp27 + 16384*x2), tmp31, xmask, sem='relaxed')
    tl.atomic_add(out_ptr3 + (tmp20 + 128*tmp27 + 16384*x2), tmp33, xmask, sem='relaxed')
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/qr/cqrr34e44owoexrc4opaa3o73dqjfmhk5fkhmrioo2stjrxyizhg.py
# Topologically Sorted Source Nodes: [constant_pad_nd_2, y], Original ATen: [aten.add, aten._to_copy, aten.im2col, aten.permute, aten.clone]
# Source node to ATen node mapping:
#   constant_pad_nd_2 => convert_element_type_301
#   y => add, iota, iota_1, unsqueeze, unsqueeze_1, unsqueeze_4, unsqueeze_5
# Graph fragment:
#   %add_2670 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%index_put_1, %index_put_2), kwargs = {})
#   %add_2671 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2670, %index_put_3), kwargs = {})
#   %add_2672 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2671, %index_put_4), kwargs = {})
#   %convert_element_type_301 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2672, torch.float16), kwargs = {})
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 4, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_1 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_1, -1), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %unsqueeze_4 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add, -1), kwargs = {})
#   %unsqueeze_5 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %index_1 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%convert_element_type_301, [None, None, %unsqueeze_5, %add]), kwargs = {})
#   %permute_139 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%index_1, [0, 1, 2, 4, 3, 5]), kwargs = {})
#   %clone_26 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_139,), kwargs = {memory_format: torch.contiguous_format})
#   %view_195 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_26, [%primals_1, 16, 1024]), kwargs = {})
#   %permute_140 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_195, [0, 2, 1]), kwargs = {})
#   %clone_27 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_140,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused__to_copy_add_clone_im2col_permute_2 = async_compile.triton('triton_poi_fused__to_copy_add_clone_im2col_permute_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_im2col_permute_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_im2col_permute_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 1024)
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4*((x1 % 32)) + 128*(x0 // 4) + 512*(x1 // 32) + 16384*x2 + ((x0 % 4))), None)
    tmp1 = tl.load(in_ptr1 + (4*((x1 % 32)) + 128*(x0 // 4) + 512*(x1 // 32) + 16384*x2 + ((x0 % 4))), None)
    tmp3 = tl.load(in_ptr2 + (4*((x1 % 32)) + 128*(x0 // 4) + 512*(x1 // 32) + 16384*x2 + ((x0 % 4))), None)
    tmp5 = tl.load(in_ptr3 + (4*((x1 % 32)) + 128*(x0 // 4) + 512*(x1 // 32) + 16384*x2 + ((x0 % 4))), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/rv/crvwhecvyns4es2ny6ytlp5tsodrqr43ydnn62eqhfdebhdksqjf.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_196, [0], True), kwargs = {dtype: torch.float32})
triton_red_fused_sum_3 = async_compile.triton('triton_red_fused_sum_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_3(in_ptr0, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 16)
    x1 = xindex // 16
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 16*r0_2 + 256*ks0*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp3, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/eh/ceh6dwq3njo53sizky3vnl5aiq63xm6gv4ogpgclkiyttxldotxx.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_196, [0], True), kwargs = {dtype: torch.float32})
triton_per_fused_sum_4 = async_compile.triton('triton_per_fused_sum_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 64},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4224, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_sum_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/de/cde6lqxnvfclrhuz42y2agwt2siubq5pdvihqvzm6lur573hcuv5.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_307 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_14, torch.float32), kwargs = {})
triton_poi_fused__to_copy_5 = async_compile.triton('triton_poi_fused__to_copy_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/6b/c6b4plxtjaduru7odbwdsbvssceuaw6vyki5ylwjmzx3buwn7dcc.py
# Topologically Sorted Source Nodes: [scalar_tensor, copy_1], Original ATen: [aten.threshold_backward]
# Source node to ATen node mapping:
#   copy_1 => where
#   scalar_tensor => full_default_5
# Graph fragment:
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%view_202, 0), kwargs = {})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %full_default_5, %view_201), kwargs = {})
triton_poi_fused_threshold_backward_6 = async_compile.triton('triton_poi_fused_threshold_backward_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_threshold_backward_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/qg/cqgvowsnv3z6dyuef53runx6fin4u42hlxn4b4ksb6gtzhehomvu.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_203, [0], True), kwargs = {dtype: torch.float32})
triton_red_fused_sum_7 = async_compile.triton('triton_red_fused_sum_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_7(in_ptr0, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 2048*ks0*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/si/csipm24fhfvkzsn7x5igzykry3hed5qrbl6bxfqpg3iy6wyhalbu.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_203, [0], True), kwargs = {dtype: torch.float32})
triton_per_fused_sum_8 = async_compile.triton('triton_per_fused_sum_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 64},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 33792, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_sum_8(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/nh/cnhzviecmirdgubfl5lszw3viwgbo4jogkv7bhs4raqulaaz6j5u.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_315 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_16, torch.float32), kwargs = {})
triton_poi_fused__to_copy_9 = async_compile.triton('triton_poi_fused__to_copy_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 983040}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/pa/cpayl5vatm6zezlaytzang65624ewbzpffo5ug2kjvspqgmopopc.py
# Topologically Sorted Source Nodes: [, convert_element_type, input_4, clone_29], Original ATen: [aten._to_copy, aten.native_dropout, aten.native_dropout_backward]
# Source node to ATen node mapping:
#    => inductor_lookup_seed_default_11, inductor_random_default
#   clone_29 => mul_2411
#   convert_element_type => convert_element_type_default_52
#   input_4 => gt_51
# Graph fragment:
#   %convert_element_type_314 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_206, torch.float32), kwargs = {})
#   %convert_element_type_317 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_314, torch.float16), kwargs = {})
#   %inductor_lookup_seed_default_11 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 11), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([%primals_1, 1024, 768], %inductor_lookup_seed_default_11, rand), kwargs = {})
#   %convert_element_type_default_52 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%inductor_random_default, torch.float16), kwargs = {})
#   %gt_51 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convert_element_type_default_52, 0.2), kwargs = {})
#   %convert_element_type_318 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_51, torch.float16), kwargs = {})
#   %mul_2410 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_318, 1.25), kwargs = {})
#   %mul_2411 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_317, %mul_2410), kwargs = {})
triton_poi_fused__to_copy_native_dropout_native_dropout_backward_10 = async_compile.triton('triton_poi_fused__to_copy_native_dropout_native_dropout_backward_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp16', 'out_ptr1': '*fp16', 'load_seed_offset': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_native_dropout_native_dropout_backward_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_native_dropout_native_dropout_backward_10(in_ptr0, in_ptr1, out_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp2.to(tl.float32)
    tmp7 = 0.2
    tmp8 = tmp6 > tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = 1.25
    tmp11 = tmp9 * tmp10
    tmp12 = tmp5 * tmp11
    tl.store(out_ptr1 + (x0), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/ht/chtvhgvnoflht6xdqvnse5mvfdhlikgoekvyxjzsmt6dxy23vbtp.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_3 => convert_element_type_277
# Graph fragment:
#   %convert_element_type_277 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_148, torch.float16), kwargs = {})
triton_poi_fused__to_copy_11 = async_compile.triton('triton_poi_fused__to_copy_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/cz/cczsytqumhmioiosme234cpnssvskxzhutdo2jvcf7cyrqnpzofj.py
# Topologically Sorted Source Nodes: [layer_norm, input_1], Original ATen: [aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   input_1 => convert_element_type_270
#   layer_norm => add_2478, add_2479, mul_2280, mul_2281, rsqrt_23, sub_618, var_mean_23
# Graph fragment:
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_2473, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_2478 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_154, 1e-05), kwargs = {})
#   %rsqrt_23 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2478,), kwargs = {})
#   %sub_618 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2473, %getitem_155), kwargs = {})
#   %mul_2280 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_618, %rsqrt_23), kwargs = {})
#   %mul_2281 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2280, %primals_144), kwargs = {})
#   %add_2479 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2281, %primals_145), kwargs = {})
#   %convert_element_type_270 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2479, torch.float16), kwargs = {})
triton_per_fused__to_copy_native_layer_norm_12 = async_compile.triton('triton_per_fused__to_copy_native_layer_norm_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_native_layer_norm_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_native_layer_norm_12(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 768, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [R0_BLOCK])
    tmp15 = tl.where(r0_mask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 768.0
    tmp19 = (tmp16 / tmp18)
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp27.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp28, r0_mask)
    tl.store(out_ptr0 + (x0), tmp10, None)
    tl.store(out_ptr1 + (x0), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/os/cosnq4q6pffxtvh6vwwmbouzva6cxont6nvl5k4furusejd6gtku.py
# Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.gelu_backward]
# Source node to ATen node mapping:
#   add => add_tensor_11
#   input_1 => convert_element_type_268, view_180
#   input_2 => add_2506, convert_element_type_274, convert_element_type_275, erf_11, mul_2301, mul_2302, mul_2303
# Graph fragment:
#   %convert_element_type_268 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_147, torch.float16), kwargs = {})
#   %add_tensor_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_11, %convert_element_type_268), kwargs = {})
#   %view_180 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_11, [%primals_1, 1024, 1536]), kwargs = {})
#   %convert_element_type_274 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_180, torch.float32), kwargs = {})
#   %mul_2301 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_274, 0.5), kwargs = {})
#   %mul_2302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_274, 0.7071067811865476), kwargs = {})
#   %erf_11 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_2302,), kwargs = {})
#   %add_2506 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_11, 1), kwargs = {})
#   %mul_2303 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2301, %add_2506), kwargs = {})
#   %convert_element_type_275 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2303, torch.float16), kwargs = {})
#   %convert_element_type_326 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_209, torch.float32), kwargs = {})
#   %mul_2413 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2506, 0.5), kwargs = {})
#   %mul_2414 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_274, %convert_element_type_274), kwargs = {})
#   %mul_2415 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2414, -0.5), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_2415,), kwargs = {})
#   %mul_2416 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, 0.3989422804014327), kwargs = {})
#   %mul_2417 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_274, %mul_2416), kwargs = {})
#   %add_2676 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2413, %mul_2417), kwargs = {})
#   %mul_2418 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_326, %add_2676), kwargs = {})
#   %convert_element_type_328 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2418, torch.float16), kwargs = {})
triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13 = async_compile.triton('triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1536)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = 0.7071067811865476
    tmp8 = tmp4 * tmp7
    tmp9 = libdevice.erf(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp11 * tmp5
    tmp17 = tmp4 * tmp4
    tmp18 = -0.5
    tmp19 = tmp17 * tmp18
    tmp20 = tl_math.exp(tmp19)
    tmp21 = 0.3989422804014327
    tmp22 = tmp20 * tmp21
    tmp23 = tmp4 * tmp22
    tmp24 = tmp16 + tmp23
    tmp25 = tmp15 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp13, None)
    tl.store(in_out_ptr0 + (x2), tmp26, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/dz/cdzja5bi43r7lbexju4qlghsi3xz2vkgx4fekgmk4cd44po3zk4a.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_207, [0], True), kwargs = {dtype: torch.float32})
triton_red_fused_sum_14 = async_compile.triton('triton_red_fused_sum_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_14(in_ptr0, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 768)
    x1 = xindex // 768
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_2 + 12288*ks0*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/f3/cf3xmucdr7joqhsch6z44smpwu2z73vfgbnymg5u4zcqyjoz7my4.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_207, [0], True), kwargs = {dtype: torch.float32})
triton_per_fused_sum_15 = async_compile.triton('triton_per_fused_sum_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 202752, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_sum_15(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 768
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/vc/cvcdbxexp7pvjqqi6z7nfwsmwvupbb7slv4thmczkxawu5vnbcrq.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_324 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_18, torch.float32), kwargs = {})
triton_poi_fused__to_copy_16 = async_compile.triton('triton_poi_fused__to_copy_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11796480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/uk/cuk4pfch4v7fhgls6aggheh3sragvxsefsf4ai2y4hbfsc46c5cp.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_210, [0], True), kwargs = {dtype: torch.float32})
triton_red_fused_sum_17 = async_compile.triton('triton_red_fused_sum_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_17(in_ptr0, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1536)
    x1 = xindex // 1536
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1536*r0_2 + 24576*ks0*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/n2/cn25zwliigpdfg5t3c7yd2seelireswpcdhpk3xyayvu55uf4u4r.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_210, [0], True), kwargs = {dtype: torch.float32})
triton_per_fused_sum_18 = async_compile.triton('triton_per_fused_sum_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 405504, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_sum_18(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1536
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1536*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/fm/cfmzlhuwn2r33khvmsdy2hbjdkdl7synimm72eqqqa7kcbogrd56.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm, aten.native_layer_norm_backward, aten.add]
# Source node to ATen node mapping:
#   layer_norm => add_2478, mul_2280, rsqrt_23, sub_618, var_mean_23
# Graph fragment:
#   %convert_element_type_314 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_206, torch.float32), kwargs = {})
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_2473, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_2478 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_154, 1e-05), kwargs = {})
#   %rsqrt_23 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2478,), kwargs = {})
#   %sub_618 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2473, %getitem_155), kwargs = {})
#   %mul_2280 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_618, %rsqrt_23), kwargs = {})
#   %convert_element_type_334 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_212, torch.float32), kwargs = {})
#   %mul_2420 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_334, %primals_144), kwargs = {})
#   %mul_2421 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2420, 768), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2420, [2], True), kwargs = {})
#   %mul_2422 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2420, %mul_2280), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2422, [2], True), kwargs = {})
#   %mul_2423 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2280, %sum_6), kwargs = {})
#   %sub_669 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_2421, %sum_5), kwargs = {})
#   %sub_670 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_669, %mul_2423), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_23, 768), kwargs = {})
#   %mul_2424 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %sub_670), kwargs = {})
#   %add_2677 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_314, %mul_2424), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_19 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp16', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp10 = tmp8 - tmp9
    tmp12 = 768.0
    tmp13 = (tmp11 / tmp12)
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp10 * tmp16
    tmp18 = tmp3 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [R0_BLOCK])
    tmp21 = tl.where(r0_mask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = tmp23.to(tl.float32)
    tmp25 = 0.0013020833333333333
    tmp26 = tmp16 * tmp25
    tmp27 = tmp3 * tmp12
    tmp28 = tmp27 - tmp7
    tmp29 = tmp17 * tmp22
    tmp30 = tmp28 - tmp29
    tmp31 = tmp26 * tmp30
    tmp32 = tmp24 + tmp31
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp32, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/pt/cptnr6dcvdfrmjoo32bjuk7g2mumqtfvzj235gnhs464sjhu6zxq.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm => add_2478, mul_2280, rsqrt_23, sub_618, var_mean_23
# Graph fragment:
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_2473, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_2478 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_154, 1e-05), kwargs = {})
#   %rsqrt_23 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2478,), kwargs = {})
#   %sub_618 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2473, %getitem_155), kwargs = {})
#   %mul_2280 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_618, %rsqrt_23), kwargs = {})
#   %convert_element_type_334 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_212, torch.float32), kwargs = {})
#   %mul_2425 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_334, %mul_2280), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2425, [0, 1]), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_334, [0, 1]), kwargs = {})
triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_20 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 768)
    x1 = xindex // 768
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 768*(((r0_2 + 16*ks0*x1) % (1024*ks0)))), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + 768*(((r0_2 + 16*ks0*x1) % (1024*ks0)))), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (((r0_2 + 16*ks0*x1) % (1024*ks0))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (((r0_2 + 16*ks0*x1) % (1024*ks0))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp4 = tmp2 - tmp3
        tmp6 = 768.0
        tmp7 = (tmp5 / tmp6)
        tmp8 = 1e-05
        tmp9 = tmp7 + tmp8
        tmp10 = libdevice.rsqrt(tmp9)
        tmp11 = tmp4 * tmp10
        tmp12 = tmp1 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask, tmp15, _tmp14)
        tmp16 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask, tmp18, _tmp17)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/fj/cfjx4j5tavlhc5qgek4r7uogmd2pslzts5plheq6dgx5qdl4zbme.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.clone, aten._unsafe_view]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_337 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2677, torch.float16), kwargs = {})
#   %permute_158 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_337, [1, 0, 2]), kwargs = {})
#   %clone_30 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_158,), kwargs = {memory_format: torch.contiguous_format})
#   %view_213 : [num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_30, [%mul_15, 768]), kwargs = {})
triton_poi_fused__to_copy__unsafe_view_clone_transpose_21 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_clone_transpose_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_clone_transpose_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_clone_transpose_21(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 768)
    x1 = xindex // 768
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 768*(x1 // ks0) + 786432*((x1 % ks0))), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/7b/c7bok3kvqeaz366wjrkpklcpfhxcdbcxqjdwwzmikmjmnfglpt7k.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   multi_head_attention_forward => convert_element_type_264
# Graph fragment:
#   %convert_element_type_264 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_142, torch.float16), kwargs = {})
triton_poi_fused__to_copy_22 = async_compile.triton('triton_poi_fused__to_copy_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_22(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/v5/cv5hprogm65biiepqpts2nrdhpfbkjxj5dmjx4qyruif4jesafip.py
# Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_24, convert_element_type_260
#   query => permute_124
#   y => add_2326, add_2327, mul_2136, mul_2137, rsqrt_22, sub_581, var_mean_22
# Graph fragment:
#   %var_mean_22 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_2321, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_2326 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_143, 1e-05), kwargs = {})
#   %rsqrt_22 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2326,), kwargs = {})
#   %sub_581 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2321, %getitem_144), kwargs = {})
#   %mul_2136 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_581, %rsqrt_22), kwargs = {})
#   %mul_2137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2136, %primals_138), kwargs = {})
#   %add_2327 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2137, %primals_139), kwargs = {})
#   %permute_124 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_2327, [1, 0, 2]), kwargs = {})
#   %convert_element_type_260 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_124, torch.float16), kwargs = {})
#   %clone_24 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_260,), kwargs = {memory_format: torch.contiguous_format})
triton_per_fused__to_copy_clone_native_layer_norm_transpose_23 = async_compile.triton('triton_per_fused__to_copy_clone_native_layer_norm_transpose_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp16', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_clone_native_layer_norm_transpose_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_clone_native_layer_norm_transpose_23(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    x2 = (xindex % 1024)
    x3 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 768, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [R0_BLOCK])
    tmp15 = tl.where(r0_mask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 768.0
    tmp19 = (tmp16 / tmp18)
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp27.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 768*x3 + 768*ks0*x2), tmp28, r0_mask)
    tl.store(out_ptr0 + (x0), tmp10, None)
    tl.store(out_ptr1 + (x0), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/62/c62vgqvzxsugaelwjbyern6hui6q6rq2jozwfbwqaopwb2dpjpez.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   multi_head_attention_forward => convert_element_type_259, permute_125
# Graph fragment:
#   %convert_element_type_259 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_141, torch.float16), kwargs = {})
#   %permute_125 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_259, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_24 = async_compile.triton('triton_poi_fused__to_copy_t_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 14155776}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_24(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/t7/ct76dwqr23hw7tzkjlqo44jbuiyffesyp7kfuvjbgytipbzzbyqi.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_2360, clone_25, convert_element_type_258, permute_126, squeeze_11, unsqueeze_17, view_170
# Graph fragment:
#   %convert_element_type_258 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_140, torch.float16), kwargs = {})
#   %add_2360 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_169, %convert_element_type_258), kwargs = {})
#   %view_170 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_2360, [1024, %primals_1, 3, 768]), kwargs = {})
#   %unsqueeze_17 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_170, 0), kwargs = {})
#   %permute_126 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_17, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_11 : [num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_126, -2), kwargs = {})
#   %clone_25 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_11,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25 = async_compile.triton('triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp16', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 768)
    x1 = ((xindex // 768) % ks0)
    x2 = xindex // ks1
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 768*x2 + 2304*x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + 768*x2), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tl.store(out_ptr0 + (x3), tmp3, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/ze/czevsknzcsoxjn4libefc7yxjkymnemahudowluf7asu4m6in5hw.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_2360, clone_25, convert_element_type_258, permute_126, permute_127, permute_128, permute_129, select_33, select_34, select_35, squeeze_11, unsqueeze_17, view_170, view_171, view_172, view_173, view_174, view_175, view_176
# Graph fragment:
#   %convert_element_type_258 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_140, torch.float16), kwargs = {})
#   %add_2360 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_169, %convert_element_type_258), kwargs = {})
#   %view_170 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_2360, [1024, %primals_1, 3, 768]), kwargs = {})
#   %unsqueeze_17 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_170, 0), kwargs = {})
#   %permute_126 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_17, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_11 : [num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_126, -2), kwargs = {})
#   %clone_25 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_11,), kwargs = {memory_format: torch.contiguous_format})
#   %select_33 : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_25, 0, 0), kwargs = {})
#   %select_34 : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_25, 0, 1), kwargs = {})
#   %select_35 : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_25, 0, 2), kwargs = {})
#   %view_171 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_33, [1024, %mul_127, 96]), kwargs = {})
#   %permute_127 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_171, [1, 0, 2]), kwargs = {})
#   %view_172 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_34, [1024, %mul_127, 96]), kwargs = {})
#   %permute_128 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_172, [1, 0, 2]), kwargs = {})
#   %view_173 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_35, [1024, %mul_127, 96]), kwargs = {})
#   %permute_129 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_173, [1, 0, 2]), kwargs = {})
#   %view_174 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_127, [%primals_1, 8, 1024, 96]), kwargs = {})
#   %view_175 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_128, [%primals_1, 8, 1024, 96]), kwargs = {})
#   %view_176 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_129, [%primals_1, 8, 1024, 96]), kwargs = {})
#   %graphsafe_run_with_rng_state_11 : [num_users=4] = call_function[target=torch.ops.higher_order.graphsafe_run_with_rng_state](args = (aten._scaled_dot_product_flash_attention.default, %view_174, %view_175, %view_176, 0.2), kwargs = {scale: 0.10206207261596577, rng_state: %bwd_rng_state_11})
#   %_scaled_dot_product_flash_attention_backward : [num_users=3] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention_backward.default](args = (%permute_163, %view_174, %view_175, %view_176, %getitem_145, %getitem_146, None, None, 1024, 1024, 0.2, False, %getitem_151, %getitem_152), kwargs = {scale: 0.10206207261596577})
triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_26 = async_compile.triton('triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_26(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 8)
    x2 = ((xindex // 768) % ks0)
    x3 = xindex // ks1
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 96*x1 + 768*((((x0 + 96*x1 + 768*x2) // 768) % ks0)) + 768*ks0*((((x0 + 96*x1 + 768*x2 + 768*ks0*x3) // (768*ks0)) % 1024))), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0 + 96*x1 + 768*((((x0 + 96*x1 + 768*x2) // 768) % ks0)) + 768*ks0*((((x0 + 96*x1 + 768*x2 + 768*ks0*x3) // ks1) % 1024))), None, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
    tl.store(out_ptr1 + (x4), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/5k/c5kqrjhpocynabvuxiiaj2cwsxip6nzx5y5xuhqvis2abpj7rvdi.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_2360, clone_25, convert_element_type_258, permute_126, permute_127, permute_128, permute_129, select_33, select_34, select_35, squeeze_11, unsqueeze_17, view_170, view_171, view_172, view_173, view_174, view_175, view_176
# Graph fragment:
#   %convert_element_type_258 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_140, torch.float16), kwargs = {})
#   %add_2360 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_169, %convert_element_type_258), kwargs = {})
#   %view_170 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_2360, [1024, %primals_1, 3, 768]), kwargs = {})
#   %unsqueeze_17 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_170, 0), kwargs = {})
#   %permute_126 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_17, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_11 : [num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_126, -2), kwargs = {})
#   %clone_25 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_11,), kwargs = {memory_format: torch.contiguous_format})
#   %select_33 : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_25, 0, 0), kwargs = {})
#   %select_34 : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_25, 0, 1), kwargs = {})
#   %select_35 : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_25, 0, 2), kwargs = {})
#   %view_171 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_33, [1024, %mul_127, 96]), kwargs = {})
#   %permute_127 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_171, [1, 0, 2]), kwargs = {})
#   %view_172 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_34, [1024, %mul_127, 96]), kwargs = {})
#   %permute_128 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_172, [1, 0, 2]), kwargs = {})
#   %view_173 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_35, [1024, %mul_127, 96]), kwargs = {})
#   %permute_129 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_173, [1, 0, 2]), kwargs = {})
#   %view_174 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_127, [%primals_1, 8, 1024, 96]), kwargs = {})
#   %view_175 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_128, [%primals_1, 8, 1024, 96]), kwargs = {})
#   %view_176 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_129, [%primals_1, 8, 1024, 96]), kwargs = {})
#   %graphsafe_run_with_rng_state_11 : [num_users=4] = call_function[target=torch.ops.higher_order.graphsafe_run_with_rng_state](args = (aten._scaled_dot_product_flash_attention.default, %view_174, %view_175, %view_176, 0.2), kwargs = {scale: 0.10206207261596577, rng_state: %bwd_rng_state_11})
#   %_scaled_dot_product_flash_attention_backward : [num_users=3] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention_backward.default](args = (%permute_163, %view_174, %view_175, %view_176, %getitem_145, %getitem_146, None, None, 1024, 1024, 0.2, False, %getitem_151, %getitem_152), kwargs = {scale: 0.10206207261596577})
triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27 = async_compile.triton('triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 8)
    x2 = ((xindex // 768) % ks0)
    x3 = xindex // ks1
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (ks2 + x0 + 96*x1 + 768*((((x0 + 96*x1 + 768*x2) // 768) % ks0)) + 768*ks0*((((x0 + 96*x1 + 768*x2 + 768*ks0*x3) // ks1) % 1024))), None, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
    tl.store(out_ptr1 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/ls/cls64r4h5gzaqufyljt7l6beogviragifjbltx3vtspfg4ue2xr7.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_2360, clone_25, convert_element_type_258, permute_126, permute_127, permute_128, permute_129, select_33, select_34, select_35, squeeze_11, unsqueeze_17, view_170, view_171, view_172, view_173, view_174, view_175, view_176
# Graph fragment:
#   %convert_element_type_258 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_140, torch.float16), kwargs = {})
#   %add_2360 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_169, %convert_element_type_258), kwargs = {})
#   %view_170 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_2360, [1024, %primals_1, 3, 768]), kwargs = {})
#   %unsqueeze_17 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_170, 0), kwargs = {})
#   %permute_126 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_17, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_11 : [num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_126, -2), kwargs = {})
#   %clone_25 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_11,), kwargs = {memory_format: torch.contiguous_format})
#   %select_33 : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_25, 0, 0), kwargs = {})
#   %select_34 : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_25, 0, 1), kwargs = {})
#   %select_35 : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_25, 0, 2), kwargs = {})
#   %view_171 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_33, [1024, %mul_127, 96]), kwargs = {})
#   %permute_127 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_171, [1, 0, 2]), kwargs = {})
#   %view_172 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_34, [1024, %mul_127, 96]), kwargs = {})
#   %permute_128 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_172, [1, 0, 2]), kwargs = {})
#   %view_173 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_35, [1024, %mul_127, 96]), kwargs = {})
#   %permute_129 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_173, [1, 0, 2]), kwargs = {})
#   %view_174 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_127, [%primals_1, 8, 1024, 96]), kwargs = {})
#   %view_175 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_128, [%primals_1, 8, 1024, 96]), kwargs = {})
#   %view_176 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_129, [%primals_1, 8, 1024, 96]), kwargs = {})
#   %graphsafe_run_with_rng_state_11 : [num_users=4] = call_function[target=torch.ops.higher_order.graphsafe_run_with_rng_state](args = (aten._scaled_dot_product_flash_attention.default, %view_174, %view_175, %view_176, 0.2), kwargs = {scale: 0.10206207261596577, rng_state: %bwd_rng_state_11})
#   %_scaled_dot_product_flash_attention_backward : [num_users=3] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention_backward.default](args = (%permute_163, %view_174, %view_175, %view_176, %getitem_145, %getitem_146, None, None, 1024, 1024, 0.2, False, %getitem_151, %getitem_152), kwargs = {scale: 0.10206207261596577})
triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28 = async_compile.triton('triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 8)
    x2 = ((xindex // 768) % ks0)
    x3 = xindex // ks1
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 96*x1 + 768*((((x0 + 96*x1 + 768*x2) // 768) % ks0)) + 1572864*ks0 + 768*ks0*((((x0 + 96*x1 + 768*x2 + 768*ks0*x3) // ks1) % 1024))), None, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
    tl.store(out_ptr1 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/nr/cnryysqfbihi7el2s7jjdy4cdzr242sqor5fbw6iznojs6bwiwjx.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_343 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_22, torch.float32), kwargs = {})
triton_poi_fused__to_copy_29 = async_compile.triton('triton_poi_fused__to_copy_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5898240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_29(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/bx/cbxfika2a33acw7ic6l2ix5cjadmtl4c2syjmj4ugryeaussxltt.py
# Topologically Sorted Source Nodes: [full_5, _generalized_scatter, _generalized_scatter_1, _generalized_scatter_2], Original ATen: [aten.view, aten.transpose, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
# Source node to ATen node mapping:
#   _generalized_scatter => select_scatter_default
#   _generalized_scatter_1 => select_scatter_default_1
#   _generalized_scatter_2 => select_scatter_default_2
#   full_5 => full_default_6
# Graph fragment:
#   %view_216 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_158, [%mul_127, 1024, 96]), kwargs = {})
#   %view_217 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_157, [%mul_127, 1024, 96]), kwargs = {})
#   %view_218 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_156, [%mul_127, 1024, 96]), kwargs = {})
#   %permute_164 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_216, [1, 0, 2]), kwargs = {})
#   %view_219 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_164, [1024, %primals_1, 768]), kwargs = {})
#   %permute_165 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_217, [1, 0, 2]), kwargs = {})
#   %view_220 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_165, [1024, %primals_1, 768]), kwargs = {})
#   %permute_166 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_218, [1, 0, 2]), kwargs = {})
#   %view_221 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_166, [1024, %primals_1, 768]), kwargs = {})
#   %full_default_6 : [num_users=36] = call_function[target=torch.ops.aten.full.default](args = ([3, 1024, %primals_1, 768], 0), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %view_219, 0, 2), kwargs = {})
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %view_220, 0, 1), kwargs = {})
#   %add_2678 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %view_221, 0, 0), kwargs = {})
#   %add_2679 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2678, %select_scatter_default_2), kwargs = {})
#   %unsqueeze_30 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_2679, 3), kwargs = {})
#   %permute_167 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_30, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_12 : [num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_167, 0), kwargs = {})
#   %clone_31 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_12,), kwargs = {memory_format: torch.contiguous_format})
#   %view_222 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_31, [1024, %primals_1, 2304]), kwargs = {})
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_222, [0, 1], True), kwargs = {dtype: torch.float32})
triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30 = async_compile.triton('triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 73728
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 2304)
    x1 = xindex // 2304
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp3 = tl.load(in_ptr0 + (96*((((((768*((r0_2 % ks1)) + ((x0 % 768))) // 96) % (8*ks1))) % (8*ks1))) + 768*ks1*((((768*((r0_2 % ks1)) + 768*ks1*((((r0_2 + 32*ks1*x1) // ks1) % 1024)) + ((x0 % 768))) // ks0) % 1024)) + ((((x0 % 768)) % 96))), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr1 + (96*((((((768*((r0_2 % ks1)) + ((x0 % 768))) // 96) % (8*ks1))) % (8*ks1))) + 768*ks1*((((768*((r0_2 % ks1)) + 768*ks1*((((r0_2 + 32*ks1*x1) // ks1) % 1024)) + ((x0 % 768))) // ks0) % 1024)) + ((((x0 % 768)) % 96))), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr2 + (96*((((((768*((r0_2 % ks1)) + ((x0 % 768))) // 96) % (8*ks1))) % (8*ks1))) + 768*ks1*((((768*((r0_2 % ks1)) + 768*ks1*((((r0_2 + 32*ks1*x1) // ks1) % 1024)) + ((x0 % 768))) // ks0) % 1024)) + ((((x0 % 768)) % 96))), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp0 = x0 // 768
        tmp1 = tl.full([1, 1], 2, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.full([1, 1], 1, tl.int32)
        tmp7 = tmp0 == tmp6
        tmp9 = tl.where(tmp7, tmp8, tmp4)
        tmp10 = tmp5 + tmp9
        tmp11 = tl.full([1, 1], 0, tl.int32)
        tmp12 = tmp0 == tmp11
        tmp14 = tl.where(tmp12, tmp13, tmp4)
        tmp15 = tmp10 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(r0_mask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/64/c64x3kmaq6wchzymi2e6vaglvkz7v5zp3j6lpxcd344cj3oh2u27.py
# Topologically Sorted Source Nodes: [full_5, _generalized_scatter, _generalized_scatter_1, _generalized_scatter_2], Original ATen: [aten.view, aten.transpose, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
# Source node to ATen node mapping:
#   _generalized_scatter => select_scatter_default
#   _generalized_scatter_1 => select_scatter_default_1
#   _generalized_scatter_2 => select_scatter_default_2
#   full_5 => full_default_6
# Graph fragment:
#   %view_216 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_158, [%mul_127, 1024, 96]), kwargs = {})
#   %view_217 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_157, [%mul_127, 1024, 96]), kwargs = {})
#   %view_218 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_156, [%mul_127, 1024, 96]), kwargs = {})
#   %permute_164 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_216, [1, 0, 2]), kwargs = {})
#   %view_219 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_164, [1024, %primals_1, 768]), kwargs = {})
#   %permute_165 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_217, [1, 0, 2]), kwargs = {})
#   %view_220 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_165, [1024, %primals_1, 768]), kwargs = {})
#   %permute_166 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_218, [1, 0, 2]), kwargs = {})
#   %view_221 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_166, [1024, %primals_1, 768]), kwargs = {})
#   %full_default_6 : [num_users=36] = call_function[target=torch.ops.aten.full.default](args = ([3, 1024, %primals_1, 768], 0), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %view_219, 0, 2), kwargs = {})
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %view_220, 0, 1), kwargs = {})
#   %add_2678 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %view_221, 0, 0), kwargs = {})
#   %add_2679 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2678, %select_scatter_default_2), kwargs = {})
#   %unsqueeze_30 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_2679, 3), kwargs = {})
#   %permute_167 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_30, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_12 : [num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_167, 0), kwargs = {})
#   %clone_31 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_12,), kwargs = {memory_format: torch.contiguous_format})
#   %view_222 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_31, [1024, %primals_1, 2304]), kwargs = {})
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_222, [0, 1], True), kwargs = {dtype: torch.float32})
triton_per_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_31 = async_compile.triton('triton_per_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 32},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 313344, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_31(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2304
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2304*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/w6/cw6q6gia6g4qfkrxz7s26uwxf4pemoxqxyxjreyxna4i4gshtrur.py
# Topologically Sorted Source Nodes: [full_5, _generalized_scatter, _generalized_scatter_1, _generalized_scatter_2], Original ATen: [aten.view, aten.transpose, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.clone]
# Source node to ATen node mapping:
#   _generalized_scatter => select_scatter_default
#   _generalized_scatter_1 => select_scatter_default_1
#   _generalized_scatter_2 => select_scatter_default_2
#   full_5 => full_default_6
# Graph fragment:
#   %view_216 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_158, [%mul_127, 1024, 96]), kwargs = {})
#   %view_217 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_157, [%mul_127, 1024, 96]), kwargs = {})
#   %view_218 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_156, [%mul_127, 1024, 96]), kwargs = {})
#   %permute_164 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_216, [1, 0, 2]), kwargs = {})
#   %view_219 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_164, [1024, %primals_1, 768]), kwargs = {})
#   %permute_165 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_217, [1, 0, 2]), kwargs = {})
#   %view_220 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_165, [1024, %primals_1, 768]), kwargs = {})
#   %permute_166 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_218, [1, 0, 2]), kwargs = {})
#   %view_221 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_166, [1024, %primals_1, 768]), kwargs = {})
#   %full_default_6 : [num_users=36] = call_function[target=torch.ops.aten.full.default](args = ([3, 1024, %primals_1, 768], 0), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %view_219, 0, 2), kwargs = {})
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %view_220, 0, 1), kwargs = {})
#   %add_2678 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %view_221, 0, 0), kwargs = {})
#   %add_2679 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2678, %select_scatter_default_2), kwargs = {})
#   %unsqueeze_30 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_2679, 3), kwargs = {})
#   %permute_167 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_30, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_12 : [num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_167, 0), kwargs = {})
#   %clone_31 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_12,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32 = async_compile.triton('triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 768) % 3)
    x0 = (xindex % 768)
    x2 = ((xindex // 2304) % ks0)
    x3 = xindex // ks1
    x4 = xindex
    tmp3 = tl.load(in_ptr0 + (96*((((((x0 + 768*x2) // 96) % (8*ks0))) % (8*ks0))) + 768*ks0*((((x0 + 768*x2 + 768*ks0*x3) // ks2) % 1024)) + ((x0 % 96))), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (96*((((((x0 + 768*x2) // 96) % (8*ks0))) % (8*ks0))) + 768*ks0*((((x0 + 768*x2 + 768*ks0*x3) // ks2) % 1024)) + ((x0 % 96))), None, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (96*((((((x0 + 768*x2) // 96) % (8*ks0))) % (8*ks0))) + 768*ks0*((((x0 + 768*x2 + 768*ks0*x3) // ks2) % 1024)) + ((x0 % 96))), None, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tl.full([1], 1, tl.int32)
    tmp7 = tmp0 == tmp6
    tmp9 = tl.where(tmp7, tmp8, tmp4)
    tmp10 = tmp5 + tmp9
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp14 = tl.where(tmp12, tmp13, tmp4)
    tmp15 = tmp10 + tmp14
    tl.store(out_ptr0 + (x4), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/fs/cfswzyjgb6wnuvvwomidqagb7jl5fdlgoda75zgqnago75bulgj3.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_351 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_23, torch.float32), kwargs = {})
triton_poi_fused__to_copy_33 = async_compile.triton('triton_poi_fused__to_copy_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 17694720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/3f/c3fm2et5kynrm2nrzjbq3m76uu7tpndk3gica47waxqewmlr6ctn.py
# Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   y => add_2326, mul_2136, rsqrt_22, sub_581, var_mean_22
# Graph fragment:
#   %var_mean_22 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_2321, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_2326 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_143, 1e-05), kwargs = {})
#   %rsqrt_22 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2326,), kwargs = {})
#   %sub_581 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2321, %getitem_144), kwargs = {})
#   %mul_2136 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_581, %rsqrt_22), kwargs = {})
#   %convert_element_type_350 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_225, torch.float32), kwargs = {})
#   %permute_172 : [num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_350, [1, 0, 2]), kwargs = {})
#   %mul_2427 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_172, %primals_138), kwargs = {})
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2427, [2], True), kwargs = {})
#   %mul_2429 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2427, %mul_2136), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2429, [2], True), kwargs = {})
triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x2 = (xindex % 1024)
    x3 = xindex // 1024
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp22 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr0 + (r0_1 + 768*x3 + 768*ks0*x2), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 * tmp2
        tmp12 = tmp10 - tmp11
        tmp14 = 768.0
        tmp15 = (tmp13 / tmp14)
        tmp16 = 1e-05
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp12 * tmp18
        tmp20 = tmp9 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(r0_mask & xmask, tmp23, _tmp22)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp5, xmask)
    tl.store(out_ptr1 + (x0), tmp22, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/6c/c6cnes7fkjcy2jnxnniwhe74zcnlnvbel5nrxbyu6qwbfoxbfptn.py
# Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   y => add_2326, mul_2136, rsqrt_22, sub_581, var_mean_22
# Graph fragment:
#   %var_mean_22 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_2321, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_2326 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_143, 1e-05), kwargs = {})
#   %rsqrt_22 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2326,), kwargs = {})
#   %sub_581 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2321, %getitem_144), kwargs = {})
#   %mul_2136 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_581, %rsqrt_22), kwargs = {})
#   %convert_element_type_350 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_225, torch.float32), kwargs = {})
#   %permute_172 : [num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_350, [1, 0, 2]), kwargs = {})
#   %mul_2432 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_172, %mul_2136), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2432, [0, 1]), kwargs = {})
#   %sum_14 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute_172, [0, 1]), kwargs = {})
triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 768)
    x1 = xindex // 768
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 768*((((r0_2 + 16*ks0*x1) // 1024) % ks0)) + 768*ks0*(((r0_2 + 16*ks0*x1) % 1024))), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + 768*(((r0_2 + 16*ks0*x1) % ks1))), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (((r0_2 + 16*ks0*x1) % ks1)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (((r0_2 + 16*ks0*x1) % ks1)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp4 = tmp2 - tmp3
        tmp6 = 768.0
        tmp7 = (tmp5 / tmp6)
        tmp8 = 1e-05
        tmp9 = tmp7 + tmp8
        tmp10 = libdevice.rsqrt(tmp9)
        tmp11 = tmp4 * tmp10
        tmp12 = tmp1 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask, tmp15, _tmp14)
        tmp16 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask, tmp18, _tmp17)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/n7/cn7kvzw7g2b3ib6ai5zwmlv3bd4mexyeokix2ffxzy6iv5ein4gx.py
# Topologically Sorted Source Nodes: [y, , convert_element_type, input_4, clone_32], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout, aten.native_dropout_backward]
# Source node to ATen node mapping:
#    => inductor_lookup_seed_default_10, inductor_random_default_1
#   clone_32 => mul_2434
#   convert_element_type => convert_element_type_default_53
#   input_4 => gt_47
#   y => add_2326, mul_2136, rsqrt_22, sub_581, var_mean_22
# Graph fragment:
#   %var_mean_22 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_2321, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_2326 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_143, 1e-05), kwargs = {})
#   %rsqrt_22 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2326,), kwargs = {})
#   %sub_581 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2321, %getitem_144), kwargs = {})
#   %mul_2136 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_581, %rsqrt_22), kwargs = {})
#   %convert_element_type_350 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_225, torch.float32), kwargs = {})
#   %permute_172 : [num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_350, [1, 0, 2]), kwargs = {})
#   %mul_2427 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_172, %primals_138), kwargs = {})
#   %mul_2428 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2427, 768), kwargs = {})
#   %mul_2430 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2136, %sum_12), kwargs = {})
#   %sub_672 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_2428, %sum_11), kwargs = {})
#   %sub_673 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_672, %mul_2430), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_22, 768), kwargs = {})
#   %mul_2431 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %sub_673), kwargs = {})
#   %add_2680 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2677, %mul_2431), kwargs = {})
#   %convert_element_type_353 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2680, torch.float16), kwargs = {})
#   %inductor_lookup_seed_default_10 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 10), kwargs = {})
#   %inductor_random_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([%primals_1, 1024, 768], %inductor_lookup_seed_default_10, rand), kwargs = {})
#   %convert_element_type_default_53 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%inductor_random_default_1, torch.float16), kwargs = {})
#   %gt_47 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convert_element_type_default_53, 0.2), kwargs = {})
#   %convert_element_type_354 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_47, torch.float16), kwargs = {})
#   %mul_2433 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_354, 1.25), kwargs = {})
#   %mul_2434 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_353, %mul_2433), kwargs = {})
triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36 = async_compile.triton('triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*i64', 'out_ptr1': '*fp16', 'ks0': 'i64', 'load_seed_offset': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, ks0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 768
    x0 = (xindex % 768)
    x1 = ((xindex // 768) % 1024)
    x2 = xindex // 786432
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x0 + 768*x2 + 768*ks0*x1), None).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2 + ks0*x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x3), None)
    tmp17 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x4), None, eviction_policy='evict_last')
    tmp2 = 768.0
    tmp3 = (tmp1 / tmp2)
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.rsqrt(tmp5)
    tmp7 = 0.0013020833333333333
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp12 * tmp2
    tmp15 = tmp13 - tmp14
    tmp18 = tmp16 - tmp17
    tmp19 = tmp18 * tmp6
    tmp21 = tmp19 * tmp20
    tmp22 = tmp15 - tmp21
    tmp23 = tmp8 * tmp22
    tmp24 = tmp0 + tmp23
    tmp25 = tl.load(in_ptr7 + load_seed_offset)
    tmp26 = x3
    tmp27 = tl.rand(tmp25, (tmp26).to(tl.uint32))
    tmp28 = tmp24.to(tl.float32)
    tmp29 = tmp27.to(tl.float32)
    tmp30 = 0.2
    tmp31 = tmp29 > tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = 1.25
    tmp34 = tmp32 * tmp33
    tmp35 = tmp28 * tmp34
    tl.store(in_out_ptr0 + (x3), tmp24, None)
    tl.store(out_ptr1 + (x3), tmp35, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/bk/cbk26wm4mze2qyjx5dgyzxxk4pnmileku5crjvunmjtxlaiv5dek.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward, aten.add]
# Source node to ATen node mapping:
#   layer_norm => add_2270, mul_2090, rsqrt_21, sub_566, var_mean_21
# Graph fragment:
#   %var_mean_21 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_2265, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_2270 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_141, 1e-05), kwargs = {})
#   %rsqrt_21 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2270,), kwargs = {})
#   %sub_566 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2265, %getitem_142), kwargs = {})
#   %mul_2090 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_566, %rsqrt_21), kwargs = {})
#   %convert_element_type_370 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_231, torch.float32), kwargs = {})
#   %mul_2443 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_370, %primals_132), kwargs = {})
#   %mul_2444 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2443, 768), kwargs = {})
#   %sum_17 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2443, [2], True), kwargs = {})
#   %mul_2445 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2443, %mul_2090), kwargs = {})
#   %sum_18 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2445, [2], True), kwargs = {})
#   %mul_2446 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2090, %sum_18), kwargs = {})
#   %sub_675 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_2444, %sum_17), kwargs = {})
#   %sub_676 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_675, %mul_2446), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_21, 768), kwargs = {})
#   %mul_2447 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %sub_676), kwargs = {})
#   %add_2683 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2680, %mul_2447), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp10 = tmp8 - tmp9
    tmp12 = 768.0
    tmp13 = (tmp11 / tmp12)
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp10 * tmp16
    tmp18 = tmp3 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [R0_BLOCK])
    tmp21 = tl.where(r0_mask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = 0.0013020833333333333
    tmp25 = tmp16 * tmp24
    tmp26 = tmp3 * tmp12
    tmp27 = tmp26 - tmp7
    tmp28 = tmp17 * tmp22
    tmp29 = tmp27 - tmp28
    tmp30 = tmp25 * tmp29
    tmp31 = tmp23 + tmp30
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp31, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/lt/cltuzivejppazn5uoioclk3th2b26xtw5edwhbrvkqcsjckoql5e.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm => add_2270, mul_2090, rsqrt_21, sub_566, var_mean_21
# Graph fragment:
#   %var_mean_21 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_2265, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_2270 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_141, 1e-05), kwargs = {})
#   %rsqrt_21 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2270,), kwargs = {})
#   %sub_566 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2265, %getitem_142), kwargs = {})
#   %mul_2090 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_566, %rsqrt_21), kwargs = {})
#   %convert_element_type_370 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_231, torch.float32), kwargs = {})
#   %mul_2448 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_370, %mul_2090), kwargs = {})
#   %sum_19 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2448, [0, 1]), kwargs = {})
#   %sum_20 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_370, [0, 1]), kwargs = {})
triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 768)
    x1 = xindex // 768
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 768*(((r0_2 + 16*ks1*x1) % ks0))), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + 768*(((r0_2 + 16*ks1*x1) % ks0))), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (((r0_2 + 16*ks1*x1) % ks0)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (((r0_2 + 16*ks1*x1) % ks0)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp4 = tmp2 - tmp3
        tmp6 = 768.0
        tmp7 = (tmp5 / tmp6)
        tmp8 = 1e-05
        tmp9 = tmp7 + tmp8
        tmp10 = libdevice.rsqrt(tmp9)
        tmp11 = tmp4 * tmp10
        tmp12 = tmp1 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask, tmp15, _tmp14)
        tmp16 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask, tmp18, _tmp17)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/4f/c4foe3t7smxdkqyqjtc6hami5ytunhirycb3hdfu3zcbvxfrqya2.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_2152, clone_23, convert_element_type_235, permute_115, permute_116, permute_117, permute_118, select_30, select_31, select_32, squeeze_10, unsqueeze_16, view_155, view_156, view_157, view_158, view_159, view_160, view_161
# Graph fragment:
#   %convert_element_type_235 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_128, torch.float16), kwargs = {})
#   %add_2152 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_154, %convert_element_type_235), kwargs = {})
#   %view_155 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_2152, [1024, %primals_1, 3, 768]), kwargs = {})
#   %unsqueeze_16 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_155, 0), kwargs = {})
#   %permute_115 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_16, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_10 : [num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_115, -2), kwargs = {})
#   %clone_23 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_10,), kwargs = {memory_format: torch.contiguous_format})
#   %select_30 : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_23, 0, 0), kwargs = {})
#   %select_31 : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_23, 0, 1), kwargs = {})
#   %select_32 : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_23, 0, 2), kwargs = {})
#   %view_156 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_30, [1024, %mul_127, 96]), kwargs = {})
#   %permute_116 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_156, [1, 0, 2]), kwargs = {})
#   %view_157 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_31, [1024, %mul_127, 96]), kwargs = {})
#   %permute_117 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_157, [1, 0, 2]), kwargs = {})
#   %view_158 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_32, [1024, %mul_127, 96]), kwargs = {})
#   %permute_118 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_158, [1, 0, 2]), kwargs = {})
#   %view_159 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_116, [%primals_1, 8, 1024, 96]), kwargs = {})
#   %view_160 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_117, [%primals_1, 8, 1024, 96]), kwargs = {})
#   %view_161 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_118, [%primals_1, 8, 1024, 96]), kwargs = {})
#   %graphsafe_run_with_rng_state_10 : [num_users=4] = call_function[target=torch.ops.higher_order.graphsafe_run_with_rng_state](args = (aten._scaled_dot_product_flash_attention.default, %view_159, %view_160, %view_161, 0.2), kwargs = {scale: 0.10206207261596577, rng_state: %bwd_rng_state_10})
#   %_scaled_dot_product_flash_attention_backward_1 : [num_users=3] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention_backward.default](args = (%permute_186, %view_159, %view_160, %view_161, %getitem_132, %getitem_133, None, None, 1024, 1024, 0.2, False, %getitem_138, %getitem_139), kwargs = {scale: 0.10206207261596577})
triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39 = async_compile.triton('triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 8)
    x2 = ((xindex // 768) % ks0)
    x3 = xindex // ks1
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 96*x1 + 768*((((x0 + 96*x1 + 768*x2) // 768) % ks0)) + 768*ks0*((((x0 + 96*x1 + 768*x2 + 768*ks0*x3) // ks1) % 1024))), None, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
    tl.store(out_ptr1 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/jl/cjl3dxub2lrn4r72ex4viwrbpvlgsry7fhhvs6ogjrzfh6lwp7x4.py
# Topologically Sorted Source Nodes: [y, , convert_element_type, input_4, clone_59], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout, aten.native_dropout_backward]
# Source node to ATen node mapping:
#    => inductor_lookup_seed_default_1, inductor_random_default_10
#   clone_59 => mul_2641
#   convert_element_type => convert_element_type_default_62
#   input_4 => gt_11
#   y => add_454, mul_426, rsqrt_4, sub_113, var_mean_4
# Graph fragment:
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_449, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_454 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_454,), kwargs = {})
#   %sub_113 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_449, %getitem_27), kwargs = {})
#   %mul_426 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_113, %rsqrt_4), kwargs = {})
#   %convert_element_type_674 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_396, torch.float32), kwargs = {})
#   %permute_379 : [num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_674, [1, 0, 2]), kwargs = {})
#   %mul_2634 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_379, %primals_30), kwargs = {})
#   %mul_2635 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2634, 768), kwargs = {})
#   %mul_2637 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_426, %sum_120), kwargs = {})
#   %sub_726 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_2635, %sum_119), kwargs = {})
#   %sub_727 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_726, %mul_2637), kwargs = {})
#   %div_19 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_4, 768), kwargs = {})
#   %mul_2638 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_19, %sub_727), kwargs = {})
#   %add_2734 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2731, %mul_2638), kwargs = {})
#   %convert_element_type_677 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2734, torch.float16), kwargs = {})
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default_10 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([%primals_1, 1024, 768], %inductor_lookup_seed_default_1, rand), kwargs = {})
#   %convert_element_type_default_62 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%inductor_random_default_10, torch.float16), kwargs = {})
#   %gt_11 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convert_element_type_default_62, 0.2), kwargs = {})
#   %convert_element_type_678 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_11, torch.float16), kwargs = {})
#   %mul_2640 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_678, 1.25), kwargs = {})
#   %mul_2641 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_677, %mul_2640), kwargs = {})
triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_40 = async_compile.triton('triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*i64', 'out_ptr1': '*fp16', 'ks0': 'i64', 'load_seed_offset': 'constexpr', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'load_seed_offset': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, ks0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 768
    x0 = (xindex % 768)
    x1 = ((xindex // 768) % 1024)
    x2 = xindex // 786432
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x0 + 768*x2 + 768*ks0*x1), None).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2 + ks0*x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x3), None)
    tmp17 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x4), None, eviction_policy='evict_last')
    tmp2 = 768.0
    tmp3 = (tmp1 / tmp2)
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.rsqrt(tmp5)
    tmp7 = 0.0013020833333333333
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp12 * tmp2
    tmp15 = tmp13 - tmp14
    tmp18 = tmp16 - tmp17
    tmp19 = tmp18 * tmp6
    tmp21 = tmp19 * tmp20
    tmp22 = tmp15 - tmp21
    tmp23 = tmp8 * tmp22
    tmp24 = tmp0 + tmp23
    tmp25 = tl.load(in_ptr7 + load_seed_offset)
    tmp26 = x3
    tmp27 = tl.rand(tmp25, (tmp26).to(tl.uint32))
    tmp28 = tmp24.to(tl.float32)
    tmp29 = tmp27.to(tl.float32)
    tmp30 = 0.2
    tmp31 = tmp29 > tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = 1.25
    tmp34 = tmp32 * tmp33
    tmp35 = tmp28 * tmp34
    tl.store(in_out_ptr0 + (x3), tmp24, None)
    tl.store(out_ptr1 + (x3), tmp35, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/zk/czkis72kfmhku7idihlmf2h25yhhtf56i3cr4rnzkfwvj5ara3qp.py
# Topologically Sorted Source Nodes: [x, x_1, y], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x => add_28, convert_element_type
#   x_1 => add_33
#   y => add_38, mul_46, rsqrt, sub_9, var_mean
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_4, torch.float16), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2, %convert_element_type), kwargs = {})
#   %add_33 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_28, %primals_5), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_33, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_38,), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_33, %getitem_1), kwargs = {})
#   %mul_46 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %rsqrt), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_41 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_41(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, xnumel, r0_numel):
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (r0_2 + 768*x3), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_2 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 768, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = (tmp13 / tmp15)
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [R0_BLOCK])
    tmp21 = tl.where(r0_mask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 768.0
    tmp25 = (tmp22 / tmp24)
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tl.store(out_ptr2 + (r0_2 + 768*x3), tmp29, r0_mask)
    tl.store(out_ptr1 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/qv/cqvvf43tduymnnobz4t2kqmlu22pyddet4goick5mzfmxgdq2ieo.py
# Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_2, convert_element_type_7
#   query => permute_3
#   y => add_39, mul_47
# Graph fragment:
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %primals_6), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %primals_7), kwargs = {})
#   %permute_3 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_39, [1, 0, 2]), kwargs = {})
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_3, torch.float16), kwargs = {})
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_7,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused__to_copy_clone_native_layer_norm_transpose_42 = async_compile.triton('triton_poi_fused__to_copy_clone_native_layer_norm_transpose_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp16', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_native_layer_norm_transpose_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_native_layer_norm_transpose_42(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 768)
    x1 = ((xindex // 768) % ks0)
    x2 = xindex // ks1
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 768*x2 + 786432*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/ey/ceybfavhncftiifsw6bwvnfei2cz5ghlos4kgujybdlnoc3tche5.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_746 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_434, torch.float32), kwargs = {})
#   %permute_425 : [num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_746, [1, 0, 2]), kwargs = {})
#   %mul_2680 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_425, %primals_6), kwargs = {})
#   %sum_143 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2680, [2], True), kwargs = {})
#   %mul_2682 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2680, %mul_46), kwargs = {})
#   %sum_144 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2682, [2], True), kwargs = {})
triton_red_fused__to_copy_native_layer_norm_backward_transpose_43 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_backward_transpose_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_backward_transpose_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_backward_transpose_43(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x2 = (xindex % 1024)
    x3 = xindex // 1024
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr0 + (r0_1 + 768*x3 + 768*ks0*x2), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 * tmp2
        tmp11 = tmp9 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask & xmask, tmp14, _tmp13)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp5, xmask)
    tl.store(out_ptr1 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/of/cofkxgxv2ndqyn5ylysacjoyexxlqbeqabt7yqyo3hlula3xnu4v.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_746 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_434, torch.float32), kwargs = {})
#   %permute_425 : [num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_746, [1, 0, 2]), kwargs = {})
#   %mul_2685 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_425, %mul_46), kwargs = {})
#   %sum_145 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2685, [0, 1]), kwargs = {})
#   %sum_146 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute_425, [0, 1]), kwargs = {})
triton_red_fused__to_copy_native_layer_norm_backward_transpose_44 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_backward_transpose_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_backward_transpose_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_backward_transpose_44(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 768)
    x1 = xindex // 768
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 768*((((r0_2 + 16*ks0*x1) // 1024) % ks0)) + 768*ks0*(((r0_2 + 16*ks0*x1) % 1024))), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + 768*(((r0_2 + 16*ks0*x1) % ks1))), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask, tmp6, _tmp5)
        tmp7 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, None)
    tl.store(out_ptr1 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/2j/c2j5eulbo5bm64sbny377qjatlhf4dtuslejbb2rcobce3iz2ges.py
# Topologically Sorted Source Nodes: [x, x_1, y], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm, aten.transpose, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   x => add_28, convert_element_type
#   x_1 => add_33
#   y => add_38, rsqrt, var_mean
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_4, torch.float16), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2, %convert_element_type), kwargs = {})
#   %add_33 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_28, %primals_5), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_33, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_38,), kwargs = {})
#   %convert_element_type_746 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_434, torch.float32), kwargs = {})
#   %permute_425 : [num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_746, [1, 0, 2]), kwargs = {})
#   %mul_2680 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_425, %primals_6), kwargs = {})
#   %mul_2681 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2680, 768), kwargs = {})
#   %mul_2683 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %sum_144), kwargs = {})
#   %sub_738 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_2681, %sum_143), kwargs = {})
#   %sub_739 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_738, %mul_2683), kwargs = {})
#   %div_23 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 768), kwargs = {})
#   %mul_2684 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_23, %sub_739), kwargs = {})
#   %add_2746 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2743, %mul_2684), kwargs = {})
#   %convert_element_type_749 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2746, torch.float16), kwargs = {})
triton_poi_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_transpose_45 = async_compile.triton('triton_poi_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_transpose_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp16', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_transpose_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_transpose_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 768
    x0 = (xindex % 768)
    x1 = ((xindex // 768) % 1024)
    x2 = xindex // 786432
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x0 + 768*x2 + 768*ks0*x1), None).to(tl.float32)
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x2 + ks0*x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x3), None)
    tmp17 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
    tmp2 = 768.0
    tmp3 = (tmp1 / tmp2)
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.rsqrt(tmp5)
    tmp7 = 0.0013020833333333333
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp12 * tmp2
    tmp15 = tmp13 - tmp14
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 - tmp18
    tmp20 = tmp8 * tmp19
    tmp21 = tmp0 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tl.store(in_out_ptr0 + (x3), tmp21, None)
    tl.store(out_ptr0 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/vx/cvxrjq34kqlwukwinrh2oyyb2btfykfh2w324heg4fq7iqv4yt4m.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_147 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_2746, [0], True), kwargs = {dtype: torch.float32})
triton_red_fused_sum_46 = async_compile.triton('triton_red_fused_sum_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1048576, 'r0_': 8},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_46(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 786432
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 786432*r0_1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/xz/cxzk7tqw3bxvb4iui6kbba6jbacov3exkwkydth6gzahoct2iaee.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_749 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2746, torch.float16), kwargs = {})
#   %sum_148 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_749, [0, 1], True), kwargs = {dtype: torch.float32})
triton_red_fused__to_copy_sum_47 = async_compile.triton('triton_red_fused__to_copy_sum_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_sum_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_sum_47(in_ptr0, out_ptr0, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 768)
    x1 = xindex // 768
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 768*(((r0_2 + 16*ks1*x1) % ks0))), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/hl/chlcgmgopcwx5spqjovdc6odobwcodcp7pqbq6pq2dvami6hoboy.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_753 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_113, torch.float32), kwargs = {})
triton_poi_fused__to_copy_48 = async_compile.triton('triton_poi_fused__to_copy_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 122880}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_48(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, mul_15, mul_127, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, primals_16, primals_18, primals_19, primals_20, primals_21, primals_22, primals_24, primals_25, primals_26, primals_27, primals_28, primals_30, primals_31, primals_32, primals_33, primals_34, primals_36, primals_37, primals_38, primals_39, primals_40, primals_42, primals_43, primals_44, primals_45, primals_46, primals_48, primals_49, primals_50, primals_51, primals_52, primals_54, primals_55, primals_56, primals_57, primals_58, primals_60, primals_61, primals_62, primals_63, primals_64, primals_66, primals_67, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_75, primals_76, primals_78, primals_79, primals_80, primals_81, primals_82, primals_84, primals_85, primals_86, primals_87, primals_88, primals_90, primals_91, primals_92, primals_93, primals_94, primals_96, primals_97, primals_98, primals_99, primals_100, primals_102, primals_103, primals_104, primals_105, primals_106, primals_108, primals_109, primals_110, primals_111, primals_112, primals_114, primals_115, primals_116, primals_117, primals_118, primals_120, primals_121, primals_122, primals_123, primals_124, primals_126, primals_127, primals_128, primals_129, primals_130, primals_132, primals_133, primals_134, primals_135, primals_136, primals_138, primals_139, primals_140, primals_141, primals_142, primals_144, primals_145, primals_146, primals_147, primals_148, view_1, mm, add_185, inductor_seeds_default, add_241, add_393, add_449, add_601, add_657, add_809, add_865, add_1017, add_1073, add_1225, add_1281, add_1433, add_1489, add_1641, add_1697, add_1849, add_1905, add_2057, add_2113, add_2265, add_2321, add_2473, view_183, view_185, full_default, convert_element_type_296, clamp_max, convert_element_type_298, clamp_max_1, clamp_max_2, clamp_max_3, permute_141, permute_145, tangents_1, bwd_rng_state_0, bwd_rng_state_1, bwd_rng_state_2, bwd_rng_state_3, bwd_rng_state_4, bwd_rng_state_5, bwd_rng_state_6, bwd_rng_state_7, bwd_rng_state_8, bwd_rng_state_9, bwd_rng_state_10, bwd_rng_state_11 = args
    args.clear()
    s77 = primals_1
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (1024, 768), (768, 1))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (2304, ), (1, ))
    assert_size_stride(primals_9, (2304, 768), (768, 1))
    assert_size_stride(primals_10, (768, 768), (768, 1))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_14, (1536, 768), (768, 1))
    assert_size_stride(primals_15, (1536, ), (1, ))
    assert_size_stride(primals_16, (768, 1536), (1536, 1))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (2304, ), (1, ))
    assert_size_stride(primals_21, (2304, 768), (768, 1))
    assert_size_stride(primals_22, (768, 768), (768, 1))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_26, (1536, 768), (768, 1))
    assert_size_stride(primals_27, (1536, ), (1, ))
    assert_size_stride(primals_28, (768, 1536), (1536, 1))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (2304, ), (1, ))
    assert_size_stride(primals_33, (2304, 768), (768, 1))
    assert_size_stride(primals_34, (768, 768), (768, 1))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_38, (1536, 768), (768, 1))
    assert_size_stride(primals_39, (1536, ), (1, ))
    assert_size_stride(primals_40, (768, 1536), (1536, 1))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (2304, ), (1, ))
    assert_size_stride(primals_45, (2304, 768), (768, 1))
    assert_size_stride(primals_46, (768, 768), (768, 1))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_50, (1536, 768), (768, 1))
    assert_size_stride(primals_51, (1536, ), (1, ))
    assert_size_stride(primals_52, (768, 1536), (1536, 1))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (2304, ), (1, ))
    assert_size_stride(primals_57, (2304, 768), (768, 1))
    assert_size_stride(primals_58, (768, 768), (768, 1))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (1536, 768), (768, 1))
    assert_size_stride(primals_63, (1536, ), (1, ))
    assert_size_stride(primals_64, (768, 1536), (1536, 1))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (2304, ), (1, ))
    assert_size_stride(primals_69, (2304, 768), (768, 1))
    assert_size_stride(primals_70, (768, 768), (768, 1))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_74, (1536, 768), (768, 1))
    assert_size_stride(primals_75, (1536, ), (1, ))
    assert_size_stride(primals_76, (768, 1536), (1536, 1))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (2304, ), (1, ))
    assert_size_stride(primals_81, (2304, 768), (768, 1))
    assert_size_stride(primals_82, (768, 768), (768, 1))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (768, ), (1, ))
    assert_size_stride(primals_86, (1536, 768), (768, 1))
    assert_size_stride(primals_87, (1536, ), (1, ))
    assert_size_stride(primals_88, (768, 1536), (1536, 1))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_91, (768, ), (1, ))
    assert_size_stride(primals_92, (2304, ), (1, ))
    assert_size_stride(primals_93, (2304, 768), (768, 1))
    assert_size_stride(primals_94, (768, 768), (768, 1))
    assert_size_stride(primals_96, (768, ), (1, ))
    assert_size_stride(primals_97, (768, ), (1, ))
    assert_size_stride(primals_98, (1536, 768), (768, 1))
    assert_size_stride(primals_99, (1536, ), (1, ))
    assert_size_stride(primals_100, (768, 1536), (1536, 1))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_104, (2304, ), (1, ))
    assert_size_stride(primals_105, (2304, 768), (768, 1))
    assert_size_stride(primals_106, (768, 768), (768, 1))
    assert_size_stride(primals_108, (768, ), (1, ))
    assert_size_stride(primals_109, (768, ), (1, ))
    assert_size_stride(primals_110, (1536, 768), (768, 1))
    assert_size_stride(primals_111, (1536, ), (1, ))
    assert_size_stride(primals_112, (768, 1536), (1536, 1))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (768, ), (1, ))
    assert_size_stride(primals_116, (2304, ), (1, ))
    assert_size_stride(primals_117, (2304, 768), (768, 1))
    assert_size_stride(primals_118, (768, 768), (768, 1))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_122, (1536, 768), (768, 1))
    assert_size_stride(primals_123, (1536, ), (1, ))
    assert_size_stride(primals_124, (768, 1536), (1536, 1))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_128, (2304, ), (1, ))
    assert_size_stride(primals_129, (2304, 768), (768, 1))
    assert_size_stride(primals_130, (768, 768), (768, 1))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_133, (768, ), (1, ))
    assert_size_stride(primals_134, (1536, 768), (768, 1))
    assert_size_stride(primals_135, (1536, ), (1, ))
    assert_size_stride(primals_136, (768, 1536), (1536, 1))
    assert_size_stride(primals_138, (768, ), (1, ))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_140, (2304, ), (1, ))
    assert_size_stride(primals_141, (2304, 768), (768, 1))
    assert_size_stride(primals_142, (768, 768), (768, 1))
    assert_size_stride(primals_144, (768, ), (1, ))
    assert_size_stride(primals_145, (768, ), (1, ))
    assert_size_stride(primals_146, (1536, 768), (768, 1))
    assert_size_stride(primals_147, (1536, ), (1, ))
    assert_size_stride(primals_148, (768, 1536), (1536, 1))
    assert_size_stride(view_1, (1024*s77, 16), (16, 1))
    assert_size_stride(mm, (1024*s77, 768), (768, 1))
    assert_size_stride(add_185, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(inductor_seeds_default, (12, ), (1, ))
    assert_size_stride(add_241, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_393, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_449, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_601, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_657, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_809, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_865, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_1017, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_1073, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_1225, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_1281, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_1433, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_1489, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_1641, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_1697, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_1849, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_1905, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_2057, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_2113, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_2265, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_2321, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(add_2473, (s77, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_183, (1024*s77, 768), (768, 1))
    assert_size_stride(view_185, (1024*s77, 128), (128, 1))
    assert_size_stride(full_default, (s77, 1, 128, 128), (16384, 16384, 128, 1))
    assert_size_stride(convert_element_type_296, (32, 1), (1, 1))
    assert_size_stride(clamp_max, (32, 1), (1, 1))
    assert_size_stride(convert_element_type_298, (32, ), (1, ))
    assert_size_stride(clamp_max_1, (32, ), (1, ))
    assert_size_stride(clamp_max_2, (32, ), (1, ))
    assert_size_stride(clamp_max_3, (32, 1), (1, 1))
    assert_size_stride(permute_141, (16, 128), (128, 1))
    assert_size_stride(permute_145, (128, 768), (768, 1))
    assert_size_stride(tangents_1, (s77, 1, 32, 32), (1024, 1024, 32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((s77, 1, 128, 128), (16384, 16384*s77, 128, 1), torch.float32)
        buf2 = empty_strided_cuda((s77, 1, 128, 128), (16384, 16384*s77, 128, 1), torch.float32)
        buf4 = empty_strided_cuda((s77, 1, 128, 128), (16384, 16384*s77, 128, 1), torch.float32)
        buf6 = empty_strided_cuda((s77, 1, 128, 128), (16384, 16384*s77, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.neg, aten.add, aten._unsafe_index_put]
        triton_poi_fused__to_copy__unsafe_index_put_add_mul_neg_0_xnumel = 16384*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_put_add_mul_neg_0.run(full_default, buf0, buf2, buf4, buf6, triton_poi_fused__to_copy__unsafe_index_put_add_mul_neg_0_xnumel, stream=stream0)
        del full_default
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.neg, aten.add, aten._unsafe_index_put]
        triton_poi_fused__to_copy__unsafe_index_put_add_mul_neg_1_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_put_add_mul_neg_1.run(clamp_max, clamp_max_1, tangents_1, clamp_max_3, clamp_max_2, convert_element_type_298, convert_element_type_296, buf0, buf2, buf4, buf6, triton_poi_fused__to_copy__unsafe_index_put_add_mul_neg_1_xnumel, stream=stream0)
        del clamp_max
        del clamp_max_1
        del clamp_max_2
        del clamp_max_3
        del convert_element_type_296
        del convert_element_type_298
        del tangents_1
        buf8 = empty_strided_cuda((s77, 1024, 16), (16384, 16, 1), torch.float16)
        # Topologically Sorted Source Nodes: [constant_pad_nd_2, y], Original ATen: [aten.add, aten._to_copy, aten.im2col, aten.permute, aten.clone]
        triton_poi_fused__to_copy_add_clone_im2col_permute_2_xnumel = 16384*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_im2col_permute_2.run(buf0, buf2, buf4, buf6, buf8, triton_poi_fused__to_copy_add_clone_im2col_permute_2_xnumel, stream=stream0)
        del buf0
        del buf2
        del buf4
        del buf6
        buf9 = empty_strided_cuda((1024*s77, 128), (128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [view_199], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (1024*s77, 16), (16, 1), 0), permute_141, out=buf9)
        del permute_141
        buf10 = empty_strided_cuda((16, 128), (128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [permute_144], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (16, 1024*s77), (1, 16), 0), view_185, out=buf10)
        buf11 = empty_strided_cuda((1, 16, 64), (1024, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_3.run(buf8, buf11, s77, 1024, triton_red_fused_sum_3_r0_numel, stream=stream0)
        del buf8
        buf12 = empty_strided_cuda((1, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_4.run(buf11, buf12, 16, 64, stream=stream0)
        del buf11
        buf13 = empty_strided_cuda((16, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(buf10, buf13, 2048, stream=stream0)
        del buf10
        buf14 = reinterpret_tensor(view_185, (s77, 1024, 128), (131072, 128, 1), 0); del view_185  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, copy_1], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_6_xnumel = 131072*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused_threshold_backward_6.run(buf14, buf9, triton_poi_fused_threshold_backward_6_xnumel, stream=stream0)
        del buf9
        buf15 = empty_strided_cuda((1024*s77, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (1024*s77, 128), (128, 1), 0), permute_145, out=buf15)
        del permute_145
        buf16 = empty_strided_cuda((128, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [permute_149], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (128, 1024*s77), (1, 128), 0), view_183, out=buf16)
        del view_183
        buf17 = empty_strided_cuda((1, 128, 64), (8192, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_7.run(buf14, buf17, s77, 8192, triton_red_fused_sum_7_r0_numel, stream=stream0)
        del buf14
        buf18 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_8.run(buf17, buf18, 128, 64, stream=stream0)
        del buf17
        buf19 = empty_strided_cuda((128, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_9.run(buf16, buf19, 98304, stream=stream0)
        del buf16
        buf21 = empty_strided_cuda((s77, 1024, 768), (786432, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [, convert_element_type, input_4, clone_29], Original ATen: [aten._to_copy, aten.native_dropout, aten.native_dropout_backward]
        triton_poi_fused__to_copy_native_dropout_native_dropout_backward_10_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_native_dropout_native_dropout_backward_10.run(inductor_seeds_default, buf15, buf21, 11, triton_poi_fused__to_copy_native_dropout_native_dropout_backward_10_xnumel, stream=stream0)
        buf22 = empty_strided_cuda((768, 1536), (1536, 1), torch.float16)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_148, buf22, 1179648, stream=stream0)
        del primals_148
        buf23 = empty_strided_cuda((1024*s77, 1536), (1536, 1), torch.float16)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (1024*s77, 768), (768, 1), 0), buf22, out=buf23)
        buf24 = empty_strided_cuda((s77, 1024, 1), (1024, 1, 1024*s77), torch.float32)
        buf25 = empty_strided_cuda((s77, 1024, 1), (1024, 1, 1024*s77), torch.float32)
        buf27 = empty_strided_cuda((s77, 1024, 768), (786432, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [layer_norm, input_1], Original ATen: [aten.native_layer_norm, aten._to_copy]
        triton_per_fused__to_copy_native_layer_norm_12_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_12.run(add_2473, primals_144, primals_145, buf24, buf25, buf27, triton_per_fused__to_copy_native_layer_norm_12_xnumel, 768, stream=stream0)
        del primals_145
        buf28 = reinterpret_tensor(buf22, (768, 1536), (1, 768), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_146, buf28, 1179648, stream=stream0)
        del primals_146
        buf29 = empty_strided_cuda((1024*s77, 1536), (1536, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf27, (1024*s77, 768), (768, 1), 0), buf28, out=buf29)
        buf30 = empty_strided_cuda((s77, 1024, 1536), (1572864, 1536, 1), torch.float16)
        buf35 = reinterpret_tensor(buf23, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.gelu_backward]
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel = 1572864*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13.run(buf35, buf29, primals_147, buf30, triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel, stream=stream0)
        del primals_147
        buf31 = empty_strided_cuda((768, 1536), (1536, 1), torch.float16)
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, permute_153], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf30, (1024*s77, 1536), (1536, 1), 0), out=buf31)
        buf32 = empty_strided_cuda((1, 768, 64), (49152, 1, 768), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf21, buf32, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        buf33 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf32, buf33, 768, 64, stream=stream0)
        buf34 = empty_strided_cuda((768, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf31, buf34, 1179648, stream=stream0)
        buf36 = reinterpret_tensor(buf21, (1024*s77, 768), (768, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (1024*s77, 1536), (1536, 1), 0), reinterpret_tensor(buf28, (1536, 768), (768, 1), 0), out=buf36)
        buf37 = reinterpret_tensor(buf28, (1536, 768), (768, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [permute_157], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (1536, 1024*s77), (1, 1536), 0), reinterpret_tensor(buf27, (1024*s77, 768), (768, 1), 0), out=buf37)
        buf38 = empty_strided_cuda((1, 1536, 64), (98304, 1, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_17.run(buf35, buf38, s77, 98304, triton_red_fused_sum_17_r0_numel, stream=stream0)
        buf39 = empty_strided_cuda((1, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_18.run(buf38, buf39, 1536, 64, stream=stream0)
        buf40 = empty_strided_cuda((1536, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf37, buf40, 1179648, stream=stream0)
        buf47 = empty_strided_cuda((s77, 1024, 768), (786432, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm, aten.native_layer_norm_backward, aten.add]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_19_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_19.run(buf36, primals_144, add_2473, buf24, buf25, buf15, buf47, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_19_xnumel, 768, stream=stream0)
        del primals_144
        buf43 = reinterpret_tensor(buf32, (768, 64), (1, 768), 0); del buf32  # reuse
        buf45 = empty_strided_cuda((768, 64), (1, 768), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_20_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_20.run(buf36, add_2473, buf24, buf25, buf43, buf45, s77, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_20_r0_numel, stream=stream0)
        del add_2473
        buf44 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf43, buf44, 768, 64, stream=stream0)
        buf46 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf45, buf46, 768, 64, stream=stream0)
        buf48 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.clone, aten._unsafe_view]
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21.run(buf47, buf48, s77, triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel, stream=stream0)
        buf49 = empty_strided_cuda((768, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_22.run(primals_142, buf49, 589824, stream=stream0)
        del primals_142
        buf50 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(buf48, buf49, out=buf50)
        buf51 = buf25; del buf25  # reuse
        buf52 = buf24; del buf24  # reuse
        buf55 = reinterpret_tensor(buf27, (1024, s77, 768), (768*s77, 768, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.clone]
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23.run(add_2321, primals_138, primals_139, buf51, buf52, buf55, s77, triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel, 768, stream=stream0)
        del primals_139
        buf54 = empty_strided_cuda((768, 2304), (1, 768), torch.float16)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_24.run(primals_141, buf54, 1769472, stream=stream0)
        del primals_141
        buf56 = empty_strided_cuda((1024*s77, 2304), (2304, 1), torch.float16)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf55, (1024*s77, 768), (768, 1), 0), buf54, out=buf56)
        ps0 = 1024*s77
        ps1 = 786432*s77
        buf57 = empty_strided_cuda((3, 1024, s77, 768), (786432*s77, 768*s77, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25.run(buf56, primals_140, buf57, ps0, ps1, triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel, stream=stream0)
        del primals_140
        ps2 = 768*s77
        buf58 = empty_strided_cuda((s77, 8, 1024, 96), (768, 96, 768*s77, 1), torch.float16)
        buf71 = empty_strided_cuda((s77, 8, 1024, 96), (768, 96, 768*s77, 1), torch.float16)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_26_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_26.run(buf57, buf58, buf71, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_26_xnumel, stream=stream0)
        buf59 = empty_strided_cuda((s77, 8, 1024, 96), (768, 96, 768*s77, 1), torch.float16)
        buf72 = empty_strided_cuda((s77, 8, 1024, 96), (768, 96, 768*s77, 1), torch.float16)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27.run(buf57, buf59, buf72, s77, ps2, ps1, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel, stream=stream0)
        buf60 = empty_strided_cuda((s77, 8, 1024, 96), (768, 96, 768*s77, 1), torch.float16)
        buf73 = empty_strided_cuda((s77, 8, 1024, 96), (768, 96, 768*s77, 1), torch.float16)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28.run(buf57, buf60, buf73, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf61 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf58, buf59, buf60, 0.2, scale=0.10206207261596577, rng_state=bwd_rng_state_11)
        del buf58
        del buf59
        del buf60
        buf62 = buf61[0]
        assert_size_stride(buf62, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf62, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf63 = buf61[1]
        assert_size_stride(buf63, (s77, 8, 1024), (8192, 1024, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf63, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf64 = buf61[6]
        assert_size_stride(buf64, (2, ), (1, ), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf64, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf65 = buf61[7]
        assert_size_stride(buf65, (), (), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf65, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf61
        buf67 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [permute_162], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf48, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf62, (1024*s77, 768), (768, 1), 0), out=buf67)
        buf68 = reinterpret_tensor(buf45, (1, 768, 64), (49152, 1, 768), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf48, buf68, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        del buf48
        buf69 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf68, buf69, 768, 64, stream=stream0)
        buf70 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(buf67, buf70, 589824, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        buf74 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf50, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0), buf71, buf72, buf73, buf62, buf63, None, None, 1024, 1024, 0.2, False, buf64, buf65, scale=0.10206207261596577)
        del buf63
        del buf64
        del buf65
        buf75 = buf74[0]
        assert_size_stride(buf75, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf75, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf76 = buf74[1]
        assert_size_stride(buf76, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf76, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf77 = buf74[2]
        assert_size_stride(buf77, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf77, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        del buf74
        buf78 = empty_strided_cuda((1, 1, 2304, 32), (73728, 73728, 1, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter, _generalized_scatter_1, _generalized_scatter_2], Original ATen: [aten.view, aten.transpose, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel = 32*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30.run(buf77, buf76, buf75, buf78, ps2, s77, 73728, triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel, stream=stream0)
        buf79 = empty_strided_cuda((1, 1, 2304), (2304, 2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter, _generalized_scatter_1, _generalized_scatter_2], Original ATen: [aten.view, aten.transpose, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_31.run(buf78, buf79, 2304, 32, stream=stream0)
        ps3 = 2304*s77
        buf80 = reinterpret_tensor(buf57, (1024, s77, 3, 768), (2304*s77, 2304, 768, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter, _generalized_scatter_1, _generalized_scatter_2], Original ATen: [aten.view, aten.transpose, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.clone]
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32.run(buf77, buf76, buf75, buf80, s77, ps3, ps2, triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel, stream=stream0)
        buf81 = empty_strided_cuda((2304, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [permute_171], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (2304, 1024*s77), (1, 2304), 0), reinterpret_tensor(buf55, (1024*s77, 768), (768, 1), 0), out=buf81)
        buf82 = reinterpret_tensor(buf55, (1024*s77, 768), (768, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (1024*s77, 2304), (2304, 1), 0), reinterpret_tensor(buf54, (2304, 768), (768, 1), 0), out=buf82)
        buf83 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_33.run(buf81, buf83, 1769472, stream=stream0)
        buf84 = empty_strided_cuda((s77, 1024, 1), (1, s77, 1024*s77), torch.float32)
        buf85 = empty_strided_cuda((s77, 1024, 1), (1024, 1, 1024*s77), torch.float32)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34.run(buf82, primals_138, add_2321, buf51, buf52, buf84, buf85, s77, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel, 768, stream=stream0)
        buf86 = reinterpret_tensor(buf68, (768, 64), (1, 768), 0); del buf68  # reuse
        buf88 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35.run(buf82, add_2321, buf51, buf52, buf86, buf88, s77, ps0, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel, stream=stream0)
        buf87 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf86, buf87, 768, 64, stream=stream0)
        buf89 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf88, buf89, 768, 64, stream=stream0)
        buf90 = buf47; del buf47  # reuse
        buf92 = reinterpret_tensor(buf77, (s77, 1024, 768), (786432, 768, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [y, , convert_element_type, input_4, clone_32], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout, aten.native_dropout_backward]
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36.run(buf90, buf52, buf82, primals_138, buf84, add_2321, buf51, buf85, inductor_seeds_default, buf92, s77, 10, triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36_xnumel, stream=stream0)
        del add_2321
        del primals_138
        buf93 = reinterpret_tensor(buf37, (768, 1536), (1536, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_136, buf93, 1179648, stream=stream0)
        del primals_136
        buf94 = reinterpret_tensor(buf35, (1024*s77, 1536), (1536, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf92, (1024*s77, 768), (768, 1), 0), buf93, out=buf94)
        buf95 = buf85; del buf85  # reuse
        buf96 = reinterpret_tensor(buf84, (s77, 1024, 1), (1024, 1, 1024*s77), 0); del buf84  # reuse
        buf98 = reinterpret_tensor(buf82, (s77, 1024, 768), (786432, 768, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1], Original ATen: [aten.native_layer_norm, aten._to_copy]
        triton_per_fused__to_copy_native_layer_norm_12_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_12.run(add_2265, primals_132, primals_133, buf95, buf96, buf98, triton_per_fused__to_copy_native_layer_norm_12_xnumel, 768, stream=stream0)
        del primals_133
        buf99 = reinterpret_tensor(buf93, (768, 1536), (1, 768), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_134, buf99, 1179648, stream=stream0)
        del primals_134
        buf100 = reinterpret_tensor(buf30, (1024*s77, 1536), (1536, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf98, (1024*s77, 768), (768, 1), 0), buf99, out=buf100)
        buf101 = reinterpret_tensor(buf29, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf29  # reuse
        buf106 = reinterpret_tensor(buf94, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.gelu_backward]
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel = 1572864*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13.run(buf106, buf100, primals_135, buf101, triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel, stream=stream0)
        del primals_135
        buf102 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, permute_176], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf92, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf101, (1024*s77, 1536), (1536, 1), 0), out=buf102)
        buf103 = reinterpret_tensor(buf88, (1, 768, 64), (49152, 1, 768), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf92, buf103, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        buf104 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf103, buf104, 768, 64, stream=stream0)
        buf105 = empty_strided_cuda((768, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf102, buf105, 1179648, stream=stream0)
        buf107 = reinterpret_tensor(buf92, (1024*s77, 768), (768, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (1024*s77, 1536), (1536, 1), 0), reinterpret_tensor(buf99, (1536, 768), (768, 1), 0), out=buf107)
        buf108 = reinterpret_tensor(buf99, (1536, 768), (768, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [permute_180], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (1536, 1024*s77), (1, 1536), 0), reinterpret_tensor(buf98, (1024*s77, 768), (768, 1), 0), out=buf108)
        buf109 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_17.run(buf106, buf109, s77, 98304, triton_red_fused_sum_17_r0_numel, stream=stream0)
        buf110 = empty_strided_cuda((1, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_18.run(buf109, buf110, 1536, 64, stream=stream0)
        buf111 = empty_strided_cuda((1536, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf108, buf111, 1179648, stream=stream0)
        buf118 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward, aten.add]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37.run(buf118, buf107, primals_132, add_2265, buf95, buf96, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel, 768, stream=stream0)
        del primals_132
        buf114 = reinterpret_tensor(buf103, (768, 64), (1, 768), 0); del buf103  # reuse
        buf116 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38.run(buf107, add_2265, buf95, buf96, buf114, buf116, ps0, s77, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel, stream=stream0)
        del add_2265
        buf115 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf114, buf115, 768, 64, stream=stream0)
        buf117 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf116, buf117, 768, 64, stream=stream0)
        buf119 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.clone, aten._unsafe_view]
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21.run(buf118, buf119, s77, triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel, stream=stream0)
        buf120 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_22.run(primals_130, buf120, 589824, stream=stream0)
        del primals_130
        buf121 = reinterpret_tensor(buf98, (1024*s77, 768), (768, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(buf119, buf120, out=buf121)
        buf122 = buf96; del buf96  # reuse
        buf123 = buf95; del buf95  # reuse
        buf126 = reinterpret_tensor(buf76, (1024, s77, 768), (768*s77, 768, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.clone]
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23.run(add_2113, primals_126, primals_127, buf122, buf123, buf126, s77, triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel, 768, stream=stream0)
        del primals_127
        buf125 = reinterpret_tensor(buf81, (768, 2304), (1, 768), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_24.run(primals_129, buf125, 1769472, stream=stream0)
        del primals_129
        buf127 = reinterpret_tensor(buf80, (1024*s77, 2304), (2304, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf126, (1024*s77, 768), (768, 1), 0), buf125, out=buf127)
        buf128 = reinterpret_tensor(buf56, (3, 1024, s77, 768), (786432*s77, 768*s77, 768, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25.run(buf127, primals_128, buf128, ps0, ps1, triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel, stream=stream0)
        del primals_128
        buf129 = buf75; del buf75  # reuse
        buf142 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39.run(buf128, buf129, buf142, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel, stream=stream0)
        buf130 = buf72; del buf72  # reuse
        buf143 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27.run(buf128, buf130, buf143, s77, ps2, ps1, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel, stream=stream0)
        buf131 = buf62; del buf62  # reuse
        buf144 = reinterpret_tensor(buf50, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28.run(buf128, buf131, buf144, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf132 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf129, buf130, buf131, 0.2, scale=0.10206207261596577, rng_state=bwd_rng_state_10)
        del buf129
        del buf130
        del buf131
        buf133 = buf132[0]
        assert_size_stride(buf133, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf133, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf134 = buf132[1]
        assert_size_stride(buf134, (s77, 8, 1024), (8192, 1024, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf134, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf135 = buf132[6]
        assert_size_stride(buf135, (2, ), (1, ), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf135, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf136 = buf132[7]
        assert_size_stride(buf136, (), (), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf136, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf132
        buf138 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [permute_185], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf133, (1024*s77, 768), (768, 1), 0), out=buf138)
        buf139 = reinterpret_tensor(buf116, (1, 768, 64), (49152, 1, 768), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf119, buf139, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        del buf119
        buf140 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf139, buf140, 768, 64, stream=stream0)
        buf141 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(buf138, buf141, 589824, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        buf145 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf121, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0), buf142, buf143, buf144, buf133, buf134, None, None, 1024, 1024, 0.2, False, buf135, buf136, scale=0.10206207261596577)
        del buf134
        del buf135
        del buf136
        buf146 = buf145[0]
        assert_size_stride(buf146, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf146, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf147 = buf145[1]
        assert_size_stride(buf147, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf147, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf148 = buf145[2]
        assert_size_stride(buf148, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf148, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        del buf145
        buf149 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_3, _generalized_scatter_4, _generalized_scatter_5], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel = 32*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30.run(buf148, buf147, buf146, buf149, ps2, s77, 73728, triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel, stream=stream0)
        buf150 = empty_strided_cuda((1, 1, 2304), (2304, 2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_3, _generalized_scatter_4, _generalized_scatter_5], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_31.run(buf149, buf150, 2304, 32, stream=stream0)
        buf151 = reinterpret_tensor(buf128, (1024, s77, 3, 768), (2304*s77, 2304, 768, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_3, _generalized_scatter_4, _generalized_scatter_5], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone]
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32.run(buf148, buf147, buf146, buf151, s77, ps3, ps2, triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel, stream=stream0)
        buf152 = reinterpret_tensor(buf54, (2304, 768), (768, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [permute_194], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf151, (2304, 1024*s77), (1, 2304), 0), reinterpret_tensor(buf126, (1024*s77, 768), (768, 1), 0), out=buf152)
        buf153 = reinterpret_tensor(buf126, (1024*s77, 768), (768, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf151, (1024*s77, 2304), (2304, 1), 0), reinterpret_tensor(buf125, (2304, 768), (768, 1), 0), out=buf153)
        buf154 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_33.run(buf152, buf154, 1769472, stream=stream0)
        buf155 = reinterpret_tensor(buf52, (s77, 1024, 1), (1, s77, 1024*s77), 0); del buf52  # reuse
        buf156 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34.run(buf153, primals_126, add_2113, buf122, buf123, buf155, buf156, s77, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel, 768, stream=stream0)
        buf157 = reinterpret_tensor(buf139, (768, 64), (1, 768), 0); del buf139  # reuse
        buf159 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35.run(buf153, add_2113, buf122, buf123, buf157, buf159, s77, ps0, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel, stream=stream0)
        buf158 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf157, buf158, 768, 64, stream=stream0)
        buf160 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf159, buf160, 768, 64, stream=stream0)
        buf161 = buf118; del buf118  # reuse
        buf163 = reinterpret_tensor(buf148, (s77, 1024, 768), (786432, 768, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [y, , convert_element_type, input_4, clone_35], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout, aten.native_dropout_backward]
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36.run(buf161, buf123, buf153, primals_126, buf155, add_2113, buf122, buf156, inductor_seeds_default, buf163, s77, 9, triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36_xnumel, stream=stream0)
        del add_2113
        del primals_126
        buf164 = reinterpret_tensor(buf108, (768, 1536), (1536, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_124, buf164, 1179648, stream=stream0)
        del primals_124
        buf165 = reinterpret_tensor(buf106, (1024*s77, 1536), (1536, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (1024*s77, 768), (768, 1), 0), buf164, out=buf165)
        buf166 = buf156; del buf156  # reuse
        buf167 = reinterpret_tensor(buf155, (s77, 1024, 1), (1024, 1, 1024*s77), 0); del buf155  # reuse
        buf169 = reinterpret_tensor(buf153, (s77, 1024, 768), (786432, 768, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1], Original ATen: [aten.native_layer_norm, aten._to_copy]
        triton_per_fused__to_copy_native_layer_norm_12_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_12.run(add_2057, primals_120, primals_121, buf166, buf167, buf169, triton_per_fused__to_copy_native_layer_norm_12_xnumel, 768, stream=stream0)
        del primals_121
        buf170 = reinterpret_tensor(buf164, (768, 1536), (1, 768), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_122, buf170, 1179648, stream=stream0)
        del primals_122
        buf171 = reinterpret_tensor(buf101, (1024*s77, 1536), (1536, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf169, (1024*s77, 768), (768, 1), 0), buf170, out=buf171)
        buf172 = reinterpret_tensor(buf100, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf100  # reuse
        buf177 = reinterpret_tensor(buf165, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.gelu_backward]
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel = 1572864*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13.run(buf177, buf171, primals_123, buf172, triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel, stream=stream0)
        del primals_123
        buf173 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, permute_199], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf172, (1024*s77, 1536), (1536, 1), 0), out=buf173)
        buf174 = reinterpret_tensor(buf159, (1, 768, 64), (49152, 1, 768), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf163, buf174, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        buf175 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf174, buf175, 768, 64, stream=stream0)
        buf176 = empty_strided_cuda((768, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf173, buf176, 1179648, stream=stream0)
        buf178 = reinterpret_tensor(buf163, (1024*s77, 768), (768, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf177, (1024*s77, 1536), (1536, 1), 0), reinterpret_tensor(buf170, (1536, 768), (768, 1), 0), out=buf178)
        buf179 = reinterpret_tensor(buf170, (1536, 768), (768, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [permute_203], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf177, (1536, 1024*s77), (1, 1536), 0), reinterpret_tensor(buf169, (1024*s77, 768), (768, 1), 0), out=buf179)
        buf180 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_17.run(buf177, buf180, s77, 98304, triton_red_fused_sum_17_r0_numel, stream=stream0)
        buf181 = empty_strided_cuda((1, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_18.run(buf180, buf181, 1536, 64, stream=stream0)
        buf182 = empty_strided_cuda((1536, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf179, buf182, 1179648, stream=stream0)
        buf189 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward, aten.add]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37.run(buf189, buf178, primals_120, add_2057, buf166, buf167, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel, 768, stream=stream0)
        del primals_120
        buf185 = reinterpret_tensor(buf174, (768, 64), (1, 768), 0); del buf174  # reuse
        buf187 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38.run(buf178, add_2057, buf166, buf167, buf185, buf187, ps0, s77, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel, stream=stream0)
        del add_2057
        buf186 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf185, buf186, 768, 64, stream=stream0)
        buf188 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf187, buf188, 768, 64, stream=stream0)
        buf190 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.clone, aten._unsafe_view]
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21.run(buf189, buf190, s77, triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel, stream=stream0)
        buf191 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_22.run(primals_118, buf191, 589824, stream=stream0)
        del primals_118
        buf192 = reinterpret_tensor(buf169, (1024*s77, 768), (768, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(buf190, buf191, out=buf192)
        buf193 = buf167; del buf167  # reuse
        buf194 = buf166; del buf166  # reuse
        buf197 = reinterpret_tensor(buf147, (1024, s77, 768), (768*s77, 768, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.clone]
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23.run(add_1905, primals_114, primals_115, buf193, buf194, buf197, s77, triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel, 768, stream=stream0)
        del primals_115
        buf196 = reinterpret_tensor(buf152, (768, 2304), (1, 768), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_24.run(primals_117, buf196, 1769472, stream=stream0)
        del primals_117
        buf198 = reinterpret_tensor(buf151, (1024*s77, 2304), (2304, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (1024*s77, 768), (768, 1), 0), buf196, out=buf198)
        buf199 = reinterpret_tensor(buf127, (3, 1024, s77, 768), (786432*s77, 768*s77, 768, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25.run(buf198, primals_116, buf199, ps0, ps1, triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel, stream=stream0)
        del primals_116
        buf200 = buf146; del buf146  # reuse
        buf213 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39.run(buf199, buf200, buf213, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel, stream=stream0)
        buf201 = buf143; del buf143  # reuse
        buf214 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27.run(buf199, buf201, buf214, s77, ps2, ps1, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel, stream=stream0)
        buf202 = buf133; del buf133  # reuse
        buf215 = reinterpret_tensor(buf121, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28.run(buf199, buf202, buf215, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf203 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf200, buf201, buf202, 0.2, scale=0.10206207261596577, rng_state=bwd_rng_state_9)
        del buf200
        del buf201
        del buf202
        buf204 = buf203[0]
        assert_size_stride(buf204, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf204, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf205 = buf203[1]
        assert_size_stride(buf205, (s77, 8, 1024), (8192, 1024, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf205, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf206 = buf203[6]
        assert_size_stride(buf206, (2, ), (1, ), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf206, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf207 = buf203[7]
        assert_size_stride(buf207, (), (), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf207, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf203
        buf209 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [permute_208], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf204, (1024*s77, 768), (768, 1), 0), out=buf209)
        buf210 = reinterpret_tensor(buf187, (1, 768, 64), (49152, 1, 768), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf190, buf210, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        del buf190
        buf211 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf210, buf211, 768, 64, stream=stream0)
        buf212 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(buf209, buf212, 589824, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        buf216 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf192, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0), buf213, buf214, buf215, buf204, buf205, None, None, 1024, 1024, 0.2, False, buf206, buf207, scale=0.10206207261596577)
        del buf205
        del buf206
        del buf207
        buf217 = buf216[0]
        assert_size_stride(buf217, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf217, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf218 = buf216[1]
        assert_size_stride(buf218, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf218, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf219 = buf216[2]
        assert_size_stride(buf219, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf219, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        del buf216
        buf220 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_6, _generalized_scatter_7, _generalized_scatter_8], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel = 32*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30.run(buf219, buf218, buf217, buf220, ps2, s77, 73728, triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel, stream=stream0)
        buf221 = empty_strided_cuda((1, 1, 2304), (2304, 2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_6, _generalized_scatter_7, _generalized_scatter_8], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_31.run(buf220, buf221, 2304, 32, stream=stream0)
        buf222 = reinterpret_tensor(buf199, (1024, s77, 3, 768), (2304*s77, 2304, 768, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_6, _generalized_scatter_7, _generalized_scatter_8], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone]
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32.run(buf219, buf218, buf217, buf222, s77, ps3, ps2, triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel, stream=stream0)
        buf223 = reinterpret_tensor(buf125, (2304, 768), (768, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [permute_217], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf222, (2304, 1024*s77), (1, 2304), 0), reinterpret_tensor(buf197, (1024*s77, 768), (768, 1), 0), out=buf223)
        buf224 = reinterpret_tensor(buf197, (1024*s77, 768), (768, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf222, (1024*s77, 2304), (2304, 1), 0), reinterpret_tensor(buf196, (2304, 768), (768, 1), 0), out=buf224)
        buf225 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_33.run(buf223, buf225, 1769472, stream=stream0)
        buf226 = reinterpret_tensor(buf123, (s77, 1024, 1), (1, s77, 1024*s77), 0); del buf123  # reuse
        buf227 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34.run(buf224, primals_114, add_1905, buf193, buf194, buf226, buf227, s77, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel, 768, stream=stream0)
        buf228 = reinterpret_tensor(buf210, (768, 64), (1, 768), 0); del buf210  # reuse
        buf230 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35.run(buf224, add_1905, buf193, buf194, buf228, buf230, s77, ps0, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel, stream=stream0)
        buf229 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf228, buf229, 768, 64, stream=stream0)
        buf231 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf230, buf231, 768, 64, stream=stream0)
        buf232 = buf189; del buf189  # reuse
        buf234 = reinterpret_tensor(buf219, (s77, 1024, 768), (786432, 768, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [y, , convert_element_type, input_4, clone_38], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout, aten.native_dropout_backward]
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36.run(buf232, buf194, buf224, primals_114, buf226, add_1905, buf193, buf227, inductor_seeds_default, buf234, s77, 8, triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36_xnumel, stream=stream0)
        del add_1905
        del primals_114
        buf235 = reinterpret_tensor(buf179, (768, 1536), (1536, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_112, buf235, 1179648, stream=stream0)
        del primals_112
        buf236 = reinterpret_tensor(buf177, (1024*s77, 1536), (1536, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf234, (1024*s77, 768), (768, 1), 0), buf235, out=buf236)
        buf237 = buf227; del buf227  # reuse
        buf238 = reinterpret_tensor(buf226, (s77, 1024, 1), (1024, 1, 1024*s77), 0); del buf226  # reuse
        buf240 = reinterpret_tensor(buf224, (s77, 1024, 768), (786432, 768, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1], Original ATen: [aten.native_layer_norm, aten._to_copy]
        triton_per_fused__to_copy_native_layer_norm_12_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_12.run(add_1849, primals_108, primals_109, buf237, buf238, buf240, triton_per_fused__to_copy_native_layer_norm_12_xnumel, 768, stream=stream0)
        del primals_109
        buf241 = reinterpret_tensor(buf235, (768, 1536), (1, 768), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_110, buf241, 1179648, stream=stream0)
        del primals_110
        buf242 = reinterpret_tensor(buf172, (1024*s77, 1536), (1536, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf240, (1024*s77, 768), (768, 1), 0), buf241, out=buf242)
        buf243 = reinterpret_tensor(buf171, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf171  # reuse
        buf248 = reinterpret_tensor(buf236, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.gelu_backward]
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel = 1572864*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13.run(buf248, buf242, primals_111, buf243, triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel, stream=stream0)
        del primals_111
        buf244 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, permute_222], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf234, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf243, (1024*s77, 1536), (1536, 1), 0), out=buf244)
        buf245 = reinterpret_tensor(buf230, (1, 768, 64), (49152, 1, 768), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf234, buf245, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        buf246 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf245, buf246, 768, 64, stream=stream0)
        buf247 = empty_strided_cuda((768, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf244, buf247, 1179648, stream=stream0)
        buf249 = reinterpret_tensor(buf234, (1024*s77, 768), (768, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf248, (1024*s77, 1536), (1536, 1), 0), reinterpret_tensor(buf241, (1536, 768), (768, 1), 0), out=buf249)
        buf250 = reinterpret_tensor(buf241, (1536, 768), (768, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [permute_226], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf248, (1536, 1024*s77), (1, 1536), 0), reinterpret_tensor(buf240, (1024*s77, 768), (768, 1), 0), out=buf250)
        buf251 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_17.run(buf248, buf251, s77, 98304, triton_red_fused_sum_17_r0_numel, stream=stream0)
        buf252 = empty_strided_cuda((1, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_18.run(buf251, buf252, 1536, 64, stream=stream0)
        buf253 = empty_strided_cuda((1536, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf250, buf253, 1179648, stream=stream0)
        buf260 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward, aten.add]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37.run(buf260, buf249, primals_108, add_1849, buf237, buf238, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel, 768, stream=stream0)
        del primals_108
        buf256 = reinterpret_tensor(buf245, (768, 64), (1, 768), 0); del buf245  # reuse
        buf258 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38.run(buf249, add_1849, buf237, buf238, buf256, buf258, ps0, s77, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel, stream=stream0)
        del add_1849
        buf257 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf256, buf257, 768, 64, stream=stream0)
        buf259 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf258, buf259, 768, 64, stream=stream0)
        buf261 = buf249; del buf249  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.clone, aten._unsafe_view]
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21.run(buf260, buf261, s77, triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel, stream=stream0)
        buf262 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_22.run(primals_106, buf262, 589824, stream=stream0)
        del primals_106
        buf263 = reinterpret_tensor(buf240, (1024*s77, 768), (768, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(buf261, buf262, out=buf263)
        buf264 = buf238; del buf238  # reuse
        buf265 = buf237; del buf237  # reuse
        buf268 = reinterpret_tensor(buf218, (1024, s77, 768), (768*s77, 768, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.clone]
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23.run(add_1697, primals_102, primals_103, buf264, buf265, buf268, s77, triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel, 768, stream=stream0)
        del primals_103
        buf267 = reinterpret_tensor(buf223, (768, 2304), (1, 768), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_24.run(primals_105, buf267, 1769472, stream=stream0)
        del primals_105
        buf269 = reinterpret_tensor(buf222, (1024*s77, 2304), (2304, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf268, (1024*s77, 768), (768, 1), 0), buf267, out=buf269)
        buf270 = reinterpret_tensor(buf198, (3, 1024, s77, 768), (786432*s77, 768*s77, 768, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25.run(buf269, primals_104, buf270, ps0, ps1, triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel, stream=stream0)
        del primals_104
        buf271 = buf217; del buf217  # reuse
        buf284 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39.run(buf270, buf271, buf284, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel, stream=stream0)
        buf272 = buf214; del buf214  # reuse
        buf285 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27.run(buf270, buf272, buf285, s77, ps2, ps1, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel, stream=stream0)
        buf273 = buf204; del buf204  # reuse
        buf286 = reinterpret_tensor(buf192, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28.run(buf270, buf273, buf286, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf274 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf271, buf272, buf273, 0.2, scale=0.10206207261596577, rng_state=bwd_rng_state_8)
        del buf271
        del buf272
        del buf273
        buf275 = buf274[0]
        assert_size_stride(buf275, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf275, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf276 = buf274[1]
        assert_size_stride(buf276, (s77, 8, 1024), (8192, 1024, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf276, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf277 = buf274[6]
        assert_size_stride(buf277, (2, ), (1, ), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf277, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf278 = buf274[7]
        assert_size_stride(buf278, (), (), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf278, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf274
        buf280 = buf262; del buf262  # reuse
        # Topologically Sorted Source Nodes: [permute_231], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf261, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf275, (1024*s77, 768), (768, 1), 0), out=buf280)
        buf281 = reinterpret_tensor(buf258, (1, 768, 64), (49152, 1, 768), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf261, buf281, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        del buf261
        buf282 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf281, buf282, 768, 64, stream=stream0)
        buf283 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(buf280, buf283, 589824, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        buf287 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf263, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0), buf284, buf285, buf286, buf275, buf276, None, None, 1024, 1024, 0.2, False, buf277, buf278, scale=0.10206207261596577)
        del buf276
        del buf277
        del buf278
        buf288 = buf287[0]
        assert_size_stride(buf288, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf288, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf289 = buf287[1]
        assert_size_stride(buf289, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf289, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf290 = buf287[2]
        assert_size_stride(buf290, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf290, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        del buf287
        buf291 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_9, _generalized_scatter_10, _generalized_scatter_11], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel = 32*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30.run(buf290, buf289, buf288, buf291, ps2, s77, 73728, triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel, stream=stream0)
        buf292 = empty_strided_cuda((1, 1, 2304), (2304, 2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_9, _generalized_scatter_10, _generalized_scatter_11], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_31.run(buf291, buf292, 2304, 32, stream=stream0)
        buf293 = reinterpret_tensor(buf270, (1024, s77, 3, 768), (2304*s77, 2304, 768, 1), 0); del buf270  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_9, _generalized_scatter_10, _generalized_scatter_11], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone]
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32.run(buf290, buf289, buf288, buf293, s77, ps3, ps2, triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel, stream=stream0)
        buf294 = reinterpret_tensor(buf196, (2304, 768), (768, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [permute_240], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf293, (2304, 1024*s77), (1, 2304), 0), reinterpret_tensor(buf268, (1024*s77, 768), (768, 1), 0), out=buf294)
        buf295 = reinterpret_tensor(buf268, (1024*s77, 768), (768, 1), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf293, (1024*s77, 2304), (2304, 1), 0), reinterpret_tensor(buf267, (2304, 768), (768, 1), 0), out=buf295)
        buf296 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_33.run(buf294, buf296, 1769472, stream=stream0)
        buf297 = reinterpret_tensor(buf194, (s77, 1024, 1), (1, s77, 1024*s77), 0); del buf194  # reuse
        buf298 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34.run(buf295, primals_102, add_1697, buf264, buf265, buf297, buf298, s77, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel, 768, stream=stream0)
        buf299 = reinterpret_tensor(buf281, (768, 64), (1, 768), 0); del buf281  # reuse
        buf301 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35.run(buf295, add_1697, buf264, buf265, buf299, buf301, s77, ps0, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel, stream=stream0)
        buf300 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf299, buf300, 768, 64, stream=stream0)
        buf302 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf301, buf302, 768, 64, stream=stream0)
        buf303 = buf260; del buf260  # reuse
        buf305 = reinterpret_tensor(buf290, (s77, 1024, 768), (786432, 768, 1), 0); del buf290  # reuse
        # Topologically Sorted Source Nodes: [y, , convert_element_type, input_4, clone_41], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout, aten.native_dropout_backward]
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36.run(buf303, buf265, buf295, primals_102, buf297, add_1697, buf264, buf298, inductor_seeds_default, buf305, s77, 7, triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36_xnumel, stream=stream0)
        del add_1697
        del primals_102
        buf306 = reinterpret_tensor(buf250, (768, 1536), (1536, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_100, buf306, 1179648, stream=stream0)
        del primals_100
        buf307 = reinterpret_tensor(buf248, (1024*s77, 1536), (1536, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf305, (1024*s77, 768), (768, 1), 0), buf306, out=buf307)
        buf308 = buf298; del buf298  # reuse
        buf309 = reinterpret_tensor(buf297, (s77, 1024, 1), (1024, 1, 1024*s77), 0); del buf297  # reuse
        buf311 = reinterpret_tensor(buf295, (s77, 1024, 768), (786432, 768, 1), 0); del buf295  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1], Original ATen: [aten.native_layer_norm, aten._to_copy]
        triton_per_fused__to_copy_native_layer_norm_12_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_12.run(add_1641, primals_96, primals_97, buf308, buf309, buf311, triton_per_fused__to_copy_native_layer_norm_12_xnumel, 768, stream=stream0)
        del primals_97
        buf312 = reinterpret_tensor(buf306, (768, 1536), (1, 768), 0); del buf306  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_98, buf312, 1179648, stream=stream0)
        del primals_98
        buf313 = reinterpret_tensor(buf243, (1024*s77, 1536), (1536, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf311, (1024*s77, 768), (768, 1), 0), buf312, out=buf313)
        buf314 = reinterpret_tensor(buf242, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf242  # reuse
        buf319 = reinterpret_tensor(buf307, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.gelu_backward]
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel = 1572864*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13.run(buf319, buf313, primals_99, buf314, triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel, stream=stream0)
        del primals_99
        buf315 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, permute_245], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf305, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf314, (1024*s77, 1536), (1536, 1), 0), out=buf315)
        buf316 = reinterpret_tensor(buf301, (1, 768, 64), (49152, 1, 768), 0); del buf301  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf305, buf316, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        buf317 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf316, buf317, 768, 64, stream=stream0)
        buf318 = empty_strided_cuda((768, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf315, buf318, 1179648, stream=stream0)
        buf320 = reinterpret_tensor(buf305, (1024*s77, 768), (768, 1), 0); del buf305  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf319, (1024*s77, 1536), (1536, 1), 0), reinterpret_tensor(buf312, (1536, 768), (768, 1), 0), out=buf320)
        buf321 = reinterpret_tensor(buf312, (1536, 768), (768, 1), 0); del buf312  # reuse
        # Topologically Sorted Source Nodes: [permute_249], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf319, (1536, 1024*s77), (1, 1536), 0), reinterpret_tensor(buf311, (1024*s77, 768), (768, 1), 0), out=buf321)
        buf322 = buf251; del buf251  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_17.run(buf319, buf322, s77, 98304, triton_red_fused_sum_17_r0_numel, stream=stream0)
        buf323 = empty_strided_cuda((1, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_18.run(buf322, buf323, 1536, 64, stream=stream0)
        buf324 = empty_strided_cuda((1536, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf321, buf324, 1179648, stream=stream0)
        buf331 = buf303; del buf303  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward, aten.add]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37.run(buf331, buf320, primals_96, add_1641, buf308, buf309, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel, 768, stream=stream0)
        del primals_96
        buf327 = reinterpret_tensor(buf316, (768, 64), (1, 768), 0); del buf316  # reuse
        buf329 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38.run(buf320, add_1641, buf308, buf309, buf327, buf329, ps0, s77, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel, stream=stream0)
        del add_1641
        buf328 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf327, buf328, 768, 64, stream=stream0)
        buf330 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf329, buf330, 768, 64, stream=stream0)
        buf332 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.clone, aten._unsafe_view]
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21.run(buf331, buf332, s77, triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel, stream=stream0)
        buf333 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_22.run(primals_94, buf333, 589824, stream=stream0)
        del primals_94
        buf334 = reinterpret_tensor(buf311, (1024*s77, 768), (768, 1), 0); del buf311  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(buf332, buf333, out=buf334)
        buf335 = buf309; del buf309  # reuse
        buf336 = buf308; del buf308  # reuse
        buf339 = reinterpret_tensor(buf289, (1024, s77, 768), (768*s77, 768, 1), 0); del buf289  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.clone]
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23.run(add_1489, primals_90, primals_91, buf335, buf336, buf339, s77, triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel, 768, stream=stream0)
        del primals_91
        buf338 = reinterpret_tensor(buf294, (768, 2304), (1, 768), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_24.run(primals_93, buf338, 1769472, stream=stream0)
        del primals_93
        buf340 = reinterpret_tensor(buf293, (1024*s77, 2304), (2304, 1), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf339, (1024*s77, 768), (768, 1), 0), buf338, out=buf340)
        buf341 = reinterpret_tensor(buf269, (3, 1024, s77, 768), (786432*s77, 768*s77, 768, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25.run(buf340, primals_92, buf341, ps0, ps1, triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel, stream=stream0)
        del primals_92
        buf342 = buf288; del buf288  # reuse
        buf355 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39.run(buf341, buf342, buf355, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel, stream=stream0)
        buf343 = buf285; del buf285  # reuse
        buf356 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27.run(buf341, buf343, buf356, s77, ps2, ps1, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel, stream=stream0)
        buf344 = buf275; del buf275  # reuse
        buf357 = reinterpret_tensor(buf263, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0); del buf263  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28.run(buf341, buf344, buf357, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf345 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf342, buf343, buf344, 0.2, scale=0.10206207261596577, rng_state=bwd_rng_state_7)
        del buf342
        del buf343
        del buf344
        buf346 = buf345[0]
        assert_size_stride(buf346, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf346, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf347 = buf345[1]
        assert_size_stride(buf347, (s77, 8, 1024), (8192, 1024, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf347, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf348 = buf345[6]
        assert_size_stride(buf348, (2, ), (1, ), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf348, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf349 = buf345[7]
        assert_size_stride(buf349, (), (), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf349, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf345
        buf351 = buf333; del buf333  # reuse
        # Topologically Sorted Source Nodes: [permute_254], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf346, (1024*s77, 768), (768, 1), 0), out=buf351)
        buf352 = reinterpret_tensor(buf329, (1, 768, 64), (49152, 1, 768), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf332, buf352, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        del buf332
        buf353 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf352, buf353, 768, 64, stream=stream0)
        buf354 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(buf351, buf354, 589824, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        buf358 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf334, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0), buf355, buf356, buf357, buf346, buf347, None, None, 1024, 1024, 0.2, False, buf348, buf349, scale=0.10206207261596577)
        del buf347
        del buf348
        del buf349
        buf359 = buf358[0]
        assert_size_stride(buf359, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf359, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf360 = buf358[1]
        assert_size_stride(buf360, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf360, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf361 = buf358[2]
        assert_size_stride(buf361, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf361, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        del buf358
        buf362 = buf291; del buf291  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_12, _generalized_scatter_13, _generalized_scatter_14], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel = 32*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30.run(buf361, buf360, buf359, buf362, ps2, s77, 73728, triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel, stream=stream0)
        buf363 = empty_strided_cuda((1, 1, 2304), (2304, 2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_12, _generalized_scatter_13, _generalized_scatter_14], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_31.run(buf362, buf363, 2304, 32, stream=stream0)
        buf364 = reinterpret_tensor(buf341, (1024, s77, 3, 768), (2304*s77, 2304, 768, 1), 0); del buf341  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_12, _generalized_scatter_13, _generalized_scatter_14], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone]
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32.run(buf361, buf360, buf359, buf364, s77, ps3, ps2, triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel, stream=stream0)
        buf365 = reinterpret_tensor(buf267, (2304, 768), (768, 1), 0); del buf267  # reuse
        # Topologically Sorted Source Nodes: [permute_263], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf364, (2304, 1024*s77), (1, 2304), 0), reinterpret_tensor(buf339, (1024*s77, 768), (768, 1), 0), out=buf365)
        buf366 = reinterpret_tensor(buf339, (1024*s77, 768), (768, 1), 0); del buf339  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf364, (1024*s77, 2304), (2304, 1), 0), reinterpret_tensor(buf338, (2304, 768), (768, 1), 0), out=buf366)
        buf367 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_33.run(buf365, buf367, 1769472, stream=stream0)
        buf368 = reinterpret_tensor(buf265, (s77, 1024, 1), (1, s77, 1024*s77), 0); del buf265  # reuse
        buf369 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34.run(buf366, primals_90, add_1489, buf335, buf336, buf368, buf369, s77, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel, 768, stream=stream0)
        buf370 = reinterpret_tensor(buf352, (768, 64), (1, 768), 0); del buf352  # reuse
        buf372 = buf327; del buf327  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35.run(buf366, add_1489, buf335, buf336, buf370, buf372, s77, ps0, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel, stream=stream0)
        buf371 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf370, buf371, 768, 64, stream=stream0)
        buf373 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf372, buf373, 768, 64, stream=stream0)
        buf374 = buf331; del buf331  # reuse
        buf376 = reinterpret_tensor(buf361, (s77, 1024, 768), (786432, 768, 1), 0); del buf361  # reuse
        # Topologically Sorted Source Nodes: [y, , convert_element_type, input_4, clone_44], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout, aten.native_dropout_backward]
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36.run(buf374, buf336, buf366, primals_90, buf368, add_1489, buf335, buf369, inductor_seeds_default, buf376, s77, 6, triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36_xnumel, stream=stream0)
        del add_1489
        del primals_90
        buf377 = reinterpret_tensor(buf321, (768, 1536), (1536, 1), 0); del buf321  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_88, buf377, 1179648, stream=stream0)
        del primals_88
        buf378 = reinterpret_tensor(buf319, (1024*s77, 1536), (1536, 1), 0); del buf319  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf376, (1024*s77, 768), (768, 1), 0), buf377, out=buf378)
        buf379 = buf369; del buf369  # reuse
        buf380 = reinterpret_tensor(buf368, (s77, 1024, 1), (1024, 1, 1024*s77), 0); del buf368  # reuse
        buf382 = reinterpret_tensor(buf366, (s77, 1024, 768), (786432, 768, 1), 0); del buf366  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1], Original ATen: [aten.native_layer_norm, aten._to_copy]
        triton_per_fused__to_copy_native_layer_norm_12_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_12.run(add_1433, primals_84, primals_85, buf379, buf380, buf382, triton_per_fused__to_copy_native_layer_norm_12_xnumel, 768, stream=stream0)
        del primals_85
        buf383 = reinterpret_tensor(buf377, (768, 1536), (1, 768), 0); del buf377  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_86, buf383, 1179648, stream=stream0)
        del primals_86
        buf384 = reinterpret_tensor(buf314, (1024*s77, 1536), (1536, 1), 0); del buf314  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf382, (1024*s77, 768), (768, 1), 0), buf383, out=buf384)
        buf385 = reinterpret_tensor(buf313, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf313  # reuse
        buf390 = reinterpret_tensor(buf378, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf378  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.gelu_backward]
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel = 1572864*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13.run(buf390, buf384, primals_87, buf385, triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel, stream=stream0)
        del primals_87
        buf386 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, permute_268], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf376, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf385, (1024*s77, 1536), (1536, 1), 0), out=buf386)
        buf387 = reinterpret_tensor(buf372, (1, 768, 64), (49152, 1, 768), 0); del buf372  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf376, buf387, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        buf388 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf387, buf388, 768, 64, stream=stream0)
        buf389 = empty_strided_cuda((768, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf386, buf389, 1179648, stream=stream0)
        buf391 = reinterpret_tensor(buf376, (1024*s77, 768), (768, 1), 0); del buf376  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf390, (1024*s77, 1536), (1536, 1), 0), reinterpret_tensor(buf383, (1536, 768), (768, 1), 0), out=buf391)
        buf392 = reinterpret_tensor(buf383, (1536, 768), (768, 1), 0); del buf383  # reuse
        # Topologically Sorted Source Nodes: [permute_272], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf390, (1536, 1024*s77), (1, 1536), 0), reinterpret_tensor(buf382, (1024*s77, 768), (768, 1), 0), out=buf392)
        buf393 = buf322; del buf322  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_17.run(buf390, buf393, s77, 98304, triton_red_fused_sum_17_r0_numel, stream=stream0)
        buf394 = empty_strided_cuda((1, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_18.run(buf393, buf394, 1536, 64, stream=stream0)
        buf395 = empty_strided_cuda((1536, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf392, buf395, 1179648, stream=stream0)
        buf402 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward, aten.add]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37.run(buf402, buf391, primals_84, add_1433, buf379, buf380, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel, 768, stream=stream0)
        del primals_84
        buf398 = reinterpret_tensor(buf387, (768, 64), (1, 768), 0); del buf387  # reuse
        buf400 = buf370; del buf370  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38.run(buf391, add_1433, buf379, buf380, buf398, buf400, ps0, s77, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel, stream=stream0)
        del add_1433
        buf399 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf398, buf399, 768, 64, stream=stream0)
        buf401 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf400, buf401, 768, 64, stream=stream0)
        buf403 = buf391; del buf391  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.clone, aten._unsafe_view]
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21.run(buf402, buf403, s77, triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel, stream=stream0)
        buf404 = buf351; del buf351  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_22.run(primals_82, buf404, 589824, stream=stream0)
        del primals_82
        buf405 = reinterpret_tensor(buf382, (1024*s77, 768), (768, 1), 0); del buf382  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(buf403, buf404, out=buf405)
        buf406 = buf380; del buf380  # reuse
        buf407 = buf379; del buf379  # reuse
        buf410 = reinterpret_tensor(buf360, (1024, s77, 768), (768*s77, 768, 1), 0); del buf360  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.clone]
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23.run(add_1281, primals_78, primals_79, buf406, buf407, buf410, s77, triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel, 768, stream=stream0)
        del primals_79
        buf409 = reinterpret_tensor(buf365, (768, 2304), (1, 768), 0); del buf365  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_24.run(primals_81, buf409, 1769472, stream=stream0)
        del primals_81
        buf411 = reinterpret_tensor(buf364, (1024*s77, 2304), (2304, 1), 0); del buf364  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf410, (1024*s77, 768), (768, 1), 0), buf409, out=buf411)
        buf412 = reinterpret_tensor(buf340, (3, 1024, s77, 768), (786432*s77, 768*s77, 768, 1), 0); del buf340  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25.run(buf411, primals_80, buf412, ps0, ps1, triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel, stream=stream0)
        del primals_80
        buf413 = buf359; del buf359  # reuse
        buf426 = buf357; del buf357  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39.run(buf412, buf413, buf426, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel, stream=stream0)
        buf414 = buf356; del buf356  # reuse
        buf427 = buf355; del buf355  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27.run(buf412, buf414, buf427, s77, ps2, ps1, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel, stream=stream0)
        buf415 = buf346; del buf346  # reuse
        buf428 = reinterpret_tensor(buf334, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0); del buf334  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28.run(buf412, buf415, buf428, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf416 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf413, buf414, buf415, 0.2, scale=0.10206207261596577, rng_state=bwd_rng_state_6)
        del buf413
        del buf414
        del buf415
        buf417 = buf416[0]
        assert_size_stride(buf417, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf417, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf418 = buf416[1]
        assert_size_stride(buf418, (s77, 8, 1024), (8192, 1024, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf418, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf419 = buf416[6]
        assert_size_stride(buf419, (2, ), (1, ), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf419, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf420 = buf416[7]
        assert_size_stride(buf420, (), (), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf420, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf416
        buf422 = buf404; del buf404  # reuse
        # Topologically Sorted Source Nodes: [permute_277], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf403, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf417, (1024*s77, 768), (768, 1), 0), out=buf422)
        buf423 = reinterpret_tensor(buf400, (1, 768, 64), (49152, 1, 768), 0); del buf400  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf403, buf423, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        del buf403
        buf424 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf423, buf424, 768, 64, stream=stream0)
        buf425 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(buf422, buf425, 589824, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        buf429 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf405, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0), buf426, buf427, buf428, buf417, buf418, None, None, 1024, 1024, 0.2, False, buf419, buf420, scale=0.10206207261596577)
        del buf418
        del buf419
        del buf420
        buf430 = buf429[0]
        assert_size_stride(buf430, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf430, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf431 = buf429[1]
        assert_size_stride(buf431, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf431, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf432 = buf429[2]
        assert_size_stride(buf432, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf432, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        del buf429
        buf433 = buf362; del buf362  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_15, _generalized_scatter_16, _generalized_scatter_17], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel = 32*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30.run(buf432, buf431, buf430, buf433, ps2, s77, 73728, triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel, stream=stream0)
        buf434 = empty_strided_cuda((1, 1, 2304), (2304, 2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_15, _generalized_scatter_16, _generalized_scatter_17], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_31.run(buf433, buf434, 2304, 32, stream=stream0)
        buf435 = reinterpret_tensor(buf412, (1024, s77, 3, 768), (2304*s77, 2304, 768, 1), 0); del buf412  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_15, _generalized_scatter_16, _generalized_scatter_17], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone]
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32.run(buf432, buf431, buf430, buf435, s77, ps3, ps2, triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel, stream=stream0)
        buf436 = reinterpret_tensor(buf338, (2304, 768), (768, 1), 0); del buf338  # reuse
        # Topologically Sorted Source Nodes: [permute_286], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf435, (2304, 1024*s77), (1, 2304), 0), reinterpret_tensor(buf410, (1024*s77, 768), (768, 1), 0), out=buf436)
        buf437 = reinterpret_tensor(buf410, (1024*s77, 768), (768, 1), 0); del buf410  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf435, (1024*s77, 2304), (2304, 1), 0), reinterpret_tensor(buf409, (2304, 768), (768, 1), 0), out=buf437)
        buf438 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_33.run(buf436, buf438, 1769472, stream=stream0)
        buf439 = reinterpret_tensor(buf336, (s77, 1024, 1), (1, s77, 1024*s77), 0); del buf336  # reuse
        buf440 = buf335; del buf335  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34.run(buf437, primals_78, add_1281, buf406, buf407, buf439, buf440, s77, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel, 768, stream=stream0)
        buf441 = reinterpret_tensor(buf423, (768, 64), (1, 768), 0); del buf423  # reuse
        buf443 = buf398; del buf398  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35.run(buf437, add_1281, buf406, buf407, buf441, buf443, s77, ps0, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel, stream=stream0)
        buf442 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf441, buf442, 768, 64, stream=stream0)
        buf444 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf443, buf444, 768, 64, stream=stream0)
        buf445 = buf402; del buf402  # reuse
        buf447 = reinterpret_tensor(buf432, (s77, 1024, 768), (786432, 768, 1), 0); del buf432  # reuse
        # Topologically Sorted Source Nodes: [y, , convert_element_type, input_4, clone_47], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout, aten.native_dropout_backward]
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36.run(buf445, buf407, buf437, primals_78, buf439, add_1281, buf406, buf440, inductor_seeds_default, buf447, s77, 5, triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36_xnumel, stream=stream0)
        del add_1281
        del primals_78
        buf448 = reinterpret_tensor(buf392, (768, 1536), (1536, 1), 0); del buf392  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_76, buf448, 1179648, stream=stream0)
        del primals_76
        buf449 = reinterpret_tensor(buf390, (1024*s77, 1536), (1536, 1), 0); del buf390  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf447, (1024*s77, 768), (768, 1), 0), buf448, out=buf449)
        buf450 = buf440; del buf440  # reuse
        buf451 = reinterpret_tensor(buf439, (s77, 1024, 1), (1024, 1, 1024*s77), 0); del buf439  # reuse
        buf453 = reinterpret_tensor(buf437, (s77, 1024, 768), (786432, 768, 1), 0); del buf437  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1], Original ATen: [aten.native_layer_norm, aten._to_copy]
        triton_per_fused__to_copy_native_layer_norm_12_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_12.run(add_1225, primals_72, primals_73, buf450, buf451, buf453, triton_per_fused__to_copy_native_layer_norm_12_xnumel, 768, stream=stream0)
        del primals_73
        buf454 = reinterpret_tensor(buf448, (768, 1536), (1, 768), 0); del buf448  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_74, buf454, 1179648, stream=stream0)
        del primals_74
        buf455 = reinterpret_tensor(buf385, (1024*s77, 1536), (1536, 1), 0); del buf385  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf453, (1024*s77, 768), (768, 1), 0), buf454, out=buf455)
        buf456 = reinterpret_tensor(buf384, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf384  # reuse
        buf461 = reinterpret_tensor(buf449, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf449  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.gelu_backward]
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel = 1572864*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13.run(buf461, buf455, primals_75, buf456, triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel, stream=stream0)
        del primals_75
        buf457 = buf386; del buf386  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, permute_291], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf447, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf456, (1024*s77, 1536), (1536, 1), 0), out=buf457)
        buf458 = reinterpret_tensor(buf443, (1, 768, 64), (49152, 1, 768), 0); del buf443  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf447, buf458, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        buf459 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf458, buf459, 768, 64, stream=stream0)
        buf460 = empty_strided_cuda((768, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf457, buf460, 1179648, stream=stream0)
        buf462 = reinterpret_tensor(buf447, (1024*s77, 768), (768, 1), 0); del buf447  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf461, (1024*s77, 1536), (1536, 1), 0), reinterpret_tensor(buf454, (1536, 768), (768, 1), 0), out=buf462)
        buf463 = reinterpret_tensor(buf454, (1536, 768), (768, 1), 0); del buf454  # reuse
        # Topologically Sorted Source Nodes: [permute_295], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf461, (1536, 1024*s77), (1, 1536), 0), reinterpret_tensor(buf453, (1024*s77, 768), (768, 1), 0), out=buf463)
        buf464 = buf393; del buf393  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_17.run(buf461, buf464, s77, 98304, triton_red_fused_sum_17_r0_numel, stream=stream0)
        buf465 = empty_strided_cuda((1, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_18.run(buf464, buf465, 1536, 64, stream=stream0)
        buf466 = empty_strided_cuda((1536, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf463, buf466, 1179648, stream=stream0)
        buf473 = buf445; del buf445  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward, aten.add]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37.run(buf473, buf462, primals_72, add_1225, buf450, buf451, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel, 768, stream=stream0)
        del primals_72
        buf469 = reinterpret_tensor(buf458, (768, 64), (1, 768), 0); del buf458  # reuse
        buf471 = buf441; del buf441  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38.run(buf462, add_1225, buf450, buf451, buf469, buf471, ps0, s77, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel, stream=stream0)
        del add_1225
        buf470 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf469, buf470, 768, 64, stream=stream0)
        buf472 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf471, buf472, 768, 64, stream=stream0)
        buf474 = buf462; del buf462  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.clone, aten._unsafe_view]
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21.run(buf473, buf474, s77, triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel, stream=stream0)
        buf475 = buf422; del buf422  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_22.run(primals_70, buf475, 589824, stream=stream0)
        del primals_70
        buf476 = reinterpret_tensor(buf453, (1024*s77, 768), (768, 1), 0); del buf453  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(buf474, buf475, out=buf476)
        buf477 = buf451; del buf451  # reuse
        buf478 = buf450; del buf450  # reuse
        buf481 = reinterpret_tensor(buf431, (1024, s77, 768), (768*s77, 768, 1), 0); del buf431  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.clone]
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23.run(add_1073, primals_66, primals_67, buf477, buf478, buf481, s77, triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel, 768, stream=stream0)
        del primals_67
        buf480 = reinterpret_tensor(buf436, (768, 2304), (1, 768), 0); del buf436  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_24.run(primals_69, buf480, 1769472, stream=stream0)
        del primals_69
        buf482 = reinterpret_tensor(buf435, (1024*s77, 2304), (2304, 1), 0); del buf435  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf481, (1024*s77, 768), (768, 1), 0), buf480, out=buf482)
        buf483 = reinterpret_tensor(buf411, (3, 1024, s77, 768), (786432*s77, 768*s77, 768, 1), 0); del buf411  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25.run(buf482, primals_68, buf483, ps0, ps1, triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel, stream=stream0)
        del primals_68
        buf484 = buf430; del buf430  # reuse
        buf497 = buf428; del buf428  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39.run(buf483, buf484, buf497, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel, stream=stream0)
        buf485 = buf427; del buf427  # reuse
        buf498 = buf426; del buf426  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27.run(buf483, buf485, buf498, s77, ps2, ps1, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel, stream=stream0)
        buf486 = buf417; del buf417  # reuse
        buf499 = reinterpret_tensor(buf405, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0); del buf405  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28.run(buf483, buf486, buf499, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf487 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf484, buf485, buf486, 0.2, scale=0.10206207261596577, rng_state=bwd_rng_state_5)
        del buf484
        del buf485
        del buf486
        buf488 = buf487[0]
        assert_size_stride(buf488, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf488, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf489 = buf487[1]
        assert_size_stride(buf489, (s77, 8, 1024), (8192, 1024, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf489, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf490 = buf487[6]
        assert_size_stride(buf490, (2, ), (1, ), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf490, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf491 = buf487[7]
        assert_size_stride(buf491, (), (), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf491, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf487
        buf493 = buf475; del buf475  # reuse
        # Topologically Sorted Source Nodes: [permute_300], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf474, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf488, (1024*s77, 768), (768, 1), 0), out=buf493)
        buf494 = reinterpret_tensor(buf471, (1, 768, 64), (49152, 1, 768), 0); del buf471  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf474, buf494, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        del buf474
        buf495 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf494, buf495, 768, 64, stream=stream0)
        buf496 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(buf493, buf496, 589824, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        buf500 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf476, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0), buf497, buf498, buf499, buf488, buf489, None, None, 1024, 1024, 0.2, False, buf490, buf491, scale=0.10206207261596577)
        del buf489
        del buf490
        del buf491
        buf501 = buf500[0]
        assert_size_stride(buf501, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf501, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf502 = buf500[1]
        assert_size_stride(buf502, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf502, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf503 = buf500[2]
        assert_size_stride(buf503, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf503, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        del buf500
        buf504 = buf433; del buf433  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_18, _generalized_scatter_19, _generalized_scatter_20], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel = 32*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30.run(buf503, buf502, buf501, buf504, ps2, s77, 73728, triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel, stream=stream0)
        buf505 = empty_strided_cuda((1, 1, 2304), (2304, 2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_18, _generalized_scatter_19, _generalized_scatter_20], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_31.run(buf504, buf505, 2304, 32, stream=stream0)
        buf506 = reinterpret_tensor(buf483, (1024, s77, 3, 768), (2304*s77, 2304, 768, 1), 0); del buf483  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_18, _generalized_scatter_19, _generalized_scatter_20], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone]
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32.run(buf503, buf502, buf501, buf506, s77, ps3, ps2, triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel, stream=stream0)
        buf507 = reinterpret_tensor(buf409, (2304, 768), (768, 1), 0); del buf409  # reuse
        # Topologically Sorted Source Nodes: [permute_309], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf506, (2304, 1024*s77), (1, 2304), 0), reinterpret_tensor(buf481, (1024*s77, 768), (768, 1), 0), out=buf507)
        buf508 = reinterpret_tensor(buf481, (1024*s77, 768), (768, 1), 0); del buf481  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf506, (1024*s77, 2304), (2304, 1), 0), reinterpret_tensor(buf480, (2304, 768), (768, 1), 0), out=buf508)
        buf509 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_33.run(buf507, buf509, 1769472, stream=stream0)
        buf510 = reinterpret_tensor(buf407, (s77, 1024, 1), (1, s77, 1024*s77), 0); del buf407  # reuse
        buf511 = buf406; del buf406  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34.run(buf508, primals_66, add_1073, buf477, buf478, buf510, buf511, s77, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel, 768, stream=stream0)
        buf512 = reinterpret_tensor(buf494, (768, 64), (1, 768), 0); del buf494  # reuse
        buf514 = buf469; del buf469  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35.run(buf508, add_1073, buf477, buf478, buf512, buf514, s77, ps0, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel, stream=stream0)
        buf513 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf512, buf513, 768, 64, stream=stream0)
        buf515 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf514, buf515, 768, 64, stream=stream0)
        buf516 = buf473; del buf473  # reuse
        buf518 = reinterpret_tensor(buf503, (s77, 1024, 768), (786432, 768, 1), 0); del buf503  # reuse
        # Topologically Sorted Source Nodes: [y, , convert_element_type, input_4, clone_50], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout, aten.native_dropout_backward]
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36.run(buf516, buf478, buf508, primals_66, buf510, add_1073, buf477, buf511, inductor_seeds_default, buf518, s77, 4, triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36_xnumel, stream=stream0)
        del add_1073
        del primals_66
        buf519 = reinterpret_tensor(buf463, (768, 1536), (1536, 1), 0); del buf463  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_64, buf519, 1179648, stream=stream0)
        del primals_64
        buf520 = reinterpret_tensor(buf461, (1024*s77, 1536), (1536, 1), 0); del buf461  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf518, (1024*s77, 768), (768, 1), 0), buf519, out=buf520)
        buf521 = buf511; del buf511  # reuse
        buf522 = reinterpret_tensor(buf510, (s77, 1024, 1), (1024, 1, 1024*s77), 0); del buf510  # reuse
        buf524 = reinterpret_tensor(buf508, (s77, 1024, 768), (786432, 768, 1), 0); del buf508  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1], Original ATen: [aten.native_layer_norm, aten._to_copy]
        triton_per_fused__to_copy_native_layer_norm_12_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_12.run(add_1017, primals_60, primals_61, buf521, buf522, buf524, triton_per_fused__to_copy_native_layer_norm_12_xnumel, 768, stream=stream0)
        del primals_61
        buf525 = reinterpret_tensor(buf519, (768, 1536), (1, 768), 0); del buf519  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_62, buf525, 1179648, stream=stream0)
        del primals_62
        buf526 = reinterpret_tensor(buf456, (1024*s77, 1536), (1536, 1), 0); del buf456  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf524, (1024*s77, 768), (768, 1), 0), buf525, out=buf526)
        buf527 = reinterpret_tensor(buf455, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf455  # reuse
        buf532 = reinterpret_tensor(buf520, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf520  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.gelu_backward]
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel = 1572864*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13.run(buf532, buf526, primals_63, buf527, triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel, stream=stream0)
        del primals_63
        buf528 = buf457; del buf457  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, permute_314], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf518, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf527, (1024*s77, 1536), (1536, 1), 0), out=buf528)
        buf529 = reinterpret_tensor(buf514, (1, 768, 64), (49152, 1, 768), 0); del buf514  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf518, buf529, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        buf530 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf529, buf530, 768, 64, stream=stream0)
        buf531 = empty_strided_cuda((768, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf528, buf531, 1179648, stream=stream0)
        buf533 = reinterpret_tensor(buf518, (1024*s77, 768), (768, 1), 0); del buf518  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf532, (1024*s77, 1536), (1536, 1), 0), reinterpret_tensor(buf525, (1536, 768), (768, 1), 0), out=buf533)
        buf534 = reinterpret_tensor(buf525, (1536, 768), (768, 1), 0); del buf525  # reuse
        # Topologically Sorted Source Nodes: [permute_318], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf532, (1536, 1024*s77), (1, 1536), 0), reinterpret_tensor(buf524, (1024*s77, 768), (768, 1), 0), out=buf534)
        buf535 = buf464; del buf464  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_17.run(buf532, buf535, s77, 98304, triton_red_fused_sum_17_r0_numel, stream=stream0)
        buf536 = empty_strided_cuda((1, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_18.run(buf535, buf536, 1536, 64, stream=stream0)
        buf537 = empty_strided_cuda((1536, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf534, buf537, 1179648, stream=stream0)
        buf544 = buf516; del buf516  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward, aten.add]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37.run(buf544, buf533, primals_60, add_1017, buf521, buf522, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel, 768, stream=stream0)
        del primals_60
        buf540 = reinterpret_tensor(buf529, (768, 64), (1, 768), 0); del buf529  # reuse
        buf542 = buf512; del buf512  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38.run(buf533, add_1017, buf521, buf522, buf540, buf542, ps0, s77, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel, stream=stream0)
        del add_1017
        buf541 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf540, buf541, 768, 64, stream=stream0)
        buf543 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf542, buf543, 768, 64, stream=stream0)
        buf545 = buf533; del buf533  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.clone, aten._unsafe_view]
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21.run(buf544, buf545, s77, triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel, stream=stream0)
        buf546 = buf493; del buf493  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_22.run(primals_58, buf546, 589824, stream=stream0)
        del primals_58
        buf547 = reinterpret_tensor(buf524, (1024*s77, 768), (768, 1), 0); del buf524  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(buf545, buf546, out=buf547)
        buf548 = buf522; del buf522  # reuse
        buf549 = buf521; del buf521  # reuse
        buf552 = reinterpret_tensor(buf502, (1024, s77, 768), (768*s77, 768, 1), 0); del buf502  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.clone]
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23.run(add_865, primals_54, primals_55, buf548, buf549, buf552, s77, triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel, 768, stream=stream0)
        del primals_55
        buf551 = reinterpret_tensor(buf507, (768, 2304), (1, 768), 0); del buf507  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_24.run(primals_57, buf551, 1769472, stream=stream0)
        del primals_57
        buf553 = reinterpret_tensor(buf506, (1024*s77, 2304), (2304, 1), 0); del buf506  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf552, (1024*s77, 768), (768, 1), 0), buf551, out=buf553)
        buf554 = reinterpret_tensor(buf482, (3, 1024, s77, 768), (786432*s77, 768*s77, 768, 1), 0); del buf482  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25.run(buf553, primals_56, buf554, ps0, ps1, triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel, stream=stream0)
        del primals_56
        buf555 = buf501; del buf501  # reuse
        buf568 = buf499; del buf499  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39.run(buf554, buf555, buf568, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel, stream=stream0)
        buf556 = buf498; del buf498  # reuse
        buf569 = buf497; del buf497  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27.run(buf554, buf556, buf569, s77, ps2, ps1, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel, stream=stream0)
        buf557 = buf488; del buf488  # reuse
        buf570 = reinterpret_tensor(buf476, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0); del buf476  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28.run(buf554, buf557, buf570, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf558 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf555, buf556, buf557, 0.2, scale=0.10206207261596577, rng_state=bwd_rng_state_4)
        del buf555
        del buf556
        del buf557
        buf559 = buf558[0]
        assert_size_stride(buf559, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf559, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf560 = buf558[1]
        assert_size_stride(buf560, (s77, 8, 1024), (8192, 1024, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf560, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf561 = buf558[6]
        assert_size_stride(buf561, (2, ), (1, ), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf561, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf562 = buf558[7]
        assert_size_stride(buf562, (), (), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf562, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf558
        buf564 = buf546; del buf546  # reuse
        # Topologically Sorted Source Nodes: [permute_323], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf545, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf559, (1024*s77, 768), (768, 1), 0), out=buf564)
        buf565 = reinterpret_tensor(buf542, (1, 768, 64), (49152, 1, 768), 0); del buf542  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf545, buf565, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        del buf545
        buf566 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf565, buf566, 768, 64, stream=stream0)
        buf567 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(buf564, buf567, 589824, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        buf571 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf547, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0), buf568, buf569, buf570, buf559, buf560, None, None, 1024, 1024, 0.2, False, buf561, buf562, scale=0.10206207261596577)
        del buf560
        del buf561
        del buf562
        buf572 = buf571[0]
        assert_size_stride(buf572, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf572, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf573 = buf571[1]
        assert_size_stride(buf573, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf573, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf574 = buf571[2]
        assert_size_stride(buf574, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf574, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        del buf571
        buf575 = buf504; del buf504  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_21, _generalized_scatter_22, _generalized_scatter_23], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel = 32*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30.run(buf574, buf573, buf572, buf575, ps2, s77, 73728, triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel, stream=stream0)
        buf576 = empty_strided_cuda((1, 1, 2304), (2304, 2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_21, _generalized_scatter_22, _generalized_scatter_23], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_31.run(buf575, buf576, 2304, 32, stream=stream0)
        buf577 = reinterpret_tensor(buf554, (1024, s77, 3, 768), (2304*s77, 2304, 768, 1), 0); del buf554  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_21, _generalized_scatter_22, _generalized_scatter_23], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone]
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32.run(buf574, buf573, buf572, buf577, s77, ps3, ps2, triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel, stream=stream0)
        buf578 = reinterpret_tensor(buf480, (2304, 768), (768, 1), 0); del buf480  # reuse
        # Topologically Sorted Source Nodes: [permute_332], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf577, (2304, 1024*s77), (1, 2304), 0), reinterpret_tensor(buf552, (1024*s77, 768), (768, 1), 0), out=buf578)
        buf579 = reinterpret_tensor(buf552, (1024*s77, 768), (768, 1), 0); del buf552  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf577, (1024*s77, 2304), (2304, 1), 0), reinterpret_tensor(buf551, (2304, 768), (768, 1), 0), out=buf579)
        buf580 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_33.run(buf578, buf580, 1769472, stream=stream0)
        buf581 = reinterpret_tensor(buf478, (s77, 1024, 1), (1, s77, 1024*s77), 0); del buf478  # reuse
        buf582 = buf477; del buf477  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34.run(buf579, primals_54, add_865, buf548, buf549, buf581, buf582, s77, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel, 768, stream=stream0)
        buf583 = reinterpret_tensor(buf565, (768, 64), (1, 768), 0); del buf565  # reuse
        buf585 = buf540; del buf540  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35.run(buf579, add_865, buf548, buf549, buf583, buf585, s77, ps0, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel, stream=stream0)
        buf584 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf583, buf584, 768, 64, stream=stream0)
        buf586 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf585, buf586, 768, 64, stream=stream0)
        buf587 = buf544; del buf544  # reuse
        buf589 = reinterpret_tensor(buf574, (s77, 1024, 768), (786432, 768, 1), 0); del buf574  # reuse
        # Topologically Sorted Source Nodes: [y, , convert_element_type, input_4, clone_53], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout, aten.native_dropout_backward]
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36.run(buf587, buf549, buf579, primals_54, buf581, add_865, buf548, buf582, inductor_seeds_default, buf589, s77, 3, triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36_xnumel, stream=stream0)
        del add_865
        del primals_54
        buf590 = reinterpret_tensor(buf534, (768, 1536), (1536, 1), 0); del buf534  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_52, buf590, 1179648, stream=stream0)
        del primals_52
        buf591 = reinterpret_tensor(buf532, (1024*s77, 1536), (1536, 1), 0); del buf532  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf589, (1024*s77, 768), (768, 1), 0), buf590, out=buf591)
        buf592 = buf582; del buf582  # reuse
        buf593 = reinterpret_tensor(buf581, (s77, 1024, 1), (1024, 1, 1024*s77), 0); del buf581  # reuse
        buf595 = reinterpret_tensor(buf579, (s77, 1024, 768), (786432, 768, 1), 0); del buf579  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1], Original ATen: [aten.native_layer_norm, aten._to_copy]
        triton_per_fused__to_copy_native_layer_norm_12_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_12.run(add_809, primals_48, primals_49, buf592, buf593, buf595, triton_per_fused__to_copy_native_layer_norm_12_xnumel, 768, stream=stream0)
        del primals_49
        buf596 = reinterpret_tensor(buf590, (768, 1536), (1, 768), 0); del buf590  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_50, buf596, 1179648, stream=stream0)
        del primals_50
        buf597 = reinterpret_tensor(buf527, (1024*s77, 1536), (1536, 1), 0); del buf527  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf595, (1024*s77, 768), (768, 1), 0), buf596, out=buf597)
        buf598 = reinterpret_tensor(buf526, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf526  # reuse
        buf603 = reinterpret_tensor(buf591, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf591  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.gelu_backward]
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel = 1572864*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13.run(buf603, buf597, primals_51, buf598, triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel, stream=stream0)
        del primals_51
        buf599 = buf528; del buf528  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, permute_337], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf589, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf598, (1024*s77, 1536), (1536, 1), 0), out=buf599)
        buf600 = reinterpret_tensor(buf585, (1, 768, 64), (49152, 1, 768), 0); del buf585  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf589, buf600, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        buf601 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf600, buf601, 768, 64, stream=stream0)
        buf602 = empty_strided_cuda((768, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf599, buf602, 1179648, stream=stream0)
        buf604 = reinterpret_tensor(buf589, (1024*s77, 768), (768, 1), 0); del buf589  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf603, (1024*s77, 1536), (1536, 1), 0), reinterpret_tensor(buf596, (1536, 768), (768, 1), 0), out=buf604)
        buf605 = reinterpret_tensor(buf596, (1536, 768), (768, 1), 0); del buf596  # reuse
        # Topologically Sorted Source Nodes: [permute_341], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf603, (1536, 1024*s77), (1, 1536), 0), reinterpret_tensor(buf595, (1024*s77, 768), (768, 1), 0), out=buf605)
        buf606 = buf535; del buf535  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_17.run(buf603, buf606, s77, 98304, triton_red_fused_sum_17_r0_numel, stream=stream0)
        buf607 = empty_strided_cuda((1, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_18.run(buf606, buf607, 1536, 64, stream=stream0)
        buf608 = empty_strided_cuda((1536, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf605, buf608, 1179648, stream=stream0)
        buf615 = buf587; del buf587  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward, aten.add]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37.run(buf615, buf604, primals_48, add_809, buf592, buf593, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel, 768, stream=stream0)
        del primals_48
        buf611 = reinterpret_tensor(buf600, (768, 64), (1, 768), 0); del buf600  # reuse
        buf613 = buf583; del buf583  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38.run(buf604, add_809, buf592, buf593, buf611, buf613, ps0, s77, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel, stream=stream0)
        del add_809
        buf612 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf611, buf612, 768, 64, stream=stream0)
        buf614 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf613, buf614, 768, 64, stream=stream0)
        buf616 = buf604; del buf604  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.clone, aten._unsafe_view]
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21.run(buf615, buf616, s77, triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel, stream=stream0)
        buf617 = buf564; del buf564  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_22.run(primals_46, buf617, 589824, stream=stream0)
        del primals_46
        buf618 = reinterpret_tensor(buf595, (1024*s77, 768), (768, 1), 0); del buf595  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(buf616, buf617, out=buf618)
        buf619 = buf593; del buf593  # reuse
        buf620 = buf592; del buf592  # reuse
        buf623 = reinterpret_tensor(buf573, (1024, s77, 768), (768*s77, 768, 1), 0); del buf573  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.clone]
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23.run(add_657, primals_42, primals_43, buf619, buf620, buf623, s77, triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel, 768, stream=stream0)
        del primals_43
        buf622 = reinterpret_tensor(buf578, (768, 2304), (1, 768), 0); del buf578  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_24.run(primals_45, buf622, 1769472, stream=stream0)
        del primals_45
        buf624 = reinterpret_tensor(buf577, (1024*s77, 2304), (2304, 1), 0); del buf577  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf623, (1024*s77, 768), (768, 1), 0), buf622, out=buf624)
        buf625 = reinterpret_tensor(buf553, (3, 1024, s77, 768), (786432*s77, 768*s77, 768, 1), 0); del buf553  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25.run(buf624, primals_44, buf625, ps0, ps1, triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel, stream=stream0)
        del primals_44
        buf626 = buf572; del buf572  # reuse
        buf639 = buf570; del buf570  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39.run(buf625, buf626, buf639, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel, stream=stream0)
        buf627 = buf569; del buf569  # reuse
        buf640 = buf568; del buf568  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27.run(buf625, buf627, buf640, s77, ps2, ps1, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel, stream=stream0)
        buf628 = buf559; del buf559  # reuse
        buf641 = reinterpret_tensor(buf547, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0); del buf547  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28.run(buf625, buf628, buf641, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf629 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf626, buf627, buf628, 0.2, scale=0.10206207261596577, rng_state=bwd_rng_state_3)
        del buf626
        del buf627
        del buf628
        buf630 = buf629[0]
        assert_size_stride(buf630, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf630, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf631 = buf629[1]
        assert_size_stride(buf631, (s77, 8, 1024), (8192, 1024, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf631, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf632 = buf629[6]
        assert_size_stride(buf632, (2, ), (1, ), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf632, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf633 = buf629[7]
        assert_size_stride(buf633, (), (), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf633, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf629
        buf635 = buf617; del buf617  # reuse
        # Topologically Sorted Source Nodes: [permute_346], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf616, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf630, (1024*s77, 768), (768, 1), 0), out=buf635)
        buf636 = reinterpret_tensor(buf613, (1, 768, 64), (49152, 1, 768), 0); del buf613  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf616, buf636, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        del buf616
        buf637 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf636, buf637, 768, 64, stream=stream0)
        buf638 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(buf635, buf638, 589824, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        buf642 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf618, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0), buf639, buf640, buf641, buf630, buf631, None, None, 1024, 1024, 0.2, False, buf632, buf633, scale=0.10206207261596577)
        del buf631
        del buf632
        del buf633
        buf643 = buf642[0]
        assert_size_stride(buf643, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf643, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf644 = buf642[1]
        assert_size_stride(buf644, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf644, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf645 = buf642[2]
        assert_size_stride(buf645, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf645, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        del buf642
        buf646 = buf575; del buf575  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_24, _generalized_scatter_25, _generalized_scatter_26], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel = 32*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30.run(buf645, buf644, buf643, buf646, ps2, s77, 73728, triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel, stream=stream0)
        buf647 = empty_strided_cuda((1, 1, 2304), (2304, 2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_24, _generalized_scatter_25, _generalized_scatter_26], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_31.run(buf646, buf647, 2304, 32, stream=stream0)
        buf648 = reinterpret_tensor(buf625, (1024, s77, 3, 768), (2304*s77, 2304, 768, 1), 0); del buf625  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_24, _generalized_scatter_25, _generalized_scatter_26], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone]
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32.run(buf645, buf644, buf643, buf648, s77, ps3, ps2, triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel, stream=stream0)
        buf649 = reinterpret_tensor(buf551, (2304, 768), (768, 1), 0); del buf551  # reuse
        # Topologically Sorted Source Nodes: [permute_355], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf648, (2304, 1024*s77), (1, 2304), 0), reinterpret_tensor(buf623, (1024*s77, 768), (768, 1), 0), out=buf649)
        buf650 = reinterpret_tensor(buf623, (1024*s77, 768), (768, 1), 0); del buf623  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf648, (1024*s77, 2304), (2304, 1), 0), reinterpret_tensor(buf622, (2304, 768), (768, 1), 0), out=buf650)
        buf651 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_33.run(buf649, buf651, 1769472, stream=stream0)
        buf652 = reinterpret_tensor(buf549, (s77, 1024, 1), (1, s77, 1024*s77), 0); del buf549  # reuse
        buf653 = buf548; del buf548  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34.run(buf650, primals_42, add_657, buf619, buf620, buf652, buf653, s77, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel, 768, stream=stream0)
        buf654 = reinterpret_tensor(buf636, (768, 64), (1, 768), 0); del buf636  # reuse
        buf656 = buf611; del buf611  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35.run(buf650, add_657, buf619, buf620, buf654, buf656, s77, ps0, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel, stream=stream0)
        buf655 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf654, buf655, 768, 64, stream=stream0)
        buf657 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf656, buf657, 768, 64, stream=stream0)
        buf658 = buf615; del buf615  # reuse
        buf660 = reinterpret_tensor(buf645, (s77, 1024, 768), (786432, 768, 1), 0); del buf645  # reuse
        # Topologically Sorted Source Nodes: [y, , convert_element_type, input_4, clone_56], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout, aten.native_dropout_backward]
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36.run(buf658, buf620, buf650, primals_42, buf652, add_657, buf619, buf653, inductor_seeds_default, buf660, s77, 2, triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36_xnumel, stream=stream0)
        del add_657
        del primals_42
        buf661 = reinterpret_tensor(buf605, (768, 1536), (1536, 1), 0); del buf605  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_40, buf661, 1179648, stream=stream0)
        del primals_40
        buf662 = reinterpret_tensor(buf603, (1024*s77, 1536), (1536, 1), 0); del buf603  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf660, (1024*s77, 768), (768, 1), 0), buf661, out=buf662)
        buf663 = buf653; del buf653  # reuse
        buf664 = reinterpret_tensor(buf652, (s77, 1024, 1), (1024, 1, 1024*s77), 0); del buf652  # reuse
        buf666 = reinterpret_tensor(buf650, (s77, 1024, 768), (786432, 768, 1), 0); del buf650  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1], Original ATen: [aten.native_layer_norm, aten._to_copy]
        triton_per_fused__to_copy_native_layer_norm_12_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_12.run(add_601, primals_36, primals_37, buf663, buf664, buf666, triton_per_fused__to_copy_native_layer_norm_12_xnumel, 768, stream=stream0)
        del primals_37
        buf667 = reinterpret_tensor(buf661, (768, 1536), (1, 768), 0); del buf661  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_38, buf667, 1179648, stream=stream0)
        del primals_38
        buf668 = reinterpret_tensor(buf598, (1024*s77, 1536), (1536, 1), 0); del buf598  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf666, (1024*s77, 768), (768, 1), 0), buf667, out=buf668)
        buf669 = reinterpret_tensor(buf597, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf597  # reuse
        buf674 = reinterpret_tensor(buf662, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf662  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.gelu_backward]
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel = 1572864*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13.run(buf674, buf668, primals_39, buf669, triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel, stream=stream0)
        del primals_39
        buf670 = buf599; del buf599  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, permute_360], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf660, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf669, (1024*s77, 1536), (1536, 1), 0), out=buf670)
        buf671 = reinterpret_tensor(buf656, (1, 768, 64), (49152, 1, 768), 0); del buf656  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf660, buf671, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        buf672 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf671, buf672, 768, 64, stream=stream0)
        buf673 = empty_strided_cuda((768, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf670, buf673, 1179648, stream=stream0)
        buf675 = reinterpret_tensor(buf660, (1024*s77, 768), (768, 1), 0); del buf660  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf674, (1024*s77, 1536), (1536, 1), 0), reinterpret_tensor(buf667, (1536, 768), (768, 1), 0), out=buf675)
        buf676 = reinterpret_tensor(buf667, (1536, 768), (768, 1), 0); del buf667  # reuse
        # Topologically Sorted Source Nodes: [permute_364], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf674, (1536, 1024*s77), (1, 1536), 0), reinterpret_tensor(buf666, (1024*s77, 768), (768, 1), 0), out=buf676)
        buf677 = buf606; del buf606  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_17.run(buf674, buf677, s77, 98304, triton_red_fused_sum_17_r0_numel, stream=stream0)
        buf678 = empty_strided_cuda((1, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_18.run(buf677, buf678, 1536, 64, stream=stream0)
        buf679 = empty_strided_cuda((1536, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf676, buf679, 1179648, stream=stream0)
        buf686 = buf658; del buf658  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward, aten.add]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37.run(buf686, buf675, primals_36, add_601, buf663, buf664, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel, 768, stream=stream0)
        del primals_36
        buf682 = reinterpret_tensor(buf671, (768, 64), (1, 768), 0); del buf671  # reuse
        buf684 = buf654; del buf654  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38.run(buf675, add_601, buf663, buf664, buf682, buf684, ps0, s77, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel, stream=stream0)
        del add_601
        buf683 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf682, buf683, 768, 64, stream=stream0)
        buf685 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf684, buf685, 768, 64, stream=stream0)
        buf687 = buf675; del buf675  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.clone, aten._unsafe_view]
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21.run(buf686, buf687, s77, triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel, stream=stream0)
        buf688 = buf635; del buf635  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_22.run(primals_34, buf688, 589824, stream=stream0)
        del primals_34
        buf689 = reinterpret_tensor(buf666, (1024*s77, 768), (768, 1), 0); del buf666  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(buf687, buf688, out=buf689)
        buf690 = buf664; del buf664  # reuse
        buf691 = buf663; del buf663  # reuse
        buf694 = reinterpret_tensor(buf644, (1024, s77, 768), (768*s77, 768, 1), 0); del buf644  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.clone]
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23.run(add_449, primals_30, primals_31, buf690, buf691, buf694, s77, triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel, 768, stream=stream0)
        del primals_31
        buf693 = reinterpret_tensor(buf649, (768, 2304), (1, 768), 0); del buf649  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_24.run(primals_33, buf693, 1769472, stream=stream0)
        del primals_33
        buf695 = reinterpret_tensor(buf648, (1024*s77, 2304), (2304, 1), 0); del buf648  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf694, (1024*s77, 768), (768, 1), 0), buf693, out=buf695)
        buf696 = reinterpret_tensor(buf624, (3, 1024, s77, 768), (786432*s77, 768*s77, 768, 1), 0); del buf624  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25.run(buf695, primals_32, buf696, ps0, ps1, triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel, stream=stream0)
        del primals_32
        buf697 = buf643; del buf643  # reuse
        buf710 = buf641; del buf641  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39.run(buf696, buf697, buf710, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel, stream=stream0)
        buf698 = buf640; del buf640  # reuse
        buf711 = buf639; del buf639  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27.run(buf696, buf698, buf711, s77, ps2, ps1, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel, stream=stream0)
        buf699 = buf630; del buf630  # reuse
        buf712 = reinterpret_tensor(buf618, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0); del buf618  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28.run(buf696, buf699, buf712, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf700 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf697, buf698, buf699, 0.2, scale=0.10206207261596577, rng_state=bwd_rng_state_2)
        del buf697
        del buf698
        del buf699
        buf701 = buf700[0]
        assert_size_stride(buf701, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf701, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf702 = buf700[1]
        assert_size_stride(buf702, (s77, 8, 1024), (8192, 1024, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf702, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf703 = buf700[6]
        assert_size_stride(buf703, (2, ), (1, ), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf703, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf704 = buf700[7]
        assert_size_stride(buf704, (), (), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf704, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf700
        buf706 = buf688; del buf688  # reuse
        # Topologically Sorted Source Nodes: [permute_369], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf687, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf701, (1024*s77, 768), (768, 1), 0), out=buf706)
        buf707 = reinterpret_tensor(buf684, (1, 768, 64), (49152, 1, 768), 0); del buf684  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf687, buf707, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        del buf687
        buf708 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf707, buf708, 768, 64, stream=stream0)
        buf709 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(buf706, buf709, 589824, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        buf713 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf689, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0), buf710, buf711, buf712, buf701, buf702, None, None, 1024, 1024, 0.2, False, buf703, buf704, scale=0.10206207261596577)
        del buf702
        del buf703
        del buf704
        buf714 = buf713[0]
        assert_size_stride(buf714, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf714, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf715 = buf713[1]
        assert_size_stride(buf715, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf715, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf716 = buf713[2]
        assert_size_stride(buf716, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf716, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        del buf713
        buf717 = buf646; del buf646  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_27, _generalized_scatter_28, _generalized_scatter_29], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel = 32*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30.run(buf716, buf715, buf714, buf717, ps2, s77, 73728, triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel, stream=stream0)
        buf718 = empty_strided_cuda((1, 1, 2304), (2304, 2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_27, _generalized_scatter_28, _generalized_scatter_29], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_31.run(buf717, buf718, 2304, 32, stream=stream0)
        buf719 = reinterpret_tensor(buf696, (1024, s77, 3, 768), (2304*s77, 2304, 768, 1), 0); del buf696  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_27, _generalized_scatter_28, _generalized_scatter_29], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone]
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32.run(buf716, buf715, buf714, buf719, s77, ps3, ps2, triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel, stream=stream0)
        buf720 = reinterpret_tensor(buf622, (2304, 768), (768, 1), 0); del buf622  # reuse
        # Topologically Sorted Source Nodes: [permute_378], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf719, (2304, 1024*s77), (1, 2304), 0), reinterpret_tensor(buf694, (1024*s77, 768), (768, 1), 0), out=buf720)
        buf721 = reinterpret_tensor(buf694, (1024*s77, 768), (768, 1), 0); del buf694  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf719, (1024*s77, 2304), (2304, 1), 0), reinterpret_tensor(buf693, (2304, 768), (768, 1), 0), out=buf721)
        buf722 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_33.run(buf720, buf722, 1769472, stream=stream0)
        buf723 = reinterpret_tensor(buf620, (s77, 1024, 1), (1, s77, 1024*s77), 0); del buf620  # reuse
        buf724 = buf619; del buf619  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34.run(buf721, primals_30, add_449, buf690, buf691, buf723, buf724, s77, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel, 768, stream=stream0)
        buf725 = reinterpret_tensor(buf707, (768, 64), (1, 768), 0); del buf707  # reuse
        buf727 = buf682; del buf682  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35.run(buf721, add_449, buf690, buf691, buf725, buf727, s77, ps0, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel, stream=stream0)
        buf726 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf725, buf726, 768, 64, stream=stream0)
        buf728 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf727, buf728, 768, 64, stream=stream0)
        buf729 = buf686; del buf686  # reuse
        buf731 = reinterpret_tensor(buf716, (s77, 1024, 768), (786432, 768, 1), 0); del buf716  # reuse
        # Topologically Sorted Source Nodes: [y, , convert_element_type, input_4, clone_59], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout, aten.native_dropout_backward]
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_40_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_40.run(buf729, buf691, buf721, primals_30, buf723, add_449, buf690, buf724, inductor_seeds_default, buf731, s77, 1, triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_40_xnumel, stream=stream0)
        del add_449
        del primals_30
        buf732 = reinterpret_tensor(buf676, (768, 1536), (1536, 1), 0); del buf676  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_28, buf732, 1179648, stream=stream0)
        del primals_28
        buf733 = reinterpret_tensor(buf674, (1024*s77, 1536), (1536, 1), 0); del buf674  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf731, (1024*s77, 768), (768, 1), 0), buf732, out=buf733)
        buf734 = buf724; del buf724  # reuse
        buf735 = reinterpret_tensor(buf723, (s77, 1024, 1), (1024, 1, 1024*s77), 0); del buf723  # reuse
        buf737 = reinterpret_tensor(buf721, (s77, 1024, 768), (786432, 768, 1), 0); del buf721  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1], Original ATen: [aten.native_layer_norm, aten._to_copy]
        triton_per_fused__to_copy_native_layer_norm_12_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_12.run(add_393, primals_24, primals_25, buf734, buf735, buf737, triton_per_fused__to_copy_native_layer_norm_12_xnumel, 768, stream=stream0)
        del primals_25
        buf738 = reinterpret_tensor(buf732, (768, 1536), (1, 768), 0); del buf732  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_26, buf738, 1179648, stream=stream0)
        del primals_26
        buf739 = reinterpret_tensor(buf669, (1024*s77, 1536), (1536, 1), 0); del buf669  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf737, (1024*s77, 768), (768, 1), 0), buf738, out=buf739)
        buf740 = reinterpret_tensor(buf668, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf668  # reuse
        buf745 = reinterpret_tensor(buf733, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf733  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.gelu_backward]
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel = 1572864*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13.run(buf745, buf739, primals_27, buf740, triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel, stream=stream0)
        del primals_27
        buf741 = buf670; del buf670  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, permute_383], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf731, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf740, (1024*s77, 1536), (1536, 1), 0), out=buf741)
        buf742 = reinterpret_tensor(buf727, (1, 768, 64), (49152, 1, 768), 0); del buf727  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf731, buf742, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        buf743 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf742, buf743, 768, 64, stream=stream0)
        buf744 = empty_strided_cuda((768, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf741, buf744, 1179648, stream=stream0)
        buf746 = reinterpret_tensor(buf731, (1024*s77, 768), (768, 1), 0); del buf731  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf745, (1024*s77, 1536), (1536, 1), 0), reinterpret_tensor(buf738, (1536, 768), (768, 1), 0), out=buf746)
        buf747 = reinterpret_tensor(buf738, (1536, 768), (768, 1), 0); del buf738  # reuse
        # Topologically Sorted Source Nodes: [permute_387], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf745, (1536, 1024*s77), (1, 1536), 0), reinterpret_tensor(buf737, (1024*s77, 768), (768, 1), 0), out=buf747)
        buf748 = buf677; del buf677  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_17.run(buf745, buf748, s77, 98304, triton_red_fused_sum_17_r0_numel, stream=stream0)
        buf749 = empty_strided_cuda((1, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_18.run(buf748, buf749, 1536, 64, stream=stream0)
        buf750 = empty_strided_cuda((1536, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf747, buf750, 1179648, stream=stream0)
        buf757 = buf729; del buf729  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward, aten.add]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37.run(buf757, buf746, primals_24, add_393, buf734, buf735, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel, 768, stream=stream0)
        del primals_24
        buf753 = reinterpret_tensor(buf742, (768, 64), (1, 768), 0); del buf742  # reuse
        buf755 = buf725; del buf725  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38.run(buf746, add_393, buf734, buf735, buf753, buf755, ps0, s77, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel, stream=stream0)
        del add_393
        buf754 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf753, buf754, 768, 64, stream=stream0)
        buf756 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf755, buf756, 768, 64, stream=stream0)
        buf758 = buf746; del buf746  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.clone, aten._unsafe_view]
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21.run(buf757, buf758, s77, triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel, stream=stream0)
        buf759 = buf706; del buf706  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_22.run(primals_22, buf759, 589824, stream=stream0)
        del primals_22
        buf760 = reinterpret_tensor(buf737, (1024*s77, 768), (768, 1), 0); del buf737  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(buf758, buf759, out=buf760)
        buf761 = buf735; del buf735  # reuse
        buf762 = buf734; del buf734  # reuse
        buf765 = reinterpret_tensor(buf715, (1024, s77, 768), (768*s77, 768, 1), 0); del buf715  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.clone]
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_clone_native_layer_norm_transpose_23.run(add_241, primals_18, primals_19, buf761, buf762, buf765, s77, triton_per_fused__to_copy_clone_native_layer_norm_transpose_23_xnumel, 768, stream=stream0)
        del primals_19
        buf764 = reinterpret_tensor(buf720, (768, 2304), (1, 768), 0); del buf720  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_24.run(primals_21, buf764, 1769472, stream=stream0)
        del primals_21
        buf766 = reinterpret_tensor(buf719, (1024*s77, 2304), (2304, 1), 0); del buf719  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf765, (1024*s77, 768), (768, 1), 0), buf764, out=buf766)
        buf767 = reinterpret_tensor(buf695, (3, 1024, s77, 768), (786432*s77, 768*s77, 768, 1), 0); del buf695  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25.run(buf766, primals_20, buf767, ps0, ps1, triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel, stream=stream0)
        del primals_20
        buf768 = buf714; del buf714  # reuse
        buf781 = buf712; del buf712  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39.run(buf767, buf768, buf781, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel, stream=stream0)
        buf769 = buf711; del buf711  # reuse
        buf782 = buf710; del buf710  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27.run(buf767, buf769, buf782, s77, ps2, ps1, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel, stream=stream0)
        buf770 = buf701; del buf701  # reuse
        buf783 = reinterpret_tensor(buf689, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0); del buf689  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28.run(buf767, buf770, buf783, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf771 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf768, buf769, buf770, 0.2, scale=0.10206207261596577, rng_state=bwd_rng_state_1)
        del buf768
        del buf769
        del buf770
        buf772 = buf771[0]
        assert_size_stride(buf772, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf772, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf773 = buf771[1]
        assert_size_stride(buf773, (s77, 8, 1024), (8192, 1024, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf773, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf774 = buf771[6]
        assert_size_stride(buf774, (2, ), (1, ), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf774, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf775 = buf771[7]
        assert_size_stride(buf775, (), (), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf775, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf771
        buf777 = buf759; del buf759  # reuse
        # Topologically Sorted Source Nodes: [permute_392], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf758, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf772, (1024*s77, 768), (768, 1), 0), out=buf777)
        buf778 = reinterpret_tensor(buf755, (1, 768, 64), (49152, 1, 768), 0); del buf755  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf758, buf778, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        del buf758
        buf779 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf778, buf779, 768, 64, stream=stream0)
        buf780 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(buf777, buf780, 589824, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        buf784 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf760, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0), buf781, buf782, buf783, buf772, buf773, None, None, 1024, 1024, 0.2, False, buf774, buf775, scale=0.10206207261596577)
        del buf773
        del buf774
        del buf775
        buf785 = buf784[0]
        assert_size_stride(buf785, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf785, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf786 = buf784[1]
        assert_size_stride(buf786, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf786, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf787 = buf784[2]
        assert_size_stride(buf787, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf787, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        del buf784
        buf788 = buf717; del buf717  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_30, _generalized_scatter_31, _generalized_scatter_32], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel = 32*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30.run(buf787, buf786, buf785, buf788, ps2, s77, 73728, triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel, stream=stream0)
        buf789 = empty_strided_cuda((1, 1, 2304), (2304, 2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_30, _generalized_scatter_31, _generalized_scatter_32], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_31.run(buf788, buf789, 2304, 32, stream=stream0)
        buf790 = reinterpret_tensor(buf767, (1024, s77, 3, 768), (2304*s77, 2304, 768, 1), 0); del buf767  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_30, _generalized_scatter_31, _generalized_scatter_32], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone]
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32.run(buf787, buf786, buf785, buf790, s77, ps3, ps2, triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel, stream=stream0)
        buf791 = reinterpret_tensor(buf693, (2304, 768), (768, 1), 0); del buf693  # reuse
        # Topologically Sorted Source Nodes: [permute_401], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf790, (2304, 1024*s77), (1, 2304), 0), reinterpret_tensor(buf765, (1024*s77, 768), (768, 1), 0), out=buf791)
        buf792 = reinterpret_tensor(buf765, (1024*s77, 768), (768, 1), 0); del buf765  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf790, (1024*s77, 2304), (2304, 1), 0), reinterpret_tensor(buf764, (2304, 768), (768, 1), 0), out=buf792)
        buf793 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_33.run(buf791, buf793, 1769472, stream=stream0)
        buf794 = reinterpret_tensor(buf691, (s77, 1024, 1), (1, s77, 1024*s77), 0); del buf691  # reuse
        buf795 = buf690; del buf690  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34.run(buf792, primals_18, add_241, buf761, buf762, buf794, buf795, s77, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_34_xnumel, 768, stream=stream0)
        buf796 = reinterpret_tensor(buf778, (768, 64), (1, 768), 0); del buf778  # reuse
        buf798 = buf753; del buf753  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35.run(buf792, add_241, buf761, buf762, buf796, buf798, s77, ps0, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_transpose_35_r0_numel, stream=stream0)
        buf797 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf796, buf797, 768, 64, stream=stream0)
        buf799 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf798, buf799, 768, 64, stream=stream0)
        buf800 = buf757; del buf757  # reuse
        buf802 = reinterpret_tensor(buf787, (s77, 1024, 768), (786432, 768, 1), 0); del buf787  # reuse
        # Topologically Sorted Source Nodes: [y, , convert_element_type, input_4, clone_62], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout, aten.native_dropout_backward]
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36.run(buf800, buf762, buf792, primals_18, buf794, add_241, buf761, buf795, inductor_seeds_default, buf802, s77, 0, triton_poi_fused__to_copy_add_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_transpose_36_xnumel, stream=stream0)
        del add_241
        del buf761
        del inductor_seeds_default
        del primals_18
        buf803 = reinterpret_tensor(buf747, (768, 1536), (1536, 1), 0); del buf747  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_16, buf803, 1179648, stream=stream0)
        del primals_16
        buf804 = reinterpret_tensor(buf745, (1024*s77, 1536), (1536, 1), 0); del buf745  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf802, (1024*s77, 768), (768, 1), 0), buf803, out=buf804)
        buf805 = buf795; del buf795  # reuse
        buf806 = reinterpret_tensor(buf794, (s77, 1024, 1), (1024, 1, 1024*s77), 0); del buf794  # reuse
        buf808 = reinterpret_tensor(buf792, (s77, 1024, 768), (786432, 768, 1), 0); del buf792  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1], Original ATen: [aten.native_layer_norm, aten._to_copy]
        triton_per_fused__to_copy_native_layer_norm_12_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_12.run(add_185, primals_12, primals_13, buf805, buf806, buf808, triton_per_fused__to_copy_native_layer_norm_12_xnumel, 768, stream=stream0)
        del primals_13
        buf809 = reinterpret_tensor(buf803, (768, 1536), (1, 768), 0); del buf803  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_11.run(primals_14, buf809, 1179648, stream=stream0)
        del primals_14
        buf810 = reinterpret_tensor(buf740, (1024*s77, 1536), (1536, 1), 0); del buf740  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf808, (1024*s77, 768), (768, 1), 0), buf809, out=buf810)
        buf811 = reinterpret_tensor(buf739, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf739  # reuse
        buf816 = reinterpret_tensor(buf804, (s77, 1024, 1536), (1572864, 1536, 1), 0); del buf804  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.gelu_backward]
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel = 1572864*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13.run(buf816, buf810, primals_15, buf811, triton_poi_fused__to_copy_addmm_gelu_gelu_backward_view_13_xnumel, stream=stream0)
        del buf810
        del primals_15
        buf812 = buf741; del buf741  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, permute_406], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf802, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf811, (1024*s77, 1536), (1536, 1), 0), out=buf812)
        del buf811
        buf813 = reinterpret_tensor(buf798, (1, 768, 64), (49152, 1, 768), 0); del buf798  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf802, buf813, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        buf814 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf813, buf814, 768, 64, stream=stream0)
        buf815 = empty_strided_cuda((768, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf812, buf815, 1179648, stream=stream0)
        del buf812
        buf817 = reinterpret_tensor(buf802, (1024*s77, 768), (768, 1), 0); del buf802  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf816, (1024*s77, 1536), (1536, 1), 0), reinterpret_tensor(buf809, (1536, 768), (768, 1), 0), out=buf817)
        buf818 = reinterpret_tensor(buf809, (1536, 768), (768, 1), 0); del buf809  # reuse
        # Topologically Sorted Source Nodes: [permute_410], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf816, (1536, 1024*s77), (1, 1536), 0), reinterpret_tensor(buf808, (1024*s77, 768), (768, 1), 0), out=buf818)
        buf819 = buf748; del buf748  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_17.run(buf816, buf819, s77, 98304, triton_red_fused_sum_17_r0_numel, stream=stream0)
        del buf816
        buf820 = empty_strided_cuda((1, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_18.run(buf819, buf820, 1536, 64, stream=stream0)
        del buf819
        buf821 = empty_strided_cuda((1536, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf818, buf821, 1179648, stream=stream0)
        del buf818
        buf828 = buf800; del buf800  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward, aten.add]
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37.run(buf828, buf817, primals_12, add_185, buf805, buf806, triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_37_xnumel, 768, stream=stream0)
        del primals_12
        buf824 = reinterpret_tensor(buf813, (768, 64), (1, 768), 0); del buf813  # reuse
        buf826 = buf796; del buf796  # reuse
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38.run(buf817, add_185, buf805, buf806, buf824, buf826, ps0, s77, 49152, triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_38_r0_numel, stream=stream0)
        del add_185
        buf825 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf824, buf825, 768, 64, stream=stream0)
        buf827 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf826, buf827, 768, 64, stream=stream0)
        buf829 = buf817; del buf817  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.clone, aten._unsafe_view]
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_view_clone_transpose_21.run(buf828, buf829, s77, triton_poi_fused__to_copy__unsafe_view_clone_transpose_21_xnumel, stream=stream0)
        buf830 = buf777; del buf777  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_22.run(primals_10, buf830, 589824, stream=stream0)
        del primals_10
        buf831 = reinterpret_tensor(buf808, (1024*s77, 768), (768, 1), 0); del buf808  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(buf829, buf830, out=buf831)
        buf833 = buf806; del buf806  # reuse
        buf835 = empty_strided_cuda((s77, 1024, 768), (786432, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, y], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm]
        triton_per_fused__to_copy_add_native_layer_norm_41_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_41.run(mm, primals_4, primals_5, buf833, buf835, triton_per_fused__to_copy_add_native_layer_norm_41_xnumel, 768, stream=stream0)
        del mm
        del primals_4
        del primals_5
        buf836 = reinterpret_tensor(buf791, (768, 2304), (1, 768), 0); del buf791  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_24.run(primals_9, buf836, 1769472, stream=stream0)
        del primals_9
        buf837 = reinterpret_tensor(buf786, (1024, s77, 768), (768*s77, 768, 1), 0); del buf786  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.clone]
        triton_poi_fused__to_copy_clone_native_layer_norm_transpose_42_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clone_native_layer_norm_transpose_42.run(buf835, primals_6, primals_7, buf837, s77, ps2, triton_poi_fused__to_copy_clone_native_layer_norm_transpose_42_xnumel, stream=stream0)
        del primals_7
        buf838 = reinterpret_tensor(buf790, (1024*s77, 2304), (2304, 1), 0); del buf790  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf837, (1024*s77, 768), (768, 1), 0), buf836, out=buf838)
        buf839 = reinterpret_tensor(buf766, (3, 1024, s77, 768), (786432*s77, 768*s77, 768, 1), 0); del buf766  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25.run(buf838, primals_8, buf839, ps0, ps1, triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_25_xnumel, stream=stream0)
        del buf838
        del primals_8
        buf840 = buf785; del buf785  # reuse
        buf853 = buf783; del buf783  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39.run(buf839, buf840, buf853, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_39_xnumel, stream=stream0)
        buf841 = buf782; del buf782  # reuse
        buf854 = buf781; del buf781  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27.run(buf839, buf841, buf854, s77, ps2, ps1, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_27_xnumel, stream=stream0)
        buf842 = buf772; del buf772  # reuse
        buf855 = reinterpret_tensor(buf760, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0); del buf760  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28.run(buf839, buf842, buf855, s77, ps2, triton_poi_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_28_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf843 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf840, buf841, buf842, 0.2, scale=0.10206207261596577, rng_state=bwd_rng_state_0)
        del buf840
        del buf841
        del buf842
        buf844 = buf843[0]
        assert_size_stride(buf844, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf844, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf845 = buf843[1]
        assert_size_stride(buf845, (s77, 8, 1024), (8192, 1024, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf845, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf846 = buf843[6]
        assert_size_stride(buf846, (2, ), (1, ), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf846, 16, 'torch.ops.graphsafe_run_with_rng_state')
        buf847 = buf843[7]
        assert_size_stride(buf847, (), (), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf847, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf843
        buf849 = buf830; del buf830  # reuse
        # Topologically Sorted Source Nodes: [permute_415], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf829, (768, 1024*s77), (1, 768), 0), reinterpret_tensor(buf844, (1024*s77, 768), (768, 1), 0), out=buf849)
        buf850 = reinterpret_tensor(buf826, (1, 768, 64), (49152, 1, 768), 0); del buf826  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_14_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf829, buf850, s77, 49152, triton_red_fused_sum_14_r0_numel, stream=stream0)
        del buf829
        buf851 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf850, buf851, 768, 64, stream=stream0)
        buf852 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(buf849, buf852, 589824, stream=stream0)
        del buf849
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention_backward]
        buf856 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf831, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 0), buf853, buf854, buf855, buf844, buf845, None, None, 1024, 1024, 0.2, False, buf846, buf847, scale=0.10206207261596577)
        del buf831
        del buf844
        del buf845
        del buf846
        del buf847
        del buf853
        del buf854
        del buf855
        buf857 = buf856[0]
        assert_size_stride(buf857, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf857, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf858 = buf856[1]
        assert_size_stride(buf858, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf858, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        buf859 = buf856[2]
        assert_size_stride(buf859, (s77, 8, 1024, 96), (768, 96, 768*s77, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        assert_alignment(buf859, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default')
        del buf856
        buf860 = buf788; del buf788  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_33, _generalized_scatter_34, _generalized_scatter_35], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel = 32*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30.run(buf859, buf858, buf857, buf860, ps2, s77, 73728, triton_red_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_30_r0_numel, stream=stream0)
        buf861 = empty_strided_cuda((1, 1, 2304), (2304, 2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_33, _generalized_scatter_34, _generalized_scatter_35], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone, aten._unsafe_view, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__unsafe_view_add_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_31.run(buf860, buf861, 2304, 32, stream=stream0)
        del buf860
        buf862 = reinterpret_tensor(buf839, (1024, s77, 3, 768), (2304*s77, 2304, 768, 1), 0); del buf839  # reuse
        # Topologically Sorted Source Nodes: [full_5, _generalized_scatter_33, _generalized_scatter_34, _generalized_scatter_35], Original ATen: [aten.select_backward, aten.view, aten.transpose, aten.add, aten.unsqueeze, aten.squeeze, aten.clone]
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel = 2359296*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32.run(buf859, buf858, buf857, buf862, s77, ps3, ps2, triton_poi_fused_add_clone_select_backward_squeeze_transpose_unsqueeze_view_32_xnumel, stream=stream0)
        del buf857
        del buf858
        buf863 = reinterpret_tensor(buf764, (2304, 768), (768, 1), 0); del buf764  # reuse
        # Topologically Sorted Source Nodes: [permute_424], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf862, (2304, 1024*s77), (1, 2304), 0), reinterpret_tensor(buf837, (1024*s77, 768), (768, 1), 0), out=buf863)
        buf864 = reinterpret_tensor(buf837, (1024*s77, 768), (768, 1), 0); del buf837  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf862, (1024*s77, 2304), (2304, 1), 0), reinterpret_tensor(buf836, (2304, 768), (768, 1), 0), out=buf864)
        del buf836
        del buf862
        buf865 = empty_strided_cuda((2304, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_33.run(buf863, buf865, 1769472, stream=stream0)
        del buf863
        buf866 = reinterpret_tensor(buf805, (s77, 1024, 1), (1, s77, 1024*s77), 0); del buf805  # reuse
        buf867 = buf762; del buf762  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_backward_transpose_43_xnumel = 1024*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_transpose_43.run(buf864, primals_6, buf835, buf866, buf867, s77, triton_red_fused__to_copy_native_layer_norm_backward_transpose_43_xnumel, 768, stream=stream0)
        buf868 = reinterpret_tensor(buf850, (768, 64), (1, 768), 0); del buf850  # reuse
        buf870 = buf824; del buf824  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        triton_red_fused__to_copy_native_layer_norm_backward_transpose_44_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_transpose_44.run(buf864, buf835, buf868, buf870, s77, ps0, 49152, triton_red_fused__to_copy_native_layer_norm_backward_transpose_44_r0_numel, stream=stream0)
        buf869 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf868, buf869, 768, 64, stream=stream0)
        del buf868
        buf871 = empty_strided_cuda((768, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.transpose, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf870, buf871, 768, 64, stream=stream0)
        buf872 = buf828; del buf828  # reuse
        buf876 = reinterpret_tensor(buf859, (s77, 1024, 768), (786432, 768, 1), 0); del buf859  # reuse
        # Topologically Sorted Source Nodes: [x, x_1, y], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm, aten.transpose, aten.native_layer_norm_backward]
        triton_poi_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_transpose_45_xnumel = 786432*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_transpose_45.run(buf872, buf833, buf864, primals_6, buf866, buf835, buf867, buf876, s77, triton_poi_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_transpose_45_xnumel, stream=stream0)
        del buf833
        del buf835
        del buf864
        del buf866
        del buf867
        del primals_6
        buf873 = empty_strided_cuda((1, 1024, 768), (786432, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_46.run(buf872, buf873, 786432, s77, stream=stream0)
        buf874 = reinterpret_tensor(buf870, (1, 1, 768, 64), (49152, 49152, 1, 768), 0); del buf870  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.sum]
        triton_red_fused__to_copy_sum_47_r0_numel = 16*s77
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_sum_47.run(buf872, buf874, ps0, s77, 49152, triton_red_fused__to_copy_sum_47_r0_numel, stream=stream0)
        del buf872
        buf875 = empty_strided_cuda((1, 1, 768), (768, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_15.run(buf874, buf875, 768, 64, stream=stream0)
        del buf874
        buf877 = empty_strided_cuda((768, 16), (16, 1), torch.float16)
        # Topologically Sorted Source Nodes: [permute_428], Original ATen: [aten._to_copy, aten.view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf876, (768, 1024*s77), (1, 768), 0), view_1, out=buf877)
        del buf876
        del view_1
        buf878 = empty_strided_cuda((768, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_48.run(buf877, buf878, 12288, stream=stream0)
        del buf877
    return (None, None, buf878, reinterpret_tensor(buf875, (768, ), (1, ), 0), reinterpret_tensor(buf873, (1024, 768), (768, 1), 0), buf869, buf871, reinterpret_tensor(buf861, (2304, ), (1, ), 0), buf865, buf852, reinterpret_tensor(buf851, (768, ), (1, ), 0), buf825, buf827, buf821, reinterpret_tensor(buf820, (1536, ), (1, ), 0), buf815, reinterpret_tensor(buf814, (768, ), (1, ), 0), buf797, buf799, reinterpret_tensor(buf789, (2304, ), (1, ), 0), buf793, buf780, reinterpret_tensor(buf779, (768, ), (1, ), 0), buf754, buf756, buf750, reinterpret_tensor(buf749, (1536, ), (1, ), 0), buf744, reinterpret_tensor(buf743, (768, ), (1, ), 0), buf726, buf728, reinterpret_tensor(buf718, (2304, ), (1, ), 0), buf722, buf709, reinterpret_tensor(buf708, (768, ), (1, ), 0), buf683, buf685, buf679, reinterpret_tensor(buf678, (1536, ), (1, ), 0), buf673, reinterpret_tensor(buf672, (768, ), (1, ), 0), buf655, buf657, reinterpret_tensor(buf647, (2304, ), (1, ), 0), buf651, buf638, reinterpret_tensor(buf637, (768, ), (1, ), 0), buf612, buf614, buf608, reinterpret_tensor(buf607, (1536, ), (1, ), 0), buf602, reinterpret_tensor(buf601, (768, ), (1, ), 0), buf584, buf586, reinterpret_tensor(buf576, (2304, ), (1, ), 0), buf580, buf567, reinterpret_tensor(buf566, (768, ), (1, ), 0), buf541, buf543, buf537, reinterpret_tensor(buf536, (1536, ), (1, ), 0), buf531, reinterpret_tensor(buf530, (768, ), (1, ), 0), buf513, buf515, reinterpret_tensor(buf505, (2304, ), (1, ), 0), buf509, buf496, reinterpret_tensor(buf495, (768, ), (1, ), 0), buf470, buf472, buf466, reinterpret_tensor(buf465, (1536, ), (1, ), 0), buf460, reinterpret_tensor(buf459, (768, ), (1, ), 0), buf442, buf444, reinterpret_tensor(buf434, (2304, ), (1, ), 0), buf438, buf425, reinterpret_tensor(buf424, (768, ), (1, ), 0), buf399, buf401, buf395, reinterpret_tensor(buf394, (1536, ), (1, ), 0), buf389, reinterpret_tensor(buf388, (768, ), (1, ), 0), buf371, buf373, reinterpret_tensor(buf363, (2304, ), (1, ), 0), buf367, buf354, reinterpret_tensor(buf353, (768, ), (1, ), 0), buf328, buf330, buf324, reinterpret_tensor(buf323, (1536, ), (1, ), 0), buf318, reinterpret_tensor(buf317, (768, ), (1, ), 0), buf300, buf302, reinterpret_tensor(buf292, (2304, ), (1, ), 0), buf296, buf283, reinterpret_tensor(buf282, (768, ), (1, ), 0), buf257, buf259, buf253, reinterpret_tensor(buf252, (1536, ), (1, ), 0), buf247, reinterpret_tensor(buf246, (768, ), (1, ), 0), buf229, buf231, reinterpret_tensor(buf221, (2304, ), (1, ), 0), buf225, buf212, reinterpret_tensor(buf211, (768, ), (1, ), 0), buf186, buf188, buf182, reinterpret_tensor(buf181, (1536, ), (1, ), 0), buf176, reinterpret_tensor(buf175, (768, ), (1, ), 0), buf158, buf160, reinterpret_tensor(buf150, (2304, ), (1, ), 0), buf154, buf141, reinterpret_tensor(buf140, (768, ), (1, ), 0), buf115, buf117, buf111, reinterpret_tensor(buf110, (1536, ), (1, ), 0), buf105, reinterpret_tensor(buf104, (768, ), (1, ), 0), buf87, buf89, reinterpret_tensor(buf79, (2304, ), (1, ), 0), buf83, buf70, reinterpret_tensor(buf69, (768, ), (1, ), 0), buf44, buf46, buf40, reinterpret_tensor(buf39, (1536, ), (1, ), 0), buf34, reinterpret_tensor(buf33, (768, ), (1, ), 0), buf19, reinterpret_tensor(buf18, (128, ), (1, ), 0), buf13, reinterpret_tensor(buf12, (16, ), (1, ), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 8
    mul_15 = 8192
    mul_127 = 64
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((8192, 16), (16, 1), device='cuda:0', dtype=torch.float16)
    mm = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    add_185 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    inductor_seeds_default = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.int64)
    add_241 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_393 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_449 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_601 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_657 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_809 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_865 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_1017 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_1073 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_1225 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_1281 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_1433 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_1489 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_1641 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_1697 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_1849 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_1905 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_2057 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_2113 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_2265 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_2321 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    add_2473 = rand_strided((8, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_183 = rand_strided((8192, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    view_185 = rand_strided((8192, 128), (128, 1), device='cuda:0', dtype=torch.float16)
    full_default = rand_strided((8, 1, 128, 128), (16384, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_296 = rand_strided((32, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    clamp_max = rand_strided((32, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    convert_element_type_298 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.int64)
    clamp_max_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.int64)
    clamp_max_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_3 = rand_strided((32, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    permute_141 = rand_strided((16, 128), (128, 1), device='cuda:0', dtype=torch.float16)
    permute_145 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    tangents_1 = rand_strided((8, 1, 32, 32), (1024, 1024, 32, 1), device='cuda:0', dtype=torch.float16)
    bwd_rng_state_0 = torch.cuda.default_generators[0].graphsafe_get_state()
    bwd_rng_state_1 = torch.cuda.default_generators[0].graphsafe_get_state()
    bwd_rng_state_2 = torch.cuda.default_generators[0].graphsafe_get_state()
    bwd_rng_state_3 = torch.cuda.default_generators[0].graphsafe_get_state()
    bwd_rng_state_4 = torch.cuda.default_generators[0].graphsafe_get_state()
    bwd_rng_state_5 = torch.cuda.default_generators[0].graphsafe_get_state()
    bwd_rng_state_6 = torch.cuda.default_generators[0].graphsafe_get_state()
    bwd_rng_state_7 = torch.cuda.default_generators[0].graphsafe_get_state()
    bwd_rng_state_8 = torch.cuda.default_generators[0].graphsafe_get_state()
    bwd_rng_state_9 = torch.cuda.default_generators[0].graphsafe_get_state()
    bwd_rng_state_10 = torch.cuda.default_generators[0].graphsafe_get_state()
    bwd_rng_state_11 = torch.cuda.default_generators[0].graphsafe_get_state()
    fn = lambda: call([primals_1, mul_15, mul_127, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, primals_16, primals_18, primals_19, primals_20, primals_21, primals_22, primals_24, primals_25, primals_26, primals_27, primals_28, primals_30, primals_31, primals_32, primals_33, primals_34, primals_36, primals_37, primals_38, primals_39, primals_40, primals_42, primals_43, primals_44, primals_45, primals_46, primals_48, primals_49, primals_50, primals_51, primals_52, primals_54, primals_55, primals_56, primals_57, primals_58, primals_60, primals_61, primals_62, primals_63, primals_64, primals_66, primals_67, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_75, primals_76, primals_78, primals_79, primals_80, primals_81, primals_82, primals_84, primals_85, primals_86, primals_87, primals_88, primals_90, primals_91, primals_92, primals_93, primals_94, primals_96, primals_97, primals_98, primals_99, primals_100, primals_102, primals_103, primals_104, primals_105, primals_106, primals_108, primals_109, primals_110, primals_111, primals_112, primals_114, primals_115, primals_116, primals_117, primals_118, primals_120, primals_121, primals_122, primals_123, primals_124, primals_126, primals_127, primals_128, primals_129, primals_130, primals_132, primals_133, primals_134, primals_135, primals_136, primals_138, primals_139, primals_140, primals_141, primals_142, primals_144, primals_145, primals_146, primals_147, primals_148, view_1, mm, add_185, inductor_seeds_default, add_241, add_393, add_449, add_601, add_657, add_809, add_865, add_1017, add_1073, add_1225, add_1281, add_1433, add_1489, add_1641, add_1697, add_1849, add_1905, add_2057, add_2113, add_2265, add_2321, add_2473, view_183, view_185, full_default, convert_element_type_296, clamp_max, convert_element_type_298, clamp_max_1, clamp_max_2, clamp_max_3, permute_141, permute_145, tangents_1, bwd_rng_state_0, bwd_rng_state_1, bwd_rng_state_2, bwd_rng_state_3, bwd_rng_state_4, bwd_rng_state_5, bwd_rng_state_6, bwd_rng_state_7, bwd_rng_state_8, bwd_rng_state_9, bwd_rng_state_10, bwd_rng_state_11])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
