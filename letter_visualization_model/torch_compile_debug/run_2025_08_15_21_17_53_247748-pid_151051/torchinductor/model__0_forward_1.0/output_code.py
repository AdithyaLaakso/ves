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


# kernel path: /tmp/torchinductor_Adithya/mu/cmuorcgmc743zklnibvsqmsd6qz4judfbeqtnym5axomvwg3fowq.py
# Topologically Sorted Source Nodes: [y, y_1, x], Original ATen: [aten.im2col, aten.permute, aten._to_copy, aten.clone]
# Source node to ATen node mapping:
#   x => clone_1, convert_element_type_2
#   y => add, clone, index, iota, iota_1, permute, unsqueeze, unsqueeze_1, unsqueeze_4, unsqueeze_5, view
#   y_1 => permute_1
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 4, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_1 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_1, -1), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %unsqueeze_4 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add, -1), kwargs = {})
#   %unsqueeze_5 : [num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%primals_1, [None, None, %unsqueeze_5, %add]), kwargs = {})
#   %permute : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%index, [0, 1, 2, 4, 3, 5]), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %view : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [20, 16, 1024]), kwargs = {})
#   %permute_1 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view, [0, 2, 1]), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_1, torch.float16), kwargs = {})
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_2,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused__to_copy_clone_im2col_permute_0 = async_compile.triton('triton_poi_fused__to_copy_clone_im2col_permute_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_im2col_permute_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2621440}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_im2col_permute_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 1024)
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4*((x1 % 32)) + 128*(x0 // 4) + 512*(x1 // 32) + 16384*x2 + ((x0 % 4))), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/5x/c5xaqrwc637s3mi4zysewptnbil32yolrfdomnphmsqanfiio4hn.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x => convert_element_type_1
# Graph fragment:
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.float16), kwargs = {})
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 98304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/qp/cqpmrxzc3h7mzbuas4u2fxbugwhrbmhzi6w53az2px6ggzptw5fr.py
# Topologically Sorted Source Nodes: [x, x_1, y, query, multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_2, convert_element_type_7
#   query => permute_3
#   x => add_2, convert_element_type
#   x_1 => add_3
#   y => add_4, add_5, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_3, torch.float16), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2, %convert_element_type), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %primals_4), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_5), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_6), kwargs = {})
#   %permute_3 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_5, [1, 0, 2]), kwargs = {})
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_3, torch.float16), kwargs = {})
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_7,), kwargs = {memory_format: torch.contiguous_format})
triton_per_fused__to_copy_add_clone_native_layer_norm_transpose_2 = async_compile.triton('triton_per_fused__to_copy_add_clone_native_layer_norm_transpose_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_native_layer_norm_transpose_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 97526784}}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_native_layer_norm_transpose_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel):
    xnumel = 20480
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
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + (r0_2 + 768*x3), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_2 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
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
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp33.to(tl.float32)
    tl.store(out_ptr2 + (r0_2 + 768*x1 + 15360*x0), tmp34, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/fe/cfegkqq3esrte2feubgrgwamgsc5mmp7jqaztdsuvincjmweodhr.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   multi_head_attention_forward => convert_element_type_6
# Graph fragment:
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_8, torch.float16), kwargs = {})
triton_poi_fused__to_copy_3 = async_compile.triton('triton_poi_fused__to_copy_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 14155776}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/mp/cmpn6snc77azitufug3ncra37r6q5vnkgzfn3gew3cfogqml6sdi.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_6, clone_3, convert_element_type_5, permute_5, squeeze, unsqueeze_6, view_5
# Graph fragment:
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_7, torch.float16), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %convert_element_type_5), kwargs = {})
#   %view_5 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_6, [1024, 20, 3, 768]), kwargs = {})
#   %unsqueeze_6 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_5, 0), kwargs = {})
#   %permute_5 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_6, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : [num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_5, -2), kwargs = {})
#   %clone_3 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_4 = async_compile.triton('triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 283124736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 47185920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 768)
    x1 = ((xindex // 768) % 20480)
    x2 = xindex // 15728640
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 768*x2 + 2304*x1), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + 768*x2), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tl.store(out_ptr0 + (x3), tmp3, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/yx/cyxfx73edxuqkohmsxwbzzlbadyklhfh46gsl3lluq4335hvnere.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_6, clone_3, convert_element_type_5, permute_5, permute_6, permute_7, permute_8, select, select_1, select_2, squeeze, unsqueeze_6, view_10, view_11, view_5, view_6, view_7, view_8, view_9
# Graph fragment:
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_7, torch.float16), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %convert_element_type_5), kwargs = {})
#   %view_5 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_6, [1024, 20, 3, 768]), kwargs = {})
#   %unsqueeze_6 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_5, 0), kwargs = {})
#   %permute_5 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_6, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : [num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_5, -2), kwargs = {})
#   %clone_3 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_3, 0, 0), kwargs = {})
#   %select_1 : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_3, 0, 1), kwargs = {})
#   %select_2 : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_3, 0, 2), kwargs = {})
#   %view_6 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select, [1024, 160, 96]), kwargs = {})
#   %permute_6 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [1, 0, 2]), kwargs = {})
#   %view_7 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [1024, 160, 96]), kwargs = {})
#   %permute_7 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_7, [1, 0, 2]), kwargs = {})
#   %view_8 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_2, [1024, 160, 96]), kwargs = {})
#   %permute_8 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [1, 0, 2]), kwargs = {})
#   %view_9 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_6, [20, 8, 1024, 96]), kwargs = {})
#   %view_10 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_7, [20, 8, 1024, 96]), kwargs = {})
#   %view_11 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_8, [20, 8, 1024, 96]), kwargs = {})
#   %graphsafe_run_with_rng_state : [num_users=1] = call_function[target=torch.ops.higher_order.graphsafe_run_with_rng_state](args = (aten._scaled_dot_product_flash_attention.default, %view_9, %view_10, %view_11, 0.2), kwargs = {scale: 0.10206207261596577, rng_state: %fwd_rng_state_0})
triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_5 = async_compile.triton('triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 94371840}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/iq/ciq7sjt6bimj7su5f5p5jywsii6ja2pme6h5fg3g4kdckxvfmbe3.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_6, clone_3, convert_element_type_5, permute_5, permute_6, permute_7, permute_8, select, select_1, select_2, squeeze, unsqueeze_6, view_10, view_11, view_5, view_6, view_7, view_8, view_9
# Graph fragment:
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_7, torch.float16), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %convert_element_type_5), kwargs = {})
#   %view_5 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_6, [1024, 20, 3, 768]), kwargs = {})
#   %unsqueeze_6 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_5, 0), kwargs = {})
#   %permute_5 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_6, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : [num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_5, -2), kwargs = {})
#   %clone_3 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_3, 0, 0), kwargs = {})
#   %select_1 : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_3, 0, 1), kwargs = {})
#   %select_2 : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_3, 0, 2), kwargs = {})
#   %view_6 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select, [1024, 160, 96]), kwargs = {})
#   %permute_6 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [1, 0, 2]), kwargs = {})
#   %view_7 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [1024, 160, 96]), kwargs = {})
#   %permute_7 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_7, [1, 0, 2]), kwargs = {})
#   %view_8 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_2, [1024, 160, 96]), kwargs = {})
#   %permute_8 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [1, 0, 2]), kwargs = {})
#   %view_9 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_6, [20, 8, 1024, 96]), kwargs = {})
#   %view_10 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_7, [20, 8, 1024, 96]), kwargs = {})
#   %view_11 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_8, [20, 8, 1024, 96]), kwargs = {})
#   %graphsafe_run_with_rng_state : [num_users=1] = call_function[target=torch.ops.higher_order.graphsafe_run_with_rng_state](args = (aten._scaled_dot_product_flash_attention.default, %view_9, %view_10, %view_11, 0.2), kwargs = {scale: 0.10206207261596577, rng_state: %fwd_rng_state_0})
triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_6 = async_compile.triton('triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 94371840}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (15728640 + x4), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/vs/cvswte72pj6g5flnesqjrdjbupgj73u5uiwpfcrvk7iout2ww4yq.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_6, clone_3, convert_element_type_5, permute_5, permute_6, permute_7, permute_8, select, select_1, select_2, squeeze, unsqueeze_6, view_10, view_11, view_5, view_6, view_7, view_8, view_9
# Graph fragment:
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_7, torch.float16), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %convert_element_type_5), kwargs = {})
#   %view_5 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_6, [1024, 20, 3, 768]), kwargs = {})
#   %unsqueeze_6 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_5, 0), kwargs = {})
#   %permute_5 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_6, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : [num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_5, -2), kwargs = {})
#   %clone_3 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_3, 0, 0), kwargs = {})
#   %select_1 : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_3, 0, 1), kwargs = {})
#   %select_2 : [num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_3, 0, 2), kwargs = {})
#   %view_6 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select, [1024, 160, 96]), kwargs = {})
#   %permute_6 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [1, 0, 2]), kwargs = {})
#   %view_7 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [1024, 160, 96]), kwargs = {})
#   %permute_7 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_7, [1, 0, 2]), kwargs = {})
#   %view_8 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_2, [1024, 160, 96]), kwargs = {})
#   %permute_8 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [1, 0, 2]), kwargs = {})
#   %view_9 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_6, [20, 8, 1024, 96]), kwargs = {})
#   %view_10 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_7, [20, 8, 1024, 96]), kwargs = {})
#   %view_11 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_8, [20, 8, 1024, 96]), kwargs = {})
#   %graphsafe_run_with_rng_state : [num_users=1] = call_function[target=torch.ops.higher_order.graphsafe_run_with_rng_state](args = (aten._scaled_dot_product_flash_attention.default, %view_9, %view_10, %view_11, 0.2), kwargs = {scale: 0.10206207261596577, rng_state: %fwd_rng_state_0})
triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_7 = async_compile.triton('triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 94371840}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (31457280 + x4), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/nd/cnd4lvnrwkgjwxj6chd4qjjfeffkee2btxe5fggm5gifsejd5rey.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   multi_head_attention_forward => convert_element_type_11
# Graph fragment:
#   %convert_element_type_11 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_9, torch.float16), kwargs = {})
triton_poi_fused__to_copy_8 = async_compile.triton('triton_poi_fused__to_copy_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/l3/cl3r5vyfsvpgxy4rseedy365u6qcxfn2wgwncd7mghthivov5ror.py
# Topologically Sorted Source Nodes: [x, x_1, multi_head_attention_forward, add, transpose_1, layer_norm, input_1], Original ATen: [aten._to_copy, aten.add, aten.addmm, aten.view, aten.transpose, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add_7, add_tensor_37
#   input_1 => convert_element_type_17
#   layer_norm => add_8, add_9, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
#   multi_head_attention_forward => convert_element_type_10, view_13
#   transpose_1 => permute_11
#   x => add_2, convert_element_type
#   x_1 => add_3
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_3, torch.float16), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2, %convert_element_type), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %primals_4), kwargs = {})
#   %convert_element_type_10 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_10, torch.float16), kwargs = {})
#   %add_tensor_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_37, %convert_element_type_10), kwargs = {})
#   %view_13 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_37, [1024, 20, 768]), kwargs = {})
#   %permute_11 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_13, [1, 0, 2]), kwargs = {})
#   %add_7 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %permute_11), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_7, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_11, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %getitem_12), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_11), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %primals_12), kwargs = {})
#   %convert_element_type_17 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9, torch.float16), kwargs = {})
triton_per_fused__to_copy_add_addmm_native_layer_norm_transpose_view_9 = async_compile.triton('triton_per_fused__to_copy_add_addmm_native_layer_norm_transpose_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_addmm_native_layer_norm_transpose_view_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 254816256}}
)
@triton.jit
def triton_per_fused__to_copy_add_addmm_native_layer_norm_transpose_view_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, r0_numel):
    xnumel = 20480
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
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + (r0_2 + 768*x3), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_2 + 768*x0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r0_2 + 768*x1 + 15360*x0), r0_mask, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr5 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr6 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 + tmp5
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp6 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [R0_BLOCK])
    tmp15 = tl.where(r0_mask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [R0_BLOCK])
    tmp18 = tl.where(r0_mask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tl.full([1], 768, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = (tmp19 / tmp21)
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [R0_BLOCK])
    tmp27 = tl.where(r0_mask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = tmp12 - tmp22
    tmp30 = 768.0
    tmp31 = (tmp28 / tmp30)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp39.to(tl.float32)
    tl.store(out_ptr0 + (r0_2 + 768*x3), tmp12, r0_mask)
    tl.store(out_ptr3 + (r0_2 + 768*x3), tmp40, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/ef/cefrz2jlpixarunv3gibpcbscnflem5i5umeh2prskitpqkybox2.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_1 => convert_element_type_16
# Graph fragment:
#   %convert_element_type_16 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_13, torch.float16), kwargs = {})
triton_poi_fused__to_copy_10 = async_compile.triton('triton_poi_fused__to_copy_10', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/eq/ceqydhwgz6eikefw45bgz3vq7z2zemw7n6xeb6iey3me2knpjzhl.py
# Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
# Source node to ATen node mapping:
#   add => add_tensor_36
#   input_1 => convert_element_type_15, view_15
#   input_2 => add_10, convert_element_type_21, convert_element_type_22, erf, mul_4, mul_5, mul_6
# Graph fragment:
#   %convert_element_type_15 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_14, torch.float16), kwargs = {})
#   %add_tensor_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_36, %convert_element_type_15), kwargs = {})
#   %view_15 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_36, [20, 1024, 1536]), kwargs = {})
#   %convert_element_type_21 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_15, torch.float32), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_21, 0.5), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_21, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_5,), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_10), kwargs = {})
#   %convert_element_type_22 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_6, torch.float16), kwargs = {})
triton_poi_fused__to_copy_addmm_gelu_view_11 = async_compile.triton('triton_poi_fused__to_copy_addmm_gelu_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_gelu_view_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 188749824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_gelu_view_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 31457280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1536)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/hg/chg2p7bcinn2qbd6zsyozk3ev4pjmetfblehmygq6h77abbodsqi.py
# Topologically Sorted Source Nodes: [input_3, add, , convert_element_type, input_4, y, query, multi_head_attention_forward], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#    => inductor_lookup_seed_default, inductor_random_default_11
#   add => add_11, add_tensor_35
#   convert_element_type => convert_element_type_default_63
#   input_3 => convert_element_type_23, view_17
#   input_4 => gt, mul_7, mul_8
#   multi_head_attention_forward => clone_4, convert_element_type_30
#   query => permute_14
#   y => add_12, add_13, mul_10, mul_9, rsqrt_2, sub_2, var_mean_2
# Graph fragment:
#   %convert_element_type_23 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_16, torch.float16), kwargs = {})
#   %add_tensor_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_35, %convert_element_type_23), kwargs = {})
#   %view_17 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_35, [20, 1024, 768]), kwargs = {})
#   %inductor_lookup_seed_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_11 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([20, 1024, 768], %inductor_lookup_seed_default, rand), kwargs = {})
#   %convert_element_type_default_63 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%inductor_random_default_11, torch.float16), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convert_element_type_default_63, 0.2), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt, %view_17), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, 1.25), kwargs = {})
#   %add_11 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %mul_8), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_11, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_13, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_11, %getitem_14), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %primals_17), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %primals_18), kwargs = {})
#   %permute_14 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_13, [1, 0, 2]), kwargs = {})
#   %convert_element_type_30 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_14, torch.float16), kwargs = {})
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_30,), kwargs = {memory_format: torch.contiguous_format})
triton_per_fused__to_copy_add_addmm_clone_native_dropout_native_layer_norm_transpose_view_12 = async_compile.triton('triton_per_fused__to_copy_add_addmm_clone_native_dropout_native_layer_norm_transpose_view_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp16', 'load_seed_offset': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_addmm_clone_native_dropout_native_layer_norm_transpose_view_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 283124736}}
)
@triton.jit
def triton_per_fused__to_copy_add_addmm_clone_native_dropout_native_layer_norm_transpose_view_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, load_seed_offset, xnumel, r0_numel):
    xnumel = 20480
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
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_1 + 768*x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp4 = tmp2.to(tl.float32)
    tmp5 = 0.2
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = tmp7 * tmp11
    tmp13 = 1.25
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp3 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [R0_BLOCK])
    tmp19 = tl.where(r0_mask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [R0_BLOCK])
    tmp22 = tl.where(r0_mask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 768, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = (tmp23 / tmp25)
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [R0_BLOCK])
    tmp31 = tl.where(r0_mask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 768.0
    tmp35 = (tmp32 / tmp34)
    tmp36 = 1e-05
    tmp37 = tmp35 + tmp36
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp43.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp16, r0_mask)
    tl.store(out_ptr2 + (r0_1 + 768*x3 + 15360*x2), tmp44, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/hi/chiiglpai4dtd5zpryci6mq67emrirsc2sn62nnsl5ljmlemnguh.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward, add, transpose_1, layer_norm, input_1], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add_15, add_tensor_34
#   input_1 => convert_element_type_40
#   layer_norm => add_16, add_17, mul_11, mul_12, rsqrt_3, sub_3, var_mean_3
#   multi_head_attention_forward => convert_element_type_33, view_28
#   transpose_1 => permute_22
# Graph fragment:
#   %convert_element_type_33 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_22, torch.float16), kwargs = {})
#   %add_tensor_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_34, %convert_element_type_33), kwargs = {})
#   %view_28 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_34, [1024, 20, 768]), kwargs = {})
#   %permute_22 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_28, [1, 0, 2]), kwargs = {})
#   %add_15 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %permute_22), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_15, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_24, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_16,), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_15, %getitem_25), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %primals_23), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %primals_24), kwargs = {})
#   %convert_element_type_40 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_17, torch.float16), kwargs = {})
triton_per_fused__to_copy_add_addmm_native_layer_norm_transpose_view_13 = async_compile.triton('triton_per_fused__to_copy_add_addmm_native_layer_norm_transpose_view_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_addmm_native_layer_norm_transpose_view_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 283124736}}
)
@triton.jit
def triton_per_fused__to_copy_add_addmm_native_layer_norm_transpose_view_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr3, xnumel, r0_numel):
    xnumel = 20480
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
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + (r0_2 + 768*x3), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 768*x1 + 15360*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp0 + tmp5
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
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp33.to(tl.float32)
    tl.store(out_ptr0 + (r0_2 + 768*x3), tmp6, r0_mask)
    tl.store(out_ptr3 + (r0_2 + 768*x3), tmp34, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/7j/c7jwrvtfj2gshilfo62ce47j2k3iz54v7fxlvnudc5seruuhirhr.py
# Topologically Sorted Source Nodes: [input_3, add, , convert_element_type, input_4, y, query, multi_head_attention_forward], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#    => inductor_lookup_seed_default_1, inductor_random_default_10
#   add => add_19, add_tensor_32
#   convert_element_type => convert_element_type_default_62
#   input_3 => convert_element_type_46, view_32
#   input_4 => gt_1, mul_16, mul_17
#   multi_head_attention_forward => clone_6, convert_element_type_53
#   query => permute_25
#   y => add_20, add_21, mul_18, mul_19, rsqrt_4, sub_4, var_mean_4
# Graph fragment:
#   %convert_element_type_46 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_28, torch.float16), kwargs = {})
#   %add_tensor_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_32, %convert_element_type_46), kwargs = {})
#   %view_32 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_32, [20, 1024, 768]), kwargs = {})
#   %inductor_lookup_seed_default_1 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default_10 : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([20, 1024, 768], %inductor_lookup_seed_default_1, rand), kwargs = {})
#   %convert_element_type_default_62 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%inductor_random_default_10, torch.float16), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convert_element_type_default_62, 0.2), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_1, %view_32), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, 1.25), kwargs = {})
#   %add_19 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %mul_17), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_19, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_20,), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_19, %getitem_27), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %primals_29), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %primals_30), kwargs = {})
#   %permute_25 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_21, [1, 0, 2]), kwargs = {})
#   %convert_element_type_53 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_25, torch.float16), kwargs = {})
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_53,), kwargs = {memory_format: torch.contiguous_format})
triton_per_fused__to_copy_add_addmm_clone_native_dropout_native_layer_norm_transpose_view_14 = async_compile.triton('triton_per_fused__to_copy_add_addmm_clone_native_dropout_native_layer_norm_transpose_view_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp16', 'load_seed_offset': 'constexpr', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'load_seed_offset': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_addmm_clone_native_dropout_native_layer_norm_transpose_view_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 283124736}}
)
@triton.jit
def triton_per_fused__to_copy_add_addmm_clone_native_dropout_native_layer_norm_transpose_view_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, load_seed_offset, xnumel, r0_numel):
    xnumel = 20480
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
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_1 + 768*x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp4 = tmp2.to(tl.float32)
    tmp5 = 0.2
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = tmp7 * tmp11
    tmp13 = 1.25
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp3 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [R0_BLOCK])
    tmp19 = tl.where(r0_mask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [R0_BLOCK])
    tmp22 = tl.where(r0_mask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 768, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = (tmp23 / tmp25)
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [R0_BLOCK])
    tmp31 = tl.where(r0_mask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 768.0
    tmp35 = (tmp32 / tmp34)
    tmp36 = 1e-05
    tmp37 = tmp35 + tmp36
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp43.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp16, r0_mask)
    tl.store(out_ptr2 + (r0_1 + 768*x3 + 15360*x2), tmp44, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/xr/cxr2rcbbyrqs5i6x3lq2wxxci7hfmmlksz6anrnjep4rgvvgcgow.py
# Topologically Sorted Source Nodes: [input_3, add, , convert_element_type, input_4, input_1], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.native_dropout, aten.add]
# Source node to ATen node mapping:
#    => inductor_lookup_seed_default_11, inductor_random_default
#   add => add_99, add_tensor_2
#   convert_element_type => convert_element_type_default_52
#   input_1 => convert_element_type_283
#   input_3 => convert_element_type_276, view_182
#   input_4 => gt_11, mul_106, mul_107
# Graph fragment:
#   %convert_element_type_276 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_148, torch.float16), kwargs = {})
#   %add_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_2, %convert_element_type_276), kwargs = {})
#   %view_182 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_2, [20, 1024, 768]), kwargs = {})
#   %inductor_lookup_seed_default_11 : [num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 11), kwargs = {})
#   %inductor_random_default : [num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([20, 1024, 768], %inductor_lookup_seed_default_11, rand), kwargs = {})
#   %convert_element_type_default_52 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%inductor_random_default, torch.float16), kwargs = {})
#   %gt_11 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convert_element_type_default_52, 0.2), kwargs = {})
#   %mul_106 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_11, %view_182), kwargs = {})
#   %mul_107 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_106, 1.25), kwargs = {})
#   %add_99 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_95, %mul_107), kwargs = {})
#   %convert_element_type_283 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_99, torch.float16), kwargs = {})
triton_poi_fused__to_copy_add_addmm_native_dropout_view_15 = async_compile.triton('triton_poi_fused__to_copy_add_addmm_native_dropout_view_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'load_seed_offset': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_native_dropout_view_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 157289472}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_native_dropout_view_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = (xindex % 768)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp8 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp4 = tmp2.to(tl.float32)
    tmp5 = 0.2
    tmp6 = tmp4 > tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = tmp7 * tmp11
    tmp13 = 1.25
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp3 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/5r/c5rxxw6i5tz2dinkjzfdhuvophoue54m5eqrmw5ksyqbvu4wvooq.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   input_1 => convert_element_type_282, permute_135
# Graph fragment:
#   %convert_element_type_282 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_149, torch.float16), kwargs = {})
#   %permute_135 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_282, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_16 = async_compile.triton('triton_poi_fused__to_copy_t_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 786432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/nq/cnqx2tme2ybil5pd5h6cu6b3yie6zg5u6m47uj45sceupptl4226.py
# Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.relu]
# Source node to ATen node mapping:
#   add => add_tensor_1
#   input_1 => convert_element_type_281, view_184
#   input_2 => relu
# Graph fragment:
#   %convert_element_type_281 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_150, torch.float16), kwargs = {})
#   %add_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_1, %convert_element_type_281), kwargs = {})
#   %view_184 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [20, 1024, 128]), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_184,), kwargs = {})
triton_poi_fused__to_copy_addmm_relu_view_17 = async_compile.triton('triton_poi_fused__to_copy_addmm_relu_view_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_relu_view_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 15729152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_relu_view_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = triton_helpers.maximum(tmp4, tmp3)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/eh/ceh26p7nywyyzesw4lrwpio3er4mh72t6xpvnmu7cdesrwmi6rhf.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   input_3 => convert_element_type_288, permute_136
# Graph fragment:
#   %convert_element_type_288 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_151, torch.float16), kwargs = {})
#   %permute_136 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_288, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_18 = async_compile.triton('triton_poi_fused__to_copy_t_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16384}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/ik/cikxbcyrt6nmz7rln2cun7ovjlxsm5gkox3vpkyjms6khf6tcioq.py
# Topologically Sorted Source Nodes: [x_27], Original ATen: [aten.col2im]
# Source node to ATen node mapping:
#   x_27 => full_default
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([20, 1, 128, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_col2im_19 = async_compile.triton('triton_poi_fused_col2im_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_col2im_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2621440}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_col2im_19(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/23/c23t27yxftqieyhzf7q6tudsedncdu4rzqwa67tir2bbzowxa663.py
# Topologically Sorted Source Nodes: [y, input_3, add, x_26, x_27], Original ATen: [aten.im2col, aten._to_copy, aten.addmm, aten.view, aten.permute, aten.col2im]
# Source node to ATen node mapping:
#   add => add_tensor
#   input_3 => convert_element_type_287, view_191
#   x_26 => permute_137
#   x_27 => convert_element_type_292, index_put, permute_138, view_192
#   y => add, iota, iota_1, unsqueeze, unsqueeze_1, unsqueeze_4, unsqueeze_5
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 4, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_1 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_1, -1), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %unsqueeze_4 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add, -1), kwargs = {})
#   %unsqueeze_5 : [num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %convert_element_type_287 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_152, torch.float16), kwargs = {})
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %convert_element_type_287), kwargs = {})
#   %view_191 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [20, 1024, 16]), kwargs = {})
#   %permute_137 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_191, [0, 2, 1]), kwargs = {})
#   %convert_element_type_292 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_137, torch.float32), kwargs = {})
#   %view_192 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_292, [20, 1, 4, 4, 32, 32]), kwargs = {})
#   %permute_138 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_192, [0, 1, 2, 4, 3, 5]), kwargs = {})
#   %index_put : [num_users=4] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default, [None, None, %unsqueeze_5, %add], %permute_138, True), kwargs = {})
triton_poi_fused__to_copy_addmm_col2im_im2col_permute_view_20 = async_compile.triton('triton_poi_fused__to_copy_addmm_col2im_im2col_permute_view_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_col2im_im2col_permute_view_20', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3276864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_col2im_im2col_permute_view_20(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 32)
    x2 = ((xindex // 128) % 4)
    x5 = xindex // 512
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*x2 + 16*x1 + 512*x5), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + 4*x2), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tl.atomic_add(out_ptr0 + (x6), tmp4, None, sem='relaxed')
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/c3/cc3bqxdjj3o72rsyooyh45ee2uget7qngitb6y57pwcroq3c7bs3.py
# Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp, aten.view]
# Source node to ATen node mapping:
#   x_28 => add_102, clamp_min, convert_element_type_295, convert_element_type_296, iota_8, mul_108, sub_24, view_193
# Graph fragment:
#   %iota_8 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_295 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_8, torch.float32), kwargs = {})
#   %add_102 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_295, 0.5), kwargs = {})
#   %mul_108 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_102, 4.0), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_108, 0.5), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_24, 0.0), kwargs = {})
#   %view_193 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clamp_min, [32, 1]), kwargs = {})
#   %convert_element_type_296 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_193, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_view_21 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_view_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_view_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_view_21(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 4.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/dd/cddbgkywxsd4sljnxi4uoll5gipzxxuhgqriwesohriwohyks43o.py
# Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   x_28 => add_103, clamp_max
# Graph fragment:
#   %add_103 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_296, 1), kwargs = {})
#   %clamp_max : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_103, 127), kwargs = {})
triton_poi_fused_add_clamp_22 = async_compile.triton('triton_poi_fused_add_clamp_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_22(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 4.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 127, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/fm/cfmnvsbjbf2ybujsk2ko4dmoehvqmsj5ocabkvkapn6cpzovro3a.py
# Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   x_28 => add_102, clamp_max_2, clamp_min, clamp_min_2, convert_element_type_295, iota_8, mul_108, sub_24, sub_26
# Graph fragment:
#   %iota_8 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_295 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_8, torch.float32), kwargs = {})
#   %add_102 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_295, 0.5), kwargs = {})
#   %mul_108 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_102, 4.0), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_108, 0.5), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_24, 0.0), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min, %convert_element_type_298), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_26, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_23 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 256}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_23(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 4.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 - tmp10
    tmp12 = triton_helpers.maximum(tmp11, tmp7)
    tmp13 = 1.0
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_Adithya/ks/ckspoeribuayt6w6rw2ipuozjayjey4jrwmymmra5fkdtmscwiq6.py
# Topologically Sorted Source Nodes: [x_28], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add, aten._to_copy]
# Source node to ATen node mapping:
#   x_28 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_106, add_107, add_108, convert_element_type_299, mul_110, mul_111, mul_112, sub_27, sub_28, sub_30
# Graph fragment:
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%index_put, [None, None, %convert_element_type_296, %convert_element_type_298]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%index_put, [None, None, %convert_element_type_296, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%index_put, [None, None, %clamp_max, %convert_element_type_298]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%index_put, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %clamp_max_2), kwargs = {})
#   %add_106 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_110), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_111 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %clamp_max_2), kwargs = {})
#   %add_107 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_111), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_107, %add_106), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %clamp_max_3), kwargs = {})
#   %add_108 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_106, %mul_112), kwargs = {})
#   %convert_element_type_299 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_108, torch.float16), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_mul_sub_24 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_mul_sub_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_mul_sub_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'C44385CD352E26CD95781CA810C847542CF8E7D835DA47390E84598F651E79BC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_mul_sub_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x2 = xindex // 1024
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 128, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 128*tmp4 + 16384*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 128*tmp4 + 16384*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp20 = tmp19 + tmp1
    tmp21 = tmp19 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp19)
    tmp23 = tl.load(in_ptr2 + (tmp8 + 128*tmp22 + 16384*x2), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (tmp13 + 128*tmp22 + 16384*x2), None, eviction_policy='evict_last')
    tmp25 = tmp24 - tmp23
    tmp26 = tmp25 * tmp16
    tmp27 = tmp23 + tmp26
    tmp28 = tmp27 - tmp18
    tmp30 = tmp28 * tmp29
    tmp31 = tmp18 + tmp30
    tmp32 = tmp31.to(tl.float32)
    tl.store(out_ptr1 + (x4), tmp32, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, fwd_rng_state_0, fwd_rng_state_1, fwd_rng_state_2, fwd_rng_state_3, fwd_rng_state_4, fwd_rng_state_5, fwd_rng_state_6, fwd_rng_state_7, fwd_rng_state_8, fwd_rng_state_9, fwd_rng_state_10, fwd_rng_state_11 = args
    args.clear()
    assert_size_stride(primals_1, (20, 1, 128, 128), (16384, 16384, 128, 1))
    assert_size_stride(primals_2, (768, 16), (16, 1))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_4, (1024, 768), (768, 1))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (2304, ), (1, ))
    assert_size_stride(primals_8, (2304, 768), (768, 1))
    assert_size_stride(primals_9, (768, 768), (768, 1))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (1536, 768), (768, 1))
    assert_size_stride(primals_14, (1536, ), (1, ))
    assert_size_stride(primals_15, (768, 1536), (1536, 1))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (2304, ), (1, ))
    assert_size_stride(primals_20, (2304, 768), (768, 1))
    assert_size_stride(primals_21, (768, 768), (768, 1))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (1536, 768), (768, 1))
    assert_size_stride(primals_26, (1536, ), (1, ))
    assert_size_stride(primals_27, (768, 1536), (1536, 1))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (2304, ), (1, ))
    assert_size_stride(primals_32, (2304, 768), (768, 1))
    assert_size_stride(primals_33, (768, 768), (768, 1))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (1536, 768), (768, 1))
    assert_size_stride(primals_38, (1536, ), (1, ))
    assert_size_stride(primals_39, (768, 1536), (1536, 1))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (2304, ), (1, ))
    assert_size_stride(primals_44, (2304, 768), (768, 1))
    assert_size_stride(primals_45, (768, 768), (768, 1))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (1536, 768), (768, 1))
    assert_size_stride(primals_50, (1536, ), (1, ))
    assert_size_stride(primals_51, (768, 1536), (1536, 1))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (2304, ), (1, ))
    assert_size_stride(primals_56, (2304, 768), (768, 1))
    assert_size_stride(primals_57, (768, 768), (768, 1))
    assert_size_stride(primals_58, (768, ), (1, ))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (1536, 768), (768, 1))
    assert_size_stride(primals_62, (1536, ), (1, ))
    assert_size_stride(primals_63, (768, 1536), (1536, 1))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (2304, ), (1, ))
    assert_size_stride(primals_68, (2304, 768), (768, 1))
    assert_size_stride(primals_69, (768, 768), (768, 1))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (1536, 768), (768, 1))
    assert_size_stride(primals_74, (1536, ), (1, ))
    assert_size_stride(primals_75, (768, 1536), (1536, 1))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_79, (2304, ), (1, ))
    assert_size_stride(primals_80, (2304, 768), (768, 1))
    assert_size_stride(primals_81, (768, 768), (768, 1))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (1536, 768), (768, 1))
    assert_size_stride(primals_86, (1536, ), (1, ))
    assert_size_stride(primals_87, (768, 1536), (1536, 1))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_91, (2304, ), (1, ))
    assert_size_stride(primals_92, (2304, 768), (768, 1))
    assert_size_stride(primals_93, (768, 768), (768, 1))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (768, ), (1, ))
    assert_size_stride(primals_97, (1536, 768), (768, 1))
    assert_size_stride(primals_98, (1536, ), (1, ))
    assert_size_stride(primals_99, (768, 1536), (1536, 1))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (768, ), (1, ))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (2304, ), (1, ))
    assert_size_stride(primals_104, (2304, 768), (768, 1))
    assert_size_stride(primals_105, (768, 768), (768, 1))
    assert_size_stride(primals_106, (768, ), (1, ))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_108, (768, ), (1, ))
    assert_size_stride(primals_109, (1536, 768), (768, 1))
    assert_size_stride(primals_110, (1536, ), (1, ))
    assert_size_stride(primals_111, (768, 1536), (1536, 1))
    assert_size_stride(primals_112, (768, ), (1, ))
    assert_size_stride(primals_113, (768, ), (1, ))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (2304, ), (1, ))
    assert_size_stride(primals_116, (2304, 768), (768, 1))
    assert_size_stride(primals_117, (768, 768), (768, 1))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (1536, 768), (768, 1))
    assert_size_stride(primals_122, (1536, ), (1, ))
    assert_size_stride(primals_123, (768, 1536), (1536, 1))
    assert_size_stride(primals_124, (768, ), (1, ))
    assert_size_stride(primals_125, (768, ), (1, ))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_127, (2304, ), (1, ))
    assert_size_stride(primals_128, (2304, 768), (768, 1))
    assert_size_stride(primals_129, (768, 768), (768, 1))
    assert_size_stride(primals_130, (768, ), (1, ))
    assert_size_stride(primals_131, (768, ), (1, ))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_133, (1536, 768), (768, 1))
    assert_size_stride(primals_134, (1536, ), (1, ))
    assert_size_stride(primals_135, (768, 1536), (1536, 1))
    assert_size_stride(primals_136, (768, ), (1, ))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_138, (768, ), (1, ))
    assert_size_stride(primals_139, (2304, ), (1, ))
    assert_size_stride(primals_140, (2304, 768), (768, 1))
    assert_size_stride(primals_141, (768, 768), (768, 1))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_144, (768, ), (1, ))
    assert_size_stride(primals_145, (1536, 768), (768, 1))
    assert_size_stride(primals_146, (1536, ), (1, ))
    assert_size_stride(primals_147, (768, 1536), (1536, 1))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (128, 768), (768, 1))
    assert_size_stride(primals_150, (128, ), (1, ))
    assert_size_stride(primals_151, (16, 128), (128, 1))
    assert_size_stride(primals_152, (16, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((20, 1024, 16), (16384, 16, 1), torch.float16)
        # Topologically Sorted Source Nodes: [y, y_1, x], Original ATen: [aten.im2col, aten.permute, aten._to_copy, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clone_im2col_permute_0.run(primals_1, buf0, 327680, stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((768, 16), (16, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(primals_2, buf1, 12288, stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((20480, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (20480, 16), (16, 1), 0), reinterpret_tensor(buf1, (16, 768), (1, 16), 0), out=buf2)
        del buf1
        buf6 = empty_strided_cuda((1024, 20, 768), (15360, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x, x_1, y, query, multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_clone_native_layer_norm_transpose_2.run(buf2, primals_3, primals_4, primals_5, primals_6, buf6, 20480, 768, stream=stream0)
        buf7 = empty_strided_cuda((2304, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(primals_8, buf7, 1769472, stream=stream0)
        buf8 = empty_strided_cuda((20480, 2304), (2304, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x, x_1, y, query, multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm, aten.transpose, aten.t, aten.clone, aten._unsafe_view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf6, (20480, 768), (768, 1), 0), reinterpret_tensor(buf7, (768, 2304), (1, 768), 0), out=buf8)
        buf9 = empty_strided_cuda((3, 1024, 20, 768), (15728640, 15360, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_4.run(buf8, primals_7, buf9, 47185920, stream=stream0)
        buf10 = reinterpret_tensor(buf6, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_5.run(buf9, buf10, 15728640, stream=stream0)
        buf11 = empty_strided_cuda((20, 8, 1024, 96), (768, 96, 15360, 1), torch.float16)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_6.run(buf9, buf11, 15728640, stream=stream0)
        buf12 = empty_strided_cuda((20, 8, 1024, 96), (768, 96, 15360, 1), torch.float16)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_7.run(buf9, buf12, 15728640, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf13 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf10, buf11, buf12, 0.2, scale=0.10206207261596577, rng_state=fwd_rng_state_0)
        del buf10
        buf14 = buf13[0]
        assert_size_stride(buf14, (20, 8, 1024, 96), (768, 96, 15360, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf14, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf13
        buf19 = empty_strided_cuda((768, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_9, buf19, 589824, stream=stream0)
        buf20 = reinterpret_tensor(buf12, (20480, 768), (768, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten._to_copy, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf14, (20480, 768), (768, 1), 0), reinterpret_tensor(buf19, (768, 768), (1, 768), 0), out=buf20)
        buf21 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf25 = reinterpret_tensor(buf14, (20, 1024, 768), (786432, 768, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [x, x_1, multi_head_attention_forward, add, transpose_1, layer_norm, input_1], Original ATen: [aten._to_copy, aten.add, aten.addmm, aten.view, aten.transpose, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_native_layer_norm_transpose_view_9.run(buf2, primals_3, primals_4, buf20, primals_10, primals_11, primals_12, buf21, buf25, 20480, 768, stream=stream0)
        del primals_10
        buf26 = empty_strided_cuda((1536, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_13, buf26, 1179648, stream=stream0)
        buf27 = empty_strided_cuda((20480, 1536), (1536, 1), torch.float16)
        # Topologically Sorted Source Nodes: [layer_norm, input_1, ], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf25, (20480, 768), (768, 1), 0), reinterpret_tensor(buf26, (768, 1536), (1, 768), 0), out=buf27)
        buf28 = reinterpret_tensor(buf27, (20, 1024, 1536), (1572864, 1536, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_view_11.run(buf28, primals_14, 31457280, stream=stream0)
        buf29 = reinterpret_tensor(buf26, (768, 1536), (1536, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_15, buf29, 1179648, stream=stream0)
        buf30 = reinterpret_tensor(buf25, (20480, 768), (768, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, ], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
        extern_kernels.mm(reinterpret_tensor(buf28, (20480, 1536), (1536, 1), 0), reinterpret_tensor(buf29, (1536, 768), (1, 1536), 0), out=buf30)
        buf31 = empty_strided_cuda((12, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [12], out=buf31)
        buf32 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf33 = buf32; del buf32  # reuse
        buf37 = reinterpret_tensor(buf20, (1024, 20, 768), (15360, 768, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [input_3, add, , convert_element_type, input_4, y, query, multi_head_attention_forward], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_clone_native_dropout_native_layer_norm_transpose_view_12.run(buf33, buf31, buf21, buf30, primals_16, primals_17, primals_18, buf37, 0, 20480, 768, stream=stream0)
        del primals_16
        buf38 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(primals_20, buf38, 1769472, stream=stream0)
        buf39 = reinterpret_tensor(buf9, (20480, 2304), (2304, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.t, aten.clone, aten._unsafe_view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (20480, 768), (768, 1), 0), reinterpret_tensor(buf38, (768, 2304), (1, 768), 0), out=buf39)
        buf40 = reinterpret_tensor(buf8, (3, 1024, 20, 768), (15728640, 15360, 768, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_4.run(buf39, primals_19, buf40, 47185920, stream=stream0)
        buf41 = reinterpret_tensor(buf37, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_5.run(buf40, buf41, 15728640, stream=stream0)
        buf42 = reinterpret_tensor(buf30, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_6.run(buf40, buf42, 15728640, stream=stream0)
        buf43 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_7.run(buf40, buf43, 15728640, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf44 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf41, buf42, buf43, 0.2, scale=0.10206207261596577, rng_state=fwd_rng_state_1)
        del buf41
        buf45 = buf44[0]
        assert_size_stride(buf45, (20, 8, 1024, 96), (768, 96, 15360, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf45, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf44
        buf50 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_21, buf50, 589824, stream=stream0)
        buf51 = reinterpret_tensor(buf43, (20480, 768), (768, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten._to_copy, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf45, (20480, 768), (768, 1), 0), reinterpret_tensor(buf50, (768, 768), (1, 768), 0), out=buf51)
        buf52 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf56 = reinterpret_tensor(buf45, (20, 1024, 768), (786432, 768, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, add, transpose_1, layer_norm, input_1], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_native_layer_norm_transpose_view_13.run(buf33, buf51, primals_22, primals_23, primals_24, buf52, buf56, 20480, 768, stream=stream0)
        del primals_22
        buf57 = reinterpret_tensor(buf29, (1536, 768), (768, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_25, buf57, 1179648, stream=stream0)
        buf58 = reinterpret_tensor(buf28, (20480, 1536), (1536, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1, ], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf56, (20480, 768), (768, 1), 0), reinterpret_tensor(buf57, (768, 1536), (1, 768), 0), out=buf58)
        buf59 = reinterpret_tensor(buf58, (20, 1024, 1536), (1572864, 1536, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_view_11.run(buf59, primals_26, 31457280, stream=stream0)
        buf60 = reinterpret_tensor(buf57, (768, 1536), (1536, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_27, buf60, 1179648, stream=stream0)
        buf61 = reinterpret_tensor(buf56, (20480, 768), (768, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, ], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
        extern_kernels.mm(reinterpret_tensor(buf59, (20480, 1536), (1536, 1), 0), reinterpret_tensor(buf60, (1536, 768), (1, 1536), 0), out=buf61)
        buf62 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf63 = buf62; del buf62  # reuse
        buf67 = reinterpret_tensor(buf51, (1024, 20, 768), (15360, 768, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [input_3, add, , convert_element_type, input_4, y, query, multi_head_attention_forward], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_clone_native_dropout_native_layer_norm_transpose_view_14.run(buf63, buf31, buf52, buf61, primals_28, primals_29, primals_30, buf67, 1, 20480, 768, stream=stream0)
        del primals_28
        buf68 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(primals_32, buf68, 1769472, stream=stream0)
        buf69 = reinterpret_tensor(buf40, (20480, 2304), (2304, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.t, aten.clone, aten._unsafe_view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (20480, 768), (768, 1), 0), reinterpret_tensor(buf68, (768, 2304), (1, 768), 0), out=buf69)
        buf70 = reinterpret_tensor(buf39, (3, 1024, 20, 768), (15728640, 15360, 768, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_4.run(buf69, primals_31, buf70, 47185920, stream=stream0)
        buf71 = reinterpret_tensor(buf67, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_5.run(buf70, buf71, 15728640, stream=stream0)
        buf72 = reinterpret_tensor(buf61, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_6.run(buf70, buf72, 15728640, stream=stream0)
        buf73 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_7.run(buf70, buf73, 15728640, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf74 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf71, buf72, buf73, 0.2, scale=0.10206207261596577, rng_state=fwd_rng_state_2)
        del buf71
        buf75 = buf74[0]
        assert_size_stride(buf75, (20, 8, 1024, 96), (768, 96, 15360, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf75, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf74
        buf80 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_33, buf80, 589824, stream=stream0)
        buf81 = reinterpret_tensor(buf73, (20480, 768), (768, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten._to_copy, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf75, (20480, 768), (768, 1), 0), reinterpret_tensor(buf80, (768, 768), (1, 768), 0), out=buf81)
        buf82 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf86 = reinterpret_tensor(buf75, (20, 1024, 768), (786432, 768, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, add, transpose_1, layer_norm, input_1], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_native_layer_norm_transpose_view_13.run(buf63, buf81, primals_34, primals_35, primals_36, buf82, buf86, 20480, 768, stream=stream0)
        del primals_34
        buf87 = reinterpret_tensor(buf60, (1536, 768), (768, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_37, buf87, 1179648, stream=stream0)
        buf88 = reinterpret_tensor(buf59, (20480, 1536), (1536, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1, ], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf86, (20480, 768), (768, 1), 0), reinterpret_tensor(buf87, (768, 1536), (1, 768), 0), out=buf88)
        buf89 = reinterpret_tensor(buf88, (20, 1024, 1536), (1572864, 1536, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_view_11.run(buf89, primals_38, 31457280, stream=stream0)
        buf90 = reinterpret_tensor(buf87, (768, 1536), (1536, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_39, buf90, 1179648, stream=stream0)
        buf91 = reinterpret_tensor(buf86, (20480, 768), (768, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, ], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
        extern_kernels.mm(reinterpret_tensor(buf89, (20480, 1536), (1536, 1), 0), reinterpret_tensor(buf90, (1536, 768), (1, 1536), 0), out=buf91)
        buf92 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf93 = buf92; del buf92  # reuse
        buf97 = reinterpret_tensor(buf81, (1024, 20, 768), (15360, 768, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [input_3, add, , convert_element_type, input_4, y, query, multi_head_attention_forward], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_clone_native_dropout_native_layer_norm_transpose_view_12.run(buf93, buf31, buf82, buf91, primals_40, primals_41, primals_42, buf97, 2, 20480, 768, stream=stream0)
        del primals_40
        buf98 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(primals_44, buf98, 1769472, stream=stream0)
        buf99 = reinterpret_tensor(buf70, (20480, 2304), (2304, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.t, aten.clone, aten._unsafe_view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf97, (20480, 768), (768, 1), 0), reinterpret_tensor(buf98, (768, 2304), (1, 768), 0), out=buf99)
        buf100 = reinterpret_tensor(buf69, (3, 1024, 20, 768), (15728640, 15360, 768, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_4.run(buf99, primals_43, buf100, 47185920, stream=stream0)
        buf101 = reinterpret_tensor(buf97, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_5.run(buf100, buf101, 15728640, stream=stream0)
        buf102 = reinterpret_tensor(buf91, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_6.run(buf100, buf102, 15728640, stream=stream0)
        buf103 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_7.run(buf100, buf103, 15728640, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf104 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf101, buf102, buf103, 0.2, scale=0.10206207261596577, rng_state=fwd_rng_state_3)
        del buf101
        buf105 = buf104[0]
        assert_size_stride(buf105, (20, 8, 1024, 96), (768, 96, 15360, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf105, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf104
        buf110 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_45, buf110, 589824, stream=stream0)
        buf111 = reinterpret_tensor(buf103, (20480, 768), (768, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten._to_copy, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf105, (20480, 768), (768, 1), 0), reinterpret_tensor(buf110, (768, 768), (1, 768), 0), out=buf111)
        buf112 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf116 = reinterpret_tensor(buf105, (20, 1024, 768), (786432, 768, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, add, transpose_1, layer_norm, input_1], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_native_layer_norm_transpose_view_13.run(buf93, buf111, primals_46, primals_47, primals_48, buf112, buf116, 20480, 768, stream=stream0)
        del primals_46
        buf117 = reinterpret_tensor(buf90, (1536, 768), (768, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_49, buf117, 1179648, stream=stream0)
        buf118 = reinterpret_tensor(buf89, (20480, 1536), (1536, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1, ], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf116, (20480, 768), (768, 1), 0), reinterpret_tensor(buf117, (768, 1536), (1, 768), 0), out=buf118)
        buf119 = reinterpret_tensor(buf118, (20, 1024, 1536), (1572864, 1536, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_view_11.run(buf119, primals_50, 31457280, stream=stream0)
        buf120 = reinterpret_tensor(buf117, (768, 1536), (1536, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_51, buf120, 1179648, stream=stream0)
        buf121 = reinterpret_tensor(buf116, (20480, 768), (768, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, ], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
        extern_kernels.mm(reinterpret_tensor(buf119, (20480, 1536), (1536, 1), 0), reinterpret_tensor(buf120, (1536, 768), (1, 1536), 0), out=buf121)
        buf122 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf123 = buf122; del buf122  # reuse
        buf127 = reinterpret_tensor(buf111, (1024, 20, 768), (15360, 768, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [input_3, add, , convert_element_type, input_4, y, query, multi_head_attention_forward], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_clone_native_dropout_native_layer_norm_transpose_view_12.run(buf123, buf31, buf112, buf121, primals_52, primals_53, primals_54, buf127, 3, 20480, 768, stream=stream0)
        del primals_52
        buf128 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(primals_56, buf128, 1769472, stream=stream0)
        buf129 = reinterpret_tensor(buf100, (20480, 2304), (2304, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.t, aten.clone, aten._unsafe_view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (20480, 768), (768, 1), 0), reinterpret_tensor(buf128, (768, 2304), (1, 768), 0), out=buf129)
        buf130 = reinterpret_tensor(buf99, (3, 1024, 20, 768), (15728640, 15360, 768, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_4.run(buf129, primals_55, buf130, 47185920, stream=stream0)
        buf131 = reinterpret_tensor(buf127, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_5.run(buf130, buf131, 15728640, stream=stream0)
        buf132 = reinterpret_tensor(buf121, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_6.run(buf130, buf132, 15728640, stream=stream0)
        buf133 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_7.run(buf130, buf133, 15728640, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf134 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf131, buf132, buf133, 0.2, scale=0.10206207261596577, rng_state=fwd_rng_state_4)
        del buf131
        buf135 = buf134[0]
        assert_size_stride(buf135, (20, 8, 1024, 96), (768, 96, 15360, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf135, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf134
        buf140 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_57, buf140, 589824, stream=stream0)
        buf141 = reinterpret_tensor(buf133, (20480, 768), (768, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten._to_copy, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf135, (20480, 768), (768, 1), 0), reinterpret_tensor(buf140, (768, 768), (1, 768), 0), out=buf141)
        buf142 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf146 = reinterpret_tensor(buf135, (20, 1024, 768), (786432, 768, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, add, transpose_1, layer_norm, input_1], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_native_layer_norm_transpose_view_13.run(buf123, buf141, primals_58, primals_59, primals_60, buf142, buf146, 20480, 768, stream=stream0)
        del primals_58
        buf147 = reinterpret_tensor(buf120, (1536, 768), (768, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_61, buf147, 1179648, stream=stream0)
        buf148 = reinterpret_tensor(buf119, (20480, 1536), (1536, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1, ], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf146, (20480, 768), (768, 1), 0), reinterpret_tensor(buf147, (768, 1536), (1, 768), 0), out=buf148)
        buf149 = reinterpret_tensor(buf148, (20, 1024, 1536), (1572864, 1536, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_view_11.run(buf149, primals_62, 31457280, stream=stream0)
        buf150 = reinterpret_tensor(buf147, (768, 1536), (1536, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_63, buf150, 1179648, stream=stream0)
        buf151 = reinterpret_tensor(buf146, (20480, 768), (768, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, ], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
        extern_kernels.mm(reinterpret_tensor(buf149, (20480, 1536), (1536, 1), 0), reinterpret_tensor(buf150, (1536, 768), (1, 1536), 0), out=buf151)
        buf152 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf153 = buf152; del buf152  # reuse
        buf157 = reinterpret_tensor(buf141, (1024, 20, 768), (15360, 768, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [input_3, add, , convert_element_type, input_4, y, query, multi_head_attention_forward], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_clone_native_dropout_native_layer_norm_transpose_view_12.run(buf153, buf31, buf142, buf151, primals_64, primals_65, primals_66, buf157, 4, 20480, 768, stream=stream0)
        del primals_64
        buf158 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(primals_68, buf158, 1769472, stream=stream0)
        buf159 = reinterpret_tensor(buf130, (20480, 2304), (2304, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.t, aten.clone, aten._unsafe_view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (20480, 768), (768, 1), 0), reinterpret_tensor(buf158, (768, 2304), (1, 768), 0), out=buf159)
        buf160 = reinterpret_tensor(buf129, (3, 1024, 20, 768), (15728640, 15360, 768, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_4.run(buf159, primals_67, buf160, 47185920, stream=stream0)
        buf161 = reinterpret_tensor(buf157, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_5.run(buf160, buf161, 15728640, stream=stream0)
        buf162 = reinterpret_tensor(buf151, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_6.run(buf160, buf162, 15728640, stream=stream0)
        buf163 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_7.run(buf160, buf163, 15728640, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf164 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf161, buf162, buf163, 0.2, scale=0.10206207261596577, rng_state=fwd_rng_state_5)
        del buf161
        buf165 = buf164[0]
        assert_size_stride(buf165, (20, 8, 1024, 96), (768, 96, 15360, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf165, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf164
        buf170 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_69, buf170, 589824, stream=stream0)
        buf171 = reinterpret_tensor(buf163, (20480, 768), (768, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten._to_copy, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf165, (20480, 768), (768, 1), 0), reinterpret_tensor(buf170, (768, 768), (1, 768), 0), out=buf171)
        buf172 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf176 = reinterpret_tensor(buf165, (20, 1024, 768), (786432, 768, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, add, transpose_1, layer_norm, input_1], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_native_layer_norm_transpose_view_13.run(buf153, buf171, primals_70, primals_71, primals_72, buf172, buf176, 20480, 768, stream=stream0)
        del primals_70
        buf177 = reinterpret_tensor(buf150, (1536, 768), (768, 1), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_73, buf177, 1179648, stream=stream0)
        buf178 = reinterpret_tensor(buf149, (20480, 1536), (1536, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1, ], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf176, (20480, 768), (768, 1), 0), reinterpret_tensor(buf177, (768, 1536), (1, 768), 0), out=buf178)
        buf179 = reinterpret_tensor(buf178, (20, 1024, 1536), (1572864, 1536, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_view_11.run(buf179, primals_74, 31457280, stream=stream0)
        buf180 = reinterpret_tensor(buf177, (768, 1536), (1536, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_75, buf180, 1179648, stream=stream0)
        buf181 = reinterpret_tensor(buf176, (20480, 768), (768, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, ], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
        extern_kernels.mm(reinterpret_tensor(buf179, (20480, 1536), (1536, 1), 0), reinterpret_tensor(buf180, (1536, 768), (1, 1536), 0), out=buf181)
        buf182 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf183 = buf182; del buf182  # reuse
        buf187 = reinterpret_tensor(buf171, (1024, 20, 768), (15360, 768, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [input_3, add, , convert_element_type, input_4, y, query, multi_head_attention_forward], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_clone_native_dropout_native_layer_norm_transpose_view_12.run(buf183, buf31, buf172, buf181, primals_76, primals_77, primals_78, buf187, 5, 20480, 768, stream=stream0)
        del primals_76
        buf188 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(primals_80, buf188, 1769472, stream=stream0)
        buf189 = reinterpret_tensor(buf160, (20480, 2304), (2304, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.t, aten.clone, aten._unsafe_view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf187, (20480, 768), (768, 1), 0), reinterpret_tensor(buf188, (768, 2304), (1, 768), 0), out=buf189)
        buf190 = reinterpret_tensor(buf159, (3, 1024, 20, 768), (15728640, 15360, 768, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_4.run(buf189, primals_79, buf190, 47185920, stream=stream0)
        buf191 = reinterpret_tensor(buf187, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_5.run(buf190, buf191, 15728640, stream=stream0)
        buf192 = reinterpret_tensor(buf181, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_6.run(buf190, buf192, 15728640, stream=stream0)
        buf193 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_7.run(buf190, buf193, 15728640, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf194 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf191, buf192, buf193, 0.2, scale=0.10206207261596577, rng_state=fwd_rng_state_6)
        del buf191
        buf195 = buf194[0]
        assert_size_stride(buf195, (20, 8, 1024, 96), (768, 96, 15360, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf195, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf194
        buf200 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_81, buf200, 589824, stream=stream0)
        buf201 = reinterpret_tensor(buf193, (20480, 768), (768, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten._to_copy, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf195, (20480, 768), (768, 1), 0), reinterpret_tensor(buf200, (768, 768), (1, 768), 0), out=buf201)
        buf202 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf206 = reinterpret_tensor(buf195, (20, 1024, 768), (786432, 768, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, add, transpose_1, layer_norm, input_1], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_native_layer_norm_transpose_view_13.run(buf183, buf201, primals_82, primals_83, primals_84, buf202, buf206, 20480, 768, stream=stream0)
        del primals_82
        buf207 = reinterpret_tensor(buf180, (1536, 768), (768, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_85, buf207, 1179648, stream=stream0)
        buf208 = reinterpret_tensor(buf179, (20480, 1536), (1536, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1, ], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf206, (20480, 768), (768, 1), 0), reinterpret_tensor(buf207, (768, 1536), (1, 768), 0), out=buf208)
        buf209 = reinterpret_tensor(buf208, (20, 1024, 1536), (1572864, 1536, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_view_11.run(buf209, primals_86, 31457280, stream=stream0)
        buf210 = reinterpret_tensor(buf207, (768, 1536), (1536, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_87, buf210, 1179648, stream=stream0)
        buf211 = reinterpret_tensor(buf206, (20480, 768), (768, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, ], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
        extern_kernels.mm(reinterpret_tensor(buf209, (20480, 1536), (1536, 1), 0), reinterpret_tensor(buf210, (1536, 768), (1, 1536), 0), out=buf211)
        buf212 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf213 = buf212; del buf212  # reuse
        buf217 = reinterpret_tensor(buf201, (1024, 20, 768), (15360, 768, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [input_3, add, , convert_element_type, input_4, y, query, multi_head_attention_forward], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_clone_native_dropout_native_layer_norm_transpose_view_12.run(buf213, buf31, buf202, buf211, primals_88, primals_89, primals_90, buf217, 6, 20480, 768, stream=stream0)
        del primals_88
        buf218 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(primals_92, buf218, 1769472, stream=stream0)
        buf219 = reinterpret_tensor(buf190, (20480, 2304), (2304, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.t, aten.clone, aten._unsafe_view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (20480, 768), (768, 1), 0), reinterpret_tensor(buf218, (768, 2304), (1, 768), 0), out=buf219)
        buf220 = reinterpret_tensor(buf189, (3, 1024, 20, 768), (15728640, 15360, 768, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_4.run(buf219, primals_91, buf220, 47185920, stream=stream0)
        buf221 = reinterpret_tensor(buf217, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_5.run(buf220, buf221, 15728640, stream=stream0)
        buf222 = reinterpret_tensor(buf211, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_6.run(buf220, buf222, 15728640, stream=stream0)
        buf223 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_7.run(buf220, buf223, 15728640, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf224 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf221, buf222, buf223, 0.2, scale=0.10206207261596577, rng_state=fwd_rng_state_7)
        del buf221
        buf225 = buf224[0]
        assert_size_stride(buf225, (20, 8, 1024, 96), (768, 96, 15360, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf225, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf224
        buf230 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_93, buf230, 589824, stream=stream0)
        buf231 = reinterpret_tensor(buf223, (20480, 768), (768, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten._to_copy, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf225, (20480, 768), (768, 1), 0), reinterpret_tensor(buf230, (768, 768), (1, 768), 0), out=buf231)
        buf232 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf236 = reinterpret_tensor(buf225, (20, 1024, 768), (786432, 768, 1), 0); del buf225  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, add, transpose_1, layer_norm, input_1], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_native_layer_norm_transpose_view_13.run(buf213, buf231, primals_94, primals_95, primals_96, buf232, buf236, 20480, 768, stream=stream0)
        del primals_94
        buf237 = reinterpret_tensor(buf210, (1536, 768), (768, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_97, buf237, 1179648, stream=stream0)
        buf238 = reinterpret_tensor(buf209, (20480, 1536), (1536, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1, ], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf236, (20480, 768), (768, 1), 0), reinterpret_tensor(buf237, (768, 1536), (1, 768), 0), out=buf238)
        buf239 = reinterpret_tensor(buf238, (20, 1024, 1536), (1572864, 1536, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_view_11.run(buf239, primals_98, 31457280, stream=stream0)
        buf240 = reinterpret_tensor(buf237, (768, 1536), (1536, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_99, buf240, 1179648, stream=stream0)
        buf241 = reinterpret_tensor(buf236, (20480, 768), (768, 1), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, ], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
        extern_kernels.mm(reinterpret_tensor(buf239, (20480, 1536), (1536, 1), 0), reinterpret_tensor(buf240, (1536, 768), (1, 1536), 0), out=buf241)
        buf242 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf243 = buf242; del buf242  # reuse
        buf247 = reinterpret_tensor(buf231, (1024, 20, 768), (15360, 768, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [input_3, add, , convert_element_type, input_4, y, query, multi_head_attention_forward], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_clone_native_dropout_native_layer_norm_transpose_view_12.run(buf243, buf31, buf232, buf241, primals_100, primals_101, primals_102, buf247, 7, 20480, 768, stream=stream0)
        del primals_100
        buf248 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(primals_104, buf248, 1769472, stream=stream0)
        buf249 = reinterpret_tensor(buf220, (20480, 2304), (2304, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.t, aten.clone, aten._unsafe_view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf247, (20480, 768), (768, 1), 0), reinterpret_tensor(buf248, (768, 2304), (1, 768), 0), out=buf249)
        buf250 = reinterpret_tensor(buf219, (3, 1024, 20, 768), (15728640, 15360, 768, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_4.run(buf249, primals_103, buf250, 47185920, stream=stream0)
        buf251 = reinterpret_tensor(buf247, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_5.run(buf250, buf251, 15728640, stream=stream0)
        buf252 = reinterpret_tensor(buf241, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_6.run(buf250, buf252, 15728640, stream=stream0)
        buf253 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_7.run(buf250, buf253, 15728640, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf254 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf251, buf252, buf253, 0.2, scale=0.10206207261596577, rng_state=fwd_rng_state_8)
        del buf251
        buf255 = buf254[0]
        assert_size_stride(buf255, (20, 8, 1024, 96), (768, 96, 15360, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf255, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf254
        buf260 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_105, buf260, 589824, stream=stream0)
        buf261 = reinterpret_tensor(buf253, (20480, 768), (768, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten._to_copy, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf255, (20480, 768), (768, 1), 0), reinterpret_tensor(buf260, (768, 768), (1, 768), 0), out=buf261)
        buf262 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf266 = reinterpret_tensor(buf255, (20, 1024, 768), (786432, 768, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, add, transpose_1, layer_norm, input_1], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_native_layer_norm_transpose_view_13.run(buf243, buf261, primals_106, primals_107, primals_108, buf262, buf266, 20480, 768, stream=stream0)
        del primals_106
        buf267 = reinterpret_tensor(buf240, (1536, 768), (768, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_109, buf267, 1179648, stream=stream0)
        buf268 = reinterpret_tensor(buf239, (20480, 1536), (1536, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1, ], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf266, (20480, 768), (768, 1), 0), reinterpret_tensor(buf267, (768, 1536), (1, 768), 0), out=buf268)
        buf269 = reinterpret_tensor(buf268, (20, 1024, 1536), (1572864, 1536, 1), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_view_11.run(buf269, primals_110, 31457280, stream=stream0)
        buf270 = reinterpret_tensor(buf267, (768, 1536), (1536, 1), 0); del buf267  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_111, buf270, 1179648, stream=stream0)
        buf271 = reinterpret_tensor(buf266, (20480, 768), (768, 1), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, ], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
        extern_kernels.mm(reinterpret_tensor(buf269, (20480, 1536), (1536, 1), 0), reinterpret_tensor(buf270, (1536, 768), (1, 1536), 0), out=buf271)
        buf272 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf273 = buf272; del buf272  # reuse
        buf277 = reinterpret_tensor(buf261, (1024, 20, 768), (15360, 768, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [input_3, add, , convert_element_type, input_4, y, query, multi_head_attention_forward], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_clone_native_dropout_native_layer_norm_transpose_view_12.run(buf273, buf31, buf262, buf271, primals_112, primals_113, primals_114, buf277, 8, 20480, 768, stream=stream0)
        del primals_112
        buf278 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(primals_116, buf278, 1769472, stream=stream0)
        buf279 = reinterpret_tensor(buf250, (20480, 2304), (2304, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.t, aten.clone, aten._unsafe_view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf277, (20480, 768), (768, 1), 0), reinterpret_tensor(buf278, (768, 2304), (1, 768), 0), out=buf279)
        buf280 = reinterpret_tensor(buf249, (3, 1024, 20, 768), (15728640, 15360, 768, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_4.run(buf279, primals_115, buf280, 47185920, stream=stream0)
        buf281 = reinterpret_tensor(buf277, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf277  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_5.run(buf280, buf281, 15728640, stream=stream0)
        buf282 = reinterpret_tensor(buf271, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_6.run(buf280, buf282, 15728640, stream=stream0)
        buf283 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_7.run(buf280, buf283, 15728640, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf284 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf281, buf282, buf283, 0.2, scale=0.10206207261596577, rng_state=fwd_rng_state_9)
        del buf281
        buf285 = buf284[0]
        assert_size_stride(buf285, (20, 8, 1024, 96), (768, 96, 15360, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf285, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf284
        buf290 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_117, buf290, 589824, stream=stream0)
        buf291 = reinterpret_tensor(buf283, (20480, 768), (768, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten._to_copy, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf285, (20480, 768), (768, 1), 0), reinterpret_tensor(buf290, (768, 768), (1, 768), 0), out=buf291)
        buf292 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf296 = reinterpret_tensor(buf285, (20, 1024, 768), (786432, 768, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, add, transpose_1, layer_norm, input_1], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_native_layer_norm_transpose_view_13.run(buf273, buf291, primals_118, primals_119, primals_120, buf292, buf296, 20480, 768, stream=stream0)
        del primals_118
        buf297 = reinterpret_tensor(buf270, (1536, 768), (768, 1), 0); del buf270  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_121, buf297, 1179648, stream=stream0)
        buf298 = reinterpret_tensor(buf269, (20480, 1536), (1536, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1, ], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf296, (20480, 768), (768, 1), 0), reinterpret_tensor(buf297, (768, 1536), (1, 768), 0), out=buf298)
        buf299 = reinterpret_tensor(buf298, (20, 1024, 1536), (1572864, 1536, 1), 0); del buf298  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_view_11.run(buf299, primals_122, 31457280, stream=stream0)
        buf300 = reinterpret_tensor(buf297, (768, 1536), (1536, 1), 0); del buf297  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_123, buf300, 1179648, stream=stream0)
        buf301 = reinterpret_tensor(buf296, (20480, 768), (768, 1), 0); del buf296  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, ], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
        extern_kernels.mm(reinterpret_tensor(buf299, (20480, 1536), (1536, 1), 0), reinterpret_tensor(buf300, (1536, 768), (1, 1536), 0), out=buf301)
        buf302 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf303 = buf302; del buf302  # reuse
        buf307 = reinterpret_tensor(buf291, (1024, 20, 768), (15360, 768, 1), 0); del buf291  # reuse
        # Topologically Sorted Source Nodes: [input_3, add, , convert_element_type, input_4, y, query, multi_head_attention_forward], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_clone_native_dropout_native_layer_norm_transpose_view_12.run(buf303, buf31, buf292, buf301, primals_124, primals_125, primals_126, buf307, 9, 20480, 768, stream=stream0)
        del primals_124
        buf308 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(primals_128, buf308, 1769472, stream=stream0)
        buf309 = reinterpret_tensor(buf280, (20480, 2304), (2304, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.t, aten.clone, aten._unsafe_view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf307, (20480, 768), (768, 1), 0), reinterpret_tensor(buf308, (768, 2304), (1, 768), 0), out=buf309)
        buf310 = reinterpret_tensor(buf279, (3, 1024, 20, 768), (15728640, 15360, 768, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_4.run(buf309, primals_127, buf310, 47185920, stream=stream0)
        buf311 = reinterpret_tensor(buf307, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_5.run(buf310, buf311, 15728640, stream=stream0)
        buf312 = reinterpret_tensor(buf301, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf301  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_6.run(buf310, buf312, 15728640, stream=stream0)
        buf313 = buf282; del buf282  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_7.run(buf310, buf313, 15728640, stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf314 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf311, buf312, buf313, 0.2, scale=0.10206207261596577, rng_state=fwd_rng_state_10)
        del buf311
        buf315 = buf314[0]
        assert_size_stride(buf315, (20, 8, 1024, 96), (768, 96, 15360, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf315, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf314
        buf320 = buf290; del buf290  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_129, buf320, 589824, stream=stream0)
        buf321 = reinterpret_tensor(buf313, (20480, 768), (768, 1), 0); del buf313  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten._to_copy, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf315, (20480, 768), (768, 1), 0), reinterpret_tensor(buf320, (768, 768), (1, 768), 0), out=buf321)
        buf322 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf326 = reinterpret_tensor(buf315, (20, 1024, 768), (786432, 768, 1), 0); del buf315  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, add, transpose_1, layer_norm, input_1], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_native_layer_norm_transpose_view_13.run(buf303, buf321, primals_130, primals_131, primals_132, buf322, buf326, 20480, 768, stream=stream0)
        del primals_130
        buf327 = reinterpret_tensor(buf300, (1536, 768), (768, 1), 0); del buf300  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_133, buf327, 1179648, stream=stream0)
        buf328 = reinterpret_tensor(buf299, (20480, 1536), (1536, 1), 0); del buf299  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1, ], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf326, (20480, 768), (768, 1), 0), reinterpret_tensor(buf327, (768, 1536), (1, 768), 0), out=buf328)
        buf329 = reinterpret_tensor(buf328, (20, 1024, 1536), (1572864, 1536, 1), 0); del buf328  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_view_11.run(buf329, primals_134, 31457280, stream=stream0)
        buf330 = reinterpret_tensor(buf327, (768, 1536), (1536, 1), 0); del buf327  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_135, buf330, 1179648, stream=stream0)
        buf331 = reinterpret_tensor(buf326, (20480, 768), (768, 1), 0); del buf326  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, ], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
        extern_kernels.mm(reinterpret_tensor(buf329, (20480, 1536), (1536, 1), 0), reinterpret_tensor(buf330, (1536, 768), (1, 1536), 0), out=buf331)
        buf332 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf333 = buf332; del buf332  # reuse
        buf337 = reinterpret_tensor(buf321, (1024, 20, 768), (15360, 768, 1), 0); del buf321  # reuse
        # Topologically Sorted Source Nodes: [input_3, add, , convert_element_type, input_4, y, query, multi_head_attention_forward], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_clone_native_dropout_native_layer_norm_transpose_view_12.run(buf333, buf31, buf322, buf331, primals_136, primals_137, primals_138, buf337, 10, 20480, 768, stream=stream0)
        del primals_136
        buf338 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(primals_140, buf338, 1769472, stream=stream0)
        buf339 = reinterpret_tensor(buf310, (20480, 2304), (2304, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [y, query, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten._to_copy, aten.t, aten.clone, aten._unsafe_view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (20480, 768), (768, 1), 0), reinterpret_tensor(buf338, (768, 2304), (1, 768), 0), out=buf339)
        del buf338
        buf340 = reinterpret_tensor(buf309, (3, 1024, 20, 768), (15728640, 15360, 768, 1), 0); del buf309  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_squeeze_transpose_unsqueeze_view_4.run(buf339, primals_139, buf340, 47185920, stream=stream0)
        del buf339
        buf341 = reinterpret_tensor(buf337, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf337  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_5.run(buf340, buf341, 15728640, stream=stream0)
        buf342 = reinterpret_tensor(buf331, (20, 8, 1024, 96), (768, 96, 15360, 1), 0); del buf331  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_6.run(buf340, buf342, 15728640, stream=stream0)
        buf343 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_select_squeeze_transpose_unsqueeze_view_7.run(buf340, buf343, 15728640, stream=stream0)
        del buf340
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
        buf344 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, buf341, buf342, buf343, 0.2, scale=0.10206207261596577, rng_state=fwd_rng_state_11)
        del buf341
        del buf342
        buf345 = buf344[0]
        assert_size_stride(buf345, (20, 8, 1024, 96), (768, 96, 15360, 1), 'torch.ops.graphsafe_run_with_rng_state')
        assert_alignment(buf345, 16, 'torch.ops.graphsafe_run_with_rng_state')
        del buf344
        buf350 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_141, buf350, 589824, stream=stream0)
        buf351 = reinterpret_tensor(buf343, (20480, 768), (768, 1), 0); del buf343  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten._to_copy, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf345, (20480, 768), (768, 1), 0), reinterpret_tensor(buf350, (768, 768), (1, 768), 0), out=buf351)
        del buf350
        buf352 = empty_strided_cuda((20, 1024, 768), (786432, 768, 1), torch.float32)
        buf356 = reinterpret_tensor(buf345, (20, 1024, 768), (786432, 768, 1), 0); del buf345  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, add, transpose_1, layer_norm, input_1], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.transpose, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_addmm_native_layer_norm_transpose_view_13.run(buf333, buf351, primals_142, primals_143, primals_144, buf352, buf356, 20480, 768, stream=stream0)
        del buf351
        del primals_142
        buf357 = reinterpret_tensor(buf330, (1536, 768), (768, 1), 0); del buf330  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_145, buf357, 1179648, stream=stream0)
        buf358 = reinterpret_tensor(buf329, (20480, 1536), (1536, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [layer_norm, input_1, ], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.view, aten.t, aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf356, (20480, 768), (768, 1), 0), reinterpret_tensor(buf357, (768, 1536), (1, 768), 0), out=buf358)
        buf359 = reinterpret_tensor(buf358, (20, 1024, 1536), (1572864, 1536, 1), 0); del buf358  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_gelu_view_11.run(buf359, primals_146, 31457280, stream=stream0)
        buf360 = reinterpret_tensor(buf357, (768, 1536), (1536, 1), 0); del buf357  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(primals_147, buf360, 1179648, stream=stream0)
        buf361 = reinterpret_tensor(buf356, (20480, 768), (768, 1), 0); del buf356  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2, input_3, ], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.gelu, aten.t]
        extern_kernels.mm(reinterpret_tensor(buf359, (20480, 1536), (1536, 1), 0), reinterpret_tensor(buf360, (1536, 768), (1, 1536), 0), out=buf361)
        del buf359
        del buf360
        buf363 = reinterpret_tensor(buf361, (20, 1024, 768), (786432, 768, 1), 0); del buf361  # reuse
        # Topologically Sorted Source Nodes: [input_3, add, , convert_element_type, input_4, input_1], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.native_dropout, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_addmm_native_dropout_view_15.run(buf363, buf31, buf352, primals_148, 11, 15728640, stream=stream0)
        del primals_148
        buf364 = empty_strided_cuda((768, 128), (1, 768), torch.float16)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_16.run(primals_149, buf364, 98304, stream=stream0)
        del primals_149
        buf365 = empty_strided_cuda((20480, 128), (128, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf363, (20480, 768), (768, 1), 0), buf364, out=buf365)
        buf366 = reinterpret_tensor(buf365, (20, 1024, 128), (131072, 128, 1), 0); del buf365  # reuse
        # Topologically Sorted Source Nodes: [input_1, add, input_2], Original ATen: [aten._to_copy, aten.addmm, aten.view, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_relu_view_17.run(buf366, primals_150, 2621440, stream=stream0)
        del primals_150
        buf367 = empty_strided_cuda((128, 16), (1, 128), torch.float16)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_18.run(primals_151, buf367, 2048, stream=stream0)
        del primals_151
        buf368 = empty_strided_cuda((20480, 16), (16, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf366, (20480, 128), (128, 1), 0), buf367, out=buf368)
        buf369 = empty_strided_cuda((20, 1, 128, 128), (16384, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten.col2im]
        stream0 = get_raw_stream(0)
        triton_poi_fused_col2im_19.run(buf369, 327680, stream=stream0)
        buf370 = empty_strided_cuda((20, 1, 128, 128), (16384, 16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y, input_3, add, x_26, x_27], Original ATen: [aten.im2col, aten._to_copy, aten.addmm, aten.view, aten.permute, aten.col2im]
        stream0 = get_raw_stream(0)
        triton_poi_fused_col2im_19.run(buf370, 327680, stream=stream0)
        # Topologically Sorted Source Nodes: [y, input_3, add, x_26, x_27], Original ATen: [aten.im2col, aten._to_copy, aten.addmm, aten.view, aten.permute, aten.col2im]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_col2im_im2col_permute_view_20.run(buf368, primals_152, buf370, 327680, stream=stream0)
        del buf368
        del primals_152
        buf372 = empty_strided_cuda((32, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_view_21.run(buf372, 32, stream=stream0)
        buf373 = empty_strided_cuda((32, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_22.run(buf373, 32, stream=stream0)
        buf374 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_view_21.run(buf374, 32, stream=stream0)
        buf375 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_22.run(buf375, 32, stream=stream0)
        buf376 = empty_strided_cuda((32, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_23.run(buf376, 32, stream=stream0)
        buf378 = empty_strided_cuda((32, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_23.run(buf378, 32, stream=stream0)
        buf379 = empty_strided_cuda((20, 1, 32, 32), (1024, 1024, 32, 1), torch.float16)
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_mul_sub_24.run(buf372, buf374, buf370, buf375, buf376, buf373, buf378, buf379, 20480, stream=stream0)
        del buf370
    return (buf379, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_11, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_21, primals_23, primals_24, primals_25, primals_26, primals_27, primals_29, primals_30, primals_31, primals_32, primals_33, primals_35, primals_36, primals_37, primals_38, primals_39, primals_41, primals_42, primals_43, primals_44, primals_45, primals_47, primals_48, primals_49, primals_50, primals_51, primals_53, primals_54, primals_55, primals_56, primals_57, primals_59, primals_60, primals_61, primals_62, primals_63, primals_65, primals_66, primals_67, primals_68, primals_69, primals_71, primals_72, primals_73, primals_74, primals_75, primals_77, primals_78, primals_79, primals_80, primals_81, primals_83, primals_84, primals_85, primals_86, primals_87, primals_89, primals_90, primals_91, primals_92, primals_93, primals_95, primals_96, primals_97, primals_98, primals_99, primals_101, primals_102, primals_103, primals_104, primals_105, primals_107, primals_108, primals_109, primals_110, primals_111, primals_113, primals_114, primals_115, primals_116, primals_117, primals_119, primals_120, primals_121, primals_122, primals_123, primals_125, primals_126, primals_127, primals_128, primals_129, primals_131, primals_132, primals_133, primals_134, primals_135, primals_137, primals_138, primals_139, primals_140, primals_141, primals_143, primals_144, primals_145, primals_146, primals_147, reinterpret_tensor(buf0, (20480, 16), (16, 1), 0), buf2, buf21, buf31, buf33, buf52, buf63, buf82, buf93, buf112, buf123, buf142, buf153, buf172, buf183, buf202, buf213, buf232, buf243, buf262, buf273, buf292, buf303, buf322, buf333, buf352, reinterpret_tensor(buf363, (20480, 768), (768, 1), 0), reinterpret_tensor(buf366, (20480, 128), (128, 1), 0), buf369, buf372, buf373, buf374, buf375, buf376, buf378, reinterpret_tensor(buf367, (16, 128), (128, 1), 0), reinterpret_tensor(buf364, (128, 768), (768, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((20, 1, 128, 128), (16384, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((768, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((16, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    fwd_rng_state_0 = torch.cuda.default_generators[0].graphsafe_get_state()
    fwd_rng_state_1 = torch.cuda.default_generators[0].graphsafe_get_state()
    fwd_rng_state_2 = torch.cuda.default_generators[0].graphsafe_get_state()
    fwd_rng_state_3 = torch.cuda.default_generators[0].graphsafe_get_state()
    fwd_rng_state_4 = torch.cuda.default_generators[0].graphsafe_get_state()
    fwd_rng_state_5 = torch.cuda.default_generators[0].graphsafe_get_state()
    fwd_rng_state_6 = torch.cuda.default_generators[0].graphsafe_get_state()
    fwd_rng_state_7 = torch.cuda.default_generators[0].graphsafe_get_state()
    fwd_rng_state_8 = torch.cuda.default_generators[0].graphsafe_get_state()
    fwd_rng_state_9 = torch.cuda.default_generators[0].graphsafe_get_state()
    fwd_rng_state_10 = torch.cuda.default_generators[0].graphsafe_get_state()
    fwd_rng_state_11 = torch.cuda.default_generators[0].graphsafe_get_state()
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, fwd_rng_state_0, fwd_rng_state_1, fwd_rng_state_2, fwd_rng_state_3, fwd_rng_state_4, fwd_rng_state_5, fwd_rng_state_6, fwd_rng_state_7, fwd_rng_state_8, fwd_rng_state_9, fwd_rng_state_10, fwd_rng_state_11])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
