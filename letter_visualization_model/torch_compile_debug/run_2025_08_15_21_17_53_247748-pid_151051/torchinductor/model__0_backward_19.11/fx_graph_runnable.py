
import os
os.environ['TORCHDYNAMO_VERBOSE'] = '1'
os.environ['TORCH_TRACE'] = './logs.txt'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_LOGS'] = 'recompiles,+dynamo'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_Adithya'
os.environ['TRITON_CACHE_DIR'] = '/tmp/torchinductor_Adithya/triton/0'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims


import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.triton.store_cubin = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.test_configs.runtime_triton_dtype_assert = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = False



isolate_fails_code_str = None




# torch version: 2.9.0.dev20250715+cu129
# torch cuda version: 12.9
# torch git version: d38be5ebdb9b292398581c48d4e666fc105d92d9


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA GeForce RTX 3080 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_11, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_21, primals_23, primals_24, primals_25, primals_26, primals_27, primals_29, primals_30, primals_31, primals_32, primals_33, primals_35, primals_36, primals_37, primals_38, primals_39, primals_41, primals_42, primals_43, primals_44, primals_45, primals_47, primals_48, primals_49, primals_50, primals_51, primals_53, primals_54, primals_55, primals_56, primals_57, primals_59, primals_60, primals_61, primals_62, primals_63, primals_65, primals_66, primals_67, primals_68, primals_69, primals_71, primals_72, primals_73, primals_74, primals_75, primals_77, primals_78, primals_79, primals_80, primals_81, primals_83, primals_84, primals_85, primals_86, primals_87, primals_89, primals_90, primals_91, primals_92, primals_93, primals_95, primals_96, primals_97, primals_98, primals_99, primals_101, primals_102, primals_103, primals_104, primals_105, primals_107, primals_108, primals_109, primals_110, primals_111, primals_113, primals_114, primals_115, primals_116, primals_117, primals_119, primals_120, primals_121, primals_122, primals_123, primals_125, primals_126, primals_127, primals_128, primals_129, primals_131, primals_132, primals_133, primals_134, primals_135, primals_137, primals_138, primals_139, primals_140, primals_141, primals_143, primals_144, primals_145, primals_146, primals_147, view_1, mm, add_7, inductor_seeds_default, add_11, add_15, add_19, add_23, add_27, add_31, add_35, add_39, add_43, add_47, add_51, add_55, add_59, add_63, add_67, add_71, add_75, add_79, add_83, add_87, add_91, add_95, view_183, view_185, full_default, convert_element_type_296, clamp_max, convert_element_type_298, clamp_max_1, clamp_max_2, clamp_max_3, permute_141, permute_145, tangents_1, bwd_rng_state_0, bwd_rng_state_1, bwd_rng_state_2, bwd_rng_state_3, bwd_rng_state_4, bwd_rng_state_5, bwd_rng_state_6, bwd_rng_state_7, bwd_rng_state_8, bwd_rng_state_9, bwd_rng_state_10, bwd_rng_state_11):
        convert_element_type_300 = torch.ops.prims.convert_element_type.default(tangents_1, torch.float32);  tangents_1 = None
        mul_113 = torch.ops.aten.mul.Tensor(convert_element_type_300, clamp_max_3);  clamp_max_3 = None
        neg = torch.ops.aten.neg.default(mul_113)
        add_109 = torch.ops.aten.add.Tensor(convert_element_type_300, neg);  convert_element_type_300 = neg = None
        mul_114 = torch.ops.aten.mul.Tensor(mul_113, clamp_max_2)
        neg_1 = torch.ops.aten.neg.default(mul_114)
        add_110 = torch.ops.aten.add.Tensor(mul_113, neg_1);  mul_113 = neg_1 = None
        mul_115 = torch.ops.aten.mul.Tensor(add_109, clamp_max_2);  clamp_max_2 = None
        neg_2 = torch.ops.aten.neg.default(mul_115)
        add_111 = torch.ops.aten.add.Tensor(add_109, neg_2);  add_109 = neg_2 = None
        index_put_1 = torch.ops.aten.index_put.default(full_default, [None, None, clamp_max, clamp_max_1], mul_114, True);  mul_114 = None
        index_put_2 = torch.ops.aten.index_put.default(full_default, [None, None, clamp_max, convert_element_type_298], add_110, True);  clamp_max = add_110 = None
        add_112 = torch.ops.aten.add.Tensor(index_put_1, index_put_2);  index_put_1 = index_put_2 = None
        index_put_3 = torch.ops.aten.index_put.default(full_default, [None, None, convert_element_type_296, clamp_max_1], mul_115, True);  clamp_max_1 = mul_115 = None
        add_113 = torch.ops.aten.add.Tensor(add_112, index_put_3);  add_112 = index_put_3 = None
        index_put_4 = torch.ops.aten.index_put.default(full_default, [None, None, convert_element_type_296, convert_element_type_298], add_111, True);  full_default = convert_element_type_296 = convert_element_type_298 = add_111 = None
        add_114 = torch.ops.aten.add.Tensor(add_113, index_put_4);  add_113 = index_put_4 = None
        convert_element_type_301 = torch.ops.prims.convert_element_type.default(add_114, torch.float16);  add_114 = None
        iota = torch.ops.prims.iota.default(32, start = 0, step = 4, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        iota_1 = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        add = torch.ops.aten.add.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(add, -1)
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        index_1 = torch.ops.aten.index.Tensor(convert_element_type_301, [None, None, unsqueeze_5, add]);  convert_element_type_301 = unsqueeze_5 = add = None
        permute_139 = torch.ops.aten.permute.default(index_1, [0, 1, 2, 4, 3, 5]);  index_1 = None
        clone_26 = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
        view_195 = torch.ops.aten.view.default(clone_26, [20, 16, 1024]);  clone_26 = None
        permute_140 = torch.ops.aten.permute.default(view_195, [0, 2, 1]);  view_195 = None
        clone_27 = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
        view_196 = torch.ops.aten.view.default(clone_27, [20480, 16]);  clone_27 = None
        mm_13 = torch.ops.aten.mm.default(view_196, permute_141);  permute_141 = None
        permute_142 = torch.ops.aten.permute.default(view_196, [1, 0])
        mm_14 = torch.ops.aten.mm.default(permute_142, view_185);  permute_142 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(view_196, [0], True, dtype = torch.float32);  view_196 = None
        view_197 = torch.ops.aten.view.default(sum_1, [16]);  sum_1 = None
        convert_element_type_307 = torch.ops.prims.convert_element_type.default(mm_14, torch.float32);  mm_14 = None
        convert_element_type_default_50 = torch.ops.prims.convert_element_type.default(view_197, torch.float32);  view_197 = None
        view_201 = torch.ops.aten.view.default(mm_13, [20, 1024, 128]);  mm_13 = None
        view_202 = torch.ops.aten.view.default(view_185, [20, 1024, 128]);  view_185 = None
        le = torch.ops.aten.le.Scalar(view_202, 0);  view_202 = None
        full_default_5 = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(le, full_default_5, view_201);  le = full_default_5 = view_201 = None
        view_203 = torch.ops.aten.view.default(where, [20480, 128]);  where = None
        mm_15 = torch.ops.aten.mm.default(view_203, permute_145);  permute_145 = None
        permute_147 = torch.ops.aten.permute.default(view_203, [1, 0])
        mm_16 = torch.ops.aten.mm.default(permute_147, view_183);  permute_147 = view_183 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(view_203, [0], True, dtype = torch.float32);  view_203 = None
        view_205 = torch.ops.aten.view.default(sum_2, [128]);  sum_2 = None
        view_206 = torch.ops.aten.view.default(mm_15, [20, 1024, 768]);  mm_15 = None
        convert_element_type_314 = torch.ops.prims.convert_element_type.default(view_206, torch.float32);  view_206 = None
        convert_element_type_315 = torch.ops.prims.convert_element_type.default(mm_16, torch.float32);  mm_16 = None
        convert_element_type_default_49 = torch.ops.prims.convert_element_type.default(view_205, torch.float32);  view_205 = None
        convert_element_type_317 = torch.ops.prims.convert_element_type.default(convert_element_type_314, torch.float16)
        inductor_lookup_seed_default_11 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        convert_element_type_default_52 = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        gt_11 = torch.ops.aten.gt.Scalar(convert_element_type_default_52, 0.2);  convert_element_type_default_52 = None
        convert_element_type_318 = torch.ops.prims.convert_element_type.default(gt_11, torch.float16);  gt_11 = None
        mul_116 = torch.ops.aten.mul.Tensor(convert_element_type_318, 1.25);  convert_element_type_318 = None
        mul_117 = torch.ops.aten.mul.Tensor(convert_element_type_317, mul_116);  convert_element_type_317 = mul_116 = None
        view_207 = torch.ops.aten.view.default(mul_117, [20480, 768]);  mul_117 = None
        convert_element_type_277 = torch.ops.prims.convert_element_type.default(primals_147, torch.float16);  primals_147 = None
        permute_134 = torch.ops.aten.permute.default(convert_element_type_277, [1, 0]);  convert_element_type_277 = None
        permute_150 = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
        mm_17 = torch.ops.aten.mm.default(view_207, permute_150);  permute_150 = None
        permute_151 = torch.ops.aten.permute.default(view_207, [1, 0])
        var_mean_23 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
        getitem_154 = var_mean_23[0]
        getitem_155 = var_mean_23[1];  var_mean_23 = None
        add_96 = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_95, getitem_155);  add_95 = getitem_155 = None
        mul_101 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        mul_102 = torch.ops.aten.mul.Tensor(mul_101, primals_143)
        add_97 = torch.ops.aten.add.Tensor(mul_102, primals_144);  mul_102 = primals_144 = None
        convert_element_type_268 = torch.ops.prims.convert_element_type.default(primals_146, torch.float16);  primals_146 = None
        convert_element_type_269 = torch.ops.prims.convert_element_type.default(primals_145, torch.float16);  primals_145 = None
        convert_element_type_270 = torch.ops.prims.convert_element_type.default(add_97, torch.float16);  add_97 = None
        view_179 = torch.ops.aten.view.default(convert_element_type_270, [20480, 768]);  convert_element_type_270 = None
        permute_133 = torch.ops.aten.permute.default(convert_element_type_269, [1, 0]);  convert_element_type_269 = None
        addmm_34 = torch.ops.aten.addmm.default(convert_element_type_268, view_179, permute_133);  convert_element_type_268 = None
        view_180 = torch.ops.aten.view.default(addmm_34, [20, 1024, 1536]);  addmm_34 = None
        convert_element_type_274 = torch.ops.prims.convert_element_type.default(view_180, torch.float32);  view_180 = None
        mul_103 = torch.ops.aten.mul.Tensor(convert_element_type_274, 0.5)
        mul_104 = torch.ops.aten.mul.Tensor(convert_element_type_274, 0.7071067811865476)
        erf_11 = torch.ops.aten.erf.default(mul_104);  mul_104 = None
        add_98 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_105 = torch.ops.aten.mul.Tensor(mul_103, add_98);  mul_103 = None
        convert_element_type_275 = torch.ops.prims.convert_element_type.default(mul_105, torch.float16);  mul_105 = None
        view_181 = torch.ops.aten.view.default(convert_element_type_275, [20480, 1536]);  convert_element_type_275 = None
        mm_18 = torch.ops.aten.mm.default(permute_151, view_181);  permute_151 = view_181 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(view_207, [0], True, dtype = torch.float32);  view_207 = None
        view_208 = torch.ops.aten.view.default(sum_3, [768]);  sum_3 = None
        view_209 = torch.ops.aten.view.default(mm_17, [20, 1024, 1536]);  mm_17 = None
        convert_element_type_324 = torch.ops.prims.convert_element_type.default(mm_18, torch.float32);  mm_18 = None
        convert_element_type_default_48 = torch.ops.prims.convert_element_type.default(view_208, torch.float32);  view_208 = None
        convert_element_type_326 = torch.ops.prims.convert_element_type.default(view_209, torch.float32);  view_209 = None
        mul_119 = torch.ops.aten.mul.Tensor(add_98, 0.5);  add_98 = None
        mul_120 = torch.ops.aten.mul.Tensor(convert_element_type_274, convert_element_type_274)
        mul_121 = torch.ops.aten.mul.Tensor(mul_120, -0.5);  mul_120 = None
        exp = torch.ops.aten.exp.default(mul_121);  mul_121 = None
        mul_122 = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_123 = torch.ops.aten.mul.Tensor(convert_element_type_274, mul_122);  convert_element_type_274 = mul_122 = None
        add_118 = torch.ops.aten.add.Tensor(mul_119, mul_123);  mul_119 = mul_123 = None
        mul_124 = torch.ops.aten.mul.Tensor(convert_element_type_326, add_118);  convert_element_type_326 = add_118 = None
        convert_element_type_328 = torch.ops.prims.convert_element_type.default(mul_124, torch.float16);  mul_124 = None
        view_210 = torch.ops.aten.view.default(convert_element_type_328, [20480, 1536]);  convert_element_type_328 = None
        permute_154 = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        mm_19 = torch.ops.aten.mm.default(view_210, permute_154);  permute_154 = None
        permute_155 = torch.ops.aten.permute.default(view_210, [1, 0])
        mm_20 = torch.ops.aten.mm.default(permute_155, view_179);  permute_155 = view_179 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(view_210, [0], True, dtype = torch.float32);  view_210 = None
        view_211 = torch.ops.aten.view.default(sum_4, [1536]);  sum_4 = None
        view_212 = torch.ops.aten.view.default(mm_19, [20, 1024, 768]);  mm_19 = None
        convert_element_type_334 = torch.ops.prims.convert_element_type.default(view_212, torch.float32);  view_212 = None
        convert_element_type_335 = torch.ops.prims.convert_element_type.default(mm_20, torch.float32);  mm_20 = None
        convert_element_type_default_47 = torch.ops.prims.convert_element_type.default(view_211, torch.float32);  view_211 = None
        mul_126 = torch.ops.aten.mul.Tensor(convert_element_type_334, primals_143);  primals_143 = None
        mul_127 = torch.ops.aten.mul.Tensor(mul_126, 768)
        sum_5 = torch.ops.aten.sum.dim_IntList(mul_126, [2], True)
        mul_128 = torch.ops.aten.mul.Tensor(mul_126, mul_101);  mul_126 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(mul_128, [2], True);  mul_128 = None
        mul_129 = torch.ops.aten.mul.Tensor(mul_101, sum_6);  sum_6 = None
        sub_32 = torch.ops.aten.sub.Tensor(mul_127, sum_5);  mul_127 = sum_5 = None
        sub_33 = torch.ops.aten.sub.Tensor(sub_32, mul_129);  sub_32 = mul_129 = None
        div = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
        mul_130 = torch.ops.aten.mul.Tensor(div, sub_33);  div = sub_33 = None
        mul_131 = torch.ops.aten.mul.Tensor(convert_element_type_334, mul_101);  mul_101 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_131, [0, 1]);  mul_131 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(convert_element_type_334, [0, 1]);  convert_element_type_334 = None
        add_119 = torch.ops.aten.add.Tensor(convert_element_type_314, mul_130);  convert_element_type_314 = mul_130 = None
        convert_element_type_337 = torch.ops.prims.convert_element_type.default(add_119, torch.float16)
        permute_158 = torch.ops.aten.permute.default(convert_element_type_337, [1, 0, 2]);  convert_element_type_337 = None
        clone_30 = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
        view_213 = torch.ops.aten.view.default(clone_30, [20480, 768]);  clone_30 = None
        convert_element_type_264 = torch.ops.prims.convert_element_type.default(primals_141, torch.float16);  primals_141 = None
        permute_131 = torch.ops.aten.permute.default(convert_element_type_264, [1, 0]);  convert_element_type_264 = None
        permute_159 = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        mm_21 = torch.ops.aten.mm.default(view_213, permute_159);  permute_159 = None
        permute_160 = torch.ops.aten.permute.default(view_213, [1, 0])
        var_mean_22 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_143 = var_mean_22[0]
        getitem_144 = var_mean_22[1];  var_mean_22 = None
        add_92 = torch.ops.aten.add.Tensor(getitem_143, 1e-05);  getitem_143 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_91, getitem_144);  add_91 = getitem_144 = None
        mul_99 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        mul_100 = torch.ops.aten.mul.Tensor(mul_99, primals_137)
        add_93 = torch.ops.aten.add.Tensor(mul_100, primals_138);  mul_100 = primals_138 = None
        permute_124 = torch.ops.aten.permute.default(add_93, [1, 0, 2]);  add_93 = None
        convert_element_type_258 = torch.ops.prims.convert_element_type.default(primals_139, torch.float16);  primals_139 = None
        convert_element_type_259 = torch.ops.prims.convert_element_type.default(primals_140, torch.float16);  primals_140 = None
        convert_element_type_260 = torch.ops.prims.convert_element_type.default(permute_124, torch.float16);  permute_124 = None
        permute_125 = torch.ops.aten.permute.default(convert_element_type_259, [1, 0]);  convert_element_type_259 = None
        clone_24 = torch.ops.aten.clone.default(convert_element_type_260, memory_format = torch.contiguous_format);  convert_element_type_260 = None
        view_168 = torch.ops.aten.view.default(clone_24, [20480, 768]);  clone_24 = None
        mm_12 = torch.ops.aten.mm.default(view_168, permute_125)
        view_169 = torch.ops.aten.view.default(mm_12, [1024, 20, 2304]);  mm_12 = None
        add_94 = torch.ops.aten.add.Tensor(view_169, convert_element_type_258);  view_169 = convert_element_type_258 = None
        view_170 = torch.ops.aten.view.default(add_94, [1024, 20, 3, 768]);  add_94 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(view_170, 0);  view_170 = None
        permute_126 = torch.ops.aten.permute.default(unsqueeze_17, [3, 1, 2, 0, 4]);  unsqueeze_17 = None
        squeeze_11 = torch.ops.aten.squeeze.dim(permute_126, -2);  permute_126 = None
        clone_25 = torch.ops.aten.clone.default(squeeze_11, memory_format = torch.contiguous_format);  squeeze_11 = None
        select_33 = torch.ops.aten.select.int(clone_25, 0, 0)
        select_34 = torch.ops.aten.select.int(clone_25, 0, 1)
        select_35 = torch.ops.aten.select.int(clone_25, 0, 2);  clone_25 = None
        view_171 = torch.ops.aten.view.default(select_33, [1024, 160, 96]);  select_33 = None
        permute_127 = torch.ops.aten.permute.default(view_171, [1, 0, 2]);  view_171 = None
        view_172 = torch.ops.aten.view.default(select_34, [1024, 160, 96]);  select_34 = None
        permute_128 = torch.ops.aten.permute.default(view_172, [1, 0, 2]);  view_172 = None
        view_173 = torch.ops.aten.view.default(select_35, [1024, 160, 96]);  select_35 = None
        permute_129 = torch.ops.aten.permute.default(view_173, [1, 0, 2]);  view_173 = None
        view_174 = torch.ops.aten.view.default(permute_127, [20, 8, 1024, 96]);  permute_127 = None
        view_175 = torch.ops.aten.view.default(permute_128, [20, 8, 1024, 96]);  permute_128 = None
        view_176 = torch.ops.aten.view.default(permute_129, [20, 8, 1024, 96]);  permute_129 = None
        graphsafe_run_with_rng_state_11 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_174, view_175, view_176, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_11);  bwd_rng_state_11 = None
        getitem_145 = graphsafe_run_with_rng_state_11[0]
        getitem_146 = graphsafe_run_with_rng_state_11[1]
        getitem_151 = graphsafe_run_with_rng_state_11[6]
        getitem_152 = graphsafe_run_with_rng_state_11[7];  graphsafe_run_with_rng_state_11 = None
        permute_130 = torch.ops.aten.permute.default(getitem_145, [2, 0, 1, 3])
        view_177 = torch.ops.aten.view.default(permute_130, [20480, 768]);  permute_130 = None
        mm_22 = torch.ops.aten.mm.default(permute_160, view_177);  permute_160 = view_177 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(view_213, [0], True, dtype = torch.float32);  view_213 = None
        view_214 = torch.ops.aten.view.default(sum_9, [768]);  sum_9 = None
        convert_element_type_343 = torch.ops.prims.convert_element_type.default(mm_22, torch.float32);  mm_22 = None
        convert_element_type_default_46 = torch.ops.prims.convert_element_type.default(view_214, torch.float32);  view_214 = None
        view_215 = torch.ops.aten.view.default(mm_21, [1024, 20, 8, 96]);  mm_21 = None
        permute_163 = torch.ops.aten.permute.default(view_215, [1, 2, 0, 3]);  view_215 = None
        _scaled_dot_product_flash_attention_backward = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_163, view_174, view_175, view_176, getitem_145, getitem_146, None, None, 1024, 1024, 0.2, False, getitem_151, getitem_152, scale = 0.10206207261596577);  permute_163 = view_174 = view_175 = view_176 = getitem_145 = getitem_146 = getitem_151 = getitem_152 = None
        getitem_156 = _scaled_dot_product_flash_attention_backward[0]
        getitem_157 = _scaled_dot_product_flash_attention_backward[1]
        getitem_158 = _scaled_dot_product_flash_attention_backward[2];  _scaled_dot_product_flash_attention_backward = None
        view_216 = torch.ops.aten.view.default(getitem_158, [160, 1024, 96]);  getitem_158 = None
        view_217 = torch.ops.aten.view.default(getitem_157, [160, 1024, 96]);  getitem_157 = None
        view_218 = torch.ops.aten.view.default(getitem_156, [160, 1024, 96]);  getitem_156 = None
        permute_164 = torch.ops.aten.permute.default(view_216, [1, 0, 2]);  view_216 = None
        view_219 = torch.ops.aten.view.default(permute_164, [1024, 20, 768]);  permute_164 = None
        permute_165 = torch.ops.aten.permute.default(view_217, [1, 0, 2]);  view_217 = None
        view_220 = torch.ops.aten.view.default(permute_165, [1024, 20, 768]);  permute_165 = None
        permute_166 = torch.ops.aten.permute.default(view_218, [1, 0, 2]);  view_218 = None
        view_221 = torch.ops.aten.view.default(permute_166, [1024, 20, 768]);  permute_166 = None
        full_default_6 = torch.ops.aten.full.default([3, 1024, 20, 768], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter = torch.ops.aten.select_scatter.default(full_default_6, view_219, 0, 2);  view_219 = None
        select_scatter_1 = torch.ops.aten.select_scatter.default(full_default_6, view_220, 0, 1);  view_220 = None
        add_120 = torch.ops.aten.add.Tensor(select_scatter, select_scatter_1);  select_scatter = select_scatter_1 = None
        select_scatter_2 = torch.ops.aten.select_scatter.default(full_default_6, view_221, 0, 0);  view_221 = None
        add_121 = torch.ops.aten.add.Tensor(add_120, select_scatter_2);  add_120 = select_scatter_2 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(add_121, 3);  add_121 = None
        permute_167 = torch.ops.aten.permute.default(unsqueeze_30, [3, 1, 2, 0, 4]);  unsqueeze_30 = None
        squeeze_12 = torch.ops.aten.squeeze.dim(permute_167, 0);  permute_167 = None
        clone_31 = torch.ops.aten.clone.default(squeeze_12, memory_format = torch.contiguous_format);  squeeze_12 = None
        view_222 = torch.ops.aten.view.default(clone_31, [1024, 20, 2304]);  clone_31 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(view_222, [0, 1], True, dtype = torch.float32)
        view_223 = torch.ops.aten.view.default(sum_10, [2304]);  sum_10 = None
        view_224 = torch.ops.aten.view.default(view_222, [20480, 2304]);  view_222 = None
        permute_168 = torch.ops.aten.permute.default(view_224, [1, 0])
        mm_23 = torch.ops.aten.mm.default(permute_168, view_168);  permute_168 = view_168 = None
        permute_170 = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
        mm_24 = torch.ops.aten.mm.default(view_224, permute_170);  view_224 = permute_170 = None
        view_225 = torch.ops.aten.view.default(mm_24, [1024, 20, 768]);  mm_24 = None
        convert_element_type_350 = torch.ops.prims.convert_element_type.default(view_225, torch.float32);  view_225 = None
        convert_element_type_351 = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_default_45 = torch.ops.prims.convert_element_type.default(view_223, torch.float32);  view_223 = None
        permute_172 = torch.ops.aten.permute.default(convert_element_type_350, [1, 0, 2]);  convert_element_type_350 = None
        mul_133 = torch.ops.aten.mul.Tensor(permute_172, primals_137);  primals_137 = None
        mul_134 = torch.ops.aten.mul.Tensor(mul_133, 768)
        sum_11 = torch.ops.aten.sum.dim_IntList(mul_133, [2], True)
        mul_135 = torch.ops.aten.mul.Tensor(mul_133, mul_99);  mul_133 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(mul_135, [2], True);  mul_135 = None
        mul_136 = torch.ops.aten.mul.Tensor(mul_99, sum_12);  sum_12 = None
        sub_35 = torch.ops.aten.sub.Tensor(mul_134, sum_11);  mul_134 = sum_11 = None
        sub_36 = torch.ops.aten.sub.Tensor(sub_35, mul_136);  sub_35 = mul_136 = None
        div_1 = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
        mul_137 = torch.ops.aten.mul.Tensor(div_1, sub_36);  div_1 = sub_36 = None
        mul_138 = torch.ops.aten.mul.Tensor(permute_172, mul_99);  mul_99 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(mul_138, [0, 1]);  mul_138 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(permute_172, [0, 1]);  permute_172 = None
        add_122 = torch.ops.aten.add.Tensor(add_119, mul_137);  add_119 = mul_137 = None
        convert_element_type_353 = torch.ops.prims.convert_element_type.default(add_122, torch.float16)
        inductor_lookup_seed_default_10 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_1 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        convert_element_type_default_53 = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.float16);  inductor_random_default_1 = None
        gt_10 = torch.ops.aten.gt.Scalar(convert_element_type_default_53, 0.2);  convert_element_type_default_53 = None
        convert_element_type_354 = torch.ops.prims.convert_element_type.default(gt_10, torch.float16);  gt_10 = None
        mul_139 = torch.ops.aten.mul.Tensor(convert_element_type_354, 1.25);  convert_element_type_354 = None
        mul_140 = torch.ops.aten.mul.Tensor(convert_element_type_353, mul_139);  convert_element_type_353 = mul_139 = None
        view_226 = torch.ops.aten.view.default(mul_140, [20480, 768]);  mul_140 = None
        convert_element_type_254 = torch.ops.prims.convert_element_type.default(primals_135, torch.float16);  primals_135 = None
        permute_123 = torch.ops.aten.permute.default(convert_element_type_254, [1, 0]);  convert_element_type_254 = None
        permute_173 = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        mm_25 = torch.ops.aten.mm.default(view_226, permute_173);  permute_173 = None
        permute_174 = torch.ops.aten.permute.default(view_226, [1, 0])
        var_mean_21 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
        getitem_141 = var_mean_21[0]
        getitem_142 = var_mean_21[1];  var_mean_21 = None
        add_88 = torch.ops.aten.add.Tensor(getitem_141, 1e-05);  getitem_141 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_87, getitem_142);  add_87 = getitem_142 = None
        mul_92 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_92, primals_131)
        add_89 = torch.ops.aten.add.Tensor(mul_93, primals_132);  mul_93 = primals_132 = None
        convert_element_type_245 = torch.ops.prims.convert_element_type.default(primals_134, torch.float16);  primals_134 = None
        convert_element_type_246 = torch.ops.prims.convert_element_type.default(primals_133, torch.float16);  primals_133 = None
        convert_element_type_247 = torch.ops.prims.convert_element_type.default(add_89, torch.float16);  add_89 = None
        view_164 = torch.ops.aten.view.default(convert_element_type_247, [20480, 768]);  convert_element_type_247 = None
        permute_122 = torch.ops.aten.permute.default(convert_element_type_246, [1, 0]);  convert_element_type_246 = None
        addmm_31 = torch.ops.aten.addmm.default(convert_element_type_245, view_164, permute_122);  convert_element_type_245 = None
        view_165 = torch.ops.aten.view.default(addmm_31, [20, 1024, 1536]);  addmm_31 = None
        convert_element_type_251 = torch.ops.prims.convert_element_type.default(view_165, torch.float32);  view_165 = None
        mul_94 = torch.ops.aten.mul.Tensor(convert_element_type_251, 0.5)
        mul_95 = torch.ops.aten.mul.Tensor(convert_element_type_251, 0.7071067811865476)
        erf_10 = torch.ops.aten.erf.default(mul_95);  mul_95 = None
        add_90 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_96 = torch.ops.aten.mul.Tensor(mul_94, add_90);  mul_94 = None
        convert_element_type_252 = torch.ops.prims.convert_element_type.default(mul_96, torch.float16);  mul_96 = None
        view_166 = torch.ops.aten.view.default(convert_element_type_252, [20480, 1536]);  convert_element_type_252 = None
        mm_26 = torch.ops.aten.mm.default(permute_174, view_166);  permute_174 = view_166 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(view_226, [0], True, dtype = torch.float32);  view_226 = None
        view_227 = torch.ops.aten.view.default(sum_15, [768]);  sum_15 = None
        view_228 = torch.ops.aten.view.default(mm_25, [20, 1024, 1536]);  mm_25 = None
        convert_element_type_360 = torch.ops.prims.convert_element_type.default(mm_26, torch.float32);  mm_26 = None
        convert_element_type_default_44 = torch.ops.prims.convert_element_type.default(view_227, torch.float32);  view_227 = None
        convert_element_type_362 = torch.ops.prims.convert_element_type.default(view_228, torch.float32);  view_228 = None
        mul_142 = torch.ops.aten.mul.Tensor(add_90, 0.5);  add_90 = None
        mul_143 = torch.ops.aten.mul.Tensor(convert_element_type_251, convert_element_type_251)
        mul_144 = torch.ops.aten.mul.Tensor(mul_143, -0.5);  mul_143 = None
        exp_1 = torch.ops.aten.exp.default(mul_144);  mul_144 = None
        mul_145 = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_146 = torch.ops.aten.mul.Tensor(convert_element_type_251, mul_145);  convert_element_type_251 = mul_145 = None
        add_124 = torch.ops.aten.add.Tensor(mul_142, mul_146);  mul_142 = mul_146 = None
        mul_147 = torch.ops.aten.mul.Tensor(convert_element_type_362, add_124);  convert_element_type_362 = add_124 = None
        convert_element_type_364 = torch.ops.prims.convert_element_type.default(mul_147, torch.float16);  mul_147 = None
        view_229 = torch.ops.aten.view.default(convert_element_type_364, [20480, 1536]);  convert_element_type_364 = None
        permute_177 = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
        mm_27 = torch.ops.aten.mm.default(view_229, permute_177);  permute_177 = None
        permute_178 = torch.ops.aten.permute.default(view_229, [1, 0])
        mm_28 = torch.ops.aten.mm.default(permute_178, view_164);  permute_178 = view_164 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(view_229, [0], True, dtype = torch.float32);  view_229 = None
        view_230 = torch.ops.aten.view.default(sum_16, [1536]);  sum_16 = None
        view_231 = torch.ops.aten.view.default(mm_27, [20, 1024, 768]);  mm_27 = None
        convert_element_type_370 = torch.ops.prims.convert_element_type.default(view_231, torch.float32);  view_231 = None
        convert_element_type_371 = torch.ops.prims.convert_element_type.default(mm_28, torch.float32);  mm_28 = None
        convert_element_type_default_43 = torch.ops.prims.convert_element_type.default(view_230, torch.float32);  view_230 = None
        mul_149 = torch.ops.aten.mul.Tensor(convert_element_type_370, primals_131);  primals_131 = None
        mul_150 = torch.ops.aten.mul.Tensor(mul_149, 768)
        sum_17 = torch.ops.aten.sum.dim_IntList(mul_149, [2], True)
        mul_151 = torch.ops.aten.mul.Tensor(mul_149, mul_92);  mul_149 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(mul_151, [2], True);  mul_151 = None
        mul_152 = torch.ops.aten.mul.Tensor(mul_92, sum_18);  sum_18 = None
        sub_38 = torch.ops.aten.sub.Tensor(mul_150, sum_17);  mul_150 = sum_17 = None
        sub_39 = torch.ops.aten.sub.Tensor(sub_38, mul_152);  sub_38 = mul_152 = None
        div_2 = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
        mul_153 = torch.ops.aten.mul.Tensor(div_2, sub_39);  div_2 = sub_39 = None
        mul_154 = torch.ops.aten.mul.Tensor(convert_element_type_370, mul_92);  mul_92 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(mul_154, [0, 1]);  mul_154 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(convert_element_type_370, [0, 1]);  convert_element_type_370 = None
        add_125 = torch.ops.aten.add.Tensor(add_122, mul_153);  add_122 = mul_153 = None
        convert_element_type_373 = torch.ops.prims.convert_element_type.default(add_125, torch.float16)
        permute_181 = torch.ops.aten.permute.default(convert_element_type_373, [1, 0, 2]);  convert_element_type_373 = None
        clone_33 = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
        view_232 = torch.ops.aten.view.default(clone_33, [20480, 768]);  clone_33 = None
        convert_element_type_241 = torch.ops.prims.convert_element_type.default(primals_129, torch.float16);  primals_129 = None
        permute_120 = torch.ops.aten.permute.default(convert_element_type_241, [1, 0]);  convert_element_type_241 = None
        permute_182 = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        mm_29 = torch.ops.aten.mm.default(view_232, permute_182);  permute_182 = None
        permute_183 = torch.ops.aten.permute.default(view_232, [1, 0])
        var_mean_20 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_130 = var_mean_20[0]
        getitem_131 = var_mean_20[1];  var_mean_20 = None
        add_84 = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_83, getitem_131);  add_83 = getitem_131 = None
        mul_90 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        mul_91 = torch.ops.aten.mul.Tensor(mul_90, primals_125)
        add_85 = torch.ops.aten.add.Tensor(mul_91, primals_126);  mul_91 = primals_126 = None
        permute_113 = torch.ops.aten.permute.default(add_85, [1, 0, 2]);  add_85 = None
        convert_element_type_235 = torch.ops.prims.convert_element_type.default(primals_127, torch.float16);  primals_127 = None
        convert_element_type_236 = torch.ops.prims.convert_element_type.default(primals_128, torch.float16);  primals_128 = None
        convert_element_type_237 = torch.ops.prims.convert_element_type.default(permute_113, torch.float16);  permute_113 = None
        permute_114 = torch.ops.aten.permute.default(convert_element_type_236, [1, 0]);  convert_element_type_236 = None
        clone_22 = torch.ops.aten.clone.default(convert_element_type_237, memory_format = torch.contiguous_format);  convert_element_type_237 = None
        view_153 = torch.ops.aten.view.default(clone_22, [20480, 768]);  clone_22 = None
        mm_11 = torch.ops.aten.mm.default(view_153, permute_114)
        view_154 = torch.ops.aten.view.default(mm_11, [1024, 20, 2304]);  mm_11 = None
        add_86 = torch.ops.aten.add.Tensor(view_154, convert_element_type_235);  view_154 = convert_element_type_235 = None
        view_155 = torch.ops.aten.view.default(add_86, [1024, 20, 3, 768]);  add_86 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(view_155, 0);  view_155 = None
        permute_115 = torch.ops.aten.permute.default(unsqueeze_16, [3, 1, 2, 0, 4]);  unsqueeze_16 = None
        squeeze_10 = torch.ops.aten.squeeze.dim(permute_115, -2);  permute_115 = None
        clone_23 = torch.ops.aten.clone.default(squeeze_10, memory_format = torch.contiguous_format);  squeeze_10 = None
        select_30 = torch.ops.aten.select.int(clone_23, 0, 0)
        select_31 = torch.ops.aten.select.int(clone_23, 0, 1)
        select_32 = torch.ops.aten.select.int(clone_23, 0, 2);  clone_23 = None
        view_156 = torch.ops.aten.view.default(select_30, [1024, 160, 96]);  select_30 = None
        permute_116 = torch.ops.aten.permute.default(view_156, [1, 0, 2]);  view_156 = None
        view_157 = torch.ops.aten.view.default(select_31, [1024, 160, 96]);  select_31 = None
        permute_117 = torch.ops.aten.permute.default(view_157, [1, 0, 2]);  view_157 = None
        view_158 = torch.ops.aten.view.default(select_32, [1024, 160, 96]);  select_32 = None
        permute_118 = torch.ops.aten.permute.default(view_158, [1, 0, 2]);  view_158 = None
        view_159 = torch.ops.aten.view.default(permute_116, [20, 8, 1024, 96]);  permute_116 = None
        view_160 = torch.ops.aten.view.default(permute_117, [20, 8, 1024, 96]);  permute_117 = None
        view_161 = torch.ops.aten.view.default(permute_118, [20, 8, 1024, 96]);  permute_118 = None
        graphsafe_run_with_rng_state_10 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_159, view_160, view_161, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_10);  bwd_rng_state_10 = None
        getitem_132 = graphsafe_run_with_rng_state_10[0]
        getitem_133 = graphsafe_run_with_rng_state_10[1]
        getitem_138 = graphsafe_run_with_rng_state_10[6]
        getitem_139 = graphsafe_run_with_rng_state_10[7];  graphsafe_run_with_rng_state_10 = None
        permute_119 = torch.ops.aten.permute.default(getitem_132, [2, 0, 1, 3])
        view_162 = torch.ops.aten.view.default(permute_119, [20480, 768]);  permute_119 = None
        mm_30 = torch.ops.aten.mm.default(permute_183, view_162);  permute_183 = view_162 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(view_232, [0], True, dtype = torch.float32);  view_232 = None
        view_233 = torch.ops.aten.view.default(sum_21, [768]);  sum_21 = None
        convert_element_type_379 = torch.ops.prims.convert_element_type.default(mm_30, torch.float32);  mm_30 = None
        convert_element_type_default_42 = torch.ops.prims.convert_element_type.default(view_233, torch.float32);  view_233 = None
        view_234 = torch.ops.aten.view.default(mm_29, [1024, 20, 8, 96]);  mm_29 = None
        permute_186 = torch.ops.aten.permute.default(view_234, [1, 2, 0, 3]);  view_234 = None
        _scaled_dot_product_flash_attention_backward_1 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_186, view_159, view_160, view_161, getitem_132, getitem_133, None, None, 1024, 1024, 0.2, False, getitem_138, getitem_139, scale = 0.10206207261596577);  permute_186 = view_159 = view_160 = view_161 = getitem_132 = getitem_133 = getitem_138 = getitem_139 = None
        getitem_159 = _scaled_dot_product_flash_attention_backward_1[0]
        getitem_160 = _scaled_dot_product_flash_attention_backward_1[1]
        getitem_161 = _scaled_dot_product_flash_attention_backward_1[2];  _scaled_dot_product_flash_attention_backward_1 = None
        view_235 = torch.ops.aten.view.default(getitem_161, [160, 1024, 96]);  getitem_161 = None
        view_236 = torch.ops.aten.view.default(getitem_160, [160, 1024, 96]);  getitem_160 = None
        view_237 = torch.ops.aten.view.default(getitem_159, [160, 1024, 96]);  getitem_159 = None
        permute_187 = torch.ops.aten.permute.default(view_235, [1, 0, 2]);  view_235 = None
        view_238 = torch.ops.aten.view.default(permute_187, [1024, 20, 768]);  permute_187 = None
        permute_188 = torch.ops.aten.permute.default(view_236, [1, 0, 2]);  view_236 = None
        view_239 = torch.ops.aten.view.default(permute_188, [1024, 20, 768]);  permute_188 = None
        permute_189 = torch.ops.aten.permute.default(view_237, [1, 0, 2]);  view_237 = None
        view_240 = torch.ops.aten.view.default(permute_189, [1024, 20, 768]);  permute_189 = None
        select_scatter_3 = torch.ops.aten.select_scatter.default(full_default_6, view_238, 0, 2);  view_238 = None
        select_scatter_4 = torch.ops.aten.select_scatter.default(full_default_6, view_239, 0, 1);  view_239 = None
        add_126 = torch.ops.aten.add.Tensor(select_scatter_3, select_scatter_4);  select_scatter_3 = select_scatter_4 = None
        select_scatter_5 = torch.ops.aten.select_scatter.default(full_default_6, view_240, 0, 0);  view_240 = None
        add_127 = torch.ops.aten.add.Tensor(add_126, select_scatter_5);  add_126 = select_scatter_5 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(add_127, 3);  add_127 = None
        permute_190 = torch.ops.aten.permute.default(unsqueeze_31, [3, 1, 2, 0, 4]);  unsqueeze_31 = None
        squeeze_13 = torch.ops.aten.squeeze.dim(permute_190, 0);  permute_190 = None
        clone_34 = torch.ops.aten.clone.default(squeeze_13, memory_format = torch.contiguous_format);  squeeze_13 = None
        view_241 = torch.ops.aten.view.default(clone_34, [1024, 20, 2304]);  clone_34 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(view_241, [0, 1], True, dtype = torch.float32)
        view_242 = torch.ops.aten.view.default(sum_22, [2304]);  sum_22 = None
        view_243 = torch.ops.aten.view.default(view_241, [20480, 2304]);  view_241 = None
        permute_191 = torch.ops.aten.permute.default(view_243, [1, 0])
        mm_31 = torch.ops.aten.mm.default(permute_191, view_153);  permute_191 = view_153 = None
        permute_193 = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
        mm_32 = torch.ops.aten.mm.default(view_243, permute_193);  view_243 = permute_193 = None
        view_244 = torch.ops.aten.view.default(mm_32, [1024, 20, 768]);  mm_32 = None
        convert_element_type_386 = torch.ops.prims.convert_element_type.default(view_244, torch.float32);  view_244 = None
        convert_element_type_387 = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_default_41 = torch.ops.prims.convert_element_type.default(view_242, torch.float32);  view_242 = None
        permute_195 = torch.ops.aten.permute.default(convert_element_type_386, [1, 0, 2]);  convert_element_type_386 = None
        mul_156 = torch.ops.aten.mul.Tensor(permute_195, primals_125);  primals_125 = None
        mul_157 = torch.ops.aten.mul.Tensor(mul_156, 768)
        sum_23 = torch.ops.aten.sum.dim_IntList(mul_156, [2], True)
        mul_158 = torch.ops.aten.mul.Tensor(mul_156, mul_90);  mul_156 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(mul_158, [2], True);  mul_158 = None
        mul_159 = torch.ops.aten.mul.Tensor(mul_90, sum_24);  sum_24 = None
        sub_41 = torch.ops.aten.sub.Tensor(mul_157, sum_23);  mul_157 = sum_23 = None
        sub_42 = torch.ops.aten.sub.Tensor(sub_41, mul_159);  sub_41 = mul_159 = None
        div_3 = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
        mul_160 = torch.ops.aten.mul.Tensor(div_3, sub_42);  div_3 = sub_42 = None
        mul_161 = torch.ops.aten.mul.Tensor(permute_195, mul_90);  mul_90 = None
        sum_25 = torch.ops.aten.sum.dim_IntList(mul_161, [0, 1]);  mul_161 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(permute_195, [0, 1]);  permute_195 = None
        add_128 = torch.ops.aten.add.Tensor(add_125, mul_160);  add_125 = mul_160 = None
        convert_element_type_389 = torch.ops.prims.convert_element_type.default(add_128, torch.float16)
        inductor_lookup_seed_default_9 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_2 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        convert_element_type_default_54 = torch.ops.prims.convert_element_type.default(inductor_random_default_2, torch.float16);  inductor_random_default_2 = None
        gt_9 = torch.ops.aten.gt.Scalar(convert_element_type_default_54, 0.2);  convert_element_type_default_54 = None
        convert_element_type_390 = torch.ops.prims.convert_element_type.default(gt_9, torch.float16);  gt_9 = None
        mul_162 = torch.ops.aten.mul.Tensor(convert_element_type_390, 1.25);  convert_element_type_390 = None
        mul_163 = torch.ops.aten.mul.Tensor(convert_element_type_389, mul_162);  convert_element_type_389 = mul_162 = None
        view_245 = torch.ops.aten.view.default(mul_163, [20480, 768]);  mul_163 = None
        convert_element_type_231 = torch.ops.prims.convert_element_type.default(primals_123, torch.float16);  primals_123 = None
        permute_112 = torch.ops.aten.permute.default(convert_element_type_231, [1, 0]);  convert_element_type_231 = None
        permute_196 = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        mm_33 = torch.ops.aten.mm.default(view_245, permute_196);  permute_196 = None
        permute_197 = torch.ops.aten.permute.default(view_245, [1, 0])
        var_mean_19 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
        getitem_128 = var_mean_19[0]
        getitem_129 = var_mean_19[1];  var_mean_19 = None
        add_80 = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_79, getitem_129);  add_79 = getitem_129 = None
        mul_83 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        mul_84 = torch.ops.aten.mul.Tensor(mul_83, primals_119)
        add_81 = torch.ops.aten.add.Tensor(mul_84, primals_120);  mul_84 = primals_120 = None
        convert_element_type_222 = torch.ops.prims.convert_element_type.default(primals_122, torch.float16);  primals_122 = None
        convert_element_type_223 = torch.ops.prims.convert_element_type.default(primals_121, torch.float16);  primals_121 = None
        convert_element_type_224 = torch.ops.prims.convert_element_type.default(add_81, torch.float16);  add_81 = None
        view_149 = torch.ops.aten.view.default(convert_element_type_224, [20480, 768]);  convert_element_type_224 = None
        permute_111 = torch.ops.aten.permute.default(convert_element_type_223, [1, 0]);  convert_element_type_223 = None
        addmm_28 = torch.ops.aten.addmm.default(convert_element_type_222, view_149, permute_111);  convert_element_type_222 = None
        view_150 = torch.ops.aten.view.default(addmm_28, [20, 1024, 1536]);  addmm_28 = None
        convert_element_type_228 = torch.ops.prims.convert_element_type.default(view_150, torch.float32);  view_150 = None
        mul_85 = torch.ops.aten.mul.Tensor(convert_element_type_228, 0.5)
        mul_86 = torch.ops.aten.mul.Tensor(convert_element_type_228, 0.7071067811865476)
        erf_9 = torch.ops.aten.erf.default(mul_86);  mul_86 = None
        add_82 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_87 = torch.ops.aten.mul.Tensor(mul_85, add_82);  mul_85 = None
        convert_element_type_229 = torch.ops.prims.convert_element_type.default(mul_87, torch.float16);  mul_87 = None
        view_151 = torch.ops.aten.view.default(convert_element_type_229, [20480, 1536]);  convert_element_type_229 = None
        mm_34 = torch.ops.aten.mm.default(permute_197, view_151);  permute_197 = view_151 = None
        sum_27 = torch.ops.aten.sum.dim_IntList(view_245, [0], True, dtype = torch.float32);  view_245 = None
        view_246 = torch.ops.aten.view.default(sum_27, [768]);  sum_27 = None
        view_247 = torch.ops.aten.view.default(mm_33, [20, 1024, 1536]);  mm_33 = None
        convert_element_type_396 = torch.ops.prims.convert_element_type.default(mm_34, torch.float32);  mm_34 = None
        convert_element_type_default_40 = torch.ops.prims.convert_element_type.default(view_246, torch.float32);  view_246 = None
        convert_element_type_398 = torch.ops.prims.convert_element_type.default(view_247, torch.float32);  view_247 = None
        mul_165 = torch.ops.aten.mul.Tensor(add_82, 0.5);  add_82 = None
        mul_166 = torch.ops.aten.mul.Tensor(convert_element_type_228, convert_element_type_228)
        mul_167 = torch.ops.aten.mul.Tensor(mul_166, -0.5);  mul_166 = None
        exp_2 = torch.ops.aten.exp.default(mul_167);  mul_167 = None
        mul_168 = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_169 = torch.ops.aten.mul.Tensor(convert_element_type_228, mul_168);  convert_element_type_228 = mul_168 = None
        add_130 = torch.ops.aten.add.Tensor(mul_165, mul_169);  mul_165 = mul_169 = None
        mul_170 = torch.ops.aten.mul.Tensor(convert_element_type_398, add_130);  convert_element_type_398 = add_130 = None
        convert_element_type_400 = torch.ops.prims.convert_element_type.default(mul_170, torch.float16);  mul_170 = None
        view_248 = torch.ops.aten.view.default(convert_element_type_400, [20480, 1536]);  convert_element_type_400 = None
        permute_200 = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        mm_35 = torch.ops.aten.mm.default(view_248, permute_200);  permute_200 = None
        permute_201 = torch.ops.aten.permute.default(view_248, [1, 0])
        mm_36 = torch.ops.aten.mm.default(permute_201, view_149);  permute_201 = view_149 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(view_248, [0], True, dtype = torch.float32);  view_248 = None
        view_249 = torch.ops.aten.view.default(sum_28, [1536]);  sum_28 = None
        view_250 = torch.ops.aten.view.default(mm_35, [20, 1024, 768]);  mm_35 = None
        convert_element_type_406 = torch.ops.prims.convert_element_type.default(view_250, torch.float32);  view_250 = None
        convert_element_type_407 = torch.ops.prims.convert_element_type.default(mm_36, torch.float32);  mm_36 = None
        convert_element_type_default_39 = torch.ops.prims.convert_element_type.default(view_249, torch.float32);  view_249 = None
        mul_172 = torch.ops.aten.mul.Tensor(convert_element_type_406, primals_119);  primals_119 = None
        mul_173 = torch.ops.aten.mul.Tensor(mul_172, 768)
        sum_29 = torch.ops.aten.sum.dim_IntList(mul_172, [2], True)
        mul_174 = torch.ops.aten.mul.Tensor(mul_172, mul_83);  mul_172 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(mul_174, [2], True);  mul_174 = None
        mul_175 = torch.ops.aten.mul.Tensor(mul_83, sum_30);  sum_30 = None
        sub_44 = torch.ops.aten.sub.Tensor(mul_173, sum_29);  mul_173 = sum_29 = None
        sub_45 = torch.ops.aten.sub.Tensor(sub_44, mul_175);  sub_44 = mul_175 = None
        div_4 = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
        mul_176 = torch.ops.aten.mul.Tensor(div_4, sub_45);  div_4 = sub_45 = None
        mul_177 = torch.ops.aten.mul.Tensor(convert_element_type_406, mul_83);  mul_83 = None
        sum_31 = torch.ops.aten.sum.dim_IntList(mul_177, [0, 1]);  mul_177 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(convert_element_type_406, [0, 1]);  convert_element_type_406 = None
        add_131 = torch.ops.aten.add.Tensor(add_128, mul_176);  add_128 = mul_176 = None
        convert_element_type_409 = torch.ops.prims.convert_element_type.default(add_131, torch.float16)
        permute_204 = torch.ops.aten.permute.default(convert_element_type_409, [1, 0, 2]);  convert_element_type_409 = None
        clone_36 = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_251 = torch.ops.aten.view.default(clone_36, [20480, 768]);  clone_36 = None
        convert_element_type_218 = torch.ops.prims.convert_element_type.default(primals_117, torch.float16);  primals_117 = None
        permute_109 = torch.ops.aten.permute.default(convert_element_type_218, [1, 0]);  convert_element_type_218 = None
        permute_205 = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        mm_37 = torch.ops.aten.mm.default(view_251, permute_205);  permute_205 = None
        permute_206 = torch.ops.aten.permute.default(view_251, [1, 0])
        var_mean_18 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_117 = var_mean_18[0]
        getitem_118 = var_mean_18[1];  var_mean_18 = None
        add_76 = torch.ops.aten.add.Tensor(getitem_117, 1e-05);  getitem_117 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_75, getitem_118);  add_75 = getitem_118 = None
        mul_81 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        mul_82 = torch.ops.aten.mul.Tensor(mul_81, primals_113)
        add_77 = torch.ops.aten.add.Tensor(mul_82, primals_114);  mul_82 = primals_114 = None
        permute_102 = torch.ops.aten.permute.default(add_77, [1, 0, 2]);  add_77 = None
        convert_element_type_212 = torch.ops.prims.convert_element_type.default(primals_115, torch.float16);  primals_115 = None
        convert_element_type_213 = torch.ops.prims.convert_element_type.default(primals_116, torch.float16);  primals_116 = None
        convert_element_type_214 = torch.ops.prims.convert_element_type.default(permute_102, torch.float16);  permute_102 = None
        permute_103 = torch.ops.aten.permute.default(convert_element_type_213, [1, 0]);  convert_element_type_213 = None
        clone_20 = torch.ops.aten.clone.default(convert_element_type_214, memory_format = torch.contiguous_format);  convert_element_type_214 = None
        view_138 = torch.ops.aten.view.default(clone_20, [20480, 768]);  clone_20 = None
        mm_10 = torch.ops.aten.mm.default(view_138, permute_103)
        view_139 = torch.ops.aten.view.default(mm_10, [1024, 20, 2304]);  mm_10 = None
        add_78 = torch.ops.aten.add.Tensor(view_139, convert_element_type_212);  view_139 = convert_element_type_212 = None
        view_140 = torch.ops.aten.view.default(add_78, [1024, 20, 3, 768]);  add_78 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(view_140, 0);  view_140 = None
        permute_104 = torch.ops.aten.permute.default(unsqueeze_15, [3, 1, 2, 0, 4]);  unsqueeze_15 = None
        squeeze_9 = torch.ops.aten.squeeze.dim(permute_104, -2);  permute_104 = None
        clone_21 = torch.ops.aten.clone.default(squeeze_9, memory_format = torch.contiguous_format);  squeeze_9 = None
        select_27 = torch.ops.aten.select.int(clone_21, 0, 0)
        select_28 = torch.ops.aten.select.int(clone_21, 0, 1)
        select_29 = torch.ops.aten.select.int(clone_21, 0, 2);  clone_21 = None
        view_141 = torch.ops.aten.view.default(select_27, [1024, 160, 96]);  select_27 = None
        permute_105 = torch.ops.aten.permute.default(view_141, [1, 0, 2]);  view_141 = None
        view_142 = torch.ops.aten.view.default(select_28, [1024, 160, 96]);  select_28 = None
        permute_106 = torch.ops.aten.permute.default(view_142, [1, 0, 2]);  view_142 = None
        view_143 = torch.ops.aten.view.default(select_29, [1024, 160, 96]);  select_29 = None
        permute_107 = torch.ops.aten.permute.default(view_143, [1, 0, 2]);  view_143 = None
        view_144 = torch.ops.aten.view.default(permute_105, [20, 8, 1024, 96]);  permute_105 = None
        view_145 = torch.ops.aten.view.default(permute_106, [20, 8, 1024, 96]);  permute_106 = None
        view_146 = torch.ops.aten.view.default(permute_107, [20, 8, 1024, 96]);  permute_107 = None
        graphsafe_run_with_rng_state_9 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_144, view_145, view_146, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_9);  bwd_rng_state_9 = None
        getitem_119 = graphsafe_run_with_rng_state_9[0]
        getitem_120 = graphsafe_run_with_rng_state_9[1]
        getitem_125 = graphsafe_run_with_rng_state_9[6]
        getitem_126 = graphsafe_run_with_rng_state_9[7];  graphsafe_run_with_rng_state_9 = None
        permute_108 = torch.ops.aten.permute.default(getitem_119, [2, 0, 1, 3])
        view_147 = torch.ops.aten.view.default(permute_108, [20480, 768]);  permute_108 = None
        mm_38 = torch.ops.aten.mm.default(permute_206, view_147);  permute_206 = view_147 = None
        sum_33 = torch.ops.aten.sum.dim_IntList(view_251, [0], True, dtype = torch.float32);  view_251 = None
        view_252 = torch.ops.aten.view.default(sum_33, [768]);  sum_33 = None
        convert_element_type_415 = torch.ops.prims.convert_element_type.default(mm_38, torch.float32);  mm_38 = None
        convert_element_type_default_38 = torch.ops.prims.convert_element_type.default(view_252, torch.float32);  view_252 = None
        view_253 = torch.ops.aten.view.default(mm_37, [1024, 20, 8, 96]);  mm_37 = None
        permute_209 = torch.ops.aten.permute.default(view_253, [1, 2, 0, 3]);  view_253 = None
        _scaled_dot_product_flash_attention_backward_2 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_209, view_144, view_145, view_146, getitem_119, getitem_120, None, None, 1024, 1024, 0.2, False, getitem_125, getitem_126, scale = 0.10206207261596577);  permute_209 = view_144 = view_145 = view_146 = getitem_119 = getitem_120 = getitem_125 = getitem_126 = None
        getitem_162 = _scaled_dot_product_flash_attention_backward_2[0]
        getitem_163 = _scaled_dot_product_flash_attention_backward_2[1]
        getitem_164 = _scaled_dot_product_flash_attention_backward_2[2];  _scaled_dot_product_flash_attention_backward_2 = None
        view_254 = torch.ops.aten.view.default(getitem_164, [160, 1024, 96]);  getitem_164 = None
        view_255 = torch.ops.aten.view.default(getitem_163, [160, 1024, 96]);  getitem_163 = None
        view_256 = torch.ops.aten.view.default(getitem_162, [160, 1024, 96]);  getitem_162 = None
        permute_210 = torch.ops.aten.permute.default(view_254, [1, 0, 2]);  view_254 = None
        view_257 = torch.ops.aten.view.default(permute_210, [1024, 20, 768]);  permute_210 = None
        permute_211 = torch.ops.aten.permute.default(view_255, [1, 0, 2]);  view_255 = None
        view_258 = torch.ops.aten.view.default(permute_211, [1024, 20, 768]);  permute_211 = None
        permute_212 = torch.ops.aten.permute.default(view_256, [1, 0, 2]);  view_256 = None
        view_259 = torch.ops.aten.view.default(permute_212, [1024, 20, 768]);  permute_212 = None
        select_scatter_6 = torch.ops.aten.select_scatter.default(full_default_6, view_257, 0, 2);  view_257 = None
        select_scatter_7 = torch.ops.aten.select_scatter.default(full_default_6, view_258, 0, 1);  view_258 = None
        add_132 = torch.ops.aten.add.Tensor(select_scatter_6, select_scatter_7);  select_scatter_6 = select_scatter_7 = None
        select_scatter_8 = torch.ops.aten.select_scatter.default(full_default_6, view_259, 0, 0);  view_259 = None
        add_133 = torch.ops.aten.add.Tensor(add_132, select_scatter_8);  add_132 = select_scatter_8 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(add_133, 3);  add_133 = None
        permute_213 = torch.ops.aten.permute.default(unsqueeze_32, [3, 1, 2, 0, 4]);  unsqueeze_32 = None
        squeeze_14 = torch.ops.aten.squeeze.dim(permute_213, 0);  permute_213 = None
        clone_37 = torch.ops.aten.clone.default(squeeze_14, memory_format = torch.contiguous_format);  squeeze_14 = None
        view_260 = torch.ops.aten.view.default(clone_37, [1024, 20, 2304]);  clone_37 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(view_260, [0, 1], True, dtype = torch.float32)
        view_261 = torch.ops.aten.view.default(sum_34, [2304]);  sum_34 = None
        view_262 = torch.ops.aten.view.default(view_260, [20480, 2304]);  view_260 = None
        permute_214 = torch.ops.aten.permute.default(view_262, [1, 0])
        mm_39 = torch.ops.aten.mm.default(permute_214, view_138);  permute_214 = view_138 = None
        permute_216 = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
        mm_40 = torch.ops.aten.mm.default(view_262, permute_216);  view_262 = permute_216 = None
        view_263 = torch.ops.aten.view.default(mm_40, [1024, 20, 768]);  mm_40 = None
        convert_element_type_422 = torch.ops.prims.convert_element_type.default(view_263, torch.float32);  view_263 = None
        convert_element_type_423 = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_default_37 = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        permute_218 = torch.ops.aten.permute.default(convert_element_type_422, [1, 0, 2]);  convert_element_type_422 = None
        mul_179 = torch.ops.aten.mul.Tensor(permute_218, primals_113);  primals_113 = None
        mul_180 = torch.ops.aten.mul.Tensor(mul_179, 768)
        sum_35 = torch.ops.aten.sum.dim_IntList(mul_179, [2], True)
        mul_181 = torch.ops.aten.mul.Tensor(mul_179, mul_81);  mul_179 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(mul_181, [2], True);  mul_181 = None
        mul_182 = torch.ops.aten.mul.Tensor(mul_81, sum_36);  sum_36 = None
        sub_47 = torch.ops.aten.sub.Tensor(mul_180, sum_35);  mul_180 = sum_35 = None
        sub_48 = torch.ops.aten.sub.Tensor(sub_47, mul_182);  sub_47 = mul_182 = None
        div_5 = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
        mul_183 = torch.ops.aten.mul.Tensor(div_5, sub_48);  div_5 = sub_48 = None
        mul_184 = torch.ops.aten.mul.Tensor(permute_218, mul_81);  mul_81 = None
        sum_37 = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1]);  mul_184 = None
        sum_38 = torch.ops.aten.sum.dim_IntList(permute_218, [0, 1]);  permute_218 = None
        add_134 = torch.ops.aten.add.Tensor(add_131, mul_183);  add_131 = mul_183 = None
        convert_element_type_425 = torch.ops.prims.convert_element_type.default(add_134, torch.float16)
        inductor_lookup_seed_default_8 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_3 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        convert_element_type_default_55 = torch.ops.prims.convert_element_type.default(inductor_random_default_3, torch.float16);  inductor_random_default_3 = None
        gt_8 = torch.ops.aten.gt.Scalar(convert_element_type_default_55, 0.2);  convert_element_type_default_55 = None
        convert_element_type_426 = torch.ops.prims.convert_element_type.default(gt_8, torch.float16);  gt_8 = None
        mul_185 = torch.ops.aten.mul.Tensor(convert_element_type_426, 1.25);  convert_element_type_426 = None
        mul_186 = torch.ops.aten.mul.Tensor(convert_element_type_425, mul_185);  convert_element_type_425 = mul_185 = None
        view_264 = torch.ops.aten.view.default(mul_186, [20480, 768]);  mul_186 = None
        convert_element_type_208 = torch.ops.prims.convert_element_type.default(primals_111, torch.float16);  primals_111 = None
        permute_101 = torch.ops.aten.permute.default(convert_element_type_208, [1, 0]);  convert_element_type_208 = None
        permute_219 = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        mm_41 = torch.ops.aten.mm.default(view_264, permute_219);  permute_219 = None
        permute_220 = torch.ops.aten.permute.default(view_264, [1, 0])
        var_mean_17 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_115 = var_mean_17[0]
        getitem_116 = var_mean_17[1];  var_mean_17 = None
        add_72 = torch.ops.aten.add.Tensor(getitem_115, 1e-05);  getitem_115 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_71, getitem_116);  add_71 = getitem_116 = None
        mul_74 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        mul_75 = torch.ops.aten.mul.Tensor(mul_74, primals_107)
        add_73 = torch.ops.aten.add.Tensor(mul_75, primals_108);  mul_75 = primals_108 = None
        convert_element_type_199 = torch.ops.prims.convert_element_type.default(primals_110, torch.float16);  primals_110 = None
        convert_element_type_200 = torch.ops.prims.convert_element_type.default(primals_109, torch.float16);  primals_109 = None
        convert_element_type_201 = torch.ops.prims.convert_element_type.default(add_73, torch.float16);  add_73 = None
        view_134 = torch.ops.aten.view.default(convert_element_type_201, [20480, 768]);  convert_element_type_201 = None
        permute_100 = torch.ops.aten.permute.default(convert_element_type_200, [1, 0]);  convert_element_type_200 = None
        addmm_25 = torch.ops.aten.addmm.default(convert_element_type_199, view_134, permute_100);  convert_element_type_199 = None
        view_135 = torch.ops.aten.view.default(addmm_25, [20, 1024, 1536]);  addmm_25 = None
        convert_element_type_205 = torch.ops.prims.convert_element_type.default(view_135, torch.float32);  view_135 = None
        mul_76 = torch.ops.aten.mul.Tensor(convert_element_type_205, 0.5)
        mul_77 = torch.ops.aten.mul.Tensor(convert_element_type_205, 0.7071067811865476)
        erf_8 = torch.ops.aten.erf.default(mul_77);  mul_77 = None
        add_74 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_78 = torch.ops.aten.mul.Tensor(mul_76, add_74);  mul_76 = None
        convert_element_type_206 = torch.ops.prims.convert_element_type.default(mul_78, torch.float16);  mul_78 = None
        view_136 = torch.ops.aten.view.default(convert_element_type_206, [20480, 1536]);  convert_element_type_206 = None
        mm_42 = torch.ops.aten.mm.default(permute_220, view_136);  permute_220 = view_136 = None
        sum_39 = torch.ops.aten.sum.dim_IntList(view_264, [0], True, dtype = torch.float32);  view_264 = None
        view_265 = torch.ops.aten.view.default(sum_39, [768]);  sum_39 = None
        view_266 = torch.ops.aten.view.default(mm_41, [20, 1024, 1536]);  mm_41 = None
        convert_element_type_432 = torch.ops.prims.convert_element_type.default(mm_42, torch.float32);  mm_42 = None
        convert_element_type_default_36 = torch.ops.prims.convert_element_type.default(view_265, torch.float32);  view_265 = None
        convert_element_type_434 = torch.ops.prims.convert_element_type.default(view_266, torch.float32);  view_266 = None
        mul_188 = torch.ops.aten.mul.Tensor(add_74, 0.5);  add_74 = None
        mul_189 = torch.ops.aten.mul.Tensor(convert_element_type_205, convert_element_type_205)
        mul_190 = torch.ops.aten.mul.Tensor(mul_189, -0.5);  mul_189 = None
        exp_3 = torch.ops.aten.exp.default(mul_190);  mul_190 = None
        mul_191 = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
        mul_192 = torch.ops.aten.mul.Tensor(convert_element_type_205, mul_191);  convert_element_type_205 = mul_191 = None
        add_136 = torch.ops.aten.add.Tensor(mul_188, mul_192);  mul_188 = mul_192 = None
        mul_193 = torch.ops.aten.mul.Tensor(convert_element_type_434, add_136);  convert_element_type_434 = add_136 = None
        convert_element_type_436 = torch.ops.prims.convert_element_type.default(mul_193, torch.float16);  mul_193 = None
        view_267 = torch.ops.aten.view.default(convert_element_type_436, [20480, 1536]);  convert_element_type_436 = None
        permute_223 = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
        mm_43 = torch.ops.aten.mm.default(view_267, permute_223);  permute_223 = None
        permute_224 = torch.ops.aten.permute.default(view_267, [1, 0])
        mm_44 = torch.ops.aten.mm.default(permute_224, view_134);  permute_224 = view_134 = None
        sum_40 = torch.ops.aten.sum.dim_IntList(view_267, [0], True, dtype = torch.float32);  view_267 = None
        view_268 = torch.ops.aten.view.default(sum_40, [1536]);  sum_40 = None
        view_269 = torch.ops.aten.view.default(mm_43, [20, 1024, 768]);  mm_43 = None
        convert_element_type_442 = torch.ops.prims.convert_element_type.default(view_269, torch.float32);  view_269 = None
        convert_element_type_443 = torch.ops.prims.convert_element_type.default(mm_44, torch.float32);  mm_44 = None
        convert_element_type_default_35 = torch.ops.prims.convert_element_type.default(view_268, torch.float32);  view_268 = None
        mul_195 = torch.ops.aten.mul.Tensor(convert_element_type_442, primals_107);  primals_107 = None
        mul_196 = torch.ops.aten.mul.Tensor(mul_195, 768)
        sum_41 = torch.ops.aten.sum.dim_IntList(mul_195, [2], True)
        mul_197 = torch.ops.aten.mul.Tensor(mul_195, mul_74);  mul_195 = None
        sum_42 = torch.ops.aten.sum.dim_IntList(mul_197, [2], True);  mul_197 = None
        mul_198 = torch.ops.aten.mul.Tensor(mul_74, sum_42);  sum_42 = None
        sub_50 = torch.ops.aten.sub.Tensor(mul_196, sum_41);  mul_196 = sum_41 = None
        sub_51 = torch.ops.aten.sub.Tensor(sub_50, mul_198);  sub_50 = mul_198 = None
        div_6 = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
        mul_199 = torch.ops.aten.mul.Tensor(div_6, sub_51);  div_6 = sub_51 = None
        mul_200 = torch.ops.aten.mul.Tensor(convert_element_type_442, mul_74);  mul_74 = None
        sum_43 = torch.ops.aten.sum.dim_IntList(mul_200, [0, 1]);  mul_200 = None
        sum_44 = torch.ops.aten.sum.dim_IntList(convert_element_type_442, [0, 1]);  convert_element_type_442 = None
        add_137 = torch.ops.aten.add.Tensor(add_134, mul_199);  add_134 = mul_199 = None
        convert_element_type_445 = torch.ops.prims.convert_element_type.default(add_137, torch.float16)
        permute_227 = torch.ops.aten.permute.default(convert_element_type_445, [1, 0, 2]);  convert_element_type_445 = None
        clone_39 = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        view_270 = torch.ops.aten.view.default(clone_39, [20480, 768]);  clone_39 = None
        convert_element_type_195 = torch.ops.prims.convert_element_type.default(primals_105, torch.float16);  primals_105 = None
        permute_98 = torch.ops.aten.permute.default(convert_element_type_195, [1, 0]);  convert_element_type_195 = None
        permute_228 = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        mm_45 = torch.ops.aten.mm.default(view_270, permute_228);  permute_228 = None
        permute_229 = torch.ops.aten.permute.default(view_270, [1, 0])
        var_mean_16 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_104 = var_mean_16[0]
        getitem_105 = var_mean_16[1];  var_mean_16 = None
        add_68 = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_67, getitem_105);  add_67 = getitem_105 = None
        mul_72 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, primals_101)
        add_69 = torch.ops.aten.add.Tensor(mul_73, primals_102);  mul_73 = primals_102 = None
        permute_91 = torch.ops.aten.permute.default(add_69, [1, 0, 2]);  add_69 = None
        convert_element_type_189 = torch.ops.prims.convert_element_type.default(primals_103, torch.float16);  primals_103 = None
        convert_element_type_190 = torch.ops.prims.convert_element_type.default(primals_104, torch.float16);  primals_104 = None
        convert_element_type_191 = torch.ops.prims.convert_element_type.default(permute_91, torch.float16);  permute_91 = None
        permute_92 = torch.ops.aten.permute.default(convert_element_type_190, [1, 0]);  convert_element_type_190 = None
        clone_18 = torch.ops.aten.clone.default(convert_element_type_191, memory_format = torch.contiguous_format);  convert_element_type_191 = None
        view_123 = torch.ops.aten.view.default(clone_18, [20480, 768]);  clone_18 = None
        mm_9 = torch.ops.aten.mm.default(view_123, permute_92)
        view_124 = torch.ops.aten.view.default(mm_9, [1024, 20, 2304]);  mm_9 = None
        add_70 = torch.ops.aten.add.Tensor(view_124, convert_element_type_189);  view_124 = convert_element_type_189 = None
        view_125 = torch.ops.aten.view.default(add_70, [1024, 20, 3, 768]);  add_70 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(view_125, 0);  view_125 = None
        permute_93 = torch.ops.aten.permute.default(unsqueeze_14, [3, 1, 2, 0, 4]);  unsqueeze_14 = None
        squeeze_8 = torch.ops.aten.squeeze.dim(permute_93, -2);  permute_93 = None
        clone_19 = torch.ops.aten.clone.default(squeeze_8, memory_format = torch.contiguous_format);  squeeze_8 = None
        select_24 = torch.ops.aten.select.int(clone_19, 0, 0)
        select_25 = torch.ops.aten.select.int(clone_19, 0, 1)
        select_26 = torch.ops.aten.select.int(clone_19, 0, 2);  clone_19 = None
        view_126 = torch.ops.aten.view.default(select_24, [1024, 160, 96]);  select_24 = None
        permute_94 = torch.ops.aten.permute.default(view_126, [1, 0, 2]);  view_126 = None
        view_127 = torch.ops.aten.view.default(select_25, [1024, 160, 96]);  select_25 = None
        permute_95 = torch.ops.aten.permute.default(view_127, [1, 0, 2]);  view_127 = None
        view_128 = torch.ops.aten.view.default(select_26, [1024, 160, 96]);  select_26 = None
        permute_96 = torch.ops.aten.permute.default(view_128, [1, 0, 2]);  view_128 = None
        view_129 = torch.ops.aten.view.default(permute_94, [20, 8, 1024, 96]);  permute_94 = None
        view_130 = torch.ops.aten.view.default(permute_95, [20, 8, 1024, 96]);  permute_95 = None
        view_131 = torch.ops.aten.view.default(permute_96, [20, 8, 1024, 96]);  permute_96 = None
        graphsafe_run_with_rng_state_8 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_129, view_130, view_131, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_8);  bwd_rng_state_8 = None
        getitem_106 = graphsafe_run_with_rng_state_8[0]
        getitem_107 = graphsafe_run_with_rng_state_8[1]
        getitem_112 = graphsafe_run_with_rng_state_8[6]
        getitem_113 = graphsafe_run_with_rng_state_8[7];  graphsafe_run_with_rng_state_8 = None
        permute_97 = torch.ops.aten.permute.default(getitem_106, [2, 0, 1, 3])
        view_132 = torch.ops.aten.view.default(permute_97, [20480, 768]);  permute_97 = None
        mm_46 = torch.ops.aten.mm.default(permute_229, view_132);  permute_229 = view_132 = None
        sum_45 = torch.ops.aten.sum.dim_IntList(view_270, [0], True, dtype = torch.float32);  view_270 = None
        view_271 = torch.ops.aten.view.default(sum_45, [768]);  sum_45 = None
        convert_element_type_451 = torch.ops.prims.convert_element_type.default(mm_46, torch.float32);  mm_46 = None
        convert_element_type_default_34 = torch.ops.prims.convert_element_type.default(view_271, torch.float32);  view_271 = None
        view_272 = torch.ops.aten.view.default(mm_45, [1024, 20, 8, 96]);  mm_45 = None
        permute_232 = torch.ops.aten.permute.default(view_272, [1, 2, 0, 3]);  view_272 = None
        _scaled_dot_product_flash_attention_backward_3 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_232, view_129, view_130, view_131, getitem_106, getitem_107, None, None, 1024, 1024, 0.2, False, getitem_112, getitem_113, scale = 0.10206207261596577);  permute_232 = view_129 = view_130 = view_131 = getitem_106 = getitem_107 = getitem_112 = getitem_113 = None
        getitem_165 = _scaled_dot_product_flash_attention_backward_3[0]
        getitem_166 = _scaled_dot_product_flash_attention_backward_3[1]
        getitem_167 = _scaled_dot_product_flash_attention_backward_3[2];  _scaled_dot_product_flash_attention_backward_3 = None
        view_273 = torch.ops.aten.view.default(getitem_167, [160, 1024, 96]);  getitem_167 = None
        view_274 = torch.ops.aten.view.default(getitem_166, [160, 1024, 96]);  getitem_166 = None
        view_275 = torch.ops.aten.view.default(getitem_165, [160, 1024, 96]);  getitem_165 = None
        permute_233 = torch.ops.aten.permute.default(view_273, [1, 0, 2]);  view_273 = None
        view_276 = torch.ops.aten.view.default(permute_233, [1024, 20, 768]);  permute_233 = None
        permute_234 = torch.ops.aten.permute.default(view_274, [1, 0, 2]);  view_274 = None
        view_277 = torch.ops.aten.view.default(permute_234, [1024, 20, 768]);  permute_234 = None
        permute_235 = torch.ops.aten.permute.default(view_275, [1, 0, 2]);  view_275 = None
        view_278 = torch.ops.aten.view.default(permute_235, [1024, 20, 768]);  permute_235 = None
        select_scatter_9 = torch.ops.aten.select_scatter.default(full_default_6, view_276, 0, 2);  view_276 = None
        select_scatter_10 = torch.ops.aten.select_scatter.default(full_default_6, view_277, 0, 1);  view_277 = None
        add_138 = torch.ops.aten.add.Tensor(select_scatter_9, select_scatter_10);  select_scatter_9 = select_scatter_10 = None
        select_scatter_11 = torch.ops.aten.select_scatter.default(full_default_6, view_278, 0, 0);  view_278 = None
        add_139 = torch.ops.aten.add.Tensor(add_138, select_scatter_11);  add_138 = select_scatter_11 = None
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(add_139, 3);  add_139 = None
        permute_236 = torch.ops.aten.permute.default(unsqueeze_33, [3, 1, 2, 0, 4]);  unsqueeze_33 = None
        squeeze_15 = torch.ops.aten.squeeze.dim(permute_236, 0);  permute_236 = None
        clone_40 = torch.ops.aten.clone.default(squeeze_15, memory_format = torch.contiguous_format);  squeeze_15 = None
        view_279 = torch.ops.aten.view.default(clone_40, [1024, 20, 2304]);  clone_40 = None
        sum_46 = torch.ops.aten.sum.dim_IntList(view_279, [0, 1], True, dtype = torch.float32)
        view_280 = torch.ops.aten.view.default(sum_46, [2304]);  sum_46 = None
        view_281 = torch.ops.aten.view.default(view_279, [20480, 2304]);  view_279 = None
        permute_237 = torch.ops.aten.permute.default(view_281, [1, 0])
        mm_47 = torch.ops.aten.mm.default(permute_237, view_123);  permute_237 = view_123 = None
        permute_239 = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
        mm_48 = torch.ops.aten.mm.default(view_281, permute_239);  view_281 = permute_239 = None
        view_282 = torch.ops.aten.view.default(mm_48, [1024, 20, 768]);  mm_48 = None
        convert_element_type_458 = torch.ops.prims.convert_element_type.default(view_282, torch.float32);  view_282 = None
        convert_element_type_459 = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_default_33 = torch.ops.prims.convert_element_type.default(view_280, torch.float32);  view_280 = None
        permute_241 = torch.ops.aten.permute.default(convert_element_type_458, [1, 0, 2]);  convert_element_type_458 = None
        mul_202 = torch.ops.aten.mul.Tensor(permute_241, primals_101);  primals_101 = None
        mul_203 = torch.ops.aten.mul.Tensor(mul_202, 768)
        sum_47 = torch.ops.aten.sum.dim_IntList(mul_202, [2], True)
        mul_204 = torch.ops.aten.mul.Tensor(mul_202, mul_72);  mul_202 = None
        sum_48 = torch.ops.aten.sum.dim_IntList(mul_204, [2], True);  mul_204 = None
        mul_205 = torch.ops.aten.mul.Tensor(mul_72, sum_48);  sum_48 = None
        sub_53 = torch.ops.aten.sub.Tensor(mul_203, sum_47);  mul_203 = sum_47 = None
        sub_54 = torch.ops.aten.sub.Tensor(sub_53, mul_205);  sub_53 = mul_205 = None
        div_7 = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
        mul_206 = torch.ops.aten.mul.Tensor(div_7, sub_54);  div_7 = sub_54 = None
        mul_207 = torch.ops.aten.mul.Tensor(permute_241, mul_72);  mul_72 = None
        sum_49 = torch.ops.aten.sum.dim_IntList(mul_207, [0, 1]);  mul_207 = None
        sum_50 = torch.ops.aten.sum.dim_IntList(permute_241, [0, 1]);  permute_241 = None
        add_140 = torch.ops.aten.add.Tensor(add_137, mul_206);  add_137 = mul_206 = None
        convert_element_type_461 = torch.ops.prims.convert_element_type.default(add_140, torch.float16)
        inductor_lookup_seed_default_7 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_4 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        convert_element_type_default_56 = torch.ops.prims.convert_element_type.default(inductor_random_default_4, torch.float16);  inductor_random_default_4 = None
        gt_7 = torch.ops.aten.gt.Scalar(convert_element_type_default_56, 0.2);  convert_element_type_default_56 = None
        convert_element_type_462 = torch.ops.prims.convert_element_type.default(gt_7, torch.float16);  gt_7 = None
        mul_208 = torch.ops.aten.mul.Tensor(convert_element_type_462, 1.25);  convert_element_type_462 = None
        mul_209 = torch.ops.aten.mul.Tensor(convert_element_type_461, mul_208);  convert_element_type_461 = mul_208 = None
        view_283 = torch.ops.aten.view.default(mul_209, [20480, 768]);  mul_209 = None
        convert_element_type_185 = torch.ops.prims.convert_element_type.default(primals_99, torch.float16);  primals_99 = None
        permute_90 = torch.ops.aten.permute.default(convert_element_type_185, [1, 0]);  convert_element_type_185 = None
        permute_242 = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        mm_49 = torch.ops.aten.mm.default(view_283, permute_242);  permute_242 = None
        permute_243 = torch.ops.aten.permute.default(view_283, [1, 0])
        var_mean_15 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_102 = var_mean_15[0]
        getitem_103 = var_mean_15[1];  var_mean_15 = None
        add_64 = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_63, getitem_103);  add_63 = getitem_103 = None
        mul_65 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        mul_66 = torch.ops.aten.mul.Tensor(mul_65, primals_95)
        add_65 = torch.ops.aten.add.Tensor(mul_66, primals_96);  mul_66 = primals_96 = None
        convert_element_type_176 = torch.ops.prims.convert_element_type.default(primals_98, torch.float16);  primals_98 = None
        convert_element_type_177 = torch.ops.prims.convert_element_type.default(primals_97, torch.float16);  primals_97 = None
        convert_element_type_178 = torch.ops.prims.convert_element_type.default(add_65, torch.float16);  add_65 = None
        view_119 = torch.ops.aten.view.default(convert_element_type_178, [20480, 768]);  convert_element_type_178 = None
        permute_89 = torch.ops.aten.permute.default(convert_element_type_177, [1, 0]);  convert_element_type_177 = None
        addmm_22 = torch.ops.aten.addmm.default(convert_element_type_176, view_119, permute_89);  convert_element_type_176 = None
        view_120 = torch.ops.aten.view.default(addmm_22, [20, 1024, 1536]);  addmm_22 = None
        convert_element_type_182 = torch.ops.prims.convert_element_type.default(view_120, torch.float32);  view_120 = None
        mul_67 = torch.ops.aten.mul.Tensor(convert_element_type_182, 0.5)
        mul_68 = torch.ops.aten.mul.Tensor(convert_element_type_182, 0.7071067811865476)
        erf_7 = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_66 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_67, add_66);  mul_67 = None
        convert_element_type_183 = torch.ops.prims.convert_element_type.default(mul_69, torch.float16);  mul_69 = None
        view_121 = torch.ops.aten.view.default(convert_element_type_183, [20480, 1536]);  convert_element_type_183 = None
        mm_50 = torch.ops.aten.mm.default(permute_243, view_121);  permute_243 = view_121 = None
        sum_51 = torch.ops.aten.sum.dim_IntList(view_283, [0], True, dtype = torch.float32);  view_283 = None
        view_284 = torch.ops.aten.view.default(sum_51, [768]);  sum_51 = None
        view_285 = torch.ops.aten.view.default(mm_49, [20, 1024, 1536]);  mm_49 = None
        convert_element_type_468 = torch.ops.prims.convert_element_type.default(mm_50, torch.float32);  mm_50 = None
        convert_element_type_default_32 = torch.ops.prims.convert_element_type.default(view_284, torch.float32);  view_284 = None
        convert_element_type_470 = torch.ops.prims.convert_element_type.default(view_285, torch.float32);  view_285 = None
        mul_211 = torch.ops.aten.mul.Tensor(add_66, 0.5);  add_66 = None
        mul_212 = torch.ops.aten.mul.Tensor(convert_element_type_182, convert_element_type_182)
        mul_213 = torch.ops.aten.mul.Tensor(mul_212, -0.5);  mul_212 = None
        exp_4 = torch.ops.aten.exp.default(mul_213);  mul_213 = None
        mul_214 = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
        mul_215 = torch.ops.aten.mul.Tensor(convert_element_type_182, mul_214);  convert_element_type_182 = mul_214 = None
        add_142 = torch.ops.aten.add.Tensor(mul_211, mul_215);  mul_211 = mul_215 = None
        mul_216 = torch.ops.aten.mul.Tensor(convert_element_type_470, add_142);  convert_element_type_470 = add_142 = None
        convert_element_type_472 = torch.ops.prims.convert_element_type.default(mul_216, torch.float16);  mul_216 = None
        view_286 = torch.ops.aten.view.default(convert_element_type_472, [20480, 1536]);  convert_element_type_472 = None
        permute_246 = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        mm_51 = torch.ops.aten.mm.default(view_286, permute_246);  permute_246 = None
        permute_247 = torch.ops.aten.permute.default(view_286, [1, 0])
        mm_52 = torch.ops.aten.mm.default(permute_247, view_119);  permute_247 = view_119 = None
        sum_52 = torch.ops.aten.sum.dim_IntList(view_286, [0], True, dtype = torch.float32);  view_286 = None
        view_287 = torch.ops.aten.view.default(sum_52, [1536]);  sum_52 = None
        view_288 = torch.ops.aten.view.default(mm_51, [20, 1024, 768]);  mm_51 = None
        convert_element_type_478 = torch.ops.prims.convert_element_type.default(view_288, torch.float32);  view_288 = None
        convert_element_type_479 = torch.ops.prims.convert_element_type.default(mm_52, torch.float32);  mm_52 = None
        convert_element_type_default_31 = torch.ops.prims.convert_element_type.default(view_287, torch.float32);  view_287 = None
        mul_218 = torch.ops.aten.mul.Tensor(convert_element_type_478, primals_95);  primals_95 = None
        mul_219 = torch.ops.aten.mul.Tensor(mul_218, 768)
        sum_53 = torch.ops.aten.sum.dim_IntList(mul_218, [2], True)
        mul_220 = torch.ops.aten.mul.Tensor(mul_218, mul_65);  mul_218 = None
        sum_54 = torch.ops.aten.sum.dim_IntList(mul_220, [2], True);  mul_220 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_65, sum_54);  sum_54 = None
        sub_56 = torch.ops.aten.sub.Tensor(mul_219, sum_53);  mul_219 = sum_53 = None
        sub_57 = torch.ops.aten.sub.Tensor(sub_56, mul_221);  sub_56 = mul_221 = None
        div_8 = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
        mul_222 = torch.ops.aten.mul.Tensor(div_8, sub_57);  div_8 = sub_57 = None
        mul_223 = torch.ops.aten.mul.Tensor(convert_element_type_478, mul_65);  mul_65 = None
        sum_55 = torch.ops.aten.sum.dim_IntList(mul_223, [0, 1]);  mul_223 = None
        sum_56 = torch.ops.aten.sum.dim_IntList(convert_element_type_478, [0, 1]);  convert_element_type_478 = None
        add_143 = torch.ops.aten.add.Tensor(add_140, mul_222);  add_140 = mul_222 = None
        convert_element_type_481 = torch.ops.prims.convert_element_type.default(add_143, torch.float16)
        permute_250 = torch.ops.aten.permute.default(convert_element_type_481, [1, 0, 2]);  convert_element_type_481 = None
        clone_42 = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None
        view_289 = torch.ops.aten.view.default(clone_42, [20480, 768]);  clone_42 = None
        convert_element_type_172 = torch.ops.prims.convert_element_type.default(primals_93, torch.float16);  primals_93 = None
        permute_87 = torch.ops.aten.permute.default(convert_element_type_172, [1, 0]);  convert_element_type_172 = None
        permute_251 = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        mm_53 = torch.ops.aten.mm.default(view_289, permute_251);  permute_251 = None
        permute_252 = torch.ops.aten.permute.default(view_289, [1, 0])
        var_mean_14 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_91 = var_mean_14[0]
        getitem_92 = var_mean_14[1];  var_mean_14 = None
        add_60 = torch.ops.aten.add.Tensor(getitem_91, 1e-05);  getitem_91 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_59, getitem_92);  add_59 = getitem_92 = None
        mul_63 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        mul_64 = torch.ops.aten.mul.Tensor(mul_63, primals_89)
        add_61 = torch.ops.aten.add.Tensor(mul_64, primals_90);  mul_64 = primals_90 = None
        permute_80 = torch.ops.aten.permute.default(add_61, [1, 0, 2]);  add_61 = None
        convert_element_type_166 = torch.ops.prims.convert_element_type.default(primals_91, torch.float16);  primals_91 = None
        convert_element_type_167 = torch.ops.prims.convert_element_type.default(primals_92, torch.float16);  primals_92 = None
        convert_element_type_168 = torch.ops.prims.convert_element_type.default(permute_80, torch.float16);  permute_80 = None
        permute_81 = torch.ops.aten.permute.default(convert_element_type_167, [1, 0]);  convert_element_type_167 = None
        clone_16 = torch.ops.aten.clone.default(convert_element_type_168, memory_format = torch.contiguous_format);  convert_element_type_168 = None
        view_108 = torch.ops.aten.view.default(clone_16, [20480, 768]);  clone_16 = None
        mm_8 = torch.ops.aten.mm.default(view_108, permute_81)
        view_109 = torch.ops.aten.view.default(mm_8, [1024, 20, 2304]);  mm_8 = None
        add_62 = torch.ops.aten.add.Tensor(view_109, convert_element_type_166);  view_109 = convert_element_type_166 = None
        view_110 = torch.ops.aten.view.default(add_62, [1024, 20, 3, 768]);  add_62 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(view_110, 0);  view_110 = None
        permute_82 = torch.ops.aten.permute.default(unsqueeze_13, [3, 1, 2, 0, 4]);  unsqueeze_13 = None
        squeeze_7 = torch.ops.aten.squeeze.dim(permute_82, -2);  permute_82 = None
        clone_17 = torch.ops.aten.clone.default(squeeze_7, memory_format = torch.contiguous_format);  squeeze_7 = None
        select_21 = torch.ops.aten.select.int(clone_17, 0, 0)
        select_22 = torch.ops.aten.select.int(clone_17, 0, 1)
        select_23 = torch.ops.aten.select.int(clone_17, 0, 2);  clone_17 = None
        view_111 = torch.ops.aten.view.default(select_21, [1024, 160, 96]);  select_21 = None
        permute_83 = torch.ops.aten.permute.default(view_111, [1, 0, 2]);  view_111 = None
        view_112 = torch.ops.aten.view.default(select_22, [1024, 160, 96]);  select_22 = None
        permute_84 = torch.ops.aten.permute.default(view_112, [1, 0, 2]);  view_112 = None
        view_113 = torch.ops.aten.view.default(select_23, [1024, 160, 96]);  select_23 = None
        permute_85 = torch.ops.aten.permute.default(view_113, [1, 0, 2]);  view_113 = None
        view_114 = torch.ops.aten.view.default(permute_83, [20, 8, 1024, 96]);  permute_83 = None
        view_115 = torch.ops.aten.view.default(permute_84, [20, 8, 1024, 96]);  permute_84 = None
        view_116 = torch.ops.aten.view.default(permute_85, [20, 8, 1024, 96]);  permute_85 = None
        graphsafe_run_with_rng_state_7 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_114, view_115, view_116, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_7);  bwd_rng_state_7 = None
        getitem_93 = graphsafe_run_with_rng_state_7[0]
        getitem_94 = graphsafe_run_with_rng_state_7[1]
        getitem_99 = graphsafe_run_with_rng_state_7[6]
        getitem_100 = graphsafe_run_with_rng_state_7[7];  graphsafe_run_with_rng_state_7 = None
        permute_86 = torch.ops.aten.permute.default(getitem_93, [2, 0, 1, 3])
        view_117 = torch.ops.aten.view.default(permute_86, [20480, 768]);  permute_86 = None
        mm_54 = torch.ops.aten.mm.default(permute_252, view_117);  permute_252 = view_117 = None
        sum_57 = torch.ops.aten.sum.dim_IntList(view_289, [0], True, dtype = torch.float32);  view_289 = None
        view_290 = torch.ops.aten.view.default(sum_57, [768]);  sum_57 = None
        convert_element_type_487 = torch.ops.prims.convert_element_type.default(mm_54, torch.float32);  mm_54 = None
        convert_element_type_default_30 = torch.ops.prims.convert_element_type.default(view_290, torch.float32);  view_290 = None
        view_291 = torch.ops.aten.view.default(mm_53, [1024, 20, 8, 96]);  mm_53 = None
        permute_255 = torch.ops.aten.permute.default(view_291, [1, 2, 0, 3]);  view_291 = None
        _scaled_dot_product_flash_attention_backward_4 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_255, view_114, view_115, view_116, getitem_93, getitem_94, None, None, 1024, 1024, 0.2, False, getitem_99, getitem_100, scale = 0.10206207261596577);  permute_255 = view_114 = view_115 = view_116 = getitem_93 = getitem_94 = getitem_99 = getitem_100 = None
        getitem_168 = _scaled_dot_product_flash_attention_backward_4[0]
        getitem_169 = _scaled_dot_product_flash_attention_backward_4[1]
        getitem_170 = _scaled_dot_product_flash_attention_backward_4[2];  _scaled_dot_product_flash_attention_backward_4 = None
        view_292 = torch.ops.aten.view.default(getitem_170, [160, 1024, 96]);  getitem_170 = None
        view_293 = torch.ops.aten.view.default(getitem_169, [160, 1024, 96]);  getitem_169 = None
        view_294 = torch.ops.aten.view.default(getitem_168, [160, 1024, 96]);  getitem_168 = None
        permute_256 = torch.ops.aten.permute.default(view_292, [1, 0, 2]);  view_292 = None
        view_295 = torch.ops.aten.view.default(permute_256, [1024, 20, 768]);  permute_256 = None
        permute_257 = torch.ops.aten.permute.default(view_293, [1, 0, 2]);  view_293 = None
        view_296 = torch.ops.aten.view.default(permute_257, [1024, 20, 768]);  permute_257 = None
        permute_258 = torch.ops.aten.permute.default(view_294, [1, 0, 2]);  view_294 = None
        view_297 = torch.ops.aten.view.default(permute_258, [1024, 20, 768]);  permute_258 = None
        select_scatter_12 = torch.ops.aten.select_scatter.default(full_default_6, view_295, 0, 2);  view_295 = None
        select_scatter_13 = torch.ops.aten.select_scatter.default(full_default_6, view_296, 0, 1);  view_296 = None
        add_144 = torch.ops.aten.add.Tensor(select_scatter_12, select_scatter_13);  select_scatter_12 = select_scatter_13 = None
        select_scatter_14 = torch.ops.aten.select_scatter.default(full_default_6, view_297, 0, 0);  view_297 = None
        add_145 = torch.ops.aten.add.Tensor(add_144, select_scatter_14);  add_144 = select_scatter_14 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(add_145, 3);  add_145 = None
        permute_259 = torch.ops.aten.permute.default(unsqueeze_34, [3, 1, 2, 0, 4]);  unsqueeze_34 = None
        squeeze_16 = torch.ops.aten.squeeze.dim(permute_259, 0);  permute_259 = None
        clone_43 = torch.ops.aten.clone.default(squeeze_16, memory_format = torch.contiguous_format);  squeeze_16 = None
        view_298 = torch.ops.aten.view.default(clone_43, [1024, 20, 2304]);  clone_43 = None
        sum_58 = torch.ops.aten.sum.dim_IntList(view_298, [0, 1], True, dtype = torch.float32)
        view_299 = torch.ops.aten.view.default(sum_58, [2304]);  sum_58 = None
        view_300 = torch.ops.aten.view.default(view_298, [20480, 2304]);  view_298 = None
        permute_260 = torch.ops.aten.permute.default(view_300, [1, 0])
        mm_55 = torch.ops.aten.mm.default(permute_260, view_108);  permute_260 = view_108 = None
        permute_262 = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        mm_56 = torch.ops.aten.mm.default(view_300, permute_262);  view_300 = permute_262 = None
        view_301 = torch.ops.aten.view.default(mm_56, [1024, 20, 768]);  mm_56 = None
        convert_element_type_494 = torch.ops.prims.convert_element_type.default(view_301, torch.float32);  view_301 = None
        convert_element_type_495 = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_default_29 = torch.ops.prims.convert_element_type.default(view_299, torch.float32);  view_299 = None
        permute_264 = torch.ops.aten.permute.default(convert_element_type_494, [1, 0, 2]);  convert_element_type_494 = None
        mul_225 = torch.ops.aten.mul.Tensor(permute_264, primals_89);  primals_89 = None
        mul_226 = torch.ops.aten.mul.Tensor(mul_225, 768)
        sum_59 = torch.ops.aten.sum.dim_IntList(mul_225, [2], True)
        mul_227 = torch.ops.aten.mul.Tensor(mul_225, mul_63);  mul_225 = None
        sum_60 = torch.ops.aten.sum.dim_IntList(mul_227, [2], True);  mul_227 = None
        mul_228 = torch.ops.aten.mul.Tensor(mul_63, sum_60);  sum_60 = None
        sub_59 = torch.ops.aten.sub.Tensor(mul_226, sum_59);  mul_226 = sum_59 = None
        sub_60 = torch.ops.aten.sub.Tensor(sub_59, mul_228);  sub_59 = mul_228 = None
        div_9 = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
        mul_229 = torch.ops.aten.mul.Tensor(div_9, sub_60);  div_9 = sub_60 = None
        mul_230 = torch.ops.aten.mul.Tensor(permute_264, mul_63);  mul_63 = None
        sum_61 = torch.ops.aten.sum.dim_IntList(mul_230, [0, 1]);  mul_230 = None
        sum_62 = torch.ops.aten.sum.dim_IntList(permute_264, [0, 1]);  permute_264 = None
        add_146 = torch.ops.aten.add.Tensor(add_143, mul_229);  add_143 = mul_229 = None
        convert_element_type_497 = torch.ops.prims.convert_element_type.default(add_146, torch.float16)
        inductor_lookup_seed_default_6 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_5 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        convert_element_type_default_57 = torch.ops.prims.convert_element_type.default(inductor_random_default_5, torch.float16);  inductor_random_default_5 = None
        gt_6 = torch.ops.aten.gt.Scalar(convert_element_type_default_57, 0.2);  convert_element_type_default_57 = None
        convert_element_type_498 = torch.ops.prims.convert_element_type.default(gt_6, torch.float16);  gt_6 = None
        mul_231 = torch.ops.aten.mul.Tensor(convert_element_type_498, 1.25);  convert_element_type_498 = None
        mul_232 = torch.ops.aten.mul.Tensor(convert_element_type_497, mul_231);  convert_element_type_497 = mul_231 = None
        view_302 = torch.ops.aten.view.default(mul_232, [20480, 768]);  mul_232 = None
        convert_element_type_162 = torch.ops.prims.convert_element_type.default(primals_87, torch.float16);  primals_87 = None
        permute_79 = torch.ops.aten.permute.default(convert_element_type_162, [1, 0]);  convert_element_type_162 = None
        permute_265 = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        mm_57 = torch.ops.aten.mm.default(view_302, permute_265);  permute_265 = None
        permute_266 = torch.ops.aten.permute.default(view_302, [1, 0])
        var_mean_13 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
        getitem_89 = var_mean_13[0]
        getitem_90 = var_mean_13[1];  var_mean_13 = None
        add_56 = torch.ops.aten.add.Tensor(getitem_89, 1e-05);  getitem_89 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_55, getitem_90);  add_55 = getitem_90 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, primals_83)
        add_57 = torch.ops.aten.add.Tensor(mul_57, primals_84);  mul_57 = primals_84 = None
        convert_element_type_153 = torch.ops.prims.convert_element_type.default(primals_86, torch.float16);  primals_86 = None
        convert_element_type_154 = torch.ops.prims.convert_element_type.default(primals_85, torch.float16);  primals_85 = None
        convert_element_type_155 = torch.ops.prims.convert_element_type.default(add_57, torch.float16);  add_57 = None
        view_104 = torch.ops.aten.view.default(convert_element_type_155, [20480, 768]);  convert_element_type_155 = None
        permute_78 = torch.ops.aten.permute.default(convert_element_type_154, [1, 0]);  convert_element_type_154 = None
        addmm_19 = torch.ops.aten.addmm.default(convert_element_type_153, view_104, permute_78);  convert_element_type_153 = None
        view_105 = torch.ops.aten.view.default(addmm_19, [20, 1024, 1536]);  addmm_19 = None
        convert_element_type_159 = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None
        mul_58 = torch.ops.aten.mul.Tensor(convert_element_type_159, 0.5)
        mul_59 = torch.ops.aten.mul.Tensor(convert_element_type_159, 0.7071067811865476)
        erf_6 = torch.ops.aten.erf.default(mul_59);  mul_59 = None
        add_58 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_60 = torch.ops.aten.mul.Tensor(mul_58, add_58);  mul_58 = None
        convert_element_type_160 = torch.ops.prims.convert_element_type.default(mul_60, torch.float16);  mul_60 = None
        view_106 = torch.ops.aten.view.default(convert_element_type_160, [20480, 1536]);  convert_element_type_160 = None
        mm_58 = torch.ops.aten.mm.default(permute_266, view_106);  permute_266 = view_106 = None
        sum_63 = torch.ops.aten.sum.dim_IntList(view_302, [0], True, dtype = torch.float32);  view_302 = None
        view_303 = torch.ops.aten.view.default(sum_63, [768]);  sum_63 = None
        view_304 = torch.ops.aten.view.default(mm_57, [20, 1024, 1536]);  mm_57 = None
        convert_element_type_504 = torch.ops.prims.convert_element_type.default(mm_58, torch.float32);  mm_58 = None
        convert_element_type_default_28 = torch.ops.prims.convert_element_type.default(view_303, torch.float32);  view_303 = None
        convert_element_type_506 = torch.ops.prims.convert_element_type.default(view_304, torch.float32);  view_304 = None
        mul_234 = torch.ops.aten.mul.Tensor(add_58, 0.5);  add_58 = None
        mul_235 = torch.ops.aten.mul.Tensor(convert_element_type_159, convert_element_type_159)
        mul_236 = torch.ops.aten.mul.Tensor(mul_235, -0.5);  mul_235 = None
        exp_5 = torch.ops.aten.exp.default(mul_236);  mul_236 = None
        mul_237 = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
        mul_238 = torch.ops.aten.mul.Tensor(convert_element_type_159, mul_237);  convert_element_type_159 = mul_237 = None
        add_148 = torch.ops.aten.add.Tensor(mul_234, mul_238);  mul_234 = mul_238 = None
        mul_239 = torch.ops.aten.mul.Tensor(convert_element_type_506, add_148);  convert_element_type_506 = add_148 = None
        convert_element_type_508 = torch.ops.prims.convert_element_type.default(mul_239, torch.float16);  mul_239 = None
        view_305 = torch.ops.aten.view.default(convert_element_type_508, [20480, 1536]);  convert_element_type_508 = None
        permute_269 = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        mm_59 = torch.ops.aten.mm.default(view_305, permute_269);  permute_269 = None
        permute_270 = torch.ops.aten.permute.default(view_305, [1, 0])
        mm_60 = torch.ops.aten.mm.default(permute_270, view_104);  permute_270 = view_104 = None
        sum_64 = torch.ops.aten.sum.dim_IntList(view_305, [0], True, dtype = torch.float32);  view_305 = None
        view_306 = torch.ops.aten.view.default(sum_64, [1536]);  sum_64 = None
        view_307 = torch.ops.aten.view.default(mm_59, [20, 1024, 768]);  mm_59 = None
        convert_element_type_514 = torch.ops.prims.convert_element_type.default(view_307, torch.float32);  view_307 = None
        convert_element_type_515 = torch.ops.prims.convert_element_type.default(mm_60, torch.float32);  mm_60 = None
        convert_element_type_default_27 = torch.ops.prims.convert_element_type.default(view_306, torch.float32);  view_306 = None
        mul_241 = torch.ops.aten.mul.Tensor(convert_element_type_514, primals_83);  primals_83 = None
        mul_242 = torch.ops.aten.mul.Tensor(mul_241, 768)
        sum_65 = torch.ops.aten.sum.dim_IntList(mul_241, [2], True)
        mul_243 = torch.ops.aten.mul.Tensor(mul_241, mul_56);  mul_241 = None
        sum_66 = torch.ops.aten.sum.dim_IntList(mul_243, [2], True);  mul_243 = None
        mul_244 = torch.ops.aten.mul.Tensor(mul_56, sum_66);  sum_66 = None
        sub_62 = torch.ops.aten.sub.Tensor(mul_242, sum_65);  mul_242 = sum_65 = None
        sub_63 = torch.ops.aten.sub.Tensor(sub_62, mul_244);  sub_62 = mul_244 = None
        div_10 = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        mul_245 = torch.ops.aten.mul.Tensor(div_10, sub_63);  div_10 = sub_63 = None
        mul_246 = torch.ops.aten.mul.Tensor(convert_element_type_514, mul_56);  mul_56 = None
        sum_67 = torch.ops.aten.sum.dim_IntList(mul_246, [0, 1]);  mul_246 = None
        sum_68 = torch.ops.aten.sum.dim_IntList(convert_element_type_514, [0, 1]);  convert_element_type_514 = None
        add_149 = torch.ops.aten.add.Tensor(add_146, mul_245);  add_146 = mul_245 = None
        convert_element_type_517 = torch.ops.prims.convert_element_type.default(add_149, torch.float16)
        permute_273 = torch.ops.aten.permute.default(convert_element_type_517, [1, 0, 2]);  convert_element_type_517 = None
        clone_45 = torch.ops.aten.clone.default(permute_273, memory_format = torch.contiguous_format);  permute_273 = None
        view_308 = torch.ops.aten.view.default(clone_45, [20480, 768]);  clone_45 = None
        convert_element_type_149 = torch.ops.prims.convert_element_type.default(primals_81, torch.float16);  primals_81 = None
        permute_76 = torch.ops.aten.permute.default(convert_element_type_149, [1, 0]);  convert_element_type_149 = None
        permute_274 = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        mm_61 = torch.ops.aten.mm.default(view_308, permute_274);  permute_274 = None
        permute_275 = torch.ops.aten.permute.default(view_308, [1, 0])
        var_mean_12 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_78 = var_mean_12[0]
        getitem_79 = var_mean_12[1];  var_mean_12 = None
        add_52 = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_51, getitem_79);  add_51 = getitem_79 = None
        mul_54 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_54, primals_77)
        add_53 = torch.ops.aten.add.Tensor(mul_55, primals_78);  mul_55 = primals_78 = None
        permute_69 = torch.ops.aten.permute.default(add_53, [1, 0, 2]);  add_53 = None
        convert_element_type_143 = torch.ops.prims.convert_element_type.default(primals_79, torch.float16);  primals_79 = None
        convert_element_type_144 = torch.ops.prims.convert_element_type.default(primals_80, torch.float16);  primals_80 = None
        convert_element_type_145 = torch.ops.prims.convert_element_type.default(permute_69, torch.float16);  permute_69 = None
        permute_70 = torch.ops.aten.permute.default(convert_element_type_144, [1, 0]);  convert_element_type_144 = None
        clone_14 = torch.ops.aten.clone.default(convert_element_type_145, memory_format = torch.contiguous_format);  convert_element_type_145 = None
        view_93 = torch.ops.aten.view.default(clone_14, [20480, 768]);  clone_14 = None
        mm_7 = torch.ops.aten.mm.default(view_93, permute_70)
        view_94 = torch.ops.aten.view.default(mm_7, [1024, 20, 2304]);  mm_7 = None
        add_54 = torch.ops.aten.add.Tensor(view_94, convert_element_type_143);  view_94 = convert_element_type_143 = None
        view_95 = torch.ops.aten.view.default(add_54, [1024, 20, 3, 768]);  add_54 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(view_95, 0);  view_95 = None
        permute_71 = torch.ops.aten.permute.default(unsqueeze_12, [3, 1, 2, 0, 4]);  unsqueeze_12 = None
        squeeze_6 = torch.ops.aten.squeeze.dim(permute_71, -2);  permute_71 = None
        clone_15 = torch.ops.aten.clone.default(squeeze_6, memory_format = torch.contiguous_format);  squeeze_6 = None
        select_18 = torch.ops.aten.select.int(clone_15, 0, 0)
        select_19 = torch.ops.aten.select.int(clone_15, 0, 1)
        select_20 = torch.ops.aten.select.int(clone_15, 0, 2);  clone_15 = None
        view_96 = torch.ops.aten.view.default(select_18, [1024, 160, 96]);  select_18 = None
        permute_72 = torch.ops.aten.permute.default(view_96, [1, 0, 2]);  view_96 = None
        view_97 = torch.ops.aten.view.default(select_19, [1024, 160, 96]);  select_19 = None
        permute_73 = torch.ops.aten.permute.default(view_97, [1, 0, 2]);  view_97 = None
        view_98 = torch.ops.aten.view.default(select_20, [1024, 160, 96]);  select_20 = None
        permute_74 = torch.ops.aten.permute.default(view_98, [1, 0, 2]);  view_98 = None
        view_99 = torch.ops.aten.view.default(permute_72, [20, 8, 1024, 96]);  permute_72 = None
        view_100 = torch.ops.aten.view.default(permute_73, [20, 8, 1024, 96]);  permute_73 = None
        view_101 = torch.ops.aten.view.default(permute_74, [20, 8, 1024, 96]);  permute_74 = None
        graphsafe_run_with_rng_state_6 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_99, view_100, view_101, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_6);  bwd_rng_state_6 = None
        getitem_80 = graphsafe_run_with_rng_state_6[0]
        getitem_81 = graphsafe_run_with_rng_state_6[1]
        getitem_86 = graphsafe_run_with_rng_state_6[6]
        getitem_87 = graphsafe_run_with_rng_state_6[7];  graphsafe_run_with_rng_state_6 = None
        permute_75 = torch.ops.aten.permute.default(getitem_80, [2, 0, 1, 3])
        view_102 = torch.ops.aten.view.default(permute_75, [20480, 768]);  permute_75 = None
        mm_62 = torch.ops.aten.mm.default(permute_275, view_102);  permute_275 = view_102 = None
        sum_69 = torch.ops.aten.sum.dim_IntList(view_308, [0], True, dtype = torch.float32);  view_308 = None
        view_309 = torch.ops.aten.view.default(sum_69, [768]);  sum_69 = None
        convert_element_type_523 = torch.ops.prims.convert_element_type.default(mm_62, torch.float32);  mm_62 = None
        convert_element_type_default_26 = torch.ops.prims.convert_element_type.default(view_309, torch.float32);  view_309 = None
        view_310 = torch.ops.aten.view.default(mm_61, [1024, 20, 8, 96]);  mm_61 = None
        permute_278 = torch.ops.aten.permute.default(view_310, [1, 2, 0, 3]);  view_310 = None
        _scaled_dot_product_flash_attention_backward_5 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_278, view_99, view_100, view_101, getitem_80, getitem_81, None, None, 1024, 1024, 0.2, False, getitem_86, getitem_87, scale = 0.10206207261596577);  permute_278 = view_99 = view_100 = view_101 = getitem_80 = getitem_81 = getitem_86 = getitem_87 = None
        getitem_171 = _scaled_dot_product_flash_attention_backward_5[0]
        getitem_172 = _scaled_dot_product_flash_attention_backward_5[1]
        getitem_173 = _scaled_dot_product_flash_attention_backward_5[2];  _scaled_dot_product_flash_attention_backward_5 = None
        view_311 = torch.ops.aten.view.default(getitem_173, [160, 1024, 96]);  getitem_173 = None
        view_312 = torch.ops.aten.view.default(getitem_172, [160, 1024, 96]);  getitem_172 = None
        view_313 = torch.ops.aten.view.default(getitem_171, [160, 1024, 96]);  getitem_171 = None
        permute_279 = torch.ops.aten.permute.default(view_311, [1, 0, 2]);  view_311 = None
        view_314 = torch.ops.aten.view.default(permute_279, [1024, 20, 768]);  permute_279 = None
        permute_280 = torch.ops.aten.permute.default(view_312, [1, 0, 2]);  view_312 = None
        view_315 = torch.ops.aten.view.default(permute_280, [1024, 20, 768]);  permute_280 = None
        permute_281 = torch.ops.aten.permute.default(view_313, [1, 0, 2]);  view_313 = None
        view_316 = torch.ops.aten.view.default(permute_281, [1024, 20, 768]);  permute_281 = None
        select_scatter_15 = torch.ops.aten.select_scatter.default(full_default_6, view_314, 0, 2);  view_314 = None
        select_scatter_16 = torch.ops.aten.select_scatter.default(full_default_6, view_315, 0, 1);  view_315 = None
        add_150 = torch.ops.aten.add.Tensor(select_scatter_15, select_scatter_16);  select_scatter_15 = select_scatter_16 = None
        select_scatter_17 = torch.ops.aten.select_scatter.default(full_default_6, view_316, 0, 0);  view_316 = None
        add_151 = torch.ops.aten.add.Tensor(add_150, select_scatter_17);  add_150 = select_scatter_17 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(add_151, 3);  add_151 = None
        permute_282 = torch.ops.aten.permute.default(unsqueeze_35, [3, 1, 2, 0, 4]);  unsqueeze_35 = None
        squeeze_17 = torch.ops.aten.squeeze.dim(permute_282, 0);  permute_282 = None
        clone_46 = torch.ops.aten.clone.default(squeeze_17, memory_format = torch.contiguous_format);  squeeze_17 = None
        view_317 = torch.ops.aten.view.default(clone_46, [1024, 20, 2304]);  clone_46 = None
        sum_70 = torch.ops.aten.sum.dim_IntList(view_317, [0, 1], True, dtype = torch.float32)
        view_318 = torch.ops.aten.view.default(sum_70, [2304]);  sum_70 = None
        view_319 = torch.ops.aten.view.default(view_317, [20480, 2304]);  view_317 = None
        permute_283 = torch.ops.aten.permute.default(view_319, [1, 0])
        mm_63 = torch.ops.aten.mm.default(permute_283, view_93);  permute_283 = view_93 = None
        permute_285 = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        mm_64 = torch.ops.aten.mm.default(view_319, permute_285);  view_319 = permute_285 = None
        view_320 = torch.ops.aten.view.default(mm_64, [1024, 20, 768]);  mm_64 = None
        convert_element_type_530 = torch.ops.prims.convert_element_type.default(view_320, torch.float32);  view_320 = None
        convert_element_type_531 = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_default_25 = torch.ops.prims.convert_element_type.default(view_318, torch.float32);  view_318 = None
        permute_287 = torch.ops.aten.permute.default(convert_element_type_530, [1, 0, 2]);  convert_element_type_530 = None
        mul_248 = torch.ops.aten.mul.Tensor(permute_287, primals_77);  primals_77 = None
        mul_249 = torch.ops.aten.mul.Tensor(mul_248, 768)
        sum_71 = torch.ops.aten.sum.dim_IntList(mul_248, [2], True)
        mul_250 = torch.ops.aten.mul.Tensor(mul_248, mul_54);  mul_248 = None
        sum_72 = torch.ops.aten.sum.dim_IntList(mul_250, [2], True);  mul_250 = None
        mul_251 = torch.ops.aten.mul.Tensor(mul_54, sum_72);  sum_72 = None
        sub_65 = torch.ops.aten.sub.Tensor(mul_249, sum_71);  mul_249 = sum_71 = None
        sub_66 = torch.ops.aten.sub.Tensor(sub_65, mul_251);  sub_65 = mul_251 = None
        div_11 = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        mul_252 = torch.ops.aten.mul.Tensor(div_11, sub_66);  div_11 = sub_66 = None
        mul_253 = torch.ops.aten.mul.Tensor(permute_287, mul_54);  mul_54 = None
        sum_73 = torch.ops.aten.sum.dim_IntList(mul_253, [0, 1]);  mul_253 = None
        sum_74 = torch.ops.aten.sum.dim_IntList(permute_287, [0, 1]);  permute_287 = None
        add_152 = torch.ops.aten.add.Tensor(add_149, mul_252);  add_149 = mul_252 = None
        convert_element_type_533 = torch.ops.prims.convert_element_type.default(add_152, torch.float16)
        inductor_lookup_seed_default_5 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_6 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        convert_element_type_default_58 = torch.ops.prims.convert_element_type.default(inductor_random_default_6, torch.float16);  inductor_random_default_6 = None
        gt_5 = torch.ops.aten.gt.Scalar(convert_element_type_default_58, 0.2);  convert_element_type_default_58 = None
        convert_element_type_534 = torch.ops.prims.convert_element_type.default(gt_5, torch.float16);  gt_5 = None
        mul_254 = torch.ops.aten.mul.Tensor(convert_element_type_534, 1.25);  convert_element_type_534 = None
        mul_255 = torch.ops.aten.mul.Tensor(convert_element_type_533, mul_254);  convert_element_type_533 = mul_254 = None
        view_321 = torch.ops.aten.view.default(mul_255, [20480, 768]);  mul_255 = None
        convert_element_type_139 = torch.ops.prims.convert_element_type.default(primals_75, torch.float16);  primals_75 = None
        permute_68 = torch.ops.aten.permute.default(convert_element_type_139, [1, 0]);  convert_element_type_139 = None
        permute_288 = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        mm_65 = torch.ops.aten.mm.default(view_321, permute_288);  permute_288 = None
        permute_289 = torch.ops.aten.permute.default(view_321, [1, 0])
        var_mean_11 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
        getitem_76 = var_mean_11[0]
        getitem_77 = var_mean_11[1];  var_mean_11 = None
        add_48 = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_47, getitem_77);  add_47 = getitem_77 = None
        mul_47 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_47, primals_71)
        add_49 = torch.ops.aten.add.Tensor(mul_48, primals_72);  mul_48 = primals_72 = None
        convert_element_type_130 = torch.ops.prims.convert_element_type.default(primals_74, torch.float16);  primals_74 = None
        convert_element_type_131 = torch.ops.prims.convert_element_type.default(primals_73, torch.float16);  primals_73 = None
        convert_element_type_132 = torch.ops.prims.convert_element_type.default(add_49, torch.float16);  add_49 = None
        view_89 = torch.ops.aten.view.default(convert_element_type_132, [20480, 768]);  convert_element_type_132 = None
        permute_67 = torch.ops.aten.permute.default(convert_element_type_131, [1, 0]);  convert_element_type_131 = None
        addmm_16 = torch.ops.aten.addmm.default(convert_element_type_130, view_89, permute_67);  convert_element_type_130 = None
        view_90 = torch.ops.aten.view.default(addmm_16, [20, 1024, 1536]);  addmm_16 = None
        convert_element_type_136 = torch.ops.prims.convert_element_type.default(view_90, torch.float32);  view_90 = None
        mul_49 = torch.ops.aten.mul.Tensor(convert_element_type_136, 0.5)
        mul_50 = torch.ops.aten.mul.Tensor(convert_element_type_136, 0.7071067811865476)
        erf_5 = torch.ops.aten.erf.default(mul_50);  mul_50 = None
        add_50 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_51 = torch.ops.aten.mul.Tensor(mul_49, add_50);  mul_49 = None
        convert_element_type_137 = torch.ops.prims.convert_element_type.default(mul_51, torch.float16);  mul_51 = None
        view_91 = torch.ops.aten.view.default(convert_element_type_137, [20480, 1536]);  convert_element_type_137 = None
        mm_66 = torch.ops.aten.mm.default(permute_289, view_91);  permute_289 = view_91 = None
        sum_75 = torch.ops.aten.sum.dim_IntList(view_321, [0], True, dtype = torch.float32);  view_321 = None
        view_322 = torch.ops.aten.view.default(sum_75, [768]);  sum_75 = None
        view_323 = torch.ops.aten.view.default(mm_65, [20, 1024, 1536]);  mm_65 = None
        convert_element_type_540 = torch.ops.prims.convert_element_type.default(mm_66, torch.float32);  mm_66 = None
        convert_element_type_default_24 = torch.ops.prims.convert_element_type.default(view_322, torch.float32);  view_322 = None
        convert_element_type_542 = torch.ops.prims.convert_element_type.default(view_323, torch.float32);  view_323 = None
        mul_257 = torch.ops.aten.mul.Tensor(add_50, 0.5);  add_50 = None
        mul_258 = torch.ops.aten.mul.Tensor(convert_element_type_136, convert_element_type_136)
        mul_259 = torch.ops.aten.mul.Tensor(mul_258, -0.5);  mul_258 = None
        exp_6 = torch.ops.aten.exp.default(mul_259);  mul_259 = None
        mul_260 = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
        mul_261 = torch.ops.aten.mul.Tensor(convert_element_type_136, mul_260);  convert_element_type_136 = mul_260 = None
        add_154 = torch.ops.aten.add.Tensor(mul_257, mul_261);  mul_257 = mul_261 = None
        mul_262 = torch.ops.aten.mul.Tensor(convert_element_type_542, add_154);  convert_element_type_542 = add_154 = None
        convert_element_type_544 = torch.ops.prims.convert_element_type.default(mul_262, torch.float16);  mul_262 = None
        view_324 = torch.ops.aten.view.default(convert_element_type_544, [20480, 1536]);  convert_element_type_544 = None
        permute_292 = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        mm_67 = torch.ops.aten.mm.default(view_324, permute_292);  permute_292 = None
        permute_293 = torch.ops.aten.permute.default(view_324, [1, 0])
        mm_68 = torch.ops.aten.mm.default(permute_293, view_89);  permute_293 = view_89 = None
        sum_76 = torch.ops.aten.sum.dim_IntList(view_324, [0], True, dtype = torch.float32);  view_324 = None
        view_325 = torch.ops.aten.view.default(sum_76, [1536]);  sum_76 = None
        view_326 = torch.ops.aten.view.default(mm_67, [20, 1024, 768]);  mm_67 = None
        convert_element_type_550 = torch.ops.prims.convert_element_type.default(view_326, torch.float32);  view_326 = None
        convert_element_type_551 = torch.ops.prims.convert_element_type.default(mm_68, torch.float32);  mm_68 = None
        convert_element_type_default_23 = torch.ops.prims.convert_element_type.default(view_325, torch.float32);  view_325 = None
        mul_264 = torch.ops.aten.mul.Tensor(convert_element_type_550, primals_71);  primals_71 = None
        mul_265 = torch.ops.aten.mul.Tensor(mul_264, 768)
        sum_77 = torch.ops.aten.sum.dim_IntList(mul_264, [2], True)
        mul_266 = torch.ops.aten.mul.Tensor(mul_264, mul_47);  mul_264 = None
        sum_78 = torch.ops.aten.sum.dim_IntList(mul_266, [2], True);  mul_266 = None
        mul_267 = torch.ops.aten.mul.Tensor(mul_47, sum_78);  sum_78 = None
        sub_68 = torch.ops.aten.sub.Tensor(mul_265, sum_77);  mul_265 = sum_77 = None
        sub_69 = torch.ops.aten.sub.Tensor(sub_68, mul_267);  sub_68 = mul_267 = None
        div_12 = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        mul_268 = torch.ops.aten.mul.Tensor(div_12, sub_69);  div_12 = sub_69 = None
        mul_269 = torch.ops.aten.mul.Tensor(convert_element_type_550, mul_47);  mul_47 = None
        sum_79 = torch.ops.aten.sum.dim_IntList(mul_269, [0, 1]);  mul_269 = None
        sum_80 = torch.ops.aten.sum.dim_IntList(convert_element_type_550, [0, 1]);  convert_element_type_550 = None
        add_155 = torch.ops.aten.add.Tensor(add_152, mul_268);  add_152 = mul_268 = None
        convert_element_type_553 = torch.ops.prims.convert_element_type.default(add_155, torch.float16)
        permute_296 = torch.ops.aten.permute.default(convert_element_type_553, [1, 0, 2]);  convert_element_type_553 = None
        clone_48 = torch.ops.aten.clone.default(permute_296, memory_format = torch.contiguous_format);  permute_296 = None
        view_327 = torch.ops.aten.view.default(clone_48, [20480, 768]);  clone_48 = None
        convert_element_type_126 = torch.ops.prims.convert_element_type.default(primals_69, torch.float16);  primals_69 = None
        permute_65 = torch.ops.aten.permute.default(convert_element_type_126, [1, 0]);  convert_element_type_126 = None
        permute_297 = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_69 = torch.ops.aten.mm.default(view_327, permute_297);  permute_297 = None
        permute_298 = torch.ops.aten.permute.default(view_327, [1, 0])
        var_mean_10 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_65 = var_mean_10[0]
        getitem_66 = var_mean_10[1];  var_mean_10 = None
        add_44 = torch.ops.aten.add.Tensor(getitem_65, 1e-05);  getitem_65 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_43, getitem_66);  add_43 = getitem_66 = None
        mul_45 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_45, primals_65)
        add_45 = torch.ops.aten.add.Tensor(mul_46, primals_66);  mul_46 = primals_66 = None
        permute_58 = torch.ops.aten.permute.default(add_45, [1, 0, 2]);  add_45 = None
        convert_element_type_120 = torch.ops.prims.convert_element_type.default(primals_67, torch.float16);  primals_67 = None
        convert_element_type_121 = torch.ops.prims.convert_element_type.default(primals_68, torch.float16);  primals_68 = None
        convert_element_type_122 = torch.ops.prims.convert_element_type.default(permute_58, torch.float16);  permute_58 = None
        permute_59 = torch.ops.aten.permute.default(convert_element_type_121, [1, 0]);  convert_element_type_121 = None
        clone_12 = torch.ops.aten.clone.default(convert_element_type_122, memory_format = torch.contiguous_format);  convert_element_type_122 = None
        view_78 = torch.ops.aten.view.default(clone_12, [20480, 768]);  clone_12 = None
        mm_6 = torch.ops.aten.mm.default(view_78, permute_59)
        view_79 = torch.ops.aten.view.default(mm_6, [1024, 20, 2304]);  mm_6 = None
        add_46 = torch.ops.aten.add.Tensor(view_79, convert_element_type_120);  view_79 = convert_element_type_120 = None
        view_80 = torch.ops.aten.view.default(add_46, [1024, 20, 3, 768]);  add_46 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(view_80, 0);  view_80 = None
        permute_60 = torch.ops.aten.permute.default(unsqueeze_11, [3, 1, 2, 0, 4]);  unsqueeze_11 = None
        squeeze_5 = torch.ops.aten.squeeze.dim(permute_60, -2);  permute_60 = None
        clone_13 = torch.ops.aten.clone.default(squeeze_5, memory_format = torch.contiguous_format);  squeeze_5 = None
        select_15 = torch.ops.aten.select.int(clone_13, 0, 0)
        select_16 = torch.ops.aten.select.int(clone_13, 0, 1)
        select_17 = torch.ops.aten.select.int(clone_13, 0, 2);  clone_13 = None
        view_81 = torch.ops.aten.view.default(select_15, [1024, 160, 96]);  select_15 = None
        permute_61 = torch.ops.aten.permute.default(view_81, [1, 0, 2]);  view_81 = None
        view_82 = torch.ops.aten.view.default(select_16, [1024, 160, 96]);  select_16 = None
        permute_62 = torch.ops.aten.permute.default(view_82, [1, 0, 2]);  view_82 = None
        view_83 = torch.ops.aten.view.default(select_17, [1024, 160, 96]);  select_17 = None
        permute_63 = torch.ops.aten.permute.default(view_83, [1, 0, 2]);  view_83 = None
        view_84 = torch.ops.aten.view.default(permute_61, [20, 8, 1024, 96]);  permute_61 = None
        view_85 = torch.ops.aten.view.default(permute_62, [20, 8, 1024, 96]);  permute_62 = None
        view_86 = torch.ops.aten.view.default(permute_63, [20, 8, 1024, 96]);  permute_63 = None
        graphsafe_run_with_rng_state_5 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_84, view_85, view_86, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_5);  bwd_rng_state_5 = None
        getitem_67 = graphsafe_run_with_rng_state_5[0]
        getitem_68 = graphsafe_run_with_rng_state_5[1]
        getitem_73 = graphsafe_run_with_rng_state_5[6]
        getitem_74 = graphsafe_run_with_rng_state_5[7];  graphsafe_run_with_rng_state_5 = None
        permute_64 = torch.ops.aten.permute.default(getitem_67, [2, 0, 1, 3])
        view_87 = torch.ops.aten.view.default(permute_64, [20480, 768]);  permute_64 = None
        mm_70 = torch.ops.aten.mm.default(permute_298, view_87);  permute_298 = view_87 = None
        sum_81 = torch.ops.aten.sum.dim_IntList(view_327, [0], True, dtype = torch.float32);  view_327 = None
        view_328 = torch.ops.aten.view.default(sum_81, [768]);  sum_81 = None
        convert_element_type_559 = torch.ops.prims.convert_element_type.default(mm_70, torch.float32);  mm_70 = None
        convert_element_type_default_22 = torch.ops.prims.convert_element_type.default(view_328, torch.float32);  view_328 = None
        view_329 = torch.ops.aten.view.default(mm_69, [1024, 20, 8, 96]);  mm_69 = None
        permute_301 = torch.ops.aten.permute.default(view_329, [1, 2, 0, 3]);  view_329 = None
        _scaled_dot_product_flash_attention_backward_6 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_301, view_84, view_85, view_86, getitem_67, getitem_68, None, None, 1024, 1024, 0.2, False, getitem_73, getitem_74, scale = 0.10206207261596577);  permute_301 = view_84 = view_85 = view_86 = getitem_67 = getitem_68 = getitem_73 = getitem_74 = None
        getitem_174 = _scaled_dot_product_flash_attention_backward_6[0]
        getitem_175 = _scaled_dot_product_flash_attention_backward_6[1]
        getitem_176 = _scaled_dot_product_flash_attention_backward_6[2];  _scaled_dot_product_flash_attention_backward_6 = None
        view_330 = torch.ops.aten.view.default(getitem_176, [160, 1024, 96]);  getitem_176 = None
        view_331 = torch.ops.aten.view.default(getitem_175, [160, 1024, 96]);  getitem_175 = None
        view_332 = torch.ops.aten.view.default(getitem_174, [160, 1024, 96]);  getitem_174 = None
        permute_302 = torch.ops.aten.permute.default(view_330, [1, 0, 2]);  view_330 = None
        view_333 = torch.ops.aten.view.default(permute_302, [1024, 20, 768]);  permute_302 = None
        permute_303 = torch.ops.aten.permute.default(view_331, [1, 0, 2]);  view_331 = None
        view_334 = torch.ops.aten.view.default(permute_303, [1024, 20, 768]);  permute_303 = None
        permute_304 = torch.ops.aten.permute.default(view_332, [1, 0, 2]);  view_332 = None
        view_335 = torch.ops.aten.view.default(permute_304, [1024, 20, 768]);  permute_304 = None
        select_scatter_18 = torch.ops.aten.select_scatter.default(full_default_6, view_333, 0, 2);  view_333 = None
        select_scatter_19 = torch.ops.aten.select_scatter.default(full_default_6, view_334, 0, 1);  view_334 = None
        add_156 = torch.ops.aten.add.Tensor(select_scatter_18, select_scatter_19);  select_scatter_18 = select_scatter_19 = None
        select_scatter_20 = torch.ops.aten.select_scatter.default(full_default_6, view_335, 0, 0);  view_335 = None
        add_157 = torch.ops.aten.add.Tensor(add_156, select_scatter_20);  add_156 = select_scatter_20 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(add_157, 3);  add_157 = None
        permute_305 = torch.ops.aten.permute.default(unsqueeze_36, [3, 1, 2, 0, 4]);  unsqueeze_36 = None
        squeeze_18 = torch.ops.aten.squeeze.dim(permute_305, 0);  permute_305 = None
        clone_49 = torch.ops.aten.clone.default(squeeze_18, memory_format = torch.contiguous_format);  squeeze_18 = None
        view_336 = torch.ops.aten.view.default(clone_49, [1024, 20, 2304]);  clone_49 = None
        sum_82 = torch.ops.aten.sum.dim_IntList(view_336, [0, 1], True, dtype = torch.float32)
        view_337 = torch.ops.aten.view.default(sum_82, [2304]);  sum_82 = None
        view_338 = torch.ops.aten.view.default(view_336, [20480, 2304]);  view_336 = None
        permute_306 = torch.ops.aten.permute.default(view_338, [1, 0])
        mm_71 = torch.ops.aten.mm.default(permute_306, view_78);  permute_306 = view_78 = None
        permute_308 = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        mm_72 = torch.ops.aten.mm.default(view_338, permute_308);  view_338 = permute_308 = None
        view_339 = torch.ops.aten.view.default(mm_72, [1024, 20, 768]);  mm_72 = None
        convert_element_type_566 = torch.ops.prims.convert_element_type.default(view_339, torch.float32);  view_339 = None
        convert_element_type_567 = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_default_21 = torch.ops.prims.convert_element_type.default(view_337, torch.float32);  view_337 = None
        permute_310 = torch.ops.aten.permute.default(convert_element_type_566, [1, 0, 2]);  convert_element_type_566 = None
        mul_271 = torch.ops.aten.mul.Tensor(permute_310, primals_65);  primals_65 = None
        mul_272 = torch.ops.aten.mul.Tensor(mul_271, 768)
        sum_83 = torch.ops.aten.sum.dim_IntList(mul_271, [2], True)
        mul_273 = torch.ops.aten.mul.Tensor(mul_271, mul_45);  mul_271 = None
        sum_84 = torch.ops.aten.sum.dim_IntList(mul_273, [2], True);  mul_273 = None
        mul_274 = torch.ops.aten.mul.Tensor(mul_45, sum_84);  sum_84 = None
        sub_71 = torch.ops.aten.sub.Tensor(mul_272, sum_83);  mul_272 = sum_83 = None
        sub_72 = torch.ops.aten.sub.Tensor(sub_71, mul_274);  sub_71 = mul_274 = None
        div_13 = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        mul_275 = torch.ops.aten.mul.Tensor(div_13, sub_72);  div_13 = sub_72 = None
        mul_276 = torch.ops.aten.mul.Tensor(permute_310, mul_45);  mul_45 = None
        sum_85 = torch.ops.aten.sum.dim_IntList(mul_276, [0, 1]);  mul_276 = None
        sum_86 = torch.ops.aten.sum.dim_IntList(permute_310, [0, 1]);  permute_310 = None
        add_158 = torch.ops.aten.add.Tensor(add_155, mul_275);  add_155 = mul_275 = None
        convert_element_type_569 = torch.ops.prims.convert_element_type.default(add_158, torch.float16)
        inductor_lookup_seed_default_4 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_7 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        convert_element_type_default_59 = torch.ops.prims.convert_element_type.default(inductor_random_default_7, torch.float16);  inductor_random_default_7 = None
        gt_4 = torch.ops.aten.gt.Scalar(convert_element_type_default_59, 0.2);  convert_element_type_default_59 = None
        convert_element_type_570 = torch.ops.prims.convert_element_type.default(gt_4, torch.float16);  gt_4 = None
        mul_277 = torch.ops.aten.mul.Tensor(convert_element_type_570, 1.25);  convert_element_type_570 = None
        mul_278 = torch.ops.aten.mul.Tensor(convert_element_type_569, mul_277);  convert_element_type_569 = mul_277 = None
        view_340 = torch.ops.aten.view.default(mul_278, [20480, 768]);  mul_278 = None
        convert_element_type_116 = torch.ops.prims.convert_element_type.default(primals_63, torch.float16);  primals_63 = None
        permute_57 = torch.ops.aten.permute.default(convert_element_type_116, [1, 0]);  convert_element_type_116 = None
        permute_311 = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        mm_73 = torch.ops.aten.mm.default(view_340, permute_311);  permute_311 = None
        permute_312 = torch.ops.aten.permute.default(view_340, [1, 0])
        var_mean_9 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_63 = var_mean_9[0]
        getitem_64 = var_mean_9[1];  var_mean_9 = None
        add_40 = torch.ops.aten.add.Tensor(getitem_63, 1e-05);  getitem_63 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_39, getitem_64);  add_39 = getitem_64 = None
        mul_38 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, primals_59)
        add_41 = torch.ops.aten.add.Tensor(mul_39, primals_60);  mul_39 = primals_60 = None
        convert_element_type_107 = torch.ops.prims.convert_element_type.default(primals_62, torch.float16);  primals_62 = None
        convert_element_type_108 = torch.ops.prims.convert_element_type.default(primals_61, torch.float16);  primals_61 = None
        convert_element_type_109 = torch.ops.prims.convert_element_type.default(add_41, torch.float16);  add_41 = None
        view_74 = torch.ops.aten.view.default(convert_element_type_109, [20480, 768]);  convert_element_type_109 = None
        permute_56 = torch.ops.aten.permute.default(convert_element_type_108, [1, 0]);  convert_element_type_108 = None
        addmm_13 = torch.ops.aten.addmm.default(convert_element_type_107, view_74, permute_56);  convert_element_type_107 = None
        view_75 = torch.ops.aten.view.default(addmm_13, [20, 1024, 1536]);  addmm_13 = None
        convert_element_type_113 = torch.ops.prims.convert_element_type.default(view_75, torch.float32);  view_75 = None
        mul_40 = torch.ops.aten.mul.Tensor(convert_element_type_113, 0.5)
        mul_41 = torch.ops.aten.mul.Tensor(convert_element_type_113, 0.7071067811865476)
        erf_4 = torch.ops.aten.erf.default(mul_41);  mul_41 = None
        add_42 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_42 = torch.ops.aten.mul.Tensor(mul_40, add_42);  mul_40 = None
        convert_element_type_114 = torch.ops.prims.convert_element_type.default(mul_42, torch.float16);  mul_42 = None
        view_76 = torch.ops.aten.view.default(convert_element_type_114, [20480, 1536]);  convert_element_type_114 = None
        mm_74 = torch.ops.aten.mm.default(permute_312, view_76);  permute_312 = view_76 = None
        sum_87 = torch.ops.aten.sum.dim_IntList(view_340, [0], True, dtype = torch.float32);  view_340 = None
        view_341 = torch.ops.aten.view.default(sum_87, [768]);  sum_87 = None
        view_342 = torch.ops.aten.view.default(mm_73, [20, 1024, 1536]);  mm_73 = None
        convert_element_type_576 = torch.ops.prims.convert_element_type.default(mm_74, torch.float32);  mm_74 = None
        convert_element_type_default_20 = torch.ops.prims.convert_element_type.default(view_341, torch.float32);  view_341 = None
        convert_element_type_578 = torch.ops.prims.convert_element_type.default(view_342, torch.float32);  view_342 = None
        mul_280 = torch.ops.aten.mul.Tensor(add_42, 0.5);  add_42 = None
        mul_281 = torch.ops.aten.mul.Tensor(convert_element_type_113, convert_element_type_113)
        mul_282 = torch.ops.aten.mul.Tensor(mul_281, -0.5);  mul_281 = None
        exp_7 = torch.ops.aten.exp.default(mul_282);  mul_282 = None
        mul_283 = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
        mul_284 = torch.ops.aten.mul.Tensor(convert_element_type_113, mul_283);  convert_element_type_113 = mul_283 = None
        add_160 = torch.ops.aten.add.Tensor(mul_280, mul_284);  mul_280 = mul_284 = None
        mul_285 = torch.ops.aten.mul.Tensor(convert_element_type_578, add_160);  convert_element_type_578 = add_160 = None
        convert_element_type_580 = torch.ops.prims.convert_element_type.default(mul_285, torch.float16);  mul_285 = None
        view_343 = torch.ops.aten.view.default(convert_element_type_580, [20480, 1536]);  convert_element_type_580 = None
        permute_315 = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        mm_75 = torch.ops.aten.mm.default(view_343, permute_315);  permute_315 = None
        permute_316 = torch.ops.aten.permute.default(view_343, [1, 0])
        mm_76 = torch.ops.aten.mm.default(permute_316, view_74);  permute_316 = view_74 = None
        sum_88 = torch.ops.aten.sum.dim_IntList(view_343, [0], True, dtype = torch.float32);  view_343 = None
        view_344 = torch.ops.aten.view.default(sum_88, [1536]);  sum_88 = None
        view_345 = torch.ops.aten.view.default(mm_75, [20, 1024, 768]);  mm_75 = None
        convert_element_type_586 = torch.ops.prims.convert_element_type.default(view_345, torch.float32);  view_345 = None
        convert_element_type_587 = torch.ops.prims.convert_element_type.default(mm_76, torch.float32);  mm_76 = None
        convert_element_type_default_19 = torch.ops.prims.convert_element_type.default(view_344, torch.float32);  view_344 = None
        mul_287 = torch.ops.aten.mul.Tensor(convert_element_type_586, primals_59);  primals_59 = None
        mul_288 = torch.ops.aten.mul.Tensor(mul_287, 768)
        sum_89 = torch.ops.aten.sum.dim_IntList(mul_287, [2], True)
        mul_289 = torch.ops.aten.mul.Tensor(mul_287, mul_38);  mul_287 = None
        sum_90 = torch.ops.aten.sum.dim_IntList(mul_289, [2], True);  mul_289 = None
        mul_290 = torch.ops.aten.mul.Tensor(mul_38, sum_90);  sum_90 = None
        sub_74 = torch.ops.aten.sub.Tensor(mul_288, sum_89);  mul_288 = sum_89 = None
        sub_75 = torch.ops.aten.sub.Tensor(sub_74, mul_290);  sub_74 = mul_290 = None
        div_14 = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        mul_291 = torch.ops.aten.mul.Tensor(div_14, sub_75);  div_14 = sub_75 = None
        mul_292 = torch.ops.aten.mul.Tensor(convert_element_type_586, mul_38);  mul_38 = None
        sum_91 = torch.ops.aten.sum.dim_IntList(mul_292, [0, 1]);  mul_292 = None
        sum_92 = torch.ops.aten.sum.dim_IntList(convert_element_type_586, [0, 1]);  convert_element_type_586 = None
        add_161 = torch.ops.aten.add.Tensor(add_158, mul_291);  add_158 = mul_291 = None
        convert_element_type_589 = torch.ops.prims.convert_element_type.default(add_161, torch.float16)
        permute_319 = torch.ops.aten.permute.default(convert_element_type_589, [1, 0, 2]);  convert_element_type_589 = None
        clone_51 = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
        view_346 = torch.ops.aten.view.default(clone_51, [20480, 768]);  clone_51 = None
        convert_element_type_103 = torch.ops.prims.convert_element_type.default(primals_57, torch.float16);  primals_57 = None
        permute_54 = torch.ops.aten.permute.default(convert_element_type_103, [1, 0]);  convert_element_type_103 = None
        permute_320 = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_77 = torch.ops.aten.mm.default(view_346, permute_320);  permute_320 = None
        permute_321 = torch.ops.aten.permute.default(view_346, [1, 0])
        var_mean_8 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_52 = var_mean_8[0]
        getitem_53 = var_mean_8[1];  var_mean_8 = None
        add_36 = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_35, getitem_53);  add_35 = getitem_53 = None
        mul_36 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, primals_53)
        add_37 = torch.ops.aten.add.Tensor(mul_37, primals_54);  mul_37 = primals_54 = None
        permute_47 = torch.ops.aten.permute.default(add_37, [1, 0, 2]);  add_37 = None
        convert_element_type_97 = torch.ops.prims.convert_element_type.default(primals_55, torch.float16);  primals_55 = None
        convert_element_type_98 = torch.ops.prims.convert_element_type.default(primals_56, torch.float16);  primals_56 = None
        convert_element_type_99 = torch.ops.prims.convert_element_type.default(permute_47, torch.float16);  permute_47 = None
        permute_48 = torch.ops.aten.permute.default(convert_element_type_98, [1, 0]);  convert_element_type_98 = None
        clone_10 = torch.ops.aten.clone.default(convert_element_type_99, memory_format = torch.contiguous_format);  convert_element_type_99 = None
        view_63 = torch.ops.aten.view.default(clone_10, [20480, 768]);  clone_10 = None
        mm_5 = torch.ops.aten.mm.default(view_63, permute_48)
        view_64 = torch.ops.aten.view.default(mm_5, [1024, 20, 2304]);  mm_5 = None
        add_38 = torch.ops.aten.add.Tensor(view_64, convert_element_type_97);  view_64 = convert_element_type_97 = None
        view_65 = torch.ops.aten.view.default(add_38, [1024, 20, 3, 768]);  add_38 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(view_65, 0);  view_65 = None
        permute_49 = torch.ops.aten.permute.default(unsqueeze_10, [3, 1, 2, 0, 4]);  unsqueeze_10 = None
        squeeze_4 = torch.ops.aten.squeeze.dim(permute_49, -2);  permute_49 = None
        clone_11 = torch.ops.aten.clone.default(squeeze_4, memory_format = torch.contiguous_format);  squeeze_4 = None
        select_12 = torch.ops.aten.select.int(clone_11, 0, 0)
        select_13 = torch.ops.aten.select.int(clone_11, 0, 1)
        select_14 = torch.ops.aten.select.int(clone_11, 0, 2);  clone_11 = None
        view_66 = torch.ops.aten.view.default(select_12, [1024, 160, 96]);  select_12 = None
        permute_50 = torch.ops.aten.permute.default(view_66, [1, 0, 2]);  view_66 = None
        view_67 = torch.ops.aten.view.default(select_13, [1024, 160, 96]);  select_13 = None
        permute_51 = torch.ops.aten.permute.default(view_67, [1, 0, 2]);  view_67 = None
        view_68 = torch.ops.aten.view.default(select_14, [1024, 160, 96]);  select_14 = None
        permute_52 = torch.ops.aten.permute.default(view_68, [1, 0, 2]);  view_68 = None
        view_69 = torch.ops.aten.view.default(permute_50, [20, 8, 1024, 96]);  permute_50 = None
        view_70 = torch.ops.aten.view.default(permute_51, [20, 8, 1024, 96]);  permute_51 = None
        view_71 = torch.ops.aten.view.default(permute_52, [20, 8, 1024, 96]);  permute_52 = None
        graphsafe_run_with_rng_state_4 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_69, view_70, view_71, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_4);  bwd_rng_state_4 = None
        getitem_54 = graphsafe_run_with_rng_state_4[0]
        getitem_55 = graphsafe_run_with_rng_state_4[1]
        getitem_60 = graphsafe_run_with_rng_state_4[6]
        getitem_61 = graphsafe_run_with_rng_state_4[7];  graphsafe_run_with_rng_state_4 = None
        permute_53 = torch.ops.aten.permute.default(getitem_54, [2, 0, 1, 3])
        view_72 = torch.ops.aten.view.default(permute_53, [20480, 768]);  permute_53 = None
        mm_78 = torch.ops.aten.mm.default(permute_321, view_72);  permute_321 = view_72 = None
        sum_93 = torch.ops.aten.sum.dim_IntList(view_346, [0], True, dtype = torch.float32);  view_346 = None
        view_347 = torch.ops.aten.view.default(sum_93, [768]);  sum_93 = None
        convert_element_type_595 = torch.ops.prims.convert_element_type.default(mm_78, torch.float32);  mm_78 = None
        convert_element_type_default_18 = torch.ops.prims.convert_element_type.default(view_347, torch.float32);  view_347 = None
        view_348 = torch.ops.aten.view.default(mm_77, [1024, 20, 8, 96]);  mm_77 = None
        permute_324 = torch.ops.aten.permute.default(view_348, [1, 2, 0, 3]);  view_348 = None
        _scaled_dot_product_flash_attention_backward_7 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_324, view_69, view_70, view_71, getitem_54, getitem_55, None, None, 1024, 1024, 0.2, False, getitem_60, getitem_61, scale = 0.10206207261596577);  permute_324 = view_69 = view_70 = view_71 = getitem_54 = getitem_55 = getitem_60 = getitem_61 = None
        getitem_177 = _scaled_dot_product_flash_attention_backward_7[0]
        getitem_178 = _scaled_dot_product_flash_attention_backward_7[1]
        getitem_179 = _scaled_dot_product_flash_attention_backward_7[2];  _scaled_dot_product_flash_attention_backward_7 = None
        view_349 = torch.ops.aten.view.default(getitem_179, [160, 1024, 96]);  getitem_179 = None
        view_350 = torch.ops.aten.view.default(getitem_178, [160, 1024, 96]);  getitem_178 = None
        view_351 = torch.ops.aten.view.default(getitem_177, [160, 1024, 96]);  getitem_177 = None
        permute_325 = torch.ops.aten.permute.default(view_349, [1, 0, 2]);  view_349 = None
        view_352 = torch.ops.aten.view.default(permute_325, [1024, 20, 768]);  permute_325 = None
        permute_326 = torch.ops.aten.permute.default(view_350, [1, 0, 2]);  view_350 = None
        view_353 = torch.ops.aten.view.default(permute_326, [1024, 20, 768]);  permute_326 = None
        permute_327 = torch.ops.aten.permute.default(view_351, [1, 0, 2]);  view_351 = None
        view_354 = torch.ops.aten.view.default(permute_327, [1024, 20, 768]);  permute_327 = None
        select_scatter_21 = torch.ops.aten.select_scatter.default(full_default_6, view_352, 0, 2);  view_352 = None
        select_scatter_22 = torch.ops.aten.select_scatter.default(full_default_6, view_353, 0, 1);  view_353 = None
        add_162 = torch.ops.aten.add.Tensor(select_scatter_21, select_scatter_22);  select_scatter_21 = select_scatter_22 = None
        select_scatter_23 = torch.ops.aten.select_scatter.default(full_default_6, view_354, 0, 0);  view_354 = None
        add_163 = torch.ops.aten.add.Tensor(add_162, select_scatter_23);  add_162 = select_scatter_23 = None
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(add_163, 3);  add_163 = None
        permute_328 = torch.ops.aten.permute.default(unsqueeze_37, [3, 1, 2, 0, 4]);  unsqueeze_37 = None
        squeeze_19 = torch.ops.aten.squeeze.dim(permute_328, 0);  permute_328 = None
        clone_52 = torch.ops.aten.clone.default(squeeze_19, memory_format = torch.contiguous_format);  squeeze_19 = None
        view_355 = torch.ops.aten.view.default(clone_52, [1024, 20, 2304]);  clone_52 = None
        sum_94 = torch.ops.aten.sum.dim_IntList(view_355, [0, 1], True, dtype = torch.float32)
        view_356 = torch.ops.aten.view.default(sum_94, [2304]);  sum_94 = None
        view_357 = torch.ops.aten.view.default(view_355, [20480, 2304]);  view_355 = None
        permute_329 = torch.ops.aten.permute.default(view_357, [1, 0])
        mm_79 = torch.ops.aten.mm.default(permute_329, view_63);  permute_329 = view_63 = None
        permute_331 = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        mm_80 = torch.ops.aten.mm.default(view_357, permute_331);  view_357 = permute_331 = None
        view_358 = torch.ops.aten.view.default(mm_80, [1024, 20, 768]);  mm_80 = None
        convert_element_type_602 = torch.ops.prims.convert_element_type.default(view_358, torch.float32);  view_358 = None
        convert_element_type_603 = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_default_17 = torch.ops.prims.convert_element_type.default(view_356, torch.float32);  view_356 = None
        permute_333 = torch.ops.aten.permute.default(convert_element_type_602, [1, 0, 2]);  convert_element_type_602 = None
        mul_294 = torch.ops.aten.mul.Tensor(permute_333, primals_53);  primals_53 = None
        mul_295 = torch.ops.aten.mul.Tensor(mul_294, 768)
        sum_95 = torch.ops.aten.sum.dim_IntList(mul_294, [2], True)
        mul_296 = torch.ops.aten.mul.Tensor(mul_294, mul_36);  mul_294 = None
        sum_96 = torch.ops.aten.sum.dim_IntList(mul_296, [2], True);  mul_296 = None
        mul_297 = torch.ops.aten.mul.Tensor(mul_36, sum_96);  sum_96 = None
        sub_77 = torch.ops.aten.sub.Tensor(mul_295, sum_95);  mul_295 = sum_95 = None
        sub_78 = torch.ops.aten.sub.Tensor(sub_77, mul_297);  sub_77 = mul_297 = None
        div_15 = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        mul_298 = torch.ops.aten.mul.Tensor(div_15, sub_78);  div_15 = sub_78 = None
        mul_299 = torch.ops.aten.mul.Tensor(permute_333, mul_36);  mul_36 = None
        sum_97 = torch.ops.aten.sum.dim_IntList(mul_299, [0, 1]);  mul_299 = None
        sum_98 = torch.ops.aten.sum.dim_IntList(permute_333, [0, 1]);  permute_333 = None
        add_164 = torch.ops.aten.add.Tensor(add_161, mul_298);  add_161 = mul_298 = None
        convert_element_type_605 = torch.ops.prims.convert_element_type.default(add_164, torch.float16)
        inductor_lookup_seed_default_3 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_8 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        convert_element_type_default_60 = torch.ops.prims.convert_element_type.default(inductor_random_default_8, torch.float16);  inductor_random_default_8 = None
        gt_3 = torch.ops.aten.gt.Scalar(convert_element_type_default_60, 0.2);  convert_element_type_default_60 = None
        convert_element_type_606 = torch.ops.prims.convert_element_type.default(gt_3, torch.float16);  gt_3 = None
        mul_300 = torch.ops.aten.mul.Tensor(convert_element_type_606, 1.25);  convert_element_type_606 = None
        mul_301 = torch.ops.aten.mul.Tensor(convert_element_type_605, mul_300);  convert_element_type_605 = mul_300 = None
        view_359 = torch.ops.aten.view.default(mul_301, [20480, 768]);  mul_301 = None
        convert_element_type_93 = torch.ops.prims.convert_element_type.default(primals_51, torch.float16);  primals_51 = None
        permute_46 = torch.ops.aten.permute.default(convert_element_type_93, [1, 0]);  convert_element_type_93 = None
        permute_334 = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_81 = torch.ops.aten.mm.default(view_359, permute_334);  permute_334 = None
        permute_335 = torch.ops.aten.permute.default(view_359, [1, 0])
        var_mean_7 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_50 = var_mean_7[0]
        getitem_51 = var_mean_7[1];  var_mean_7 = None
        add_32 = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_31, getitem_51);  add_31 = getitem_51 = None
        mul_29 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        mul_30 = torch.ops.aten.mul.Tensor(mul_29, primals_47)
        add_33 = torch.ops.aten.add.Tensor(mul_30, primals_48);  mul_30 = primals_48 = None
        convert_element_type_84 = torch.ops.prims.convert_element_type.default(primals_50, torch.float16);  primals_50 = None
        convert_element_type_85 = torch.ops.prims.convert_element_type.default(primals_49, torch.float16);  primals_49 = None
        convert_element_type_86 = torch.ops.prims.convert_element_type.default(add_33, torch.float16);  add_33 = None
        view_59 = torch.ops.aten.view.default(convert_element_type_86, [20480, 768]);  convert_element_type_86 = None
        permute_45 = torch.ops.aten.permute.default(convert_element_type_85, [1, 0]);  convert_element_type_85 = None
        addmm_10 = torch.ops.aten.addmm.default(convert_element_type_84, view_59, permute_45);  convert_element_type_84 = None
        view_60 = torch.ops.aten.view.default(addmm_10, [20, 1024, 1536]);  addmm_10 = None
        convert_element_type_90 = torch.ops.prims.convert_element_type.default(view_60, torch.float32);  view_60 = None
        mul_31 = torch.ops.aten.mul.Tensor(convert_element_type_90, 0.5)
        mul_32 = torch.ops.aten.mul.Tensor(convert_element_type_90, 0.7071067811865476)
        erf_3 = torch.ops.aten.erf.default(mul_32);  mul_32 = None
        add_34 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_31, add_34);  mul_31 = None
        convert_element_type_91 = torch.ops.prims.convert_element_type.default(mul_33, torch.float16);  mul_33 = None
        view_61 = torch.ops.aten.view.default(convert_element_type_91, [20480, 1536]);  convert_element_type_91 = None
        mm_82 = torch.ops.aten.mm.default(permute_335, view_61);  permute_335 = view_61 = None
        sum_99 = torch.ops.aten.sum.dim_IntList(view_359, [0], True, dtype = torch.float32);  view_359 = None
        view_360 = torch.ops.aten.view.default(sum_99, [768]);  sum_99 = None
        view_361 = torch.ops.aten.view.default(mm_81, [20, 1024, 1536]);  mm_81 = None
        convert_element_type_612 = torch.ops.prims.convert_element_type.default(mm_82, torch.float32);  mm_82 = None
        convert_element_type_default_16 = torch.ops.prims.convert_element_type.default(view_360, torch.float32);  view_360 = None
        convert_element_type_614 = torch.ops.prims.convert_element_type.default(view_361, torch.float32);  view_361 = None
        mul_303 = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
        mul_304 = torch.ops.aten.mul.Tensor(convert_element_type_90, convert_element_type_90)
        mul_305 = torch.ops.aten.mul.Tensor(mul_304, -0.5);  mul_304 = None
        exp_8 = torch.ops.aten.exp.default(mul_305);  mul_305 = None
        mul_306 = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
        mul_307 = torch.ops.aten.mul.Tensor(convert_element_type_90, mul_306);  convert_element_type_90 = mul_306 = None
        add_166 = torch.ops.aten.add.Tensor(mul_303, mul_307);  mul_303 = mul_307 = None
        mul_308 = torch.ops.aten.mul.Tensor(convert_element_type_614, add_166);  convert_element_type_614 = add_166 = None
        convert_element_type_616 = torch.ops.prims.convert_element_type.default(mul_308, torch.float16);  mul_308 = None
        view_362 = torch.ops.aten.view.default(convert_element_type_616, [20480, 1536]);  convert_element_type_616 = None
        permute_338 = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        mm_83 = torch.ops.aten.mm.default(view_362, permute_338);  permute_338 = None
        permute_339 = torch.ops.aten.permute.default(view_362, [1, 0])
        mm_84 = torch.ops.aten.mm.default(permute_339, view_59);  permute_339 = view_59 = None
        sum_100 = torch.ops.aten.sum.dim_IntList(view_362, [0], True, dtype = torch.float32);  view_362 = None
        view_363 = torch.ops.aten.view.default(sum_100, [1536]);  sum_100 = None
        view_364 = torch.ops.aten.view.default(mm_83, [20, 1024, 768]);  mm_83 = None
        convert_element_type_622 = torch.ops.prims.convert_element_type.default(view_364, torch.float32);  view_364 = None
        convert_element_type_623 = torch.ops.prims.convert_element_type.default(mm_84, torch.float32);  mm_84 = None
        convert_element_type_default_15 = torch.ops.prims.convert_element_type.default(view_363, torch.float32);  view_363 = None
        mul_310 = torch.ops.aten.mul.Tensor(convert_element_type_622, primals_47);  primals_47 = None
        mul_311 = torch.ops.aten.mul.Tensor(mul_310, 768)
        sum_101 = torch.ops.aten.sum.dim_IntList(mul_310, [2], True)
        mul_312 = torch.ops.aten.mul.Tensor(mul_310, mul_29);  mul_310 = None
        sum_102 = torch.ops.aten.sum.dim_IntList(mul_312, [2], True);  mul_312 = None
        mul_313 = torch.ops.aten.mul.Tensor(mul_29, sum_102);  sum_102 = None
        sub_80 = torch.ops.aten.sub.Tensor(mul_311, sum_101);  mul_311 = sum_101 = None
        sub_81 = torch.ops.aten.sub.Tensor(sub_80, mul_313);  sub_80 = mul_313 = None
        div_16 = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        mul_314 = torch.ops.aten.mul.Tensor(div_16, sub_81);  div_16 = sub_81 = None
        mul_315 = torch.ops.aten.mul.Tensor(convert_element_type_622, mul_29);  mul_29 = None
        sum_103 = torch.ops.aten.sum.dim_IntList(mul_315, [0, 1]);  mul_315 = None
        sum_104 = torch.ops.aten.sum.dim_IntList(convert_element_type_622, [0, 1]);  convert_element_type_622 = None
        add_167 = torch.ops.aten.add.Tensor(add_164, mul_314);  add_164 = mul_314 = None
        convert_element_type_625 = torch.ops.prims.convert_element_type.default(add_167, torch.float16)
        permute_342 = torch.ops.aten.permute.default(convert_element_type_625, [1, 0, 2]);  convert_element_type_625 = None
        clone_54 = torch.ops.aten.clone.default(permute_342, memory_format = torch.contiguous_format);  permute_342 = None
        view_365 = torch.ops.aten.view.default(clone_54, [20480, 768]);  clone_54 = None
        convert_element_type_80 = torch.ops.prims.convert_element_type.default(primals_45, torch.float16);  primals_45 = None
        permute_43 = torch.ops.aten.permute.default(convert_element_type_80, [1, 0]);  convert_element_type_80 = None
        permute_343 = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_85 = torch.ops.aten.mm.default(view_365, permute_343);  permute_343 = None
        permute_344 = torch.ops.aten.permute.default(view_365, [1, 0])
        var_mean_6 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_39 = var_mean_6[0]
        getitem_40 = var_mean_6[1];  var_mean_6 = None
        add_28 = torch.ops.aten.add.Tensor(getitem_39, 1e-05);  getitem_39 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_27, getitem_40);  add_27 = getitem_40 = None
        mul_27 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        mul_28 = torch.ops.aten.mul.Tensor(mul_27, primals_41)
        add_29 = torch.ops.aten.add.Tensor(mul_28, primals_42);  mul_28 = primals_42 = None
        permute_36 = torch.ops.aten.permute.default(add_29, [1, 0, 2]);  add_29 = None
        convert_element_type_74 = torch.ops.prims.convert_element_type.default(primals_43, torch.float16);  primals_43 = None
        convert_element_type_75 = torch.ops.prims.convert_element_type.default(primals_44, torch.float16);  primals_44 = None
        convert_element_type_76 = torch.ops.prims.convert_element_type.default(permute_36, torch.float16);  permute_36 = None
        permute_37 = torch.ops.aten.permute.default(convert_element_type_75, [1, 0]);  convert_element_type_75 = None
        clone_8 = torch.ops.aten.clone.default(convert_element_type_76, memory_format = torch.contiguous_format);  convert_element_type_76 = None
        view_48 = torch.ops.aten.view.default(clone_8, [20480, 768]);  clone_8 = None
        mm_4 = torch.ops.aten.mm.default(view_48, permute_37)
        view_49 = torch.ops.aten.view.default(mm_4, [1024, 20, 2304]);  mm_4 = None
        add_30 = torch.ops.aten.add.Tensor(view_49, convert_element_type_74);  view_49 = convert_element_type_74 = None
        view_50 = torch.ops.aten.view.default(add_30, [1024, 20, 3, 768]);  add_30 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(view_50, 0);  view_50 = None
        permute_38 = torch.ops.aten.permute.default(unsqueeze_9, [3, 1, 2, 0, 4]);  unsqueeze_9 = None
        squeeze_3 = torch.ops.aten.squeeze.dim(permute_38, -2);  permute_38 = None
        clone_9 = torch.ops.aten.clone.default(squeeze_3, memory_format = torch.contiguous_format);  squeeze_3 = None
        select_9 = torch.ops.aten.select.int(clone_9, 0, 0)
        select_10 = torch.ops.aten.select.int(clone_9, 0, 1)
        select_11 = torch.ops.aten.select.int(clone_9, 0, 2);  clone_9 = None
        view_51 = torch.ops.aten.view.default(select_9, [1024, 160, 96]);  select_9 = None
        permute_39 = torch.ops.aten.permute.default(view_51, [1, 0, 2]);  view_51 = None
        view_52 = torch.ops.aten.view.default(select_10, [1024, 160, 96]);  select_10 = None
        permute_40 = torch.ops.aten.permute.default(view_52, [1, 0, 2]);  view_52 = None
        view_53 = torch.ops.aten.view.default(select_11, [1024, 160, 96]);  select_11 = None
        permute_41 = torch.ops.aten.permute.default(view_53, [1, 0, 2]);  view_53 = None
        view_54 = torch.ops.aten.view.default(permute_39, [20, 8, 1024, 96]);  permute_39 = None
        view_55 = torch.ops.aten.view.default(permute_40, [20, 8, 1024, 96]);  permute_40 = None
        view_56 = torch.ops.aten.view.default(permute_41, [20, 8, 1024, 96]);  permute_41 = None
        graphsafe_run_with_rng_state_3 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_54, view_55, view_56, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_3);  bwd_rng_state_3 = None
        getitem_41 = graphsafe_run_with_rng_state_3[0]
        getitem_42 = graphsafe_run_with_rng_state_3[1]
        getitem_47 = graphsafe_run_with_rng_state_3[6]
        getitem_48 = graphsafe_run_with_rng_state_3[7];  graphsafe_run_with_rng_state_3 = None
        permute_42 = torch.ops.aten.permute.default(getitem_41, [2, 0, 1, 3])
        view_57 = torch.ops.aten.view.default(permute_42, [20480, 768]);  permute_42 = None
        mm_86 = torch.ops.aten.mm.default(permute_344, view_57);  permute_344 = view_57 = None
        sum_105 = torch.ops.aten.sum.dim_IntList(view_365, [0], True, dtype = torch.float32);  view_365 = None
        view_366 = torch.ops.aten.view.default(sum_105, [768]);  sum_105 = None
        convert_element_type_631 = torch.ops.prims.convert_element_type.default(mm_86, torch.float32);  mm_86 = None
        convert_element_type_default_14 = torch.ops.prims.convert_element_type.default(view_366, torch.float32);  view_366 = None
        view_367 = torch.ops.aten.view.default(mm_85, [1024, 20, 8, 96]);  mm_85 = None
        permute_347 = torch.ops.aten.permute.default(view_367, [1, 2, 0, 3]);  view_367 = None
        _scaled_dot_product_flash_attention_backward_8 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_347, view_54, view_55, view_56, getitem_41, getitem_42, None, None, 1024, 1024, 0.2, False, getitem_47, getitem_48, scale = 0.10206207261596577);  permute_347 = view_54 = view_55 = view_56 = getitem_41 = getitem_42 = getitem_47 = getitem_48 = None
        getitem_180 = _scaled_dot_product_flash_attention_backward_8[0]
        getitem_181 = _scaled_dot_product_flash_attention_backward_8[1]
        getitem_182 = _scaled_dot_product_flash_attention_backward_8[2];  _scaled_dot_product_flash_attention_backward_8 = None
        view_368 = torch.ops.aten.view.default(getitem_182, [160, 1024, 96]);  getitem_182 = None
        view_369 = torch.ops.aten.view.default(getitem_181, [160, 1024, 96]);  getitem_181 = None
        view_370 = torch.ops.aten.view.default(getitem_180, [160, 1024, 96]);  getitem_180 = None
        permute_348 = torch.ops.aten.permute.default(view_368, [1, 0, 2]);  view_368 = None
        view_371 = torch.ops.aten.view.default(permute_348, [1024, 20, 768]);  permute_348 = None
        permute_349 = torch.ops.aten.permute.default(view_369, [1, 0, 2]);  view_369 = None
        view_372 = torch.ops.aten.view.default(permute_349, [1024, 20, 768]);  permute_349 = None
        permute_350 = torch.ops.aten.permute.default(view_370, [1, 0, 2]);  view_370 = None
        view_373 = torch.ops.aten.view.default(permute_350, [1024, 20, 768]);  permute_350 = None
        select_scatter_24 = torch.ops.aten.select_scatter.default(full_default_6, view_371, 0, 2);  view_371 = None
        select_scatter_25 = torch.ops.aten.select_scatter.default(full_default_6, view_372, 0, 1);  view_372 = None
        add_168 = torch.ops.aten.add.Tensor(select_scatter_24, select_scatter_25);  select_scatter_24 = select_scatter_25 = None
        select_scatter_26 = torch.ops.aten.select_scatter.default(full_default_6, view_373, 0, 0);  view_373 = None
        add_169 = torch.ops.aten.add.Tensor(add_168, select_scatter_26);  add_168 = select_scatter_26 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(add_169, 3);  add_169 = None
        permute_351 = torch.ops.aten.permute.default(unsqueeze_38, [3, 1, 2, 0, 4]);  unsqueeze_38 = None
        squeeze_20 = torch.ops.aten.squeeze.dim(permute_351, 0);  permute_351 = None
        clone_55 = torch.ops.aten.clone.default(squeeze_20, memory_format = torch.contiguous_format);  squeeze_20 = None
        view_374 = torch.ops.aten.view.default(clone_55, [1024, 20, 2304]);  clone_55 = None
        sum_106 = torch.ops.aten.sum.dim_IntList(view_374, [0, 1], True, dtype = torch.float32)
        view_375 = torch.ops.aten.view.default(sum_106, [2304]);  sum_106 = None
        view_376 = torch.ops.aten.view.default(view_374, [20480, 2304]);  view_374 = None
        permute_352 = torch.ops.aten.permute.default(view_376, [1, 0])
        mm_87 = torch.ops.aten.mm.default(permute_352, view_48);  permute_352 = view_48 = None
        permute_354 = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        mm_88 = torch.ops.aten.mm.default(view_376, permute_354);  view_376 = permute_354 = None
        view_377 = torch.ops.aten.view.default(mm_88, [1024, 20, 768]);  mm_88 = None
        convert_element_type_638 = torch.ops.prims.convert_element_type.default(view_377, torch.float32);  view_377 = None
        convert_element_type_639 = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_default_13 = torch.ops.prims.convert_element_type.default(view_375, torch.float32);  view_375 = None
        permute_356 = torch.ops.aten.permute.default(convert_element_type_638, [1, 0, 2]);  convert_element_type_638 = None
        mul_317 = torch.ops.aten.mul.Tensor(permute_356, primals_41);  primals_41 = None
        mul_318 = torch.ops.aten.mul.Tensor(mul_317, 768)
        sum_107 = torch.ops.aten.sum.dim_IntList(mul_317, [2], True)
        mul_319 = torch.ops.aten.mul.Tensor(mul_317, mul_27);  mul_317 = None
        sum_108 = torch.ops.aten.sum.dim_IntList(mul_319, [2], True);  mul_319 = None
        mul_320 = torch.ops.aten.mul.Tensor(mul_27, sum_108);  sum_108 = None
        sub_83 = torch.ops.aten.sub.Tensor(mul_318, sum_107);  mul_318 = sum_107 = None
        sub_84 = torch.ops.aten.sub.Tensor(sub_83, mul_320);  sub_83 = mul_320 = None
        div_17 = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        mul_321 = torch.ops.aten.mul.Tensor(div_17, sub_84);  div_17 = sub_84 = None
        mul_322 = torch.ops.aten.mul.Tensor(permute_356, mul_27);  mul_27 = None
        sum_109 = torch.ops.aten.sum.dim_IntList(mul_322, [0, 1]);  mul_322 = None
        sum_110 = torch.ops.aten.sum.dim_IntList(permute_356, [0, 1]);  permute_356 = None
        add_170 = torch.ops.aten.add.Tensor(add_167, mul_321);  add_167 = mul_321 = None
        convert_element_type_641 = torch.ops.prims.convert_element_type.default(add_170, torch.float16)
        inductor_lookup_seed_default_2 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_9 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        convert_element_type_default_61 = torch.ops.prims.convert_element_type.default(inductor_random_default_9, torch.float16);  inductor_random_default_9 = None
        gt_2 = torch.ops.aten.gt.Scalar(convert_element_type_default_61, 0.2);  convert_element_type_default_61 = None
        convert_element_type_642 = torch.ops.prims.convert_element_type.default(gt_2, torch.float16);  gt_2 = None
        mul_323 = torch.ops.aten.mul.Tensor(convert_element_type_642, 1.25);  convert_element_type_642 = None
        mul_324 = torch.ops.aten.mul.Tensor(convert_element_type_641, mul_323);  convert_element_type_641 = mul_323 = None
        view_378 = torch.ops.aten.view.default(mul_324, [20480, 768]);  mul_324 = None
        convert_element_type_70 = torch.ops.prims.convert_element_type.default(primals_39, torch.float16);  primals_39 = None
        permute_35 = torch.ops.aten.permute.default(convert_element_type_70, [1, 0]);  convert_element_type_70 = None
        permute_357 = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_89 = torch.ops.aten.mm.default(view_378, permute_357);  permute_357 = None
        permute_358 = torch.ops.aten.permute.default(view_378, [1, 0])
        var_mean_5 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
        getitem_37 = var_mean_5[0]
        getitem_38 = var_mean_5[1];  var_mean_5 = None
        add_24 = torch.ops.aten.add.Tensor(getitem_37, 1e-05);  getitem_37 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_23, getitem_38);  add_23 = getitem_38 = None
        mul_20 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, primals_35)
        add_25 = torch.ops.aten.add.Tensor(mul_21, primals_36);  mul_21 = primals_36 = None
        convert_element_type_61 = torch.ops.prims.convert_element_type.default(primals_38, torch.float16);  primals_38 = None
        convert_element_type_62 = torch.ops.prims.convert_element_type.default(primals_37, torch.float16);  primals_37 = None
        convert_element_type_63 = torch.ops.prims.convert_element_type.default(add_25, torch.float16);  add_25 = None
        view_44 = torch.ops.aten.view.default(convert_element_type_63, [20480, 768]);  convert_element_type_63 = None
        permute_34 = torch.ops.aten.permute.default(convert_element_type_62, [1, 0]);  convert_element_type_62 = None
        addmm_7 = torch.ops.aten.addmm.default(convert_element_type_61, view_44, permute_34);  convert_element_type_61 = None
        view_45 = torch.ops.aten.view.default(addmm_7, [20, 1024, 1536]);  addmm_7 = None
        convert_element_type_67 = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        mul_22 = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.5)
        mul_23 = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.7071067811865476)
        erf_2 = torch.ops.aten.erf.default(mul_23);  mul_23 = None
        add_26 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_24 = torch.ops.aten.mul.Tensor(mul_22, add_26);  mul_22 = None
        convert_element_type_68 = torch.ops.prims.convert_element_type.default(mul_24, torch.float16);  mul_24 = None
        view_46 = torch.ops.aten.view.default(convert_element_type_68, [20480, 1536]);  convert_element_type_68 = None
        mm_90 = torch.ops.aten.mm.default(permute_358, view_46);  permute_358 = view_46 = None
        sum_111 = torch.ops.aten.sum.dim_IntList(view_378, [0], True, dtype = torch.float32);  view_378 = None
        view_379 = torch.ops.aten.view.default(sum_111, [768]);  sum_111 = None
        view_380 = torch.ops.aten.view.default(mm_89, [20, 1024, 1536]);  mm_89 = None
        convert_element_type_648 = torch.ops.prims.convert_element_type.default(mm_90, torch.float32);  mm_90 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(view_379, torch.float32);  view_379 = None
        convert_element_type_650 = torch.ops.prims.convert_element_type.default(view_380, torch.float32);  view_380 = None
        mul_326 = torch.ops.aten.mul.Tensor(add_26, 0.5);  add_26 = None
        mul_327 = torch.ops.aten.mul.Tensor(convert_element_type_67, convert_element_type_67)
        mul_328 = torch.ops.aten.mul.Tensor(mul_327, -0.5);  mul_327 = None
        exp_9 = torch.ops.aten.exp.default(mul_328);  mul_328 = None
        mul_329 = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
        mul_330 = torch.ops.aten.mul.Tensor(convert_element_type_67, mul_329);  convert_element_type_67 = mul_329 = None
        add_172 = torch.ops.aten.add.Tensor(mul_326, mul_330);  mul_326 = mul_330 = None
        mul_331 = torch.ops.aten.mul.Tensor(convert_element_type_650, add_172);  convert_element_type_650 = add_172 = None
        convert_element_type_652 = torch.ops.prims.convert_element_type.default(mul_331, torch.float16);  mul_331 = None
        view_381 = torch.ops.aten.view.default(convert_element_type_652, [20480, 1536]);  convert_element_type_652 = None
        permute_361 = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        mm_91 = torch.ops.aten.mm.default(view_381, permute_361);  permute_361 = None
        permute_362 = torch.ops.aten.permute.default(view_381, [1, 0])
        mm_92 = torch.ops.aten.mm.default(permute_362, view_44);  permute_362 = view_44 = None
        sum_112 = torch.ops.aten.sum.dim_IntList(view_381, [0], True, dtype = torch.float32);  view_381 = None
        view_382 = torch.ops.aten.view.default(sum_112, [1536]);  sum_112 = None
        view_383 = torch.ops.aten.view.default(mm_91, [20, 1024, 768]);  mm_91 = None
        convert_element_type_658 = torch.ops.prims.convert_element_type.default(view_383, torch.float32);  view_383 = None
        convert_element_type_659 = torch.ops.prims.convert_element_type.default(mm_92, torch.float32);  mm_92 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(view_382, torch.float32);  view_382 = None
        mul_333 = torch.ops.aten.mul.Tensor(convert_element_type_658, primals_35);  primals_35 = None
        mul_334 = torch.ops.aten.mul.Tensor(mul_333, 768)
        sum_113 = torch.ops.aten.sum.dim_IntList(mul_333, [2], True)
        mul_335 = torch.ops.aten.mul.Tensor(mul_333, mul_20);  mul_333 = None
        sum_114 = torch.ops.aten.sum.dim_IntList(mul_335, [2], True);  mul_335 = None
        mul_336 = torch.ops.aten.mul.Tensor(mul_20, sum_114);  sum_114 = None
        sub_86 = torch.ops.aten.sub.Tensor(mul_334, sum_113);  mul_334 = sum_113 = None
        sub_87 = torch.ops.aten.sub.Tensor(sub_86, mul_336);  sub_86 = mul_336 = None
        div_18 = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        mul_337 = torch.ops.aten.mul.Tensor(div_18, sub_87);  div_18 = sub_87 = None
        mul_338 = torch.ops.aten.mul.Tensor(convert_element_type_658, mul_20);  mul_20 = None
        sum_115 = torch.ops.aten.sum.dim_IntList(mul_338, [0, 1]);  mul_338 = None
        sum_116 = torch.ops.aten.sum.dim_IntList(convert_element_type_658, [0, 1]);  convert_element_type_658 = None
        add_173 = torch.ops.aten.add.Tensor(add_170, mul_337);  add_170 = mul_337 = None
        convert_element_type_661 = torch.ops.prims.convert_element_type.default(add_173, torch.float16)
        permute_365 = torch.ops.aten.permute.default(convert_element_type_661, [1, 0, 2]);  convert_element_type_661 = None
        clone_57 = torch.ops.aten.clone.default(permute_365, memory_format = torch.contiguous_format);  permute_365 = None
        view_384 = torch.ops.aten.view.default(clone_57, [20480, 768]);  clone_57 = None
        convert_element_type_57 = torch.ops.prims.convert_element_type.default(primals_33, torch.float16);  primals_33 = None
        permute_32 = torch.ops.aten.permute.default(convert_element_type_57, [1, 0]);  convert_element_type_57 = None
        permute_366 = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_93 = torch.ops.aten.mm.default(view_384, permute_366);  permute_366 = None
        permute_367 = torch.ops.aten.permute.default(view_384, [1, 0])
        var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_4[0]
        getitem_27 = var_mean_4[1];  var_mean_4 = None
        add_20 = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_19, getitem_27);  add_19 = getitem_27 = None
        mul_18 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, primals_29)
        add_21 = torch.ops.aten.add.Tensor(mul_19, primals_30);  mul_19 = primals_30 = None
        permute_25 = torch.ops.aten.permute.default(add_21, [1, 0, 2]);  add_21 = None
        convert_element_type_51 = torch.ops.prims.convert_element_type.default(primals_31, torch.float16);  primals_31 = None
        convert_element_type_52 = torch.ops.prims.convert_element_type.default(primals_32, torch.float16);  primals_32 = None
        convert_element_type_53 = torch.ops.prims.convert_element_type.default(permute_25, torch.float16);  permute_25 = None
        permute_26 = torch.ops.aten.permute.default(convert_element_type_52, [1, 0]);  convert_element_type_52 = None
        clone_6 = torch.ops.aten.clone.default(convert_element_type_53, memory_format = torch.contiguous_format);  convert_element_type_53 = None
        view_33 = torch.ops.aten.view.default(clone_6, [20480, 768]);  clone_6 = None
        mm_3 = torch.ops.aten.mm.default(view_33, permute_26)
        view_34 = torch.ops.aten.view.default(mm_3, [1024, 20, 2304]);  mm_3 = None
        add_22 = torch.ops.aten.add.Tensor(view_34, convert_element_type_51);  view_34 = convert_element_type_51 = None
        view_35 = torch.ops.aten.view.default(add_22, [1024, 20, 3, 768]);  add_22 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(view_35, 0);  view_35 = None
        permute_27 = torch.ops.aten.permute.default(unsqueeze_8, [3, 1, 2, 0, 4]);  unsqueeze_8 = None
        squeeze_2 = torch.ops.aten.squeeze.dim(permute_27, -2);  permute_27 = None
        clone_7 = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
        select_6 = torch.ops.aten.select.int(clone_7, 0, 0)
        select_7 = torch.ops.aten.select.int(clone_7, 0, 1)
        select_8 = torch.ops.aten.select.int(clone_7, 0, 2);  clone_7 = None
        view_36 = torch.ops.aten.view.default(select_6, [1024, 160, 96]);  select_6 = None
        permute_28 = torch.ops.aten.permute.default(view_36, [1, 0, 2]);  view_36 = None
        view_37 = torch.ops.aten.view.default(select_7, [1024, 160, 96]);  select_7 = None
        permute_29 = torch.ops.aten.permute.default(view_37, [1, 0, 2]);  view_37 = None
        view_38 = torch.ops.aten.view.default(select_8, [1024, 160, 96]);  select_8 = None
        permute_30 = torch.ops.aten.permute.default(view_38, [1, 0, 2]);  view_38 = None
        view_39 = torch.ops.aten.view.default(permute_28, [20, 8, 1024, 96]);  permute_28 = None
        view_40 = torch.ops.aten.view.default(permute_29, [20, 8, 1024, 96]);  permute_29 = None
        view_41 = torch.ops.aten.view.default(permute_30, [20, 8, 1024, 96]);  permute_30 = None
        graphsafe_run_with_rng_state_2 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_39, view_40, view_41, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_2);  bwd_rng_state_2 = None
        getitem_28 = graphsafe_run_with_rng_state_2[0]
        getitem_29 = graphsafe_run_with_rng_state_2[1]
        getitem_34 = graphsafe_run_with_rng_state_2[6]
        getitem_35 = graphsafe_run_with_rng_state_2[7];  graphsafe_run_with_rng_state_2 = None
        permute_31 = torch.ops.aten.permute.default(getitem_28, [2, 0, 1, 3])
        view_42 = torch.ops.aten.view.default(permute_31, [20480, 768]);  permute_31 = None
        mm_94 = torch.ops.aten.mm.default(permute_367, view_42);  permute_367 = view_42 = None
        sum_117 = torch.ops.aten.sum.dim_IntList(view_384, [0], True, dtype = torch.float32);  view_384 = None
        view_385 = torch.ops.aten.view.default(sum_117, [768]);  sum_117 = None
        convert_element_type_667 = torch.ops.prims.convert_element_type.default(mm_94, torch.float32);  mm_94 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(view_385, torch.float32);  view_385 = None
        view_386 = torch.ops.aten.view.default(mm_93, [1024, 20, 8, 96]);  mm_93 = None
        permute_370 = torch.ops.aten.permute.default(view_386, [1, 2, 0, 3]);  view_386 = None
        _scaled_dot_product_flash_attention_backward_9 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_370, view_39, view_40, view_41, getitem_28, getitem_29, None, None, 1024, 1024, 0.2, False, getitem_34, getitem_35, scale = 0.10206207261596577);  permute_370 = view_39 = view_40 = view_41 = getitem_28 = getitem_29 = getitem_34 = getitem_35 = None
        getitem_183 = _scaled_dot_product_flash_attention_backward_9[0]
        getitem_184 = _scaled_dot_product_flash_attention_backward_9[1]
        getitem_185 = _scaled_dot_product_flash_attention_backward_9[2];  _scaled_dot_product_flash_attention_backward_9 = None
        view_387 = torch.ops.aten.view.default(getitem_185, [160, 1024, 96]);  getitem_185 = None
        view_388 = torch.ops.aten.view.default(getitem_184, [160, 1024, 96]);  getitem_184 = None
        view_389 = torch.ops.aten.view.default(getitem_183, [160, 1024, 96]);  getitem_183 = None
        permute_371 = torch.ops.aten.permute.default(view_387, [1, 0, 2]);  view_387 = None
        view_390 = torch.ops.aten.view.default(permute_371, [1024, 20, 768]);  permute_371 = None
        permute_372 = torch.ops.aten.permute.default(view_388, [1, 0, 2]);  view_388 = None
        view_391 = torch.ops.aten.view.default(permute_372, [1024, 20, 768]);  permute_372 = None
        permute_373 = torch.ops.aten.permute.default(view_389, [1, 0, 2]);  view_389 = None
        view_392 = torch.ops.aten.view.default(permute_373, [1024, 20, 768]);  permute_373 = None
        select_scatter_27 = torch.ops.aten.select_scatter.default(full_default_6, view_390, 0, 2);  view_390 = None
        select_scatter_28 = torch.ops.aten.select_scatter.default(full_default_6, view_391, 0, 1);  view_391 = None
        add_174 = torch.ops.aten.add.Tensor(select_scatter_27, select_scatter_28);  select_scatter_27 = select_scatter_28 = None
        select_scatter_29 = torch.ops.aten.select_scatter.default(full_default_6, view_392, 0, 0);  view_392 = None
        add_175 = torch.ops.aten.add.Tensor(add_174, select_scatter_29);  add_174 = select_scatter_29 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(add_175, 3);  add_175 = None
        permute_374 = torch.ops.aten.permute.default(unsqueeze_39, [3, 1, 2, 0, 4]);  unsqueeze_39 = None
        squeeze_21 = torch.ops.aten.squeeze.dim(permute_374, 0);  permute_374 = None
        clone_58 = torch.ops.aten.clone.default(squeeze_21, memory_format = torch.contiguous_format);  squeeze_21 = None
        view_393 = torch.ops.aten.view.default(clone_58, [1024, 20, 2304]);  clone_58 = None
        sum_118 = torch.ops.aten.sum.dim_IntList(view_393, [0, 1], True, dtype = torch.float32)
        view_394 = torch.ops.aten.view.default(sum_118, [2304]);  sum_118 = None
        view_395 = torch.ops.aten.view.default(view_393, [20480, 2304]);  view_393 = None
        permute_375 = torch.ops.aten.permute.default(view_395, [1, 0])
        mm_95 = torch.ops.aten.mm.default(permute_375, view_33);  permute_375 = view_33 = None
        permute_377 = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        mm_96 = torch.ops.aten.mm.default(view_395, permute_377);  view_395 = permute_377 = None
        view_396 = torch.ops.aten.view.default(mm_96, [1024, 20, 768]);  mm_96 = None
        convert_element_type_674 = torch.ops.prims.convert_element_type.default(view_396, torch.float32);  view_396 = None
        convert_element_type_675 = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(view_394, torch.float32);  view_394 = None
        permute_379 = torch.ops.aten.permute.default(convert_element_type_674, [1, 0, 2]);  convert_element_type_674 = None
        mul_340 = torch.ops.aten.mul.Tensor(permute_379, primals_29);  primals_29 = None
        mul_341 = torch.ops.aten.mul.Tensor(mul_340, 768)
        sum_119 = torch.ops.aten.sum.dim_IntList(mul_340, [2], True)
        mul_342 = torch.ops.aten.mul.Tensor(mul_340, mul_18);  mul_340 = None
        sum_120 = torch.ops.aten.sum.dim_IntList(mul_342, [2], True);  mul_342 = None
        mul_343 = torch.ops.aten.mul.Tensor(mul_18, sum_120);  sum_120 = None
        sub_89 = torch.ops.aten.sub.Tensor(mul_341, sum_119);  mul_341 = sum_119 = None
        sub_90 = torch.ops.aten.sub.Tensor(sub_89, mul_343);  sub_89 = mul_343 = None
        div_19 = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        mul_344 = torch.ops.aten.mul.Tensor(div_19, sub_90);  div_19 = sub_90 = None
        mul_345 = torch.ops.aten.mul.Tensor(permute_379, mul_18);  mul_18 = None
        sum_121 = torch.ops.aten.sum.dim_IntList(mul_345, [0, 1]);  mul_345 = None
        sum_122 = torch.ops.aten.sum.dim_IntList(permute_379, [0, 1]);  permute_379 = None
        add_176 = torch.ops.aten.add.Tensor(add_173, mul_344);  add_173 = mul_344 = None
        convert_element_type_677 = torch.ops.prims.convert_element_type.default(add_176, torch.float16)
        inductor_lookup_seed_default_1 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_10 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default_62 = torch.ops.prims.convert_element_type.default(inductor_random_default_10, torch.float16);  inductor_random_default_10 = None
        gt_1 = torch.ops.aten.gt.Scalar(convert_element_type_default_62, 0.2);  convert_element_type_default_62 = None
        convert_element_type_678 = torch.ops.prims.convert_element_type.default(gt_1, torch.float16);  gt_1 = None
        mul_346 = torch.ops.aten.mul.Tensor(convert_element_type_678, 1.25);  convert_element_type_678 = None
        mul_347 = torch.ops.aten.mul.Tensor(convert_element_type_677, mul_346);  convert_element_type_677 = mul_346 = None
        view_397 = torch.ops.aten.view.default(mul_347, [20480, 768]);  mul_347 = None
        convert_element_type_47 = torch.ops.prims.convert_element_type.default(primals_27, torch.float16);  primals_27 = None
        permute_24 = torch.ops.aten.permute.default(convert_element_type_47, [1, 0]);  convert_element_type_47 = None
        permute_380 = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_97 = torch.ops.aten.mm.default(view_397, permute_380);  permute_380 = None
        permute_381 = torch.ops.aten.permute.default(view_397, [1, 0])
        var_mean_3 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_3[0]
        getitem_25 = var_mean_3[1];  var_mean_3 = None
        add_16 = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_15, getitem_25);  add_15 = getitem_25 = None
        mul_11 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        mul_12 = torch.ops.aten.mul.Tensor(mul_11, primals_23)
        add_17 = torch.ops.aten.add.Tensor(mul_12, primals_24);  mul_12 = primals_24 = None
        convert_element_type_38 = torch.ops.prims.convert_element_type.default(primals_26, torch.float16);  primals_26 = None
        convert_element_type_39 = torch.ops.prims.convert_element_type.default(primals_25, torch.float16);  primals_25 = None
        convert_element_type_40 = torch.ops.prims.convert_element_type.default(add_17, torch.float16);  add_17 = None
        view_29 = torch.ops.aten.view.default(convert_element_type_40, [20480, 768]);  convert_element_type_40 = None
        permute_23 = torch.ops.aten.permute.default(convert_element_type_39, [1, 0]);  convert_element_type_39 = None
        addmm_4 = torch.ops.aten.addmm.default(convert_element_type_38, view_29, permute_23);  convert_element_type_38 = None
        view_30 = torch.ops.aten.view.default(addmm_4, [20, 1024, 1536]);  addmm_4 = None
        convert_element_type_44 = torch.ops.prims.convert_element_type.default(view_30, torch.float32);  view_30 = None
        mul_13 = torch.ops.aten.mul.Tensor(convert_element_type_44, 0.5)
        mul_14 = torch.ops.aten.mul.Tensor(convert_element_type_44, 0.7071067811865476)
        erf_1 = torch.ops.aten.erf.default(mul_14);  mul_14 = None
        add_18 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_13, add_18);  mul_13 = None
        convert_element_type_45 = torch.ops.prims.convert_element_type.default(mul_15, torch.float16);  mul_15 = None
        view_31 = torch.ops.aten.view.default(convert_element_type_45, [20480, 1536]);  convert_element_type_45 = None
        mm_98 = torch.ops.aten.mm.default(permute_381, view_31);  permute_381 = view_31 = None
        sum_123 = torch.ops.aten.sum.dim_IntList(view_397, [0], True, dtype = torch.float32);  view_397 = None
        view_398 = torch.ops.aten.view.default(sum_123, [768]);  sum_123 = None
        view_399 = torch.ops.aten.view.default(mm_97, [20, 1024, 1536]);  mm_97 = None
        convert_element_type_684 = torch.ops.prims.convert_element_type.default(mm_98, torch.float32);  mm_98 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(view_398, torch.float32);  view_398 = None
        convert_element_type_686 = torch.ops.prims.convert_element_type.default(view_399, torch.float32);  view_399 = None
        mul_349 = torch.ops.aten.mul.Tensor(add_18, 0.5);  add_18 = None
        mul_350 = torch.ops.aten.mul.Tensor(convert_element_type_44, convert_element_type_44)
        mul_351 = torch.ops.aten.mul.Tensor(mul_350, -0.5);  mul_350 = None
        exp_10 = torch.ops.aten.exp.default(mul_351);  mul_351 = None
        mul_352 = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
        mul_353 = torch.ops.aten.mul.Tensor(convert_element_type_44, mul_352);  convert_element_type_44 = mul_352 = None
        add_178 = torch.ops.aten.add.Tensor(mul_349, mul_353);  mul_349 = mul_353 = None
        mul_354 = torch.ops.aten.mul.Tensor(convert_element_type_686, add_178);  convert_element_type_686 = add_178 = None
        convert_element_type_688 = torch.ops.prims.convert_element_type.default(mul_354, torch.float16);  mul_354 = None
        view_400 = torch.ops.aten.view.default(convert_element_type_688, [20480, 1536]);  convert_element_type_688 = None
        permute_384 = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_99 = torch.ops.aten.mm.default(view_400, permute_384);  permute_384 = None
        permute_385 = torch.ops.aten.permute.default(view_400, [1, 0])
        mm_100 = torch.ops.aten.mm.default(permute_385, view_29);  permute_385 = view_29 = None
        sum_124 = torch.ops.aten.sum.dim_IntList(view_400, [0], True, dtype = torch.float32);  view_400 = None
        view_401 = torch.ops.aten.view.default(sum_124, [1536]);  sum_124 = None
        view_402 = torch.ops.aten.view.default(mm_99, [20, 1024, 768]);  mm_99 = None
        convert_element_type_694 = torch.ops.prims.convert_element_type.default(view_402, torch.float32);  view_402 = None
        convert_element_type_695 = torch.ops.prims.convert_element_type.default(mm_100, torch.float32);  mm_100 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(view_401, torch.float32);  view_401 = None
        mul_356 = torch.ops.aten.mul.Tensor(convert_element_type_694, primals_23);  primals_23 = None
        mul_357 = torch.ops.aten.mul.Tensor(mul_356, 768)
        sum_125 = torch.ops.aten.sum.dim_IntList(mul_356, [2], True)
        mul_358 = torch.ops.aten.mul.Tensor(mul_356, mul_11);  mul_356 = None
        sum_126 = torch.ops.aten.sum.dim_IntList(mul_358, [2], True);  mul_358 = None
        mul_359 = torch.ops.aten.mul.Tensor(mul_11, sum_126);  sum_126 = None
        sub_92 = torch.ops.aten.sub.Tensor(mul_357, sum_125);  mul_357 = sum_125 = None
        sub_93 = torch.ops.aten.sub.Tensor(sub_92, mul_359);  sub_92 = mul_359 = None
        div_20 = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        mul_360 = torch.ops.aten.mul.Tensor(div_20, sub_93);  div_20 = sub_93 = None
        mul_361 = torch.ops.aten.mul.Tensor(convert_element_type_694, mul_11);  mul_11 = None
        sum_127 = torch.ops.aten.sum.dim_IntList(mul_361, [0, 1]);  mul_361 = None
        sum_128 = torch.ops.aten.sum.dim_IntList(convert_element_type_694, [0, 1]);  convert_element_type_694 = None
        add_179 = torch.ops.aten.add.Tensor(add_176, mul_360);  add_176 = mul_360 = None
        convert_element_type_697 = torch.ops.prims.convert_element_type.default(add_179, torch.float16)
        permute_388 = torch.ops.aten.permute.default(convert_element_type_697, [1, 0, 2]);  convert_element_type_697 = None
        clone_60 = torch.ops.aten.clone.default(permute_388, memory_format = torch.contiguous_format);  permute_388 = None
        view_403 = torch.ops.aten.view.default(clone_60, [20480, 768]);  clone_60 = None
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(primals_21, torch.float16);  primals_21 = None
        permute_21 = torch.ops.aten.permute.default(convert_element_type_34, [1, 0]);  convert_element_type_34 = None
        permute_389 = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_101 = torch.ops.aten.mm.default(view_403, permute_389);  permute_389 = None
        permute_390 = torch.ops.aten.permute.default(view_403, [1, 0])
        var_mean_2 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_13 = var_mean_2[0]
        getitem_14 = var_mean_2[1];  var_mean_2 = None
        add_12 = torch.ops.aten.add.Tensor(getitem_13, 1e-05);  getitem_13 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_11, getitem_14);  add_11 = getitem_14 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, primals_17)
        add_13 = torch.ops.aten.add.Tensor(mul_10, primals_18);  mul_10 = primals_18 = None
        permute_14 = torch.ops.aten.permute.default(add_13, [1, 0, 2]);  add_13 = None
        convert_element_type_28 = torch.ops.prims.convert_element_type.default(primals_19, torch.float16);  primals_19 = None
        convert_element_type_29 = torch.ops.prims.convert_element_type.default(primals_20, torch.float16);  primals_20 = None
        convert_element_type_30 = torch.ops.prims.convert_element_type.default(permute_14, torch.float16);  permute_14 = None
        permute_15 = torch.ops.aten.permute.default(convert_element_type_29, [1, 0]);  convert_element_type_29 = None
        clone_4 = torch.ops.aten.clone.default(convert_element_type_30, memory_format = torch.contiguous_format);  convert_element_type_30 = None
        view_18 = torch.ops.aten.view.default(clone_4, [20480, 768]);  clone_4 = None
        mm_2 = torch.ops.aten.mm.default(view_18, permute_15)
        view_19 = torch.ops.aten.view.default(mm_2, [1024, 20, 2304]);  mm_2 = None
        add_14 = torch.ops.aten.add.Tensor(view_19, convert_element_type_28);  view_19 = convert_element_type_28 = None
        view_20 = torch.ops.aten.view.default(add_14, [1024, 20, 3, 768]);  add_14 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(view_20, 0);  view_20 = None
        permute_16 = torch.ops.aten.permute.default(unsqueeze_7, [3, 1, 2, 0, 4]);  unsqueeze_7 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(permute_16, -2);  permute_16 = None
        clone_5 = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        select_3 = torch.ops.aten.select.int(clone_5, 0, 0)
        select_4 = torch.ops.aten.select.int(clone_5, 0, 1)
        select_5 = torch.ops.aten.select.int(clone_5, 0, 2);  clone_5 = None
        view_21 = torch.ops.aten.view.default(select_3, [1024, 160, 96]);  select_3 = None
        permute_17 = torch.ops.aten.permute.default(view_21, [1, 0, 2]);  view_21 = None
        view_22 = torch.ops.aten.view.default(select_4, [1024, 160, 96]);  select_4 = None
        permute_18 = torch.ops.aten.permute.default(view_22, [1, 0, 2]);  view_22 = None
        view_23 = torch.ops.aten.view.default(select_5, [1024, 160, 96]);  select_5 = None
        permute_19 = torch.ops.aten.permute.default(view_23, [1, 0, 2]);  view_23 = None
        view_24 = torch.ops.aten.view.default(permute_17, [20, 8, 1024, 96]);  permute_17 = None
        view_25 = torch.ops.aten.view.default(permute_18, [20, 8, 1024, 96]);  permute_18 = None
        view_26 = torch.ops.aten.view.default(permute_19, [20, 8, 1024, 96]);  permute_19 = None
        graphsafe_run_with_rng_state_1 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_24, view_25, view_26, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_1);  bwd_rng_state_1 = None
        getitem_15 = graphsafe_run_with_rng_state_1[0]
        getitem_16 = graphsafe_run_with_rng_state_1[1]
        getitem_21 = graphsafe_run_with_rng_state_1[6]
        getitem_22 = graphsafe_run_with_rng_state_1[7];  graphsafe_run_with_rng_state_1 = None
        permute_20 = torch.ops.aten.permute.default(getitem_15, [2, 0, 1, 3])
        view_27 = torch.ops.aten.view.default(permute_20, [20480, 768]);  permute_20 = None
        mm_102 = torch.ops.aten.mm.default(permute_390, view_27);  permute_390 = view_27 = None
        sum_129 = torch.ops.aten.sum.dim_IntList(view_403, [0], True, dtype = torch.float32);  view_403 = None
        view_404 = torch.ops.aten.view.default(sum_129, [768]);  sum_129 = None
        convert_element_type_703 = torch.ops.prims.convert_element_type.default(mm_102, torch.float32);  mm_102 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_404, torch.float32);  view_404 = None
        view_405 = torch.ops.aten.view.default(mm_101, [1024, 20, 8, 96]);  mm_101 = None
        permute_393 = torch.ops.aten.permute.default(view_405, [1, 2, 0, 3]);  view_405 = None
        _scaled_dot_product_flash_attention_backward_10 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_393, view_24, view_25, view_26, getitem_15, getitem_16, None, None, 1024, 1024, 0.2, False, getitem_21, getitem_22, scale = 0.10206207261596577);  permute_393 = view_24 = view_25 = view_26 = getitem_15 = getitem_16 = getitem_21 = getitem_22 = None
        getitem_186 = _scaled_dot_product_flash_attention_backward_10[0]
        getitem_187 = _scaled_dot_product_flash_attention_backward_10[1]
        getitem_188 = _scaled_dot_product_flash_attention_backward_10[2];  _scaled_dot_product_flash_attention_backward_10 = None
        view_406 = torch.ops.aten.view.default(getitem_188, [160, 1024, 96]);  getitem_188 = None
        view_407 = torch.ops.aten.view.default(getitem_187, [160, 1024, 96]);  getitem_187 = None
        view_408 = torch.ops.aten.view.default(getitem_186, [160, 1024, 96]);  getitem_186 = None
        permute_394 = torch.ops.aten.permute.default(view_406, [1, 0, 2]);  view_406 = None
        view_409 = torch.ops.aten.view.default(permute_394, [1024, 20, 768]);  permute_394 = None
        permute_395 = torch.ops.aten.permute.default(view_407, [1, 0, 2]);  view_407 = None
        view_410 = torch.ops.aten.view.default(permute_395, [1024, 20, 768]);  permute_395 = None
        permute_396 = torch.ops.aten.permute.default(view_408, [1, 0, 2]);  view_408 = None
        view_411 = torch.ops.aten.view.default(permute_396, [1024, 20, 768]);  permute_396 = None
        select_scatter_30 = torch.ops.aten.select_scatter.default(full_default_6, view_409, 0, 2);  view_409 = None
        select_scatter_31 = torch.ops.aten.select_scatter.default(full_default_6, view_410, 0, 1);  view_410 = None
        add_180 = torch.ops.aten.add.Tensor(select_scatter_30, select_scatter_31);  select_scatter_30 = select_scatter_31 = None
        select_scatter_32 = torch.ops.aten.select_scatter.default(full_default_6, view_411, 0, 0);  view_411 = None
        add_181 = torch.ops.aten.add.Tensor(add_180, select_scatter_32);  add_180 = select_scatter_32 = None
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(add_181, 3);  add_181 = None
        permute_397 = torch.ops.aten.permute.default(unsqueeze_40, [3, 1, 2, 0, 4]);  unsqueeze_40 = None
        squeeze_22 = torch.ops.aten.squeeze.dim(permute_397, 0);  permute_397 = None
        clone_61 = torch.ops.aten.clone.default(squeeze_22, memory_format = torch.contiguous_format);  squeeze_22 = None
        view_412 = torch.ops.aten.view.default(clone_61, [1024, 20, 2304]);  clone_61 = None
        sum_130 = torch.ops.aten.sum.dim_IntList(view_412, [0, 1], True, dtype = torch.float32)
        view_413 = torch.ops.aten.view.default(sum_130, [2304]);  sum_130 = None
        view_414 = torch.ops.aten.view.default(view_412, [20480, 2304]);  view_412 = None
        permute_398 = torch.ops.aten.permute.default(view_414, [1, 0])
        mm_103 = torch.ops.aten.mm.default(permute_398, view_18);  permute_398 = view_18 = None
        permute_400 = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        mm_104 = torch.ops.aten.mm.default(view_414, permute_400);  view_414 = permute_400 = None
        view_415 = torch.ops.aten.view.default(mm_104, [1024, 20, 768]);  mm_104 = None
        convert_element_type_710 = torch.ops.prims.convert_element_type.default(view_415, torch.float32);  view_415 = None
        convert_element_type_711 = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_413, torch.float32);  view_413 = None
        permute_402 = torch.ops.aten.permute.default(convert_element_type_710, [1, 0, 2]);  convert_element_type_710 = None
        mul_363 = torch.ops.aten.mul.Tensor(permute_402, primals_17);  primals_17 = None
        mul_364 = torch.ops.aten.mul.Tensor(mul_363, 768)
        sum_131 = torch.ops.aten.sum.dim_IntList(mul_363, [2], True)
        mul_365 = torch.ops.aten.mul.Tensor(mul_363, mul_9);  mul_363 = None
        sum_132 = torch.ops.aten.sum.dim_IntList(mul_365, [2], True);  mul_365 = None
        mul_366 = torch.ops.aten.mul.Tensor(mul_9, sum_132);  sum_132 = None
        sub_95 = torch.ops.aten.sub.Tensor(mul_364, sum_131);  mul_364 = sum_131 = None
        sub_96 = torch.ops.aten.sub.Tensor(sub_95, mul_366);  sub_95 = mul_366 = None
        div_21 = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        mul_367 = torch.ops.aten.mul.Tensor(div_21, sub_96);  div_21 = sub_96 = None
        mul_368 = torch.ops.aten.mul.Tensor(permute_402, mul_9);  mul_9 = None
        sum_133 = torch.ops.aten.sum.dim_IntList(mul_368, [0, 1]);  mul_368 = None
        sum_134 = torch.ops.aten.sum.dim_IntList(permute_402, [0, 1]);  permute_402 = None
        add_182 = torch.ops.aten.add.Tensor(add_179, mul_367);  add_179 = mul_367 = None
        convert_element_type_713 = torch.ops.prims.convert_element_type.default(add_182, torch.float16)
        inductor_lookup_seed_default = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default_11 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default_63 = torch.ops.prims.convert_element_type.default(inductor_random_default_11, torch.float16);  inductor_random_default_11 = None
        gt = torch.ops.aten.gt.Scalar(convert_element_type_default_63, 0.2);  convert_element_type_default_63 = None
        convert_element_type_714 = torch.ops.prims.convert_element_type.default(gt, torch.float16);  gt = None
        mul_369 = torch.ops.aten.mul.Tensor(convert_element_type_714, 1.25);  convert_element_type_714 = None
        mul_370 = torch.ops.aten.mul.Tensor(convert_element_type_713, mul_369);  convert_element_type_713 = mul_369 = None
        view_416 = torch.ops.aten.view.default(mul_370, [20480, 768]);  mul_370 = None
        convert_element_type_24 = torch.ops.prims.convert_element_type.default(primals_15, torch.float16);  primals_15 = None
        permute_13 = torch.ops.aten.permute.default(convert_element_type_24, [1, 0]);  convert_element_type_24 = None
        permute_403 = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_105 = torch.ops.aten.mm.default(view_416, permute_403);  permute_403 = None
        permute_404 = torch.ops.aten.permute.default(view_416, [1, 0])
        var_mean_1 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_11 = var_mean_1[0]
        getitem_12 = var_mean_1[1];  var_mean_1 = None
        add_8 = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_7, getitem_12);  add_7 = getitem_12 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, primals_11)
        add_9 = torch.ops.aten.add.Tensor(mul_3, primals_12);  mul_3 = primals_12 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(primals_14, torch.float16);  primals_14 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(primals_13, torch.float16);  primals_13 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(add_9, torch.float16);  add_9 = None
        view_14 = torch.ops.aten.view.default(convert_element_type_17, [20480, 768]);  convert_element_type_17 = None
        permute_12 = torch.ops.aten.permute.default(convert_element_type_16, [1, 0]);  convert_element_type_16 = None
        addmm_1 = torch.ops.aten.addmm.default(convert_element_type_15, view_14, permute_12);  convert_element_type_15 = None
        view_15 = torch.ops.aten.view.default(addmm_1, [20, 1024, 1536]);  addmm_1 = None
        convert_element_type_21 = torch.ops.prims.convert_element_type.default(view_15, torch.float32);  view_15 = None
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.5)
        mul_5 = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.7071067811865476)
        erf = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_10 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_4, add_10);  mul_4 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(mul_6, torch.float16);  mul_6 = None
        view_16 = torch.ops.aten.view.default(convert_element_type_22, [20480, 1536]);  convert_element_type_22 = None
        mm_106 = torch.ops.aten.mm.default(permute_404, view_16);  permute_404 = view_16 = None
        sum_135 = torch.ops.aten.sum.dim_IntList(view_416, [0], True, dtype = torch.float32);  view_416 = None
        view_417 = torch.ops.aten.view.default(sum_135, [768]);  sum_135 = None
        view_418 = torch.ops.aten.view.default(mm_105, [20, 1024, 1536]);  mm_105 = None
        convert_element_type_720 = torch.ops.prims.convert_element_type.default(mm_106, torch.float32);  mm_106 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_417, torch.float32);  view_417 = None
        convert_element_type_722 = torch.ops.prims.convert_element_type.default(view_418, torch.float32);  view_418 = None
        mul_372 = torch.ops.aten.mul.Tensor(add_10, 0.5);  add_10 = None
        mul_373 = torch.ops.aten.mul.Tensor(convert_element_type_21, convert_element_type_21)
        mul_374 = torch.ops.aten.mul.Tensor(mul_373, -0.5);  mul_373 = None
        exp_11 = torch.ops.aten.exp.default(mul_374);  mul_374 = None
        mul_375 = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
        mul_376 = torch.ops.aten.mul.Tensor(convert_element_type_21, mul_375);  convert_element_type_21 = mul_375 = None
        add_184 = torch.ops.aten.add.Tensor(mul_372, mul_376);  mul_372 = mul_376 = None
        mul_377 = torch.ops.aten.mul.Tensor(convert_element_type_722, add_184);  convert_element_type_722 = add_184 = None
        convert_element_type_724 = torch.ops.prims.convert_element_type.default(mul_377, torch.float16);  mul_377 = None
        view_419 = torch.ops.aten.view.default(convert_element_type_724, [20480, 1536]);  convert_element_type_724 = None
        permute_407 = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_107 = torch.ops.aten.mm.default(view_419, permute_407);  permute_407 = None
        permute_408 = torch.ops.aten.permute.default(view_419, [1, 0])
        mm_108 = torch.ops.aten.mm.default(permute_408, view_14);  permute_408 = view_14 = None
        sum_136 = torch.ops.aten.sum.dim_IntList(view_419, [0], True, dtype = torch.float32);  view_419 = None
        view_420 = torch.ops.aten.view.default(sum_136, [1536]);  sum_136 = None
        view_421 = torch.ops.aten.view.default(mm_107, [20, 1024, 768]);  mm_107 = None
        convert_element_type_730 = torch.ops.prims.convert_element_type.default(view_421, torch.float32);  view_421 = None
        convert_element_type_731 = torch.ops.prims.convert_element_type.default(mm_108, torch.float32);  mm_108 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_420, torch.float32);  view_420 = None
        mul_379 = torch.ops.aten.mul.Tensor(convert_element_type_730, primals_11);  primals_11 = None
        mul_380 = torch.ops.aten.mul.Tensor(mul_379, 768)
        sum_137 = torch.ops.aten.sum.dim_IntList(mul_379, [2], True)
        mul_381 = torch.ops.aten.mul.Tensor(mul_379, mul_2);  mul_379 = None
        sum_138 = torch.ops.aten.sum.dim_IntList(mul_381, [2], True);  mul_381 = None
        mul_382 = torch.ops.aten.mul.Tensor(mul_2, sum_138);  sum_138 = None
        sub_98 = torch.ops.aten.sub.Tensor(mul_380, sum_137);  mul_380 = sum_137 = None
        sub_99 = torch.ops.aten.sub.Tensor(sub_98, mul_382);  sub_98 = mul_382 = None
        div_22 = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        mul_383 = torch.ops.aten.mul.Tensor(div_22, sub_99);  div_22 = sub_99 = None
        mul_384 = torch.ops.aten.mul.Tensor(convert_element_type_730, mul_2);  mul_2 = None
        sum_139 = torch.ops.aten.sum.dim_IntList(mul_384, [0, 1]);  mul_384 = None
        sum_140 = torch.ops.aten.sum.dim_IntList(convert_element_type_730, [0, 1]);  convert_element_type_730 = None
        add_185 = torch.ops.aten.add.Tensor(add_182, mul_383);  add_182 = mul_383 = None
        convert_element_type_733 = torch.ops.prims.convert_element_type.default(add_185, torch.float16)
        permute_411 = torch.ops.aten.permute.default(convert_element_type_733, [1, 0, 2]);  convert_element_type_733 = None
        clone_63 = torch.ops.aten.clone.default(permute_411, memory_format = torch.contiguous_format);  permute_411 = None
        view_422 = torch.ops.aten.view.default(clone_63, [20480, 768]);  clone_63 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(primals_9, torch.float16);  primals_9 = None
        permute_10 = torch.ops.aten.permute.default(convert_element_type_11, [1, 0]);  convert_element_type_11 = None
        permute_412 = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_109 = torch.ops.aten.mm.default(view_422, permute_412);  permute_412 = None
        permute_413 = torch.ops.aten.permute.default(view_422, [1, 0])
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_3, torch.float16);  primals_3 = None
        view_2 = torch.ops.aten.view.default(mm, [20, 1024, 768]);  mm = None
        add_2 = torch.ops.aten.add.Tensor(view_2, convert_element_type);  view_2 = convert_element_type = None
        add_3 = torch.ops.aten.add.Tensor(add_2, primals_4);  add_2 = primals_4 = None
        var_mean = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_4 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub = torch.ops.aten.sub.Tensor(add_3, getitem_1);  add_3 = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, primals_5)
        add_5 = torch.ops.aten.add.Tensor(mul_1, primals_6);  mul_1 = primals_6 = None
        permute_3 = torch.ops.aten.permute.default(add_5, [1, 0, 2]);  add_5 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(primals_7, torch.float16);  primals_7 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(primals_8, torch.float16);  primals_8 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(permute_3, torch.float16);  permute_3 = None
        permute_4 = torch.ops.aten.permute.default(convert_element_type_6, [1, 0]);  convert_element_type_6 = None
        clone_2 = torch.ops.aten.clone.default(convert_element_type_7, memory_format = torch.contiguous_format);  convert_element_type_7 = None
        view_3 = torch.ops.aten.view.default(clone_2, [20480, 768]);  clone_2 = None
        mm_1 = torch.ops.aten.mm.default(view_3, permute_4)
        view_4 = torch.ops.aten.view.default(mm_1, [1024, 20, 2304]);  mm_1 = None
        add_6 = torch.ops.aten.add.Tensor(view_4, convert_element_type_5);  view_4 = convert_element_type_5 = None
        view_5 = torch.ops.aten.view.default(add_6, [1024, 20, 3, 768]);  add_6 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(view_5, 0);  view_5 = None
        permute_5 = torch.ops.aten.permute.default(unsqueeze_6, [3, 1, 2, 0, 4]);  unsqueeze_6 = None
        squeeze = torch.ops.aten.squeeze.dim(permute_5, -2);  permute_5 = None
        clone_3 = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        select = torch.ops.aten.select.int(clone_3, 0, 0)
        select_1 = torch.ops.aten.select.int(clone_3, 0, 1)
        select_2 = torch.ops.aten.select.int(clone_3, 0, 2);  clone_3 = None
        view_6 = torch.ops.aten.view.default(select, [1024, 160, 96]);  select = None
        permute_6 = torch.ops.aten.permute.default(view_6, [1, 0, 2]);  view_6 = None
        view_7 = torch.ops.aten.view.default(select_1, [1024, 160, 96]);  select_1 = None
        permute_7 = torch.ops.aten.permute.default(view_7, [1, 0, 2]);  view_7 = None
        view_8 = torch.ops.aten.view.default(select_2, [1024, 160, 96]);  select_2 = None
        permute_8 = torch.ops.aten.permute.default(view_8, [1, 0, 2]);  view_8 = None
        view_9 = torch.ops.aten.view.default(permute_6, [20, 8, 1024, 96]);  permute_6 = None
        view_10 = torch.ops.aten.view.default(permute_7, [20, 8, 1024, 96]);  permute_7 = None
        view_11 = torch.ops.aten.view.default(permute_8, [20, 8, 1024, 96]);  permute_8 = None
        graphsafe_run_with_rng_state = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_9, view_10, view_11, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_0);  bwd_rng_state_0 = None
        getitem_2 = graphsafe_run_with_rng_state[0]
        getitem_3 = graphsafe_run_with_rng_state[1]
        getitem_8 = graphsafe_run_with_rng_state[6]
        getitem_9 = graphsafe_run_with_rng_state[7];  graphsafe_run_with_rng_state = None
        permute_9 = torch.ops.aten.permute.default(getitem_2, [2, 0, 1, 3])
        view_12 = torch.ops.aten.view.default(permute_9, [20480, 768]);  permute_9 = None
        mm_110 = torch.ops.aten.mm.default(permute_413, view_12);  permute_413 = view_12 = None
        sum_141 = torch.ops.aten.sum.dim_IntList(view_422, [0], True, dtype = torch.float32);  view_422 = None
        view_423 = torch.ops.aten.view.default(sum_141, [768]);  sum_141 = None
        convert_element_type_739 = torch.ops.prims.convert_element_type.default(mm_110, torch.float32);  mm_110 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_423, torch.float32);  view_423 = None
        view_424 = torch.ops.aten.view.default(mm_109, [1024, 20, 8, 96]);  mm_109 = None
        permute_416 = torch.ops.aten.permute.default(view_424, [1, 2, 0, 3]);  view_424 = None
        _scaled_dot_product_flash_attention_backward_11 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_416, view_9, view_10, view_11, getitem_2, getitem_3, None, None, 1024, 1024, 0.2, False, getitem_8, getitem_9, scale = 0.10206207261596577);  permute_416 = view_9 = view_10 = view_11 = getitem_2 = getitem_3 = getitem_8 = getitem_9 = None
        getitem_189 = _scaled_dot_product_flash_attention_backward_11[0]
        getitem_190 = _scaled_dot_product_flash_attention_backward_11[1]
        getitem_191 = _scaled_dot_product_flash_attention_backward_11[2];  _scaled_dot_product_flash_attention_backward_11 = None
        view_425 = torch.ops.aten.view.default(getitem_191, [160, 1024, 96]);  getitem_191 = None
        view_426 = torch.ops.aten.view.default(getitem_190, [160, 1024, 96]);  getitem_190 = None
        view_427 = torch.ops.aten.view.default(getitem_189, [160, 1024, 96]);  getitem_189 = None
        permute_417 = torch.ops.aten.permute.default(view_425, [1, 0, 2]);  view_425 = None
        view_428 = torch.ops.aten.view.default(permute_417, [1024, 20, 768]);  permute_417 = None
        permute_418 = torch.ops.aten.permute.default(view_426, [1, 0, 2]);  view_426 = None
        view_429 = torch.ops.aten.view.default(permute_418, [1024, 20, 768]);  permute_418 = None
        permute_419 = torch.ops.aten.permute.default(view_427, [1, 0, 2]);  view_427 = None
        view_430 = torch.ops.aten.view.default(permute_419, [1024, 20, 768]);  permute_419 = None
        select_scatter_33 = torch.ops.aten.select_scatter.default(full_default_6, view_428, 0, 2);  view_428 = None
        select_scatter_34 = torch.ops.aten.select_scatter.default(full_default_6, view_429, 0, 1);  view_429 = None
        add_186 = torch.ops.aten.add.Tensor(select_scatter_33, select_scatter_34);  select_scatter_33 = select_scatter_34 = None
        select_scatter_35 = torch.ops.aten.select_scatter.default(full_default_6, view_430, 0, 0);  full_default_6 = view_430 = None
        add_187 = torch.ops.aten.add.Tensor(add_186, select_scatter_35);  add_186 = select_scatter_35 = None
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(add_187, 3);  add_187 = None
        permute_420 = torch.ops.aten.permute.default(unsqueeze_41, [3, 1, 2, 0, 4]);  unsqueeze_41 = None
        squeeze_23 = torch.ops.aten.squeeze.dim(permute_420, 0);  permute_420 = None
        clone_64 = torch.ops.aten.clone.default(squeeze_23, memory_format = torch.contiguous_format);  squeeze_23 = None
        view_431 = torch.ops.aten.view.default(clone_64, [1024, 20, 2304]);  clone_64 = None
        sum_142 = torch.ops.aten.sum.dim_IntList(view_431, [0, 1], True, dtype = torch.float32)
        view_432 = torch.ops.aten.view.default(sum_142, [2304]);  sum_142 = None
        view_433 = torch.ops.aten.view.default(view_431, [20480, 2304]);  view_431 = None
        permute_421 = torch.ops.aten.permute.default(view_433, [1, 0])
        mm_111 = torch.ops.aten.mm.default(permute_421, view_3);  permute_421 = view_3 = None
        permute_423 = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_112 = torch.ops.aten.mm.default(view_433, permute_423);  view_433 = permute_423 = None
        view_434 = torch.ops.aten.view.default(mm_112, [1024, 20, 768]);  mm_112 = None
        convert_element_type_746 = torch.ops.prims.convert_element_type.default(view_434, torch.float32);  view_434 = None
        convert_element_type_747 = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_432, torch.float32);  view_432 = None
        permute_425 = torch.ops.aten.permute.default(convert_element_type_746, [1, 0, 2]);  convert_element_type_746 = None
        mul_386 = torch.ops.aten.mul.Tensor(permute_425, primals_5);  primals_5 = None
        mul_387 = torch.ops.aten.mul.Tensor(mul_386, 768)
        sum_143 = torch.ops.aten.sum.dim_IntList(mul_386, [2], True)
        mul_388 = torch.ops.aten.mul.Tensor(mul_386, mul);  mul_386 = None
        sum_144 = torch.ops.aten.sum.dim_IntList(mul_388, [2], True);  mul_388 = None
        mul_389 = torch.ops.aten.mul.Tensor(mul, sum_144);  sum_144 = None
        sub_101 = torch.ops.aten.sub.Tensor(mul_387, sum_143);  mul_387 = sum_143 = None
        sub_102 = torch.ops.aten.sub.Tensor(sub_101, mul_389);  sub_101 = mul_389 = None
        div_23 = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_390 = torch.ops.aten.mul.Tensor(div_23, sub_102);  div_23 = sub_102 = None
        mul_391 = torch.ops.aten.mul.Tensor(permute_425, mul);  mul = None
        sum_145 = torch.ops.aten.sum.dim_IntList(mul_391, [0, 1]);  mul_391 = None
        sum_146 = torch.ops.aten.sum.dim_IntList(permute_425, [0, 1]);  permute_425 = None
        add_188 = torch.ops.aten.add.Tensor(add_185, mul_390);  add_185 = mul_390 = None
        convert_element_type_749 = torch.ops.prims.convert_element_type.default(add_188, torch.float16)
        sum_147 = torch.ops.aten.sum.dim_IntList(add_188, [0], True, dtype = torch.float32);  add_188 = None
        view_435 = torch.ops.aten.view.default(sum_147, [1024, 768]);  sum_147 = None
        sum_148 = torch.ops.aten.sum.dim_IntList(convert_element_type_749, [0, 1], True, dtype = torch.float32)
        view_436 = torch.ops.aten.view.default(sum_148, [768]);  sum_148 = None
        view_437 = torch.ops.aten.view.default(convert_element_type_749, [20480, 768]);  convert_element_type_749 = None
        permute_426 = torch.ops.aten.permute.default(view_437, [1, 0]);  view_437 = None
        mm_113 = torch.ops.aten.mm.default(permute_426, view_1);  permute_426 = view_1 = None
        convert_element_type_753 = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_436, torch.float32);  view_436 = None
        return (None, convert_element_type_753, convert_element_type_default, view_435, sum_145, sum_146, convert_element_type_default_1, convert_element_type_747, convert_element_type_739, convert_element_type_default_2, sum_139, sum_140, convert_element_type_731, convert_element_type_default_3, convert_element_type_720, convert_element_type_default_4, sum_133, sum_134, convert_element_type_default_5, convert_element_type_711, convert_element_type_703, convert_element_type_default_6, sum_127, sum_128, convert_element_type_695, convert_element_type_default_7, convert_element_type_684, convert_element_type_default_8, sum_121, sum_122, convert_element_type_default_9, convert_element_type_675, convert_element_type_667, convert_element_type_default_10, sum_115, sum_116, convert_element_type_659, convert_element_type_default_11, convert_element_type_648, convert_element_type_default_12, sum_109, sum_110, convert_element_type_default_13, convert_element_type_639, convert_element_type_631, convert_element_type_default_14, sum_103, sum_104, convert_element_type_623, convert_element_type_default_15, convert_element_type_612, convert_element_type_default_16, sum_97, sum_98, convert_element_type_default_17, convert_element_type_603, convert_element_type_595, convert_element_type_default_18, sum_91, sum_92, convert_element_type_587, convert_element_type_default_19, convert_element_type_576, convert_element_type_default_20, sum_85, sum_86, convert_element_type_default_21, convert_element_type_567, convert_element_type_559, convert_element_type_default_22, sum_79, sum_80, convert_element_type_551, convert_element_type_default_23, convert_element_type_540, convert_element_type_default_24, sum_73, sum_74, convert_element_type_default_25, convert_element_type_531, convert_element_type_523, convert_element_type_default_26, sum_67, sum_68, convert_element_type_515, convert_element_type_default_27, convert_element_type_504, convert_element_type_default_28, sum_61, sum_62, convert_element_type_default_29, convert_element_type_495, convert_element_type_487, convert_element_type_default_30, sum_55, sum_56, convert_element_type_479, convert_element_type_default_31, convert_element_type_468, convert_element_type_default_32, sum_49, sum_50, convert_element_type_default_33, convert_element_type_459, convert_element_type_451, convert_element_type_default_34, sum_43, sum_44, convert_element_type_443, convert_element_type_default_35, convert_element_type_432, convert_element_type_default_36, sum_37, sum_38, convert_element_type_default_37, convert_element_type_423, convert_element_type_415, convert_element_type_default_38, sum_31, sum_32, convert_element_type_407, convert_element_type_default_39, convert_element_type_396, convert_element_type_default_40, sum_25, sum_26, convert_element_type_default_41, convert_element_type_387, convert_element_type_379, convert_element_type_default_42, sum_19, sum_20, convert_element_type_371, convert_element_type_default_43, convert_element_type_360, convert_element_type_default_44, sum_13, sum_14, convert_element_type_default_45, convert_element_type_351, convert_element_type_343, convert_element_type_default_46, sum_7, sum_8, convert_element_type_335, convert_element_type_default_47, convert_element_type_324, convert_element_type_default_48, convert_element_type_315, convert_element_type_default_49, convert_element_type_307, convert_element_type_default_50)
        
def load_args(reader):
    buf0 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf0, (768,), is_leaf=True)  # primals_3
    buf1 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf1, (1024, 768), is_leaf=True)  # primals_4
    buf2 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf2, (768,), is_leaf=True)  # primals_5
    buf3 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf3, (768,), is_leaf=True)  # primals_6
    buf4 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf4, (2304,), is_leaf=True)  # primals_7
    buf5 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf5, (2304, 768), is_leaf=True)  # primals_8
    buf6 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768, 768), is_leaf=True)  # primals_9
    buf7 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf7, (768,), is_leaf=True)  # primals_11
    buf8 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf8, (768,), is_leaf=True)  # primals_12
    buf9 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf9, (1536, 768), is_leaf=True)  # primals_13
    buf10 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf10, (1536,), is_leaf=True)  # primals_14
    buf11 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf11, (768, 1536), is_leaf=True)  # primals_15
    buf12 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf12, (768,), is_leaf=True)  # primals_17
    buf13 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf13, (768,), is_leaf=True)  # primals_18
    buf14 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf14, (2304,), is_leaf=True)  # primals_19
    buf15 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf15, (2304, 768), is_leaf=True)  # primals_20
    buf16 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf16, (768, 768), is_leaf=True)  # primals_21
    buf17 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf17, (768,), is_leaf=True)  # primals_23
    buf18 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768,), is_leaf=True)  # primals_24
    buf19 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf19, (1536, 768), is_leaf=True)  # primals_25
    buf20 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf20, (1536,), is_leaf=True)  # primals_26
    buf21 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768, 1536), is_leaf=True)  # primals_27
    buf22 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768,), is_leaf=True)  # primals_29
    buf23 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768,), is_leaf=True)  # primals_30
    buf24 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf24, (2304,), is_leaf=True)  # primals_31
    buf25 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf25, (2304, 768), is_leaf=True)  # primals_32
    buf26 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf26, (768, 768), is_leaf=True)  # primals_33
    buf27 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf27, (768,), is_leaf=True)  # primals_35
    buf28 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf28, (768,), is_leaf=True)  # primals_36
    buf29 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf29, (1536, 768), is_leaf=True)  # primals_37
    buf30 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf30, (1536,), is_leaf=True)  # primals_38
    buf31 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf31, (768, 1536), is_leaf=True)  # primals_39
    buf32 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf32, (768,), is_leaf=True)  # primals_41
    buf33 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf33, (768,), is_leaf=True)  # primals_42
    buf34 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf34, (2304,), is_leaf=True)  # primals_43
    buf35 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf35, (2304, 768), is_leaf=True)  # primals_44
    buf36 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf36, (768, 768), is_leaf=True)  # primals_45
    buf37 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf37, (768,), is_leaf=True)  # primals_47
    buf38 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf38, (768,), is_leaf=True)  # primals_48
    buf39 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf39, (1536, 768), is_leaf=True)  # primals_49
    buf40 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf40, (1536,), is_leaf=True)  # primals_50
    buf41 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf41, (768, 1536), is_leaf=True)  # primals_51
    buf42 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf42, (768,), is_leaf=True)  # primals_53
    buf43 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf43, (768,), is_leaf=True)  # primals_54
    buf44 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf44, (2304,), is_leaf=True)  # primals_55
    buf45 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf45, (2304, 768), is_leaf=True)  # primals_56
    buf46 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf46, (768, 768), is_leaf=True)  # primals_57
    buf47 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf47, (768,), is_leaf=True)  # primals_59
    buf48 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf48, (768,), is_leaf=True)  # primals_60
    buf49 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf49, (1536, 768), is_leaf=True)  # primals_61
    buf50 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf50, (1536,), is_leaf=True)  # primals_62
    buf51 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf51, (768, 1536), is_leaf=True)  # primals_63
    buf52 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf52, (768,), is_leaf=True)  # primals_65
    buf53 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf53, (768,), is_leaf=True)  # primals_66
    buf54 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf54, (2304,), is_leaf=True)  # primals_67
    buf55 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf55, (2304, 768), is_leaf=True)  # primals_68
    buf56 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf56, (768, 768), is_leaf=True)  # primals_69
    buf57 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf57, (768,), is_leaf=True)  # primals_71
    buf58 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf58, (768,), is_leaf=True)  # primals_72
    buf59 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf59, (1536, 768), is_leaf=True)  # primals_73
    buf60 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf60, (1536,), is_leaf=True)  # primals_74
    buf61 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf61, (768, 1536), is_leaf=True)  # primals_75
    buf62 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf62, (768,), is_leaf=True)  # primals_77
    buf63 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf63, (768,), is_leaf=True)  # primals_78
    buf64 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf64, (2304,), is_leaf=True)  # primals_79
    buf65 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf65, (2304, 768), is_leaf=True)  # primals_80
    buf66 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf66, (768, 768), is_leaf=True)  # primals_81
    buf67 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf67, (768,), is_leaf=True)  # primals_83
    buf68 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf68, (768,), is_leaf=True)  # primals_84
    buf69 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf69, (1536, 768), is_leaf=True)  # primals_85
    buf70 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf70, (1536,), is_leaf=True)  # primals_86
    buf71 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf71, (768, 1536), is_leaf=True)  # primals_87
    buf72 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf72, (768,), is_leaf=True)  # primals_89
    buf73 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf73, (768,), is_leaf=True)  # primals_90
    buf74 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf74, (2304,), is_leaf=True)  # primals_91
    buf75 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf75, (2304, 768), is_leaf=True)  # primals_92
    buf76 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf76, (768, 768), is_leaf=True)  # primals_93
    buf77 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf77, (768,), is_leaf=True)  # primals_95
    buf78 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf78, (768,), is_leaf=True)  # primals_96
    buf79 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf79, (1536, 768), is_leaf=True)  # primals_97
    buf80 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf80, (1536,), is_leaf=True)  # primals_98
    buf81 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf81, (768, 1536), is_leaf=True)  # primals_99
    buf82 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf82, (768,), is_leaf=True)  # primals_101
    buf83 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf83, (768,), is_leaf=True)  # primals_102
    buf84 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf84, (2304,), is_leaf=True)  # primals_103
    buf85 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf85, (2304, 768), is_leaf=True)  # primals_104
    buf86 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf86, (768, 768), is_leaf=True)  # primals_105
    buf87 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf87, (768,), is_leaf=True)  # primals_107
    buf88 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf88, (768,), is_leaf=True)  # primals_108
    buf89 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf89, (1536, 768), is_leaf=True)  # primals_109
    buf90 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf90, (1536,), is_leaf=True)  # primals_110
    buf91 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf91, (768, 1536), is_leaf=True)  # primals_111
    buf92 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf92, (768,), is_leaf=True)  # primals_113
    buf93 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf93, (768,), is_leaf=True)  # primals_114
    buf94 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf94, (2304,), is_leaf=True)  # primals_115
    buf95 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf95, (2304, 768), is_leaf=True)  # primals_116
    buf96 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf96, (768, 768), is_leaf=True)  # primals_117
    buf97 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf97, (768,), is_leaf=True)  # primals_119
    buf98 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf98, (768,), is_leaf=True)  # primals_120
    buf99 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf99, (1536, 768), is_leaf=True)  # primals_121
    buf100 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf100, (1536,), is_leaf=True)  # primals_122
    buf101 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf101, (768, 1536), is_leaf=True)  # primals_123
    buf102 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf102, (768,), is_leaf=True)  # primals_125
    buf103 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768,), is_leaf=True)  # primals_126
    buf104 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf104, (2304,), is_leaf=True)  # primals_127
    buf105 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf105, (2304, 768), is_leaf=True)  # primals_128
    buf106 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf106, (768, 768), is_leaf=True)  # primals_129
    buf107 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf107, (768,), is_leaf=True)  # primals_131
    buf108 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf108, (768,), is_leaf=True)  # primals_132
    buf109 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf109, (1536, 768), is_leaf=True)  # primals_133
    buf110 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf110, (1536,), is_leaf=True)  # primals_134
    buf111 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf111, (768, 1536), is_leaf=True)  # primals_135
    buf112 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf112, (768,), is_leaf=True)  # primals_137
    buf113 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf113, (768,), is_leaf=True)  # primals_138
    buf114 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf114, (2304,), is_leaf=True)  # primals_139
    buf115 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf115, (2304, 768), is_leaf=True)  # primals_140
    buf116 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf116, (768, 768), is_leaf=True)  # primals_141
    buf117 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf117, (768,), is_leaf=True)  # primals_143
    buf118 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf118, (768,), is_leaf=True)  # primals_144
    buf119 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf119, (1536, 768), is_leaf=True)  # primals_145
    buf120 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf120, (1536,), is_leaf=True)  # primals_146
    buf121 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf121, (768, 1536), is_leaf=True)  # primals_147
    buf122 = reader.storage(None, 655360, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf122, (20480, 16), dtype=torch.float16, is_leaf=True)  # view_1
    buf123 = reader.storage(None, 31457280, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf123, (20480, 768), dtype=torch.float16, is_leaf=True)  # mm
    buf124 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf124, (20, 1024, 768), is_leaf=True)  # add_7
    buf125 = reader.storage(None, 96, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf125, (12,), dtype=torch.int64, is_leaf=True)  # inductor_seeds_default
    buf126 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf126, (20, 1024, 768), is_leaf=True)  # add_11
    buf127 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf127, (20, 1024, 768), is_leaf=True)  # add_15
    buf128 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf128, (20, 1024, 768), is_leaf=True)  # add_19
    buf129 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf129, (20, 1024, 768), is_leaf=True)  # add_23
    buf130 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf130, (20, 1024, 768), is_leaf=True)  # add_27
    buf131 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf131, (20, 1024, 768), is_leaf=True)  # add_31
    buf132 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf132, (20, 1024, 768), is_leaf=True)  # add_35
    buf133 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf133, (20, 1024, 768), is_leaf=True)  # add_39
    buf134 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf134, (20, 1024, 768), is_leaf=True)  # add_43
    buf135 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf135, (20, 1024, 768), is_leaf=True)  # add_47
    buf136 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf136, (20, 1024, 768), is_leaf=True)  # add_51
    buf137 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf137, (20, 1024, 768), is_leaf=True)  # add_55
    buf138 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf138, (20, 1024, 768), is_leaf=True)  # add_59
    buf139 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf139, (20, 1024, 768), is_leaf=True)  # add_63
    buf140 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf140, (20, 1024, 768), is_leaf=True)  # add_67
    buf141 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf141, (20, 1024, 768), is_leaf=True)  # add_71
    buf142 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf142, (20, 1024, 768), is_leaf=True)  # add_75
    buf143 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf143, (20, 1024, 768), is_leaf=True)  # add_79
    buf144 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf144, (20, 1024, 768), is_leaf=True)  # add_83
    buf145 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf145, (20, 1024, 768), is_leaf=True)  # add_87
    buf146 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf146, (20, 1024, 768), is_leaf=True)  # add_91
    buf147 = reader.storage(None, 62914560, device=device(type='cuda', index=0))
    reader.tensor(buf147, (20, 1024, 768), is_leaf=True)  # add_95
    buf148 = reader.storage(None, 31457280, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf148, (20480, 768), dtype=torch.float16, is_leaf=True)  # view_183
    buf149 = reader.storage(None, 5242880, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf149, (20480, 128), dtype=torch.float16, is_leaf=True)  # view_185
    buf150 = reader.storage(None, 1310720, device=device(type='cuda', index=0))
    reader.tensor(buf150, (20, 1, 128, 128), is_leaf=True)  # full_default
    buf151 = reader.storage(None, 256, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf151, (32, 1), dtype=torch.int64, is_leaf=True)  # convert_element_type_296
    buf152 = reader.storage(None, 256, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf152, (32, 1), dtype=torch.int64, is_leaf=True)  # clamp_max
    buf153 = reader.storage(None, 256, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf153, (32,), dtype=torch.int64, is_leaf=True)  # convert_element_type_298
    buf154 = reader.storage(None, 256, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf154, (32,), dtype=torch.int64, is_leaf=True)  # clamp_max_1
    buf155 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf155, (32,), is_leaf=True)  # clamp_max_2
    buf156 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf156, (32, 1), is_leaf=True)  # clamp_max_3
    buf157 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf157, (16, 128), dtype=torch.float16, is_leaf=True)  # permute_141
    buf158 = reader.storage(None, 196608, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf158, (128, 768), dtype=torch.float16, is_leaf=True)  # permute_145
    buf159 = reader.storage(None, 40960, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf159, (20, 1, 32, 32), dtype=torch.float16, is_leaf=True)  # tangents_1
    # bwd_rng_state_0 was unsupported type for dumping: <class 'torch._C.Generator'>
    # bwd_rng_state_1 was unsupported type for dumping: <class 'torch._C.Generator'>
    # bwd_rng_state_2 was unsupported type for dumping: <class 'torch._C.Generator'>
    # bwd_rng_state_3 was unsupported type for dumping: <class 'torch._C.Generator'>
    # bwd_rng_state_4 was unsupported type for dumping: <class 'torch._C.Generator'>
    # bwd_rng_state_5 was unsupported type for dumping: <class 'torch._C.Generator'>
    # bwd_rng_state_6 was unsupported type for dumping: <class 'torch._C.Generator'>
    # bwd_rng_state_7 was unsupported type for dumping: <class 'torch._C.Generator'>
    # bwd_rng_state_8 was unsupported type for dumping: <class 'torch._C.Generator'>
    # bwd_rng_state_9 was unsupported type for dumping: <class 'torch._C.Generator'>
    # bwd_rng_state_10 was unsupported type for dumping: <class 'torch._C.Generator'>
    # bwd_rng_state_11 was unsupported type for dumping: <class 'torch._C.Generator'>
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)