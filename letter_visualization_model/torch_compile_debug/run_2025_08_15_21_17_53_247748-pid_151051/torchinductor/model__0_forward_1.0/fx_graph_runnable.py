
import os
os.environ['TORCHDYNAMO_VERBOSE'] = '1'
os.environ['TORCH_TRACE'] = './logs.txt'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_LOGS'] = 'recompiles,+dynamo'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_Adithya'

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

torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, fwd_rng_state_0, fwd_rng_state_1, fwd_rng_state_2, fwd_rng_state_3, fwd_rng_state_4, fwd_rng_state_5, fwd_rng_state_6, fwd_rng_state_7, fwd_rng_state_8, fwd_rng_state_9, fwd_rng_state_10, fwd_rng_state_11):
        iota = torch.ops.prims.iota.default(32, start = 0, step = 4, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        iota_1 = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        add = torch.ops.aten.add.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(add, -1)
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        index = torch.ops.aten.index.Tensor(primals_1, [None, None, unsqueeze_5, add]);  primals_1 = None
        permute = torch.ops.aten.permute.default(index, [0, 1, 2, 4, 3, 5]);  index = None
        clone = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view = torch.ops.aten.view.default(clone, [20, 16, 1024]);  clone = None
        permute_1 = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_3, torch.float16)
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(primals_2, torch.float16);  primals_2 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(permute_1, torch.float16);  permute_1 = None
        permute_2 = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        clone_1 = torch.ops.aten.clone.default(convert_element_type_2, memory_format = torch.contiguous_format);  convert_element_type_2 = None
        view_1 = torch.ops.aten.view.default(clone_1, [20480, 16]);  clone_1 = None
        mm = torch.ops.aten.mm.default(view_1, permute_2);  permute_2 = None
        view_2 = torch.ops.aten.view.default(mm, [20, 1024, 768])
        add_2 = torch.ops.aten.add.Tensor(view_2, convert_element_type);  view_2 = convert_element_type = None
        add_3 = torch.ops.aten.add.Tensor(add_2, primals_4);  add_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_4 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub = torch.ops.aten.sub.Tensor(add_3, getitem_1);  getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, primals_5);  mul = None
        add_5 = torch.ops.aten.add.Tensor(mul_1, primals_6);  mul_1 = None
        permute_3 = torch.ops.aten.permute.default(add_5, [1, 0, 2]);  add_5 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(primals_7, torch.float16)
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(primals_8, torch.float16)
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(permute_3, torch.float16);  permute_3 = None
        permute_4 = torch.ops.aten.permute.default(convert_element_type_6, [1, 0]);  convert_element_type_6 = None
        clone_2 = torch.ops.aten.clone.default(convert_element_type_7, memory_format = torch.contiguous_format);  convert_element_type_7 = None
        view_3 = torch.ops.aten.view.default(clone_2, [20480, 768]);  clone_2 = None
        mm_1 = torch.ops.aten.mm.default(view_3, permute_4);  view_3 = permute_4 = None
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
        graphsafe_run_with_rng_state = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_9, view_10, view_11, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_0);  view_9 = view_10 = view_11 = fwd_rng_state_0 = None
        getitem_2 = graphsafe_run_with_rng_state[0];  graphsafe_run_with_rng_state = None
        permute_9 = torch.ops.aten.permute.default(getitem_2, [2, 0, 1, 3]);  getitem_2 = None
        view_12 = torch.ops.aten.view.default(permute_9, [20480, 768]);  permute_9 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(primals_10, torch.float16);  primals_10 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(primals_9, torch.float16)
        permute_10 = torch.ops.aten.permute.default(convert_element_type_11, [1, 0]);  convert_element_type_11 = None
        addmm = torch.ops.aten.addmm.default(convert_element_type_10, view_12, permute_10);  convert_element_type_10 = view_12 = permute_10 = None
        view_13 = torch.ops.aten.view.default(addmm, [1024, 20, 768]);  addmm = None
        permute_11 = torch.ops.aten.permute.default(view_13, [1, 0, 2]);  view_13 = None
        add_7 = torch.ops.aten.add.Tensor(add_3, permute_11);  add_3 = permute_11 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_11 = var_mean_1[0]
        getitem_12 = var_mean_1[1];  var_mean_1 = None
        add_8 = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_7, getitem_12);  getitem_12 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, primals_11);  mul_2 = None
        add_9 = torch.ops.aten.add.Tensor(mul_3, primals_12);  mul_3 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(primals_14, torch.float16)
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(primals_13, torch.float16)
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(add_9, torch.float16);  add_9 = None
        view_14 = torch.ops.aten.view.default(convert_element_type_17, [20480, 768]);  convert_element_type_17 = None
        permute_12 = torch.ops.aten.permute.default(convert_element_type_16, [1, 0]);  convert_element_type_16 = None
        addmm_1 = torch.ops.aten.addmm.default(convert_element_type_15, view_14, permute_12);  convert_element_type_15 = view_14 = permute_12 = None
        view_15 = torch.ops.aten.view.default(addmm_1, [20, 1024, 1536]);  addmm_1 = None
        convert_element_type_21 = torch.ops.prims.convert_element_type.default(view_15, torch.float32);  view_15 = None
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.5)
        mul_5 = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.7071067811865476);  convert_element_type_21 = None
        erf = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_10 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_4, add_10);  mul_4 = add_10 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(mul_6, torch.float16);  mul_6 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(primals_16, torch.float16);  primals_16 = None
        convert_element_type_24 = torch.ops.prims.convert_element_type.default(primals_15, torch.float16)
        view_16 = torch.ops.aten.view.default(convert_element_type_22, [20480, 1536]);  convert_element_type_22 = None
        permute_13 = torch.ops.aten.permute.default(convert_element_type_24, [1, 0]);  convert_element_type_24 = None
        addmm_2 = torch.ops.aten.addmm.default(convert_element_type_23, view_16, permute_13);  convert_element_type_23 = view_16 = permute_13 = None
        view_17 = torch.ops.aten.view.default(addmm_2, [20, 1024, 768]);  addmm_2 = None
        inductor_seeds_default = torch.ops.prims.inductor_seeds.default(12, device(type='cuda', index=0))
        inductor_lookup_seed_default = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_11 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default_63 = torch.ops.prims.convert_element_type.default(inductor_random_default_11, torch.float16);  inductor_random_default_11 = None
        gt = torch.ops.aten.gt.Scalar(convert_element_type_default_63, 0.2);  convert_element_type_default_63 = None
        mul_7 = torch.ops.aten.mul.Tensor(gt, view_17);  gt = view_17 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, 1.25);  mul_7 = None
        add_11 = torch.ops.aten.add.Tensor(add_7, mul_8);  mul_8 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_13 = var_mean_2[0]
        getitem_14 = var_mean_2[1];  var_mean_2 = None
        add_12 = torch.ops.aten.add.Tensor(getitem_13, 1e-05);  getitem_13 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_11, getitem_14);  getitem_14 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, primals_17);  mul_9 = None
        add_13 = torch.ops.aten.add.Tensor(mul_10, primals_18);  mul_10 = None
        permute_14 = torch.ops.aten.permute.default(add_13, [1, 0, 2]);  add_13 = None
        convert_element_type_28 = torch.ops.prims.convert_element_type.default(primals_19, torch.float16)
        convert_element_type_29 = torch.ops.prims.convert_element_type.default(primals_20, torch.float16)
        convert_element_type_30 = torch.ops.prims.convert_element_type.default(permute_14, torch.float16);  permute_14 = None
        permute_15 = torch.ops.aten.permute.default(convert_element_type_29, [1, 0]);  convert_element_type_29 = None
        clone_4 = torch.ops.aten.clone.default(convert_element_type_30, memory_format = torch.contiguous_format);  convert_element_type_30 = None
        view_18 = torch.ops.aten.view.default(clone_4, [20480, 768]);  clone_4 = None
        mm_2 = torch.ops.aten.mm.default(view_18, permute_15);  view_18 = permute_15 = None
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
        graphsafe_run_with_rng_state_1 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_24, view_25, view_26, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_1);  view_24 = view_25 = view_26 = fwd_rng_state_1 = None
        getitem_15 = graphsafe_run_with_rng_state_1[0];  graphsafe_run_with_rng_state_1 = None
        permute_20 = torch.ops.aten.permute.default(getitem_15, [2, 0, 1, 3]);  getitem_15 = None
        view_27 = torch.ops.aten.view.default(permute_20, [20480, 768]);  permute_20 = None
        convert_element_type_33 = torch.ops.prims.convert_element_type.default(primals_22, torch.float16);  primals_22 = None
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(primals_21, torch.float16)
        permute_21 = torch.ops.aten.permute.default(convert_element_type_34, [1, 0]);  convert_element_type_34 = None
        addmm_3 = torch.ops.aten.addmm.default(convert_element_type_33, view_27, permute_21);  convert_element_type_33 = view_27 = permute_21 = None
        view_28 = torch.ops.aten.view.default(addmm_3, [1024, 20, 768]);  addmm_3 = None
        permute_22 = torch.ops.aten.permute.default(view_28, [1, 0, 2]);  view_28 = None
        add_15 = torch.ops.aten.add.Tensor(add_11, permute_22);  permute_22 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_3[0]
        getitem_25 = var_mean_3[1];  var_mean_3 = None
        add_16 = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_15, getitem_25);  getitem_25 = None
        mul_11 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_12 = torch.ops.aten.mul.Tensor(mul_11, primals_23);  mul_11 = None
        add_17 = torch.ops.aten.add.Tensor(mul_12, primals_24);  mul_12 = None
        convert_element_type_38 = torch.ops.prims.convert_element_type.default(primals_26, torch.float16)
        convert_element_type_39 = torch.ops.prims.convert_element_type.default(primals_25, torch.float16)
        convert_element_type_40 = torch.ops.prims.convert_element_type.default(add_17, torch.float16);  add_17 = None
        view_29 = torch.ops.aten.view.default(convert_element_type_40, [20480, 768]);  convert_element_type_40 = None
        permute_23 = torch.ops.aten.permute.default(convert_element_type_39, [1, 0]);  convert_element_type_39 = None
        addmm_4 = torch.ops.aten.addmm.default(convert_element_type_38, view_29, permute_23);  convert_element_type_38 = view_29 = permute_23 = None
        view_30 = torch.ops.aten.view.default(addmm_4, [20, 1024, 1536]);  addmm_4 = None
        convert_element_type_44 = torch.ops.prims.convert_element_type.default(view_30, torch.float32);  view_30 = None
        mul_13 = torch.ops.aten.mul.Tensor(convert_element_type_44, 0.5)
        mul_14 = torch.ops.aten.mul.Tensor(convert_element_type_44, 0.7071067811865476);  convert_element_type_44 = None
        erf_1 = torch.ops.aten.erf.default(mul_14);  mul_14 = None
        add_18 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_13, add_18);  mul_13 = add_18 = None
        convert_element_type_45 = torch.ops.prims.convert_element_type.default(mul_15, torch.float16);  mul_15 = None
        convert_element_type_46 = torch.ops.prims.convert_element_type.default(primals_28, torch.float16);  primals_28 = None
        convert_element_type_47 = torch.ops.prims.convert_element_type.default(primals_27, torch.float16)
        view_31 = torch.ops.aten.view.default(convert_element_type_45, [20480, 1536]);  convert_element_type_45 = None
        permute_24 = torch.ops.aten.permute.default(convert_element_type_47, [1, 0]);  convert_element_type_47 = None
        addmm_5 = torch.ops.aten.addmm.default(convert_element_type_46, view_31, permute_24);  convert_element_type_46 = view_31 = permute_24 = None
        view_32 = torch.ops.aten.view.default(addmm_5, [20, 1024, 768]);  addmm_5 = None
        inductor_lookup_seed_default_1 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_10 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default_62 = torch.ops.prims.convert_element_type.default(inductor_random_default_10, torch.float16);  inductor_random_default_10 = None
        gt_1 = torch.ops.aten.gt.Scalar(convert_element_type_default_62, 0.2);  convert_element_type_default_62 = None
        mul_16 = torch.ops.aten.mul.Tensor(gt_1, view_32);  gt_1 = view_32 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, 1.25);  mul_16 = None
        add_19 = torch.ops.aten.add.Tensor(add_15, mul_17);  mul_17 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_4[0]
        getitem_27 = var_mean_4[1];  var_mean_4 = None
        add_20 = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_19, getitem_27);  getitem_27 = None
        mul_18 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, primals_29);  mul_18 = None
        add_21 = torch.ops.aten.add.Tensor(mul_19, primals_30);  mul_19 = None
        permute_25 = torch.ops.aten.permute.default(add_21, [1, 0, 2]);  add_21 = None
        convert_element_type_51 = torch.ops.prims.convert_element_type.default(primals_31, torch.float16)
        convert_element_type_52 = torch.ops.prims.convert_element_type.default(primals_32, torch.float16)
        convert_element_type_53 = torch.ops.prims.convert_element_type.default(permute_25, torch.float16);  permute_25 = None
        permute_26 = torch.ops.aten.permute.default(convert_element_type_52, [1, 0]);  convert_element_type_52 = None
        clone_6 = torch.ops.aten.clone.default(convert_element_type_53, memory_format = torch.contiguous_format);  convert_element_type_53 = None
        view_33 = torch.ops.aten.view.default(clone_6, [20480, 768]);  clone_6 = None
        mm_3 = torch.ops.aten.mm.default(view_33, permute_26);  view_33 = permute_26 = None
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
        graphsafe_run_with_rng_state_2 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_39, view_40, view_41, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_2);  view_39 = view_40 = view_41 = fwd_rng_state_2 = None
        getitem_28 = graphsafe_run_with_rng_state_2[0];  graphsafe_run_with_rng_state_2 = None
        permute_31 = torch.ops.aten.permute.default(getitem_28, [2, 0, 1, 3]);  getitem_28 = None
        view_42 = torch.ops.aten.view.default(permute_31, [20480, 768]);  permute_31 = None
        convert_element_type_56 = torch.ops.prims.convert_element_type.default(primals_34, torch.float16);  primals_34 = None
        convert_element_type_57 = torch.ops.prims.convert_element_type.default(primals_33, torch.float16)
        permute_32 = torch.ops.aten.permute.default(convert_element_type_57, [1, 0]);  convert_element_type_57 = None
        addmm_6 = torch.ops.aten.addmm.default(convert_element_type_56, view_42, permute_32);  convert_element_type_56 = view_42 = permute_32 = None
        view_43 = torch.ops.aten.view.default(addmm_6, [1024, 20, 768]);  addmm_6 = None
        permute_33 = torch.ops.aten.permute.default(view_43, [1, 0, 2]);  view_43 = None
        add_23 = torch.ops.aten.add.Tensor(add_19, permute_33);  permute_33 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
        getitem_37 = var_mean_5[0]
        getitem_38 = var_mean_5[1];  var_mean_5 = None
        add_24 = torch.ops.aten.add.Tensor(getitem_37, 1e-05);  getitem_37 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_23, getitem_38);  getitem_38 = None
        mul_20 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, primals_35);  mul_20 = None
        add_25 = torch.ops.aten.add.Tensor(mul_21, primals_36);  mul_21 = None
        convert_element_type_61 = torch.ops.prims.convert_element_type.default(primals_38, torch.float16)
        convert_element_type_62 = torch.ops.prims.convert_element_type.default(primals_37, torch.float16)
        convert_element_type_63 = torch.ops.prims.convert_element_type.default(add_25, torch.float16);  add_25 = None
        view_44 = torch.ops.aten.view.default(convert_element_type_63, [20480, 768]);  convert_element_type_63 = None
        permute_34 = torch.ops.aten.permute.default(convert_element_type_62, [1, 0]);  convert_element_type_62 = None
        addmm_7 = torch.ops.aten.addmm.default(convert_element_type_61, view_44, permute_34);  convert_element_type_61 = view_44 = permute_34 = None
        view_45 = torch.ops.aten.view.default(addmm_7, [20, 1024, 1536]);  addmm_7 = None
        convert_element_type_67 = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        mul_22 = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.5)
        mul_23 = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.7071067811865476);  convert_element_type_67 = None
        erf_2 = torch.ops.aten.erf.default(mul_23);  mul_23 = None
        add_26 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_24 = torch.ops.aten.mul.Tensor(mul_22, add_26);  mul_22 = add_26 = None
        convert_element_type_68 = torch.ops.prims.convert_element_type.default(mul_24, torch.float16);  mul_24 = None
        convert_element_type_69 = torch.ops.prims.convert_element_type.default(primals_40, torch.float16);  primals_40 = None
        convert_element_type_70 = torch.ops.prims.convert_element_type.default(primals_39, torch.float16)
        view_46 = torch.ops.aten.view.default(convert_element_type_68, [20480, 1536]);  convert_element_type_68 = None
        permute_35 = torch.ops.aten.permute.default(convert_element_type_70, [1, 0]);  convert_element_type_70 = None
        addmm_8 = torch.ops.aten.addmm.default(convert_element_type_69, view_46, permute_35);  convert_element_type_69 = view_46 = permute_35 = None
        view_47 = torch.ops.aten.view.default(addmm_8, [20, 1024, 768]);  addmm_8 = None
        inductor_lookup_seed_default_2 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_9 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        convert_element_type_default_61 = torch.ops.prims.convert_element_type.default(inductor_random_default_9, torch.float16);  inductor_random_default_9 = None
        gt_2 = torch.ops.aten.gt.Scalar(convert_element_type_default_61, 0.2);  convert_element_type_default_61 = None
        mul_25 = torch.ops.aten.mul.Tensor(gt_2, view_47);  gt_2 = view_47 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_25, 1.25);  mul_25 = None
        add_27 = torch.ops.aten.add.Tensor(add_23, mul_26);  mul_26 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_39 = var_mean_6[0]
        getitem_40 = var_mean_6[1];  var_mean_6 = None
        add_28 = torch.ops.aten.add.Tensor(getitem_39, 1e-05);  getitem_39 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_27, getitem_40);  getitem_40 = None
        mul_27 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_28 = torch.ops.aten.mul.Tensor(mul_27, primals_41);  mul_27 = None
        add_29 = torch.ops.aten.add.Tensor(mul_28, primals_42);  mul_28 = None
        permute_36 = torch.ops.aten.permute.default(add_29, [1, 0, 2]);  add_29 = None
        convert_element_type_74 = torch.ops.prims.convert_element_type.default(primals_43, torch.float16)
        convert_element_type_75 = torch.ops.prims.convert_element_type.default(primals_44, torch.float16)
        convert_element_type_76 = torch.ops.prims.convert_element_type.default(permute_36, torch.float16);  permute_36 = None
        permute_37 = torch.ops.aten.permute.default(convert_element_type_75, [1, 0]);  convert_element_type_75 = None
        clone_8 = torch.ops.aten.clone.default(convert_element_type_76, memory_format = torch.contiguous_format);  convert_element_type_76 = None
        view_48 = torch.ops.aten.view.default(clone_8, [20480, 768]);  clone_8 = None
        mm_4 = torch.ops.aten.mm.default(view_48, permute_37);  view_48 = permute_37 = None
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
        graphsafe_run_with_rng_state_3 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_54, view_55, view_56, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_3);  view_54 = view_55 = view_56 = fwd_rng_state_3 = None
        getitem_41 = graphsafe_run_with_rng_state_3[0];  graphsafe_run_with_rng_state_3 = None
        permute_42 = torch.ops.aten.permute.default(getitem_41, [2, 0, 1, 3]);  getitem_41 = None
        view_57 = torch.ops.aten.view.default(permute_42, [20480, 768]);  permute_42 = None
        convert_element_type_79 = torch.ops.prims.convert_element_type.default(primals_46, torch.float16);  primals_46 = None
        convert_element_type_80 = torch.ops.prims.convert_element_type.default(primals_45, torch.float16)
        permute_43 = torch.ops.aten.permute.default(convert_element_type_80, [1, 0]);  convert_element_type_80 = None
        addmm_9 = torch.ops.aten.addmm.default(convert_element_type_79, view_57, permute_43);  convert_element_type_79 = view_57 = permute_43 = None
        view_58 = torch.ops.aten.view.default(addmm_9, [1024, 20, 768]);  addmm_9 = None
        permute_44 = torch.ops.aten.permute.default(view_58, [1, 0, 2]);  view_58 = None
        add_31 = torch.ops.aten.add.Tensor(add_27, permute_44);  permute_44 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_50 = var_mean_7[0]
        getitem_51 = var_mean_7[1];  var_mean_7 = None
        add_32 = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_31, getitem_51);  getitem_51 = None
        mul_29 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_30 = torch.ops.aten.mul.Tensor(mul_29, primals_47);  mul_29 = None
        add_33 = torch.ops.aten.add.Tensor(mul_30, primals_48);  mul_30 = None
        convert_element_type_84 = torch.ops.prims.convert_element_type.default(primals_50, torch.float16)
        convert_element_type_85 = torch.ops.prims.convert_element_type.default(primals_49, torch.float16)
        convert_element_type_86 = torch.ops.prims.convert_element_type.default(add_33, torch.float16);  add_33 = None
        view_59 = torch.ops.aten.view.default(convert_element_type_86, [20480, 768]);  convert_element_type_86 = None
        permute_45 = torch.ops.aten.permute.default(convert_element_type_85, [1, 0]);  convert_element_type_85 = None
        addmm_10 = torch.ops.aten.addmm.default(convert_element_type_84, view_59, permute_45);  convert_element_type_84 = view_59 = permute_45 = None
        view_60 = torch.ops.aten.view.default(addmm_10, [20, 1024, 1536]);  addmm_10 = None
        convert_element_type_90 = torch.ops.prims.convert_element_type.default(view_60, torch.float32);  view_60 = None
        mul_31 = torch.ops.aten.mul.Tensor(convert_element_type_90, 0.5)
        mul_32 = torch.ops.aten.mul.Tensor(convert_element_type_90, 0.7071067811865476);  convert_element_type_90 = None
        erf_3 = torch.ops.aten.erf.default(mul_32);  mul_32 = None
        add_34 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_31, add_34);  mul_31 = add_34 = None
        convert_element_type_91 = torch.ops.prims.convert_element_type.default(mul_33, torch.float16);  mul_33 = None
        convert_element_type_92 = torch.ops.prims.convert_element_type.default(primals_52, torch.float16);  primals_52 = None
        convert_element_type_93 = torch.ops.prims.convert_element_type.default(primals_51, torch.float16)
        view_61 = torch.ops.aten.view.default(convert_element_type_91, [20480, 1536]);  convert_element_type_91 = None
        permute_46 = torch.ops.aten.permute.default(convert_element_type_93, [1, 0]);  convert_element_type_93 = None
        addmm_11 = torch.ops.aten.addmm.default(convert_element_type_92, view_61, permute_46);  convert_element_type_92 = view_61 = permute_46 = None
        view_62 = torch.ops.aten.view.default(addmm_11, [20, 1024, 768]);  addmm_11 = None
        inductor_lookup_seed_default_3 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_8 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        convert_element_type_default_60 = torch.ops.prims.convert_element_type.default(inductor_random_default_8, torch.float16);  inductor_random_default_8 = None
        gt_3 = torch.ops.aten.gt.Scalar(convert_element_type_default_60, 0.2);  convert_element_type_default_60 = None
        mul_34 = torch.ops.aten.mul.Tensor(gt_3, view_62);  gt_3 = view_62 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_34, 1.25);  mul_34 = None
        add_35 = torch.ops.aten.add.Tensor(add_31, mul_35);  mul_35 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_52 = var_mean_8[0]
        getitem_53 = var_mean_8[1];  var_mean_8 = None
        add_36 = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_35, getitem_53);  getitem_53 = None
        mul_36 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, primals_53);  mul_36 = None
        add_37 = torch.ops.aten.add.Tensor(mul_37, primals_54);  mul_37 = None
        permute_47 = torch.ops.aten.permute.default(add_37, [1, 0, 2]);  add_37 = None
        convert_element_type_97 = torch.ops.prims.convert_element_type.default(primals_55, torch.float16)
        convert_element_type_98 = torch.ops.prims.convert_element_type.default(primals_56, torch.float16)
        convert_element_type_99 = torch.ops.prims.convert_element_type.default(permute_47, torch.float16);  permute_47 = None
        permute_48 = torch.ops.aten.permute.default(convert_element_type_98, [1, 0]);  convert_element_type_98 = None
        clone_10 = torch.ops.aten.clone.default(convert_element_type_99, memory_format = torch.contiguous_format);  convert_element_type_99 = None
        view_63 = torch.ops.aten.view.default(clone_10, [20480, 768]);  clone_10 = None
        mm_5 = torch.ops.aten.mm.default(view_63, permute_48);  view_63 = permute_48 = None
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
        graphsafe_run_with_rng_state_4 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_69, view_70, view_71, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_4);  view_69 = view_70 = view_71 = fwd_rng_state_4 = None
        getitem_54 = graphsafe_run_with_rng_state_4[0];  graphsafe_run_with_rng_state_4 = None
        permute_53 = torch.ops.aten.permute.default(getitem_54, [2, 0, 1, 3]);  getitem_54 = None
        view_72 = torch.ops.aten.view.default(permute_53, [20480, 768]);  permute_53 = None
        convert_element_type_102 = torch.ops.prims.convert_element_type.default(primals_58, torch.float16);  primals_58 = None
        convert_element_type_103 = torch.ops.prims.convert_element_type.default(primals_57, torch.float16)
        permute_54 = torch.ops.aten.permute.default(convert_element_type_103, [1, 0]);  convert_element_type_103 = None
        addmm_12 = torch.ops.aten.addmm.default(convert_element_type_102, view_72, permute_54);  convert_element_type_102 = view_72 = permute_54 = None
        view_73 = torch.ops.aten.view.default(addmm_12, [1024, 20, 768]);  addmm_12 = None
        permute_55 = torch.ops.aten.permute.default(view_73, [1, 0, 2]);  view_73 = None
        add_39 = torch.ops.aten.add.Tensor(add_35, permute_55);  permute_55 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_63 = var_mean_9[0]
        getitem_64 = var_mean_9[1];  var_mean_9 = None
        add_40 = torch.ops.aten.add.Tensor(getitem_63, 1e-05);  getitem_63 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_39, getitem_64);  getitem_64 = None
        mul_38 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, primals_59);  mul_38 = None
        add_41 = torch.ops.aten.add.Tensor(mul_39, primals_60);  mul_39 = None
        convert_element_type_107 = torch.ops.prims.convert_element_type.default(primals_62, torch.float16)
        convert_element_type_108 = torch.ops.prims.convert_element_type.default(primals_61, torch.float16)
        convert_element_type_109 = torch.ops.prims.convert_element_type.default(add_41, torch.float16);  add_41 = None
        view_74 = torch.ops.aten.view.default(convert_element_type_109, [20480, 768]);  convert_element_type_109 = None
        permute_56 = torch.ops.aten.permute.default(convert_element_type_108, [1, 0]);  convert_element_type_108 = None
        addmm_13 = torch.ops.aten.addmm.default(convert_element_type_107, view_74, permute_56);  convert_element_type_107 = view_74 = permute_56 = None
        view_75 = torch.ops.aten.view.default(addmm_13, [20, 1024, 1536]);  addmm_13 = None
        convert_element_type_113 = torch.ops.prims.convert_element_type.default(view_75, torch.float32);  view_75 = None
        mul_40 = torch.ops.aten.mul.Tensor(convert_element_type_113, 0.5)
        mul_41 = torch.ops.aten.mul.Tensor(convert_element_type_113, 0.7071067811865476);  convert_element_type_113 = None
        erf_4 = torch.ops.aten.erf.default(mul_41);  mul_41 = None
        add_42 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_42 = torch.ops.aten.mul.Tensor(mul_40, add_42);  mul_40 = add_42 = None
        convert_element_type_114 = torch.ops.prims.convert_element_type.default(mul_42, torch.float16);  mul_42 = None
        convert_element_type_115 = torch.ops.prims.convert_element_type.default(primals_64, torch.float16);  primals_64 = None
        convert_element_type_116 = torch.ops.prims.convert_element_type.default(primals_63, torch.float16)
        view_76 = torch.ops.aten.view.default(convert_element_type_114, [20480, 1536]);  convert_element_type_114 = None
        permute_57 = torch.ops.aten.permute.default(convert_element_type_116, [1, 0]);  convert_element_type_116 = None
        addmm_14 = torch.ops.aten.addmm.default(convert_element_type_115, view_76, permute_57);  convert_element_type_115 = view_76 = permute_57 = None
        view_77 = torch.ops.aten.view.default(addmm_14, [20, 1024, 768]);  addmm_14 = None
        inductor_lookup_seed_default_4 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_7 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        convert_element_type_default_59 = torch.ops.prims.convert_element_type.default(inductor_random_default_7, torch.float16);  inductor_random_default_7 = None
        gt_4 = torch.ops.aten.gt.Scalar(convert_element_type_default_59, 0.2);  convert_element_type_default_59 = None
        mul_43 = torch.ops.aten.mul.Tensor(gt_4, view_77);  gt_4 = view_77 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_43, 1.25);  mul_43 = None
        add_43 = torch.ops.aten.add.Tensor(add_39, mul_44);  mul_44 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_65 = var_mean_10[0]
        getitem_66 = var_mean_10[1];  var_mean_10 = None
        add_44 = torch.ops.aten.add.Tensor(getitem_65, 1e-05);  getitem_65 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_43, getitem_66);  getitem_66 = None
        mul_45 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_45, primals_65);  mul_45 = None
        add_45 = torch.ops.aten.add.Tensor(mul_46, primals_66);  mul_46 = None
        permute_58 = torch.ops.aten.permute.default(add_45, [1, 0, 2]);  add_45 = None
        convert_element_type_120 = torch.ops.prims.convert_element_type.default(primals_67, torch.float16)
        convert_element_type_121 = torch.ops.prims.convert_element_type.default(primals_68, torch.float16)
        convert_element_type_122 = torch.ops.prims.convert_element_type.default(permute_58, torch.float16);  permute_58 = None
        permute_59 = torch.ops.aten.permute.default(convert_element_type_121, [1, 0]);  convert_element_type_121 = None
        clone_12 = torch.ops.aten.clone.default(convert_element_type_122, memory_format = torch.contiguous_format);  convert_element_type_122 = None
        view_78 = torch.ops.aten.view.default(clone_12, [20480, 768]);  clone_12 = None
        mm_6 = torch.ops.aten.mm.default(view_78, permute_59);  view_78 = permute_59 = None
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
        graphsafe_run_with_rng_state_5 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_84, view_85, view_86, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_5);  view_84 = view_85 = view_86 = fwd_rng_state_5 = None
        getitem_67 = graphsafe_run_with_rng_state_5[0];  graphsafe_run_with_rng_state_5 = None
        permute_64 = torch.ops.aten.permute.default(getitem_67, [2, 0, 1, 3]);  getitem_67 = None
        view_87 = torch.ops.aten.view.default(permute_64, [20480, 768]);  permute_64 = None
        convert_element_type_125 = torch.ops.prims.convert_element_type.default(primals_70, torch.float16);  primals_70 = None
        convert_element_type_126 = torch.ops.prims.convert_element_type.default(primals_69, torch.float16)
        permute_65 = torch.ops.aten.permute.default(convert_element_type_126, [1, 0]);  convert_element_type_126 = None
        addmm_15 = torch.ops.aten.addmm.default(convert_element_type_125, view_87, permute_65);  convert_element_type_125 = view_87 = permute_65 = None
        view_88 = torch.ops.aten.view.default(addmm_15, [1024, 20, 768]);  addmm_15 = None
        permute_66 = torch.ops.aten.permute.default(view_88, [1, 0, 2]);  view_88 = None
        add_47 = torch.ops.aten.add.Tensor(add_43, permute_66);  permute_66 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
        getitem_76 = var_mean_11[0]
        getitem_77 = var_mean_11[1];  var_mean_11 = None
        add_48 = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_47, getitem_77);  getitem_77 = None
        mul_47 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_47, primals_71);  mul_47 = None
        add_49 = torch.ops.aten.add.Tensor(mul_48, primals_72);  mul_48 = None
        convert_element_type_130 = torch.ops.prims.convert_element_type.default(primals_74, torch.float16)
        convert_element_type_131 = torch.ops.prims.convert_element_type.default(primals_73, torch.float16)
        convert_element_type_132 = torch.ops.prims.convert_element_type.default(add_49, torch.float16);  add_49 = None
        view_89 = torch.ops.aten.view.default(convert_element_type_132, [20480, 768]);  convert_element_type_132 = None
        permute_67 = torch.ops.aten.permute.default(convert_element_type_131, [1, 0]);  convert_element_type_131 = None
        addmm_16 = torch.ops.aten.addmm.default(convert_element_type_130, view_89, permute_67);  convert_element_type_130 = view_89 = permute_67 = None
        view_90 = torch.ops.aten.view.default(addmm_16, [20, 1024, 1536]);  addmm_16 = None
        convert_element_type_136 = torch.ops.prims.convert_element_type.default(view_90, torch.float32);  view_90 = None
        mul_49 = torch.ops.aten.mul.Tensor(convert_element_type_136, 0.5)
        mul_50 = torch.ops.aten.mul.Tensor(convert_element_type_136, 0.7071067811865476);  convert_element_type_136 = None
        erf_5 = torch.ops.aten.erf.default(mul_50);  mul_50 = None
        add_50 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_51 = torch.ops.aten.mul.Tensor(mul_49, add_50);  mul_49 = add_50 = None
        convert_element_type_137 = torch.ops.prims.convert_element_type.default(mul_51, torch.float16);  mul_51 = None
        convert_element_type_138 = torch.ops.prims.convert_element_type.default(primals_76, torch.float16);  primals_76 = None
        convert_element_type_139 = torch.ops.prims.convert_element_type.default(primals_75, torch.float16)
        view_91 = torch.ops.aten.view.default(convert_element_type_137, [20480, 1536]);  convert_element_type_137 = None
        permute_68 = torch.ops.aten.permute.default(convert_element_type_139, [1, 0]);  convert_element_type_139 = None
        addmm_17 = torch.ops.aten.addmm.default(convert_element_type_138, view_91, permute_68);  convert_element_type_138 = view_91 = permute_68 = None
        view_92 = torch.ops.aten.view.default(addmm_17, [20, 1024, 768]);  addmm_17 = None
        inductor_lookup_seed_default_5 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_6 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        convert_element_type_default_58 = torch.ops.prims.convert_element_type.default(inductor_random_default_6, torch.float16);  inductor_random_default_6 = None
        gt_5 = torch.ops.aten.gt.Scalar(convert_element_type_default_58, 0.2);  convert_element_type_default_58 = None
        mul_52 = torch.ops.aten.mul.Tensor(gt_5, view_92);  gt_5 = view_92 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, 1.25);  mul_52 = None
        add_51 = torch.ops.aten.add.Tensor(add_47, mul_53);  mul_53 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_78 = var_mean_12[0]
        getitem_79 = var_mean_12[1];  var_mean_12 = None
        add_52 = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_51, getitem_79);  getitem_79 = None
        mul_54 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_54, primals_77);  mul_54 = None
        add_53 = torch.ops.aten.add.Tensor(mul_55, primals_78);  mul_55 = None
        permute_69 = torch.ops.aten.permute.default(add_53, [1, 0, 2]);  add_53 = None
        convert_element_type_143 = torch.ops.prims.convert_element_type.default(primals_79, torch.float16)
        convert_element_type_144 = torch.ops.prims.convert_element_type.default(primals_80, torch.float16)
        convert_element_type_145 = torch.ops.prims.convert_element_type.default(permute_69, torch.float16);  permute_69 = None
        permute_70 = torch.ops.aten.permute.default(convert_element_type_144, [1, 0]);  convert_element_type_144 = None
        clone_14 = torch.ops.aten.clone.default(convert_element_type_145, memory_format = torch.contiguous_format);  convert_element_type_145 = None
        view_93 = torch.ops.aten.view.default(clone_14, [20480, 768]);  clone_14 = None
        mm_7 = torch.ops.aten.mm.default(view_93, permute_70);  view_93 = permute_70 = None
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
        graphsafe_run_with_rng_state_6 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_99, view_100, view_101, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_6);  view_99 = view_100 = view_101 = fwd_rng_state_6 = None
        getitem_80 = graphsafe_run_with_rng_state_6[0];  graphsafe_run_with_rng_state_6 = None
        permute_75 = torch.ops.aten.permute.default(getitem_80, [2, 0, 1, 3]);  getitem_80 = None
        view_102 = torch.ops.aten.view.default(permute_75, [20480, 768]);  permute_75 = None
        convert_element_type_148 = torch.ops.prims.convert_element_type.default(primals_82, torch.float16);  primals_82 = None
        convert_element_type_149 = torch.ops.prims.convert_element_type.default(primals_81, torch.float16)
        permute_76 = torch.ops.aten.permute.default(convert_element_type_149, [1, 0]);  convert_element_type_149 = None
        addmm_18 = torch.ops.aten.addmm.default(convert_element_type_148, view_102, permute_76);  convert_element_type_148 = view_102 = permute_76 = None
        view_103 = torch.ops.aten.view.default(addmm_18, [1024, 20, 768]);  addmm_18 = None
        permute_77 = torch.ops.aten.permute.default(view_103, [1, 0, 2]);  view_103 = None
        add_55 = torch.ops.aten.add.Tensor(add_51, permute_77);  permute_77 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
        getitem_89 = var_mean_13[0]
        getitem_90 = var_mean_13[1];  var_mean_13 = None
        add_56 = torch.ops.aten.add.Tensor(getitem_89, 1e-05);  getitem_89 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_55, getitem_90);  getitem_90 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, primals_83);  mul_56 = None
        add_57 = torch.ops.aten.add.Tensor(mul_57, primals_84);  mul_57 = None
        convert_element_type_153 = torch.ops.prims.convert_element_type.default(primals_86, torch.float16)
        convert_element_type_154 = torch.ops.prims.convert_element_type.default(primals_85, torch.float16)
        convert_element_type_155 = torch.ops.prims.convert_element_type.default(add_57, torch.float16);  add_57 = None
        view_104 = torch.ops.aten.view.default(convert_element_type_155, [20480, 768]);  convert_element_type_155 = None
        permute_78 = torch.ops.aten.permute.default(convert_element_type_154, [1, 0]);  convert_element_type_154 = None
        addmm_19 = torch.ops.aten.addmm.default(convert_element_type_153, view_104, permute_78);  convert_element_type_153 = view_104 = permute_78 = None
        view_105 = torch.ops.aten.view.default(addmm_19, [20, 1024, 1536]);  addmm_19 = None
        convert_element_type_159 = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None
        mul_58 = torch.ops.aten.mul.Tensor(convert_element_type_159, 0.5)
        mul_59 = torch.ops.aten.mul.Tensor(convert_element_type_159, 0.7071067811865476);  convert_element_type_159 = None
        erf_6 = torch.ops.aten.erf.default(mul_59);  mul_59 = None
        add_58 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_60 = torch.ops.aten.mul.Tensor(mul_58, add_58);  mul_58 = add_58 = None
        convert_element_type_160 = torch.ops.prims.convert_element_type.default(mul_60, torch.float16);  mul_60 = None
        convert_element_type_161 = torch.ops.prims.convert_element_type.default(primals_88, torch.float16);  primals_88 = None
        convert_element_type_162 = torch.ops.prims.convert_element_type.default(primals_87, torch.float16)
        view_106 = torch.ops.aten.view.default(convert_element_type_160, [20480, 1536]);  convert_element_type_160 = None
        permute_79 = torch.ops.aten.permute.default(convert_element_type_162, [1, 0]);  convert_element_type_162 = None
        addmm_20 = torch.ops.aten.addmm.default(convert_element_type_161, view_106, permute_79);  convert_element_type_161 = view_106 = permute_79 = None
        view_107 = torch.ops.aten.view.default(addmm_20, [20, 1024, 768]);  addmm_20 = None
        inductor_lookup_seed_default_6 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_5 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        convert_element_type_default_57 = torch.ops.prims.convert_element_type.default(inductor_random_default_5, torch.float16);  inductor_random_default_5 = None
        gt_6 = torch.ops.aten.gt.Scalar(convert_element_type_default_57, 0.2);  convert_element_type_default_57 = None
        mul_61 = torch.ops.aten.mul.Tensor(gt_6, view_107);  gt_6 = view_107 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul_61, 1.25);  mul_61 = None
        add_59 = torch.ops.aten.add.Tensor(add_55, mul_62);  mul_62 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_91 = var_mean_14[0]
        getitem_92 = var_mean_14[1];  var_mean_14 = None
        add_60 = torch.ops.aten.add.Tensor(getitem_91, 1e-05);  getitem_91 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_59, getitem_92);  getitem_92 = None
        mul_63 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_64 = torch.ops.aten.mul.Tensor(mul_63, primals_89);  mul_63 = None
        add_61 = torch.ops.aten.add.Tensor(mul_64, primals_90);  mul_64 = None
        permute_80 = torch.ops.aten.permute.default(add_61, [1, 0, 2]);  add_61 = None
        convert_element_type_166 = torch.ops.prims.convert_element_type.default(primals_91, torch.float16)
        convert_element_type_167 = torch.ops.prims.convert_element_type.default(primals_92, torch.float16)
        convert_element_type_168 = torch.ops.prims.convert_element_type.default(permute_80, torch.float16);  permute_80 = None
        permute_81 = torch.ops.aten.permute.default(convert_element_type_167, [1, 0]);  convert_element_type_167 = None
        clone_16 = torch.ops.aten.clone.default(convert_element_type_168, memory_format = torch.contiguous_format);  convert_element_type_168 = None
        view_108 = torch.ops.aten.view.default(clone_16, [20480, 768]);  clone_16 = None
        mm_8 = torch.ops.aten.mm.default(view_108, permute_81);  view_108 = permute_81 = None
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
        graphsafe_run_with_rng_state_7 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_114, view_115, view_116, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_7);  view_114 = view_115 = view_116 = fwd_rng_state_7 = None
        getitem_93 = graphsafe_run_with_rng_state_7[0];  graphsafe_run_with_rng_state_7 = None
        permute_86 = torch.ops.aten.permute.default(getitem_93, [2, 0, 1, 3]);  getitem_93 = None
        view_117 = torch.ops.aten.view.default(permute_86, [20480, 768]);  permute_86 = None
        convert_element_type_171 = torch.ops.prims.convert_element_type.default(primals_94, torch.float16);  primals_94 = None
        convert_element_type_172 = torch.ops.prims.convert_element_type.default(primals_93, torch.float16)
        permute_87 = torch.ops.aten.permute.default(convert_element_type_172, [1, 0]);  convert_element_type_172 = None
        addmm_21 = torch.ops.aten.addmm.default(convert_element_type_171, view_117, permute_87);  convert_element_type_171 = view_117 = permute_87 = None
        view_118 = torch.ops.aten.view.default(addmm_21, [1024, 20, 768]);  addmm_21 = None
        permute_88 = torch.ops.aten.permute.default(view_118, [1, 0, 2]);  view_118 = None
        add_63 = torch.ops.aten.add.Tensor(add_59, permute_88);  permute_88 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_102 = var_mean_15[0]
        getitem_103 = var_mean_15[1];  var_mean_15 = None
        add_64 = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_63, getitem_103);  getitem_103 = None
        mul_65 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_66 = torch.ops.aten.mul.Tensor(mul_65, primals_95);  mul_65 = None
        add_65 = torch.ops.aten.add.Tensor(mul_66, primals_96);  mul_66 = None
        convert_element_type_176 = torch.ops.prims.convert_element_type.default(primals_98, torch.float16)
        convert_element_type_177 = torch.ops.prims.convert_element_type.default(primals_97, torch.float16)
        convert_element_type_178 = torch.ops.prims.convert_element_type.default(add_65, torch.float16);  add_65 = None
        view_119 = torch.ops.aten.view.default(convert_element_type_178, [20480, 768]);  convert_element_type_178 = None
        permute_89 = torch.ops.aten.permute.default(convert_element_type_177, [1, 0]);  convert_element_type_177 = None
        addmm_22 = torch.ops.aten.addmm.default(convert_element_type_176, view_119, permute_89);  convert_element_type_176 = view_119 = permute_89 = None
        view_120 = torch.ops.aten.view.default(addmm_22, [20, 1024, 1536]);  addmm_22 = None
        convert_element_type_182 = torch.ops.prims.convert_element_type.default(view_120, torch.float32);  view_120 = None
        mul_67 = torch.ops.aten.mul.Tensor(convert_element_type_182, 0.5)
        mul_68 = torch.ops.aten.mul.Tensor(convert_element_type_182, 0.7071067811865476);  convert_element_type_182 = None
        erf_7 = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_66 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_67, add_66);  mul_67 = add_66 = None
        convert_element_type_183 = torch.ops.prims.convert_element_type.default(mul_69, torch.float16);  mul_69 = None
        convert_element_type_184 = torch.ops.prims.convert_element_type.default(primals_100, torch.float16);  primals_100 = None
        convert_element_type_185 = torch.ops.prims.convert_element_type.default(primals_99, torch.float16)
        view_121 = torch.ops.aten.view.default(convert_element_type_183, [20480, 1536]);  convert_element_type_183 = None
        permute_90 = torch.ops.aten.permute.default(convert_element_type_185, [1, 0]);  convert_element_type_185 = None
        addmm_23 = torch.ops.aten.addmm.default(convert_element_type_184, view_121, permute_90);  convert_element_type_184 = view_121 = permute_90 = None
        view_122 = torch.ops.aten.view.default(addmm_23, [20, 1024, 768]);  addmm_23 = None
        inductor_lookup_seed_default_7 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_4 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        convert_element_type_default_56 = torch.ops.prims.convert_element_type.default(inductor_random_default_4, torch.float16);  inductor_random_default_4 = None
        gt_7 = torch.ops.aten.gt.Scalar(convert_element_type_default_56, 0.2);  convert_element_type_default_56 = None
        mul_70 = torch.ops.aten.mul.Tensor(gt_7, view_122);  gt_7 = view_122 = None
        mul_71 = torch.ops.aten.mul.Tensor(mul_70, 1.25);  mul_70 = None
        add_67 = torch.ops.aten.add.Tensor(add_63, mul_71);  mul_71 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_104 = var_mean_16[0]
        getitem_105 = var_mean_16[1];  var_mean_16 = None
        add_68 = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_67, getitem_105);  getitem_105 = None
        mul_72 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, primals_101);  mul_72 = None
        add_69 = torch.ops.aten.add.Tensor(mul_73, primals_102);  mul_73 = None
        permute_91 = torch.ops.aten.permute.default(add_69, [1, 0, 2]);  add_69 = None
        convert_element_type_189 = torch.ops.prims.convert_element_type.default(primals_103, torch.float16)
        convert_element_type_190 = torch.ops.prims.convert_element_type.default(primals_104, torch.float16)
        convert_element_type_191 = torch.ops.prims.convert_element_type.default(permute_91, torch.float16);  permute_91 = None
        permute_92 = torch.ops.aten.permute.default(convert_element_type_190, [1, 0]);  convert_element_type_190 = None
        clone_18 = torch.ops.aten.clone.default(convert_element_type_191, memory_format = torch.contiguous_format);  convert_element_type_191 = None
        view_123 = torch.ops.aten.view.default(clone_18, [20480, 768]);  clone_18 = None
        mm_9 = torch.ops.aten.mm.default(view_123, permute_92);  view_123 = permute_92 = None
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
        graphsafe_run_with_rng_state_8 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_129, view_130, view_131, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_8);  view_129 = view_130 = view_131 = fwd_rng_state_8 = None
        getitem_106 = graphsafe_run_with_rng_state_8[0];  graphsafe_run_with_rng_state_8 = None
        permute_97 = torch.ops.aten.permute.default(getitem_106, [2, 0, 1, 3]);  getitem_106 = None
        view_132 = torch.ops.aten.view.default(permute_97, [20480, 768]);  permute_97 = None
        convert_element_type_194 = torch.ops.prims.convert_element_type.default(primals_106, torch.float16);  primals_106 = None
        convert_element_type_195 = torch.ops.prims.convert_element_type.default(primals_105, torch.float16)
        permute_98 = torch.ops.aten.permute.default(convert_element_type_195, [1, 0]);  convert_element_type_195 = None
        addmm_24 = torch.ops.aten.addmm.default(convert_element_type_194, view_132, permute_98);  convert_element_type_194 = view_132 = permute_98 = None
        view_133 = torch.ops.aten.view.default(addmm_24, [1024, 20, 768]);  addmm_24 = None
        permute_99 = torch.ops.aten.permute.default(view_133, [1, 0, 2]);  view_133 = None
        add_71 = torch.ops.aten.add.Tensor(add_67, permute_99);  permute_99 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_115 = var_mean_17[0]
        getitem_116 = var_mean_17[1];  var_mean_17 = None
        add_72 = torch.ops.aten.add.Tensor(getitem_115, 1e-05);  getitem_115 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_71, getitem_116);  getitem_116 = None
        mul_74 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_75 = torch.ops.aten.mul.Tensor(mul_74, primals_107);  mul_74 = None
        add_73 = torch.ops.aten.add.Tensor(mul_75, primals_108);  mul_75 = None
        convert_element_type_199 = torch.ops.prims.convert_element_type.default(primals_110, torch.float16)
        convert_element_type_200 = torch.ops.prims.convert_element_type.default(primals_109, torch.float16)
        convert_element_type_201 = torch.ops.prims.convert_element_type.default(add_73, torch.float16);  add_73 = None
        view_134 = torch.ops.aten.view.default(convert_element_type_201, [20480, 768]);  convert_element_type_201 = None
        permute_100 = torch.ops.aten.permute.default(convert_element_type_200, [1, 0]);  convert_element_type_200 = None
        addmm_25 = torch.ops.aten.addmm.default(convert_element_type_199, view_134, permute_100);  convert_element_type_199 = view_134 = permute_100 = None
        view_135 = torch.ops.aten.view.default(addmm_25, [20, 1024, 1536]);  addmm_25 = None
        convert_element_type_205 = torch.ops.prims.convert_element_type.default(view_135, torch.float32);  view_135 = None
        mul_76 = torch.ops.aten.mul.Tensor(convert_element_type_205, 0.5)
        mul_77 = torch.ops.aten.mul.Tensor(convert_element_type_205, 0.7071067811865476);  convert_element_type_205 = None
        erf_8 = torch.ops.aten.erf.default(mul_77);  mul_77 = None
        add_74 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_78 = torch.ops.aten.mul.Tensor(mul_76, add_74);  mul_76 = add_74 = None
        convert_element_type_206 = torch.ops.prims.convert_element_type.default(mul_78, torch.float16);  mul_78 = None
        convert_element_type_207 = torch.ops.prims.convert_element_type.default(primals_112, torch.float16);  primals_112 = None
        convert_element_type_208 = torch.ops.prims.convert_element_type.default(primals_111, torch.float16)
        view_136 = torch.ops.aten.view.default(convert_element_type_206, [20480, 1536]);  convert_element_type_206 = None
        permute_101 = torch.ops.aten.permute.default(convert_element_type_208, [1, 0]);  convert_element_type_208 = None
        addmm_26 = torch.ops.aten.addmm.default(convert_element_type_207, view_136, permute_101);  convert_element_type_207 = view_136 = permute_101 = None
        view_137 = torch.ops.aten.view.default(addmm_26, [20, 1024, 768]);  addmm_26 = None
        inductor_lookup_seed_default_8 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_3 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        convert_element_type_default_55 = torch.ops.prims.convert_element_type.default(inductor_random_default_3, torch.float16);  inductor_random_default_3 = None
        gt_8 = torch.ops.aten.gt.Scalar(convert_element_type_default_55, 0.2);  convert_element_type_default_55 = None
        mul_79 = torch.ops.aten.mul.Tensor(gt_8, view_137);  gt_8 = view_137 = None
        mul_80 = torch.ops.aten.mul.Tensor(mul_79, 1.25);  mul_79 = None
        add_75 = torch.ops.aten.add.Tensor(add_71, mul_80);  mul_80 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_117 = var_mean_18[0]
        getitem_118 = var_mean_18[1];  var_mean_18 = None
        add_76 = torch.ops.aten.add.Tensor(getitem_117, 1e-05);  getitem_117 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_75, getitem_118);  getitem_118 = None
        mul_81 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_82 = torch.ops.aten.mul.Tensor(mul_81, primals_113);  mul_81 = None
        add_77 = torch.ops.aten.add.Tensor(mul_82, primals_114);  mul_82 = None
        permute_102 = torch.ops.aten.permute.default(add_77, [1, 0, 2]);  add_77 = None
        convert_element_type_212 = torch.ops.prims.convert_element_type.default(primals_115, torch.float16)
        convert_element_type_213 = torch.ops.prims.convert_element_type.default(primals_116, torch.float16)
        convert_element_type_214 = torch.ops.prims.convert_element_type.default(permute_102, torch.float16);  permute_102 = None
        permute_103 = torch.ops.aten.permute.default(convert_element_type_213, [1, 0]);  convert_element_type_213 = None
        clone_20 = torch.ops.aten.clone.default(convert_element_type_214, memory_format = torch.contiguous_format);  convert_element_type_214 = None
        view_138 = torch.ops.aten.view.default(clone_20, [20480, 768]);  clone_20 = None
        mm_10 = torch.ops.aten.mm.default(view_138, permute_103);  view_138 = permute_103 = None
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
        graphsafe_run_with_rng_state_9 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_144, view_145, view_146, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_9);  view_144 = view_145 = view_146 = fwd_rng_state_9 = None
        getitem_119 = graphsafe_run_with_rng_state_9[0];  graphsafe_run_with_rng_state_9 = None
        permute_108 = torch.ops.aten.permute.default(getitem_119, [2, 0, 1, 3]);  getitem_119 = None
        view_147 = torch.ops.aten.view.default(permute_108, [20480, 768]);  permute_108 = None
        convert_element_type_217 = torch.ops.prims.convert_element_type.default(primals_118, torch.float16);  primals_118 = None
        convert_element_type_218 = torch.ops.prims.convert_element_type.default(primals_117, torch.float16)
        permute_109 = torch.ops.aten.permute.default(convert_element_type_218, [1, 0]);  convert_element_type_218 = None
        addmm_27 = torch.ops.aten.addmm.default(convert_element_type_217, view_147, permute_109);  convert_element_type_217 = view_147 = permute_109 = None
        view_148 = torch.ops.aten.view.default(addmm_27, [1024, 20, 768]);  addmm_27 = None
        permute_110 = torch.ops.aten.permute.default(view_148, [1, 0, 2]);  view_148 = None
        add_79 = torch.ops.aten.add.Tensor(add_75, permute_110);  permute_110 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
        getitem_128 = var_mean_19[0]
        getitem_129 = var_mean_19[1];  var_mean_19 = None
        add_80 = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_79, getitem_129);  getitem_129 = None
        mul_83 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_84 = torch.ops.aten.mul.Tensor(mul_83, primals_119);  mul_83 = None
        add_81 = torch.ops.aten.add.Tensor(mul_84, primals_120);  mul_84 = None
        convert_element_type_222 = torch.ops.prims.convert_element_type.default(primals_122, torch.float16)
        convert_element_type_223 = torch.ops.prims.convert_element_type.default(primals_121, torch.float16)
        convert_element_type_224 = torch.ops.prims.convert_element_type.default(add_81, torch.float16);  add_81 = None
        view_149 = torch.ops.aten.view.default(convert_element_type_224, [20480, 768]);  convert_element_type_224 = None
        permute_111 = torch.ops.aten.permute.default(convert_element_type_223, [1, 0]);  convert_element_type_223 = None
        addmm_28 = torch.ops.aten.addmm.default(convert_element_type_222, view_149, permute_111);  convert_element_type_222 = view_149 = permute_111 = None
        view_150 = torch.ops.aten.view.default(addmm_28, [20, 1024, 1536]);  addmm_28 = None
        convert_element_type_228 = torch.ops.prims.convert_element_type.default(view_150, torch.float32);  view_150 = None
        mul_85 = torch.ops.aten.mul.Tensor(convert_element_type_228, 0.5)
        mul_86 = torch.ops.aten.mul.Tensor(convert_element_type_228, 0.7071067811865476);  convert_element_type_228 = None
        erf_9 = torch.ops.aten.erf.default(mul_86);  mul_86 = None
        add_82 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_87 = torch.ops.aten.mul.Tensor(mul_85, add_82);  mul_85 = add_82 = None
        convert_element_type_229 = torch.ops.prims.convert_element_type.default(mul_87, torch.float16);  mul_87 = None
        convert_element_type_230 = torch.ops.prims.convert_element_type.default(primals_124, torch.float16);  primals_124 = None
        convert_element_type_231 = torch.ops.prims.convert_element_type.default(primals_123, torch.float16)
        view_151 = torch.ops.aten.view.default(convert_element_type_229, [20480, 1536]);  convert_element_type_229 = None
        permute_112 = torch.ops.aten.permute.default(convert_element_type_231, [1, 0]);  convert_element_type_231 = None
        addmm_29 = torch.ops.aten.addmm.default(convert_element_type_230, view_151, permute_112);  convert_element_type_230 = view_151 = permute_112 = None
        view_152 = torch.ops.aten.view.default(addmm_29, [20, 1024, 768]);  addmm_29 = None
        inductor_lookup_seed_default_9 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_2 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        convert_element_type_default_54 = torch.ops.prims.convert_element_type.default(inductor_random_default_2, torch.float16);  inductor_random_default_2 = None
        gt_9 = torch.ops.aten.gt.Scalar(convert_element_type_default_54, 0.2);  convert_element_type_default_54 = None
        mul_88 = torch.ops.aten.mul.Tensor(gt_9, view_152);  gt_9 = view_152 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_88, 1.25);  mul_88 = None
        add_83 = torch.ops.aten.add.Tensor(add_79, mul_89);  mul_89 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_130 = var_mean_20[0]
        getitem_131 = var_mean_20[1];  var_mean_20 = None
        add_84 = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_83, getitem_131);  getitem_131 = None
        mul_90 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_91 = torch.ops.aten.mul.Tensor(mul_90, primals_125);  mul_90 = None
        add_85 = torch.ops.aten.add.Tensor(mul_91, primals_126);  mul_91 = None
        permute_113 = torch.ops.aten.permute.default(add_85, [1, 0, 2]);  add_85 = None
        convert_element_type_235 = torch.ops.prims.convert_element_type.default(primals_127, torch.float16)
        convert_element_type_236 = torch.ops.prims.convert_element_type.default(primals_128, torch.float16)
        convert_element_type_237 = torch.ops.prims.convert_element_type.default(permute_113, torch.float16);  permute_113 = None
        permute_114 = torch.ops.aten.permute.default(convert_element_type_236, [1, 0]);  convert_element_type_236 = None
        clone_22 = torch.ops.aten.clone.default(convert_element_type_237, memory_format = torch.contiguous_format);  convert_element_type_237 = None
        view_153 = torch.ops.aten.view.default(clone_22, [20480, 768]);  clone_22 = None
        mm_11 = torch.ops.aten.mm.default(view_153, permute_114);  view_153 = permute_114 = None
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
        graphsafe_run_with_rng_state_10 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_159, view_160, view_161, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_10);  view_159 = view_160 = view_161 = fwd_rng_state_10 = None
        getitem_132 = graphsafe_run_with_rng_state_10[0];  graphsafe_run_with_rng_state_10 = None
        permute_119 = torch.ops.aten.permute.default(getitem_132, [2, 0, 1, 3]);  getitem_132 = None
        view_162 = torch.ops.aten.view.default(permute_119, [20480, 768]);  permute_119 = None
        convert_element_type_240 = torch.ops.prims.convert_element_type.default(primals_130, torch.float16);  primals_130 = None
        convert_element_type_241 = torch.ops.prims.convert_element_type.default(primals_129, torch.float16)
        permute_120 = torch.ops.aten.permute.default(convert_element_type_241, [1, 0]);  convert_element_type_241 = None
        addmm_30 = torch.ops.aten.addmm.default(convert_element_type_240, view_162, permute_120);  convert_element_type_240 = view_162 = permute_120 = None
        view_163 = torch.ops.aten.view.default(addmm_30, [1024, 20, 768]);  addmm_30 = None
        permute_121 = torch.ops.aten.permute.default(view_163, [1, 0, 2]);  view_163 = None
        add_87 = torch.ops.aten.add.Tensor(add_83, permute_121);  permute_121 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
        getitem_141 = var_mean_21[0]
        getitem_142 = var_mean_21[1];  var_mean_21 = None
        add_88 = torch.ops.aten.add.Tensor(getitem_141, 1e-05);  getitem_141 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_87, getitem_142);  getitem_142 = None
        mul_92 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_92, primals_131);  mul_92 = None
        add_89 = torch.ops.aten.add.Tensor(mul_93, primals_132);  mul_93 = None
        convert_element_type_245 = torch.ops.prims.convert_element_type.default(primals_134, torch.float16)
        convert_element_type_246 = torch.ops.prims.convert_element_type.default(primals_133, torch.float16)
        convert_element_type_247 = torch.ops.prims.convert_element_type.default(add_89, torch.float16);  add_89 = None
        view_164 = torch.ops.aten.view.default(convert_element_type_247, [20480, 768]);  convert_element_type_247 = None
        permute_122 = torch.ops.aten.permute.default(convert_element_type_246, [1, 0]);  convert_element_type_246 = None
        addmm_31 = torch.ops.aten.addmm.default(convert_element_type_245, view_164, permute_122);  convert_element_type_245 = view_164 = permute_122 = None
        view_165 = torch.ops.aten.view.default(addmm_31, [20, 1024, 1536]);  addmm_31 = None
        convert_element_type_251 = torch.ops.prims.convert_element_type.default(view_165, torch.float32);  view_165 = None
        mul_94 = torch.ops.aten.mul.Tensor(convert_element_type_251, 0.5)
        mul_95 = torch.ops.aten.mul.Tensor(convert_element_type_251, 0.7071067811865476);  convert_element_type_251 = None
        erf_10 = torch.ops.aten.erf.default(mul_95);  mul_95 = None
        add_90 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_96 = torch.ops.aten.mul.Tensor(mul_94, add_90);  mul_94 = add_90 = None
        convert_element_type_252 = torch.ops.prims.convert_element_type.default(mul_96, torch.float16);  mul_96 = None
        convert_element_type_253 = torch.ops.prims.convert_element_type.default(primals_136, torch.float16);  primals_136 = None
        convert_element_type_254 = torch.ops.prims.convert_element_type.default(primals_135, torch.float16)
        view_166 = torch.ops.aten.view.default(convert_element_type_252, [20480, 1536]);  convert_element_type_252 = None
        permute_123 = torch.ops.aten.permute.default(convert_element_type_254, [1, 0]);  convert_element_type_254 = None
        addmm_32 = torch.ops.aten.addmm.default(convert_element_type_253, view_166, permute_123);  convert_element_type_253 = view_166 = permute_123 = None
        view_167 = torch.ops.aten.view.default(addmm_32, [20, 1024, 768]);  addmm_32 = None
        inductor_lookup_seed_default_10 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_1 = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        convert_element_type_default_53 = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.float16);  inductor_random_default_1 = None
        gt_10 = torch.ops.aten.gt.Scalar(convert_element_type_default_53, 0.2);  convert_element_type_default_53 = None
        mul_97 = torch.ops.aten.mul.Tensor(gt_10, view_167);  gt_10 = view_167 = None
        mul_98 = torch.ops.aten.mul.Tensor(mul_97, 1.25);  mul_97 = None
        add_91 = torch.ops.aten.add.Tensor(add_87, mul_98);  mul_98 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_143 = var_mean_22[0]
        getitem_144 = var_mean_22[1];  var_mean_22 = None
        add_92 = torch.ops.aten.add.Tensor(getitem_143, 1e-05);  getitem_143 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_91, getitem_144);  getitem_144 = None
        mul_99 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_100 = torch.ops.aten.mul.Tensor(mul_99, primals_137);  mul_99 = None
        add_93 = torch.ops.aten.add.Tensor(mul_100, primals_138);  mul_100 = None
        permute_124 = torch.ops.aten.permute.default(add_93, [1, 0, 2]);  add_93 = None
        convert_element_type_258 = torch.ops.prims.convert_element_type.default(primals_139, torch.float16)
        convert_element_type_259 = torch.ops.prims.convert_element_type.default(primals_140, torch.float16)
        convert_element_type_260 = torch.ops.prims.convert_element_type.default(permute_124, torch.float16);  permute_124 = None
        permute_125 = torch.ops.aten.permute.default(convert_element_type_259, [1, 0]);  convert_element_type_259 = None
        clone_24 = torch.ops.aten.clone.default(convert_element_type_260, memory_format = torch.contiguous_format);  convert_element_type_260 = None
        view_168 = torch.ops.aten.view.default(clone_24, [20480, 768]);  clone_24 = None
        mm_12 = torch.ops.aten.mm.default(view_168, permute_125);  view_168 = permute_125 = None
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
        graphsafe_run_with_rng_state_11 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_174, view_175, view_176, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_11);  view_174 = view_175 = view_176 = fwd_rng_state_11 = None
        getitem_145 = graphsafe_run_with_rng_state_11[0];  graphsafe_run_with_rng_state_11 = None
        permute_130 = torch.ops.aten.permute.default(getitem_145, [2, 0, 1, 3]);  getitem_145 = None
        view_177 = torch.ops.aten.view.default(permute_130, [20480, 768]);  permute_130 = None
        convert_element_type_263 = torch.ops.prims.convert_element_type.default(primals_142, torch.float16);  primals_142 = None
        convert_element_type_264 = torch.ops.prims.convert_element_type.default(primals_141, torch.float16)
        permute_131 = torch.ops.aten.permute.default(convert_element_type_264, [1, 0]);  convert_element_type_264 = None
        addmm_33 = torch.ops.aten.addmm.default(convert_element_type_263, view_177, permute_131);  convert_element_type_263 = view_177 = permute_131 = None
        view_178 = torch.ops.aten.view.default(addmm_33, [1024, 20, 768]);  addmm_33 = None
        permute_132 = torch.ops.aten.permute.default(view_178, [1, 0, 2]);  view_178 = None
        add_95 = torch.ops.aten.add.Tensor(add_91, permute_132);  permute_132 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
        getitem_154 = var_mean_23[0]
        getitem_155 = var_mean_23[1];  var_mean_23 = None
        add_96 = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_95, getitem_155);  getitem_155 = None
        mul_101 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_102 = torch.ops.aten.mul.Tensor(mul_101, primals_143);  mul_101 = None
        add_97 = torch.ops.aten.add.Tensor(mul_102, primals_144);  mul_102 = None
        convert_element_type_268 = torch.ops.prims.convert_element_type.default(primals_146, torch.float16)
        convert_element_type_269 = torch.ops.prims.convert_element_type.default(primals_145, torch.float16)
        convert_element_type_270 = torch.ops.prims.convert_element_type.default(add_97, torch.float16);  add_97 = None
        view_179 = torch.ops.aten.view.default(convert_element_type_270, [20480, 768]);  convert_element_type_270 = None
        permute_133 = torch.ops.aten.permute.default(convert_element_type_269, [1, 0]);  convert_element_type_269 = None
        addmm_34 = torch.ops.aten.addmm.default(convert_element_type_268, view_179, permute_133);  convert_element_type_268 = view_179 = permute_133 = None
        view_180 = torch.ops.aten.view.default(addmm_34, [20, 1024, 1536]);  addmm_34 = None
        convert_element_type_274 = torch.ops.prims.convert_element_type.default(view_180, torch.float32);  view_180 = None
        mul_103 = torch.ops.aten.mul.Tensor(convert_element_type_274, 0.5)
        mul_104 = torch.ops.aten.mul.Tensor(convert_element_type_274, 0.7071067811865476);  convert_element_type_274 = None
        erf_11 = torch.ops.aten.erf.default(mul_104);  mul_104 = None
        add_98 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_105 = torch.ops.aten.mul.Tensor(mul_103, add_98);  mul_103 = add_98 = None
        convert_element_type_275 = torch.ops.prims.convert_element_type.default(mul_105, torch.float16);  mul_105 = None
        convert_element_type_276 = torch.ops.prims.convert_element_type.default(primals_148, torch.float16);  primals_148 = None
        convert_element_type_277 = torch.ops.prims.convert_element_type.default(primals_147, torch.float16)
        view_181 = torch.ops.aten.view.default(convert_element_type_275, [20480, 1536]);  convert_element_type_275 = None
        permute_134 = torch.ops.aten.permute.default(convert_element_type_277, [1, 0]);  convert_element_type_277 = None
        addmm_35 = torch.ops.aten.addmm.default(convert_element_type_276, view_181, permute_134);  convert_element_type_276 = view_181 = permute_134 = None
        view_182 = torch.ops.aten.view.default(addmm_35, [20, 1024, 768]);  addmm_35 = None
        inductor_lookup_seed_default_11 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default = torch.ops.prims.inductor_random.default([20, 1024, 768], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        convert_element_type_default_52 = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        gt_11 = torch.ops.aten.gt.Scalar(convert_element_type_default_52, 0.2);  convert_element_type_default_52 = None
        mul_106 = torch.ops.aten.mul.Tensor(gt_11, view_182);  gt_11 = view_182 = None
        mul_107 = torch.ops.aten.mul.Tensor(mul_106, 1.25);  mul_106 = None
        add_99 = torch.ops.aten.add.Tensor(add_95, mul_107);  mul_107 = None
        convert_element_type_281 = torch.ops.prims.convert_element_type.default(primals_150, torch.float16);  primals_150 = None
        convert_element_type_282 = torch.ops.prims.convert_element_type.default(primals_149, torch.float16);  primals_149 = None
        convert_element_type_283 = torch.ops.prims.convert_element_type.default(add_99, torch.float16);  add_99 = None
        view_183 = torch.ops.aten.view.default(convert_element_type_283, [20480, 768]);  convert_element_type_283 = None
        permute_135 = torch.ops.aten.permute.default(convert_element_type_282, [1, 0]);  convert_element_type_282 = None
        addmm_36 = torch.ops.aten.addmm.default(convert_element_type_281, view_183, permute_135);  convert_element_type_281 = None
        view_184 = torch.ops.aten.view.default(addmm_36, [20, 1024, 128]);  addmm_36 = None
        relu = torch.ops.aten.relu.default(view_184);  view_184 = None
        view_185 = torch.ops.aten.view.default(relu, [20480, 128]);  relu = None
        convert_element_type_287 = torch.ops.prims.convert_element_type.default(primals_152, torch.float16);  primals_152 = None
        convert_element_type_288 = torch.ops.prims.convert_element_type.default(primals_151, torch.float16);  primals_151 = None
        permute_136 = torch.ops.aten.permute.default(convert_element_type_288, [1, 0]);  convert_element_type_288 = None
        addmm_37 = torch.ops.aten.addmm.default(convert_element_type_287, view_185, permute_136);  convert_element_type_287 = None
        view_191 = torch.ops.aten.view.default(addmm_37, [20, 1024, 16]);  addmm_37 = None
        permute_137 = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
        convert_element_type_292 = torch.ops.prims.convert_element_type.default(permute_137, torch.float32);  permute_137 = None
        view_192 = torch.ops.aten.view.default(convert_element_type_292, [20, 1, 4, 4, 32, 32]);  convert_element_type_292 = None
        permute_138 = torch.ops.aten.permute.default(view_192, [0, 1, 2, 4, 3, 5]);  view_192 = None
        full_default = torch.ops.aten.full.default([20, 1, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put = torch.ops.aten.index_put.default(full_default, [None, None, unsqueeze_5, add], permute_138, True);  unsqueeze_5 = add = permute_138 = None
        convert_element_type_default_51 = torch.ops.prims.convert_element_type.default(index_put, torch.float32);  index_put = None
        iota_8 = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_295 = torch.ops.prims.convert_element_type.default(iota_8, torch.float32);  iota_8 = None
        add_102 = torch.ops.aten.add.Tensor(convert_element_type_295, 0.5);  convert_element_type_295 = None
        mul_108 = torch.ops.aten.mul.Tensor(add_102, 4.0);  add_102 = None
        sub_24 = torch.ops.aten.sub.Tensor(mul_108, 0.5);  mul_108 = None
        clamp_min = torch.ops.aten.clamp_min.default(sub_24, 0.0);  sub_24 = None
        view_193 = torch.ops.aten.view.default(clamp_min, [32, 1])
        convert_element_type_296 = torch.ops.prims.convert_element_type.default(view_193, torch.int64)
        add_103 = torch.ops.aten.add.Tensor(convert_element_type_296, 1)
        clamp_max = torch.ops.aten.clamp_max.default(add_103, 127);  add_103 = None
        convert_element_type_298 = torch.ops.prims.convert_element_type.default(clamp_min, torch.int64)
        add_105 = torch.ops.aten.add.Tensor(convert_element_type_298, 1)
        clamp_max_1 = torch.ops.aten.clamp_max.default(add_105, 127);  add_105 = None
        _unsafe_index = torch.ops.aten._unsafe_index.Tensor(convert_element_type_default_51, [None, None, convert_element_type_296, convert_element_type_298])
        _unsafe_index_1 = torch.ops.aten._unsafe_index.Tensor(convert_element_type_default_51, [None, None, convert_element_type_296, clamp_max_1])
        _unsafe_index_2 = torch.ops.aten._unsafe_index.Tensor(convert_element_type_default_51, [None, None, clamp_max, convert_element_type_298])
        _unsafe_index_3 = torch.ops.aten._unsafe_index.Tensor(convert_element_type_default_51, [None, None, clamp_max, clamp_max_1]);  convert_element_type_default_51 = None
        sub_26 = torch.ops.aten.sub.Tensor(clamp_min, convert_element_type_298);  clamp_min = None
        clamp_min_2 = torch.ops.aten.clamp_min.default(sub_26, 0.0);  sub_26 = None
        clamp_max_2 = torch.ops.aten.clamp_max.default(clamp_min_2, 1.0);  clamp_min_2 = None
        sub_27 = torch.ops.aten.sub.Tensor(_unsafe_index_1, _unsafe_index);  _unsafe_index_1 = None
        mul_110 = torch.ops.aten.mul.Tensor(sub_27, clamp_max_2);  sub_27 = None
        add_106 = torch.ops.aten.add.Tensor(_unsafe_index, mul_110);  _unsafe_index = mul_110 = None
        sub_28 = torch.ops.aten.sub.Tensor(_unsafe_index_3, _unsafe_index_2);  _unsafe_index_3 = None
        mul_111 = torch.ops.aten.mul.Tensor(sub_28, clamp_max_2);  sub_28 = None
        add_107 = torch.ops.aten.add.Tensor(_unsafe_index_2, mul_111);  _unsafe_index_2 = mul_111 = None
        sub_29 = torch.ops.aten.sub.Tensor(view_193, convert_element_type_296);  view_193 = None
        clamp_min_3 = torch.ops.aten.clamp_min.default(sub_29, 0.0);  sub_29 = None
        clamp_max_3 = torch.ops.aten.clamp_max.default(clamp_min_3, 1.0);  clamp_min_3 = None
        sub_30 = torch.ops.aten.sub.Tensor(add_107, add_106);  add_107 = None
        mul_112 = torch.ops.aten.mul.Tensor(sub_30, clamp_max_3);  sub_30 = None
        add_108 = torch.ops.aten.add.Tensor(add_106, mul_112);  add_106 = mul_112 = None
        convert_element_type_299 = torch.ops.prims.convert_element_type.default(add_108, torch.float16);  add_108 = None
        permute_141 = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
        permute_145 = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
        return (convert_element_type_299, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_11, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_21, primals_23, primals_24, primals_25, primals_26, primals_27, primals_29, primals_30, primals_31, primals_32, primals_33, primals_35, primals_36, primals_37, primals_38, primals_39, primals_41, primals_42, primals_43, primals_44, primals_45, primals_47, primals_48, primals_49, primals_50, primals_51, primals_53, primals_54, primals_55, primals_56, primals_57, primals_59, primals_60, primals_61, primals_62, primals_63, primals_65, primals_66, primals_67, primals_68, primals_69, primals_71, primals_72, primals_73, primals_74, primals_75, primals_77, primals_78, primals_79, primals_80, primals_81, primals_83, primals_84, primals_85, primals_86, primals_87, primals_89, primals_90, primals_91, primals_92, primals_93, primals_95, primals_96, primals_97, primals_98, primals_99, primals_101, primals_102, primals_103, primals_104, primals_105, primals_107, primals_108, primals_109, primals_110, primals_111, primals_113, primals_114, primals_115, primals_116, primals_117, primals_119, primals_120, primals_121, primals_122, primals_123, primals_125, primals_126, primals_127, primals_128, primals_129, primals_131, primals_132, primals_133, primals_134, primals_135, primals_137, primals_138, primals_139, primals_140, primals_141, primals_143, primals_144, primals_145, primals_146, primals_147, view_1, mm, add_7, inductor_seeds_default, add_11, add_15, add_19, add_23, add_27, add_31, add_35, add_39, add_43, add_47, add_51, add_55, add_59, add_63, add_67, add_71, add_75, add_79, add_83, add_87, add_91, add_95, view_183, view_185, full_default, convert_element_type_296, clamp_max, convert_element_type_298, clamp_max_1, clamp_max_2, clamp_max_3, permute_141, permute_145)
        
def load_args(reader):
    buf0 = reader.storage(None, 1310720, device=device(type='cuda', index=0))
    reader.tensor(buf0, (20, 1, 128, 128), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 49152, device=device(type='cuda', index=0))
    reader.tensor(buf1, (768, 16), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf2, (768,), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1024, 768), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf4, (768,), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf5, (768,), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf6, (2304,), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf7, (2304, 768), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf8, (768, 768), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf9, (768,), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf10, (768,), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf11, (768,), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf12, (1536, 768), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf13, (1536,), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf14, (768, 1536), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf15, (768,), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf16, (768,), is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf17, (768,), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf18, (2304,), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf19, (2304, 768), is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf20, (768, 768), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768,), is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768,), is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768,), is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf24, (1536, 768), is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf25, (1536,), is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf26, (768, 1536), is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf27, (768,), is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf28, (768,), is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf29, (768,), is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf30, (2304,), is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf31, (2304, 768), is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf32, (768, 768), is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf33, (768,), is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf34, (768,), is_leaf=True)  # primals_35
    buf35 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf35, (768,), is_leaf=True)  # primals_36
    buf36 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf36, (1536, 768), is_leaf=True)  # primals_37
    buf37 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf37, (1536,), is_leaf=True)  # primals_38
    buf38 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf38, (768, 1536), is_leaf=True)  # primals_39
    buf39 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf39, (768,), is_leaf=True)  # primals_40
    buf40 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf40, (768,), is_leaf=True)  # primals_41
    buf41 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf41, (768,), is_leaf=True)  # primals_42
    buf42 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf42, (2304,), is_leaf=True)  # primals_43
    buf43 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf43, (2304, 768), is_leaf=True)  # primals_44
    buf44 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf44, (768, 768), is_leaf=True)  # primals_45
    buf45 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf45, (768,), is_leaf=True)  # primals_46
    buf46 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf46, (768,), is_leaf=True)  # primals_47
    buf47 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf47, (768,), is_leaf=True)  # primals_48
    buf48 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf48, (1536, 768), is_leaf=True)  # primals_49
    buf49 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf49, (1536,), is_leaf=True)  # primals_50
    buf50 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf50, (768, 1536), is_leaf=True)  # primals_51
    buf51 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf51, (768,), is_leaf=True)  # primals_52
    buf52 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf52, (768,), is_leaf=True)  # primals_53
    buf53 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf53, (768,), is_leaf=True)  # primals_54
    buf54 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf54, (2304,), is_leaf=True)  # primals_55
    buf55 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf55, (2304, 768), is_leaf=True)  # primals_56
    buf56 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf56, (768, 768), is_leaf=True)  # primals_57
    buf57 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf57, (768,), is_leaf=True)  # primals_58
    buf58 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf58, (768,), is_leaf=True)  # primals_59
    buf59 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf59, (768,), is_leaf=True)  # primals_60
    buf60 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf60, (1536, 768), is_leaf=True)  # primals_61
    buf61 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf61, (1536,), is_leaf=True)  # primals_62
    buf62 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf62, (768, 1536), is_leaf=True)  # primals_63
    buf63 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf63, (768,), is_leaf=True)  # primals_64
    buf64 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf64, (768,), is_leaf=True)  # primals_65
    buf65 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf65, (768,), is_leaf=True)  # primals_66
    buf66 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf66, (2304,), is_leaf=True)  # primals_67
    buf67 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf67, (2304, 768), is_leaf=True)  # primals_68
    buf68 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf68, (768, 768), is_leaf=True)  # primals_69
    buf69 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf69, (768,), is_leaf=True)  # primals_70
    buf70 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf70, (768,), is_leaf=True)  # primals_71
    buf71 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf71, (768,), is_leaf=True)  # primals_72
    buf72 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf72, (1536, 768), is_leaf=True)  # primals_73
    buf73 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf73, (1536,), is_leaf=True)  # primals_74
    buf74 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf74, (768, 1536), is_leaf=True)  # primals_75
    buf75 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf75, (768,), is_leaf=True)  # primals_76
    buf76 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf76, (768,), is_leaf=True)  # primals_77
    buf77 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf77, (768,), is_leaf=True)  # primals_78
    buf78 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf78, (2304,), is_leaf=True)  # primals_79
    buf79 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf79, (2304, 768), is_leaf=True)  # primals_80
    buf80 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf80, (768, 768), is_leaf=True)  # primals_81
    buf81 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf81, (768,), is_leaf=True)  # primals_82
    buf82 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf82, (768,), is_leaf=True)  # primals_83
    buf83 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf83, (768,), is_leaf=True)  # primals_84
    buf84 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf84, (1536, 768), is_leaf=True)  # primals_85
    buf85 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf85, (1536,), is_leaf=True)  # primals_86
    buf86 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf86, (768, 1536), is_leaf=True)  # primals_87
    buf87 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf87, (768,), is_leaf=True)  # primals_88
    buf88 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf88, (768,), is_leaf=True)  # primals_89
    buf89 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf89, (768,), is_leaf=True)  # primals_90
    buf90 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf90, (2304,), is_leaf=True)  # primals_91
    buf91 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf91, (2304, 768), is_leaf=True)  # primals_92
    buf92 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf92, (768, 768), is_leaf=True)  # primals_93
    buf93 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf93, (768,), is_leaf=True)  # primals_94
    buf94 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf94, (768,), is_leaf=True)  # primals_95
    buf95 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf95, (768,), is_leaf=True)  # primals_96
    buf96 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf96, (1536, 768), is_leaf=True)  # primals_97
    buf97 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf97, (1536,), is_leaf=True)  # primals_98
    buf98 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf98, (768, 1536), is_leaf=True)  # primals_99
    buf99 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf99, (768,), is_leaf=True)  # primals_100
    buf100 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf100, (768,), is_leaf=True)  # primals_101
    buf101 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf101, (768,), is_leaf=True)  # primals_102
    buf102 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf102, (2304,), is_leaf=True)  # primals_103
    buf103 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf103, (2304, 768), is_leaf=True)  # primals_104
    buf104 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf104, (768, 768), is_leaf=True)  # primals_105
    buf105 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf105, (768,), is_leaf=True)  # primals_106
    buf106 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf106, (768,), is_leaf=True)  # primals_107
    buf107 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf107, (768,), is_leaf=True)  # primals_108
    buf108 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf108, (1536, 768), is_leaf=True)  # primals_109
    buf109 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf109, (1536,), is_leaf=True)  # primals_110
    buf110 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf110, (768, 1536), is_leaf=True)  # primals_111
    buf111 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf111, (768,), is_leaf=True)  # primals_112
    buf112 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf112, (768,), is_leaf=True)  # primals_113
    buf113 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf113, (768,), is_leaf=True)  # primals_114
    buf114 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf114, (2304,), is_leaf=True)  # primals_115
    buf115 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf115, (2304, 768), is_leaf=True)  # primals_116
    buf116 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf116, (768, 768), is_leaf=True)  # primals_117
    buf117 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf117, (768,), is_leaf=True)  # primals_118
    buf118 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf118, (768,), is_leaf=True)  # primals_119
    buf119 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf119, (768,), is_leaf=True)  # primals_120
    buf120 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf120, (1536, 768), is_leaf=True)  # primals_121
    buf121 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf121, (1536,), is_leaf=True)  # primals_122
    buf122 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf122, (768, 1536), is_leaf=True)  # primals_123
    buf123 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf123, (768,), is_leaf=True)  # primals_124
    buf124 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf124, (768,), is_leaf=True)  # primals_125
    buf125 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf125, (768,), is_leaf=True)  # primals_126
    buf126 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf126, (2304,), is_leaf=True)  # primals_127
    buf127 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf127, (2304, 768), is_leaf=True)  # primals_128
    buf128 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf128, (768, 768), is_leaf=True)  # primals_129
    buf129 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf129, (768,), is_leaf=True)  # primals_130
    buf130 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf130, (768,), is_leaf=True)  # primals_131
    buf131 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf131, (768,), is_leaf=True)  # primals_132
    buf132 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf132, (1536, 768), is_leaf=True)  # primals_133
    buf133 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf133, (1536,), is_leaf=True)  # primals_134
    buf134 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf134, (768, 1536), is_leaf=True)  # primals_135
    buf135 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf135, (768,), is_leaf=True)  # primals_136
    buf136 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf136, (768,), is_leaf=True)  # primals_137
    buf137 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf137, (768,), is_leaf=True)  # primals_138
    buf138 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf138, (2304,), is_leaf=True)  # primals_139
    buf139 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf139, (2304, 768), is_leaf=True)  # primals_140
    buf140 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf140, (768, 768), is_leaf=True)  # primals_141
    buf141 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf141, (768,), is_leaf=True)  # primals_142
    buf142 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf142, (768,), is_leaf=True)  # primals_143
    buf143 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf143, (768,), is_leaf=True)  # primals_144
    buf144 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf144, (1536, 768), is_leaf=True)  # primals_145
    buf145 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf145, (1536,), is_leaf=True)  # primals_146
    buf146 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf146, (768, 1536), is_leaf=True)  # primals_147
    buf147 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf147, (768,), is_leaf=True)  # primals_148
    buf148 = reader.storage(None, 393216, device=device(type='cuda', index=0))
    reader.tensor(buf148, (128, 768), is_leaf=True)  # primals_149
    buf149 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf149, (128,), is_leaf=True)  # primals_150
    buf150 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf150, (16, 128), is_leaf=True)  # primals_151
    buf151 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf151, (16,), is_leaf=True)  # primals_152
    # fwd_rng_state_0 was unsupported type for dumping: <class 'torch._C.Generator'>
    # fwd_rng_state_1 was unsupported type for dumping: <class 'torch._C.Generator'>
    # fwd_rng_state_2 was unsupported type for dumping: <class 'torch._C.Generator'>
    # fwd_rng_state_3 was unsupported type for dumping: <class 'torch._C.Generator'>
    # fwd_rng_state_4 was unsupported type for dumping: <class 'torch._C.Generator'>
    # fwd_rng_state_5 was unsupported type for dumping: <class 'torch._C.Generator'>
    # fwd_rng_state_6 was unsupported type for dumping: <class 'torch._C.Generator'>
    # fwd_rng_state_7 was unsupported type for dumping: <class 'torch._C.Generator'>
    # fwd_rng_state_8 was unsupported type for dumping: <class 'torch._C.Generator'>
    # fwd_rng_state_9 was unsupported type for dumping: <class 'torch._C.Generator'>
    # fwd_rng_state_10 was unsupported type for dumping: <class 'torch._C.Generator'>
    # fwd_rng_state_11 was unsupported type for dumping: <class 'torch._C.Generator'>
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)