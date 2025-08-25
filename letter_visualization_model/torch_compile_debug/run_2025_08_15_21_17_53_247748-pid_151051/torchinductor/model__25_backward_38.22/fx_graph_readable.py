class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(s77)", mul_15: "Sym(1024*s77)", mul_127: "Sym(8*s77)", primals_4: "f32[768]", primals_5: "f32[1024, 768]", primals_6: "f32[768]", primals_7: "f32[768]", primals_8: "f32[2304]", primals_9: "f32[2304, 768]", primals_10: "f32[768, 768]", primals_12: "f32[768]", primals_13: "f32[768]", primals_14: "f32[1536, 768]", primals_15: "f32[1536]", primals_16: "f32[768, 1536]", primals_18: "f32[768]", primals_19: "f32[768]", primals_20: "f32[2304]", primals_21: "f32[2304, 768]", primals_22: "f32[768, 768]", primals_24: "f32[768]", primals_25: "f32[768]", primals_26: "f32[1536, 768]", primals_27: "f32[1536]", primals_28: "f32[768, 1536]", primals_30: "f32[768]", primals_31: "f32[768]", primals_32: "f32[2304]", primals_33: "f32[2304, 768]", primals_34: "f32[768, 768]", primals_36: "f32[768]", primals_37: "f32[768]", primals_38: "f32[1536, 768]", primals_39: "f32[1536]", primals_40: "f32[768, 1536]", primals_42: "f32[768]", primals_43: "f32[768]", primals_44: "f32[2304]", primals_45: "f32[2304, 768]", primals_46: "f32[768, 768]", primals_48: "f32[768]", primals_49: "f32[768]", primals_50: "f32[1536, 768]", primals_51: "f32[1536]", primals_52: "f32[768, 1536]", primals_54: "f32[768]", primals_55: "f32[768]", primals_56: "f32[2304]", primals_57: "f32[2304, 768]", primals_58: "f32[768, 768]", primals_60: "f32[768]", primals_61: "f32[768]", primals_62: "f32[1536, 768]", primals_63: "f32[1536]", primals_64: "f32[768, 1536]", primals_66: "f32[768]", primals_67: "f32[768]", primals_68: "f32[2304]", primals_69: "f32[2304, 768]", primals_70: "f32[768, 768]", primals_72: "f32[768]", primals_73: "f32[768]", primals_74: "f32[1536, 768]", primals_75: "f32[1536]", primals_76: "f32[768, 1536]", primals_78: "f32[768]", primals_79: "f32[768]", primals_80: "f32[2304]", primals_81: "f32[2304, 768]", primals_82: "f32[768, 768]", primals_84: "f32[768]", primals_85: "f32[768]", primals_86: "f32[1536, 768]", primals_87: "f32[1536]", primals_88: "f32[768, 1536]", primals_90: "f32[768]", primals_91: "f32[768]", primals_92: "f32[2304]", primals_93: "f32[2304, 768]", primals_94: "f32[768, 768]", primals_96: "f32[768]", primals_97: "f32[768]", primals_98: "f32[1536, 768]", primals_99: "f32[1536]", primals_100: "f32[768, 1536]", primals_102: "f32[768]", primals_103: "f32[768]", primals_104: "f32[2304]", primals_105: "f32[2304, 768]", primals_106: "f32[768, 768]", primals_108: "f32[768]", primals_109: "f32[768]", primals_110: "f32[1536, 768]", primals_111: "f32[1536]", primals_112: "f32[768, 1536]", primals_114: "f32[768]", primals_115: "f32[768]", primals_116: "f32[2304]", primals_117: "f32[2304, 768]", primals_118: "f32[768, 768]", primals_120: "f32[768]", primals_121: "f32[768]", primals_122: "f32[1536, 768]", primals_123: "f32[1536]", primals_124: "f32[768, 1536]", primals_126: "f32[768]", primals_127: "f32[768]", primals_128: "f32[2304]", primals_129: "f32[2304, 768]", primals_130: "f32[768, 768]", primals_132: "f32[768]", primals_133: "f32[768]", primals_134: "f32[1536, 768]", primals_135: "f32[1536]", primals_136: "f32[768, 1536]", primals_138: "f32[768]", primals_139: "f32[768]", primals_140: "f32[2304]", primals_141: "f32[2304, 768]", primals_142: "f32[768, 768]", primals_144: "f32[768]", primals_145: "f32[768]", primals_146: "f32[1536, 768]", primals_147: "f32[1536]", primals_148: "f32[768, 1536]", view_1: "f16[1024*s77, 16]", mm: "f16[1024*s77, 768]", add_185: "f32[s77, 1024, 768]", inductor_seeds_default: "i64[12]", add_241: "f32[s77, 1024, 768]", add_393: "f32[s77, 1024, 768]", add_449: "f32[s77, 1024, 768]", add_601: "f32[s77, 1024, 768]", add_657: "f32[s77, 1024, 768]", add_809: "f32[s77, 1024, 768]", add_865: "f32[s77, 1024, 768]", add_1017: "f32[s77, 1024, 768]", add_1073: "f32[s77, 1024, 768]", add_1225: "f32[s77, 1024, 768]", add_1281: "f32[s77, 1024, 768]", add_1433: "f32[s77, 1024, 768]", add_1489: "f32[s77, 1024, 768]", add_1641: "f32[s77, 1024, 768]", add_1697: "f32[s77, 1024, 768]", add_1849: "f32[s77, 1024, 768]", add_1905: "f32[s77, 1024, 768]", add_2057: "f32[s77, 1024, 768]", add_2113: "f32[s77, 1024, 768]", add_2265: "f32[s77, 1024, 768]", add_2321: "f32[s77, 1024, 768]", add_2473: "f32[s77, 1024, 768]", view_183: "f16[1024*s77, 768]", view_185: "f16[1024*s77, 128]", full_default: "f32[s77, 1, 128, 128]", convert_element_type_296: "i64[32, 1]", clamp_max: "i64[32, 1]", convert_element_type_298: "i64[32]", clamp_max_1: "i64[32]", clamp_max_2: "f32[32]", clamp_max_3: "f32[32, 1]", permute_141: "f16[16, 128]", permute_145: "f16[128, 768]", tangents_1: "f16[s77, 1, 32, 32]", bwd_rng_state_0, bwd_rng_state_1, bwd_rng_state_2, bwd_rng_state_3, bwd_rng_state_4, bwd_rng_state_5, bwd_rng_state_6, bwd_rng_state_7, bwd_rng_state_8, bwd_rng_state_9, bwd_rng_state_10, bwd_rng_state_11):
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:179 in forward, code: x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        convert_element_type_300: "f32[s77, 1, 32, 32]" = torch.ops.prims.convert_element_type.default(tangents_1, torch.float32);  tangents_1 = None
        mul_2407: "f32[s77, 1, 32, 32]" = torch.ops.aten.mul.Tensor(convert_element_type_300, clamp_max_3);  clamp_max_3 = None
        neg: "f32[s77, 1, 32, 32]" = torch.ops.aten.neg.default(mul_2407)
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:179 in forward, code: x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        add_2667: "f32[s77, 1, 32, 32]" = torch.ops.aten.add.Tensor(convert_element_type_300, neg);  convert_element_type_300 = neg = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:179 in forward, code: x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        mul_2408: "f32[s77, 1, 32, 32]" = torch.ops.aten.mul.Tensor(mul_2407, clamp_max_2)
        neg_1: "f32[s77, 1, 32, 32]" = torch.ops.aten.neg.default(mul_2408)
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:179 in forward, code: x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        add_2668: "f32[s77, 1, 32, 32]" = torch.ops.aten.add.Tensor(mul_2407, neg_1);  mul_2407 = neg_1 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:179 in forward, code: x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        mul_2409: "f32[s77, 1, 32, 32]" = torch.ops.aten.mul.Tensor(add_2667, clamp_max_2);  clamp_max_2 = None
        neg_2: "f32[s77, 1, 32, 32]" = torch.ops.aten.neg.default(mul_2409)
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:179 in forward, code: x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        add_2669: "f32[s77, 1, 32, 32]" = torch.ops.aten.add.Tensor(add_2667, neg_2);  add_2667 = neg_2 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:179 in forward, code: x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        index_put_1: "f32[s77, 1, 128, 128]" = torch.ops.aten.index_put.default(full_default, [None, None, clamp_max, clamp_max_1], mul_2408, True);  mul_2408 = None
        index_put_2: "f32[s77, 1, 128, 128]" = torch.ops.aten.index_put.default(full_default, [None, None, clamp_max, convert_element_type_298], add_2668, True);  clamp_max = add_2668 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:179 in forward, code: x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        add_2670: "f32[s77, 1, 128, 128]" = torch.ops.aten.add.Tensor(index_put_1, index_put_2);  index_put_1 = index_put_2 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:179 in forward, code: x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        index_put_3: "f32[s77, 1, 128, 128]" = torch.ops.aten.index_put.default(full_default, [None, None, convert_element_type_296, clamp_max_1], mul_2409, True);  clamp_max_1 = mul_2409 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:179 in forward, code: x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        add_2671: "f32[s77, 1, 128, 128]" = torch.ops.aten.add.Tensor(add_2670, index_put_3);  add_2670 = index_put_3 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:179 in forward, code: x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        index_put_4: "f32[s77, 1, 128, 128]" = torch.ops.aten.index_put.default(full_default, [None, None, convert_element_type_296, convert_element_type_298], add_2669, True);  full_default = convert_element_type_296 = convert_element_type_298 = add_2669 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:179 in forward, code: x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        add_2672: "f32[s77, 1, 128, 128]" = torch.ops.aten.add.Tensor(add_2671, index_put_4);  add_2671 = index_put_4 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:179 in forward, code: x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        convert_element_type_301: "f16[s77, 1, 128, 128]" = torch.ops.prims.convert_element_type.default(add_2672, torch.float16);  add_2672 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:22 in forward, code: y = self.unfold(x)
        iota: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 4, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[1, 32]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        iota_1: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_1: "i64[4, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        add: "i64[4, 32]" = torch.ops.aten.add.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        unsqueeze_4: "i64[4, 32, 1]" = torch.ops.aten.unsqueeze.default(add, -1)
        unsqueeze_5: "i64[4, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:175 in forward, code: x = self.fold(x)
        index_1: "f16[s77, 1, 4, 32, 4, 32]" = torch.ops.aten.index.Tensor(convert_element_type_301, [None, None, unsqueeze_5, add]);  convert_element_type_301 = unsqueeze_5 = add = None
        permute_139: "f16[s77, 1, 4, 4, 32, 32]" = torch.ops.aten.permute.default(index_1, [0, 1, 2, 4, 3, 5]);  index_1 = None
        clone_26: "f16[s77, 1, 4, 4, 32, 32]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
        view_195: "f16[s77, 16, 1024]" = torch.ops.aten.view.default(clone_26, [primals_1, 16, 1024]);  clone_26 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:174 in forward, code: x = x.permute(0, 2, 1)
        permute_140: "f16[s77, 1024, 16]" = torch.ops.aten.permute.default(view_195, [0, 2, 1]);  view_195 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:165 in forward, code: x = self.progressive_projection(x)
        clone_27: "f16[s77, 1024, 16]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
        view_196: "f16[1024*s77, 16]" = torch.ops.aten.view.default(clone_27, [mul_15, 16]);  clone_27 = None
        mm_13: "f16[1024*s77, 128]" = torch.ops.aten.mm.default(view_196, permute_141);  permute_141 = None
        permute_142: "f16[16, 1024*s77]" = torch.ops.aten.permute.default(view_196, [1, 0])
        mm_14: "f16[16, 128]" = torch.ops.aten.mm.default(permute_142, view_185);  permute_142 = None
        sum_1: "f32[1, 16]" = torch.ops.aten.sum.dim_IntList(view_196, [0], True, dtype = torch.float32);  view_196 = None
        view_197: "f32[16]" = torch.ops.aten.view.default(sum_1, [16]);  sum_1 = None
        convert_element_type_307: "f32[16, 128]" = torch.ops.prims.convert_element_type.default(mm_14, torch.float32);  mm_14 = None
        convert_element_type_default_50: "f32[16]" = torch.ops.prims.convert_element_type.default(view_197, torch.float32);  view_197 = None
        view_201: "f16[s77, 1024, 128]" = torch.ops.aten.view.default(mm_13, [primals_1, 1024, 128]);  mm_13 = None
        view_202: "f16[s77, 1024, 128]" = torch.ops.aten.view.default(view_185, [primals_1, 1024, 128]);  view_185 = None
        le_1: "b8[s77, 1024, 128]" = torch.ops.aten.le.Scalar(view_202, 0);  view_202 = None
        full_default_5: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f16[s77, 1024, 128]" = torch.ops.aten.where.self(le_1, full_default_5, view_201);  le_1 = full_default_5 = view_201 = None
        view_203: "f16[1024*s77, 128]" = torch.ops.aten.view.default(where, [mul_15, 128]);  where = None
        mm_15: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_203, permute_145);  permute_145 = None
        permute_147: "f16[128, 1024*s77]" = torch.ops.aten.permute.default(view_203, [1, 0])
        mm_16: "f16[128, 768]" = torch.ops.aten.mm.default(permute_147, view_183);  permute_147 = view_183 = None
        sum_2: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_203, [0], True, dtype = torch.float32);  view_203 = None
        view_205: "f32[128]" = torch.ops.aten.view.default(sum_2, [128]);  sum_2 = None
        view_206: "f16[s77, 1024, 768]" = torch.ops.aten.view.default(mm_15, [primals_1, 1024, 768]);  mm_15 = None
        convert_element_type_314: "f32[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_206, torch.float32);  view_206 = None
        convert_element_type_315: "f32[128, 768]" = torch.ops.prims.convert_element_type.default(mm_16, torch.float32);  mm_16 = None
        convert_element_type_default_49: "f32[128]" = torch.ops.prims.convert_element_type.default(view_205, torch.float32);  view_205 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        convert_element_type_317: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_314, torch.float16)
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        convert_element_type_default_52: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_51: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_52, 0.2);  convert_element_type_default_52 = None
        convert_element_type_318: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_51, torch.float16);  gt_51 = None
        mul_2410: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_318, 1.25);  convert_element_type_318 = None
        mul_2411: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_317, mul_2410);  convert_element_type_317 = mul_2410 = None
        view_207: "f16[1024*s77, 768]" = torch.ops.aten.view.default(mul_2411, [mul_15, 768]);  mul_2411 = None
        convert_element_type_277: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_148, torch.float16);  primals_148 = None
        permute_134: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_277, [1, 0]);  convert_element_type_277 = None
        permute_150: "f16[768, 1536]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
        mm_17: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_207, permute_150);  permute_150 = None
        permute_151: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_207, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_23 = torch.ops.aten.var_mean.correction(add_2473, [2], correction = 0, keepdim = True)
        getitem_154: "f32[s77, 1024, 1]" = var_mean_23[0]
        getitem_155: "f32[s77, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
        add_2478: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
        rsqrt_23: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_2478);  add_2478 = None
        sub_618: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_2473, getitem_155);  add_2473 = getitem_155 = None
        mul_2280: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_618, rsqrt_23);  sub_618 = None
        mul_2281: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2280, primals_144)
        add_2479: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_2281, primals_145);  mul_2281 = primals_145 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_268: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_147, torch.float16);  primals_147 = None
        convert_element_type_269: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_146, torch.float16);  primals_146 = None
        convert_element_type_270: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2479, torch.float16);  add_2479 = None
        view_179: "f16[1024*s77, 768]" = torch.ops.aten.view.default(convert_element_type_270, [mul_15, 768]);  convert_element_type_270 = None
        permute_133: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_269, [1, 0]);  convert_element_type_269 = None
        addmm_34: "f16[1024*s77, 1536]" = torch.ops.aten.addmm.default(convert_element_type_268, view_179, permute_133);  convert_element_type_268 = None
        view_180: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(addmm_34, [primals_1, 1024, 1536]);  addmm_34 = None
        convert_element_type_274: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_180, torch.float32);  view_180 = None
        mul_2301: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_274, 0.5)
        mul_2302: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_274, 0.7071067811865476)
        erf_11: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_2302);  mul_2302 = None
        add_2506: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_2303: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_2301, add_2506);  mul_2301 = None
        convert_element_type_275: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_2303, torch.float16);  mul_2303 = None
        view_181: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_275, [mul_15, 1536]);  convert_element_type_275 = None
        mm_18: "f16[768, 1536]" = torch.ops.aten.mm.default(permute_151, view_181);  permute_151 = view_181 = None
        sum_3: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_207, [0], True, dtype = torch.float32);  view_207 = None
        view_208: "f32[768]" = torch.ops.aten.view.default(sum_3, [768]);  sum_3 = None
        view_209: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(mm_17, [primals_1, 1024, 1536]);  mm_17 = None
        convert_element_type_324: "f32[768, 1536]" = torch.ops.prims.convert_element_type.default(mm_18, torch.float32);  mm_18 = None
        convert_element_type_default_48: "f32[768]" = torch.ops.prims.convert_element_type.default(view_208, torch.float32);  view_208 = None
        convert_element_type_326: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_209, torch.float32);  view_209 = None
        mul_2413: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(add_2506, 0.5);  add_2506 = None
        mul_2414: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_274, convert_element_type_274)
        mul_2415: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_2414, -0.5);  mul_2414 = None
        exp: "f32[s77, 1024, 1536]" = torch.ops.aten.exp.default(mul_2415);  mul_2415 = None
        mul_2416: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_2417: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_274, mul_2416);  convert_element_type_274 = mul_2416 = None
        add_2676: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(mul_2413, mul_2417);  mul_2413 = mul_2417 = None
        mul_2418: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_326, add_2676);  convert_element_type_326 = add_2676 = None
        convert_element_type_328: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_2418, torch.float16);  mul_2418 = None
        view_210: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_328, [mul_15, 1536]);  convert_element_type_328 = None
        permute_154: "f16[1536, 768]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        mm_19: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_210, permute_154);  permute_154 = None
        permute_155: "f16[1536, 1024*s77]" = torch.ops.aten.permute.default(view_210, [1, 0])
        mm_20: "f16[1536, 768]" = torch.ops.aten.mm.default(permute_155, view_179);  permute_155 = view_179 = None
        sum_4: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_210, [0], True, dtype = torch.float32);  view_210 = None
        view_211: "f32[1536]" = torch.ops.aten.view.default(sum_4, [1536]);  sum_4 = None
        view_212: "f16[s77, 1024, 768]" = torch.ops.aten.view.default(mm_19, [primals_1, 1024, 768]);  mm_19 = None
        convert_element_type_334: "f32[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_212, torch.float32);  view_212 = None
        convert_element_type_335: "f32[1536, 768]" = torch.ops.prims.convert_element_type.default(mm_20, torch.float32);  mm_20 = None
        convert_element_type_default_47: "f32[1536]" = torch.ops.prims.convert_element_type.default(view_211, torch.float32);  view_211 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        mul_2420: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_334, primals_144);  primals_144 = None
        mul_2421: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2420, 768)
        sum_5: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2420, [2], True)
        mul_2422: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2420, mul_2280);  mul_2420 = None
        sum_6: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2422, [2], True);  mul_2422 = None
        mul_2423: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2280, sum_6);  sum_6 = None
        sub_669: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2421, sum_5);  mul_2421 = sum_5 = None
        sub_670: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_669, mul_2423);  sub_669 = mul_2423 = None
        div: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
        mul_2424: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div, sub_670);  div = sub_670 = None
        mul_2425: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_334, mul_2280);  mul_2280 = None
        sum_7: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2425, [0, 1]);  mul_2425 = None
        sum_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_334, [0, 1]);  convert_element_type_334 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_2677: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(convert_element_type_314, mul_2424);  convert_element_type_314 = mul_2424 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        convert_element_type_337: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2677, torch.float16)
        permute_158: "f16[1024, s77, 768]" = torch.ops.aten.permute.default(convert_element_type_337, [1, 0, 2]);  convert_element_type_337 = None
        clone_30: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
        view_213: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_30, [mul_15, 768]);  clone_30 = None
        convert_element_type_264: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_142, torch.float16);  primals_142 = None
        permute_131: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_264, [1, 0]);  convert_element_type_264 = None
        permute_159: "f16[768, 768]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        mm_21: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_213, permute_159);  permute_159 = None
        permute_160: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_213, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_2321, [2], correction = 0, keepdim = True)
        getitem_143: "f32[s77, 1024, 1]" = var_mean_22[0]
        getitem_144: "f32[s77, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
        add_2326: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_143, 1e-05);  getitem_143 = None
        rsqrt_22: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_2326);  add_2326 = None
        sub_581: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_2321, getitem_144);  add_2321 = getitem_144 = None
        mul_2136: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_581, rsqrt_22);  sub_581 = None
        mul_2137: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2136, primals_138)
        add_2327: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_2137, primals_139);  mul_2137 = primals_139 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_124: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_2327, [1, 0, 2]);  add_2327 = None
        convert_element_type_258: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_140, torch.float16);  primals_140 = None
        convert_element_type_259: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_141, torch.float16);  primals_141 = None
        convert_element_type_260: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_124, torch.float16);  permute_124 = None
        permute_125: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_259, [1, 0]);  convert_element_type_259 = None
        clone_24: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_260, memory_format = torch.contiguous_format);  convert_element_type_260 = None
        view_168: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_24, [mul_15, 768]);  clone_24 = None
        mm_12: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_168, permute_125)
        view_169: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(mm_12, [1024, primals_1, 2304]);  mm_12 = None
        add_2360: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_169, convert_element_type_258);  view_169 = convert_element_type_258 = None
        view_170: "f16[1024, s77, 3, 768]" = torch.ops.aten.view.default(add_2360, [1024, primals_1, 3, 768]);  add_2360 = None
        unsqueeze_17: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_170, 0);  view_170 = None
        permute_126: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_17, [3, 1, 2, 0, 4]);  unsqueeze_17 = None
        squeeze_11: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_126, -2);  permute_126 = None
        clone_25: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_11, memory_format = torch.contiguous_format);  squeeze_11 = None
        select_33: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_25, 0, 0)
        select_34: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_25, 0, 1)
        select_35: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_25, 0, 2);  clone_25 = None
        view_171: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_33, [1024, mul_127, 96]);  select_33 = None
        permute_127: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_171, [1, 0, 2]);  view_171 = None
        view_172: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_34, [1024, mul_127, 96]);  select_34 = None
        permute_128: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_172, [1, 0, 2]);  view_172 = None
        view_173: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_35, [1024, mul_127, 96]);  select_35 = None
        permute_129: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_173, [1, 0, 2]);  view_173 = None
        view_174: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_127, [primals_1, 8, 1024, 96]);  permute_127 = None
        view_175: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_128, [primals_1, 8, 1024, 96]);  permute_128 = None
        view_176: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_129, [primals_1, 8, 1024, 96]);  permute_129 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_11 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_174, view_175, view_176, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_11);  bwd_rng_state_11 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_145: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_11[0]
        getitem_146: "f32[s77, 8, 1024]" = graphsafe_run_with_rng_state_11[1]
        getitem_151: "u64[2]" = graphsafe_run_with_rng_state_11[6]
        getitem_152: "u64[]" = graphsafe_run_with_rng_state_11[7];  graphsafe_run_with_rng_state_11 = None
        permute_130: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_145, [2, 0, 1, 3])
        view_177: "f16[1024*s77, 768]" = torch.ops.aten.view.default(permute_130, [mul_15, 768]);  permute_130 = None
        mm_22: "f16[768, 768]" = torch.ops.aten.mm.default(permute_160, view_177);  permute_160 = view_177 = None
        sum_9: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_213, [0], True, dtype = torch.float32);  view_213 = None
        view_214: "f32[768]" = torch.ops.aten.view.default(sum_9, [768]);  sum_9 = None
        convert_element_type_343: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_22, torch.float32);  mm_22 = None
        convert_element_type_default_46: "f32[768]" = torch.ops.prims.convert_element_type.default(view_214, torch.float32);  view_214 = None
        view_215: "f16[1024, s77, 8, 96]" = torch.ops.aten.view.default(mm_21, [1024, primals_1, 8, 96]);  mm_21 = None
        permute_163: "f16[s77, 8, 1024, 96]" = torch.ops.aten.permute.default(view_215, [1, 2, 0, 3]);  view_215 = None
        _scaled_dot_product_flash_attention_backward = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_163, view_174, view_175, view_176, getitem_145, getitem_146, None, None, 1024, 1024, 0.2, False, getitem_151, getitem_152, scale = 0.10206207261596577);  permute_163 = view_174 = view_175 = view_176 = getitem_145 = getitem_146 = getitem_151 = getitem_152 = None
        getitem_156: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward[0]
        getitem_157: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward[1]
        getitem_158: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward[2];  _scaled_dot_product_flash_attention_backward = None
        view_216: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_158, [mul_127, 1024, 96]);  getitem_158 = None
        view_217: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_157, [mul_127, 1024, 96]);  getitem_157 = None
        view_218: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_156, [mul_127, 1024, 96]);  getitem_156 = None
        permute_164: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_216, [1, 0, 2]);  view_216 = None
        view_219: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_164, [1024, primals_1, 768]);  permute_164 = None
        permute_165: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_217, [1, 0, 2]);  view_217 = None
        view_220: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_165, [1024, primals_1, 768]);  permute_165 = None
        permute_166: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_218, [1, 0, 2]);  view_218 = None
        view_221: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_166, [1024, primals_1, 768]);  permute_166 = None
        full_default_6: "f16[3, 1024, s77, 768]" = torch.ops.aten.full.default([3, 1024, primals_1, 768], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_219, 0, 2);  view_219 = None
        select_scatter_1: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_220, 0, 1);  view_220 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2678: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(select_scatter, select_scatter_1);  select_scatter = select_scatter_1 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        select_scatter_2: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_221, 0, 0);  view_221 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2679: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(add_2678, select_scatter_2);  add_2678 = select_scatter_2 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        unsqueeze_30: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.unsqueeze.default(add_2679, 3);  add_2679 = None
        permute_167: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_30, [3, 1, 2, 0, 4]);  unsqueeze_30 = None
        squeeze_12: "f16[1024, s77, 3, 768]" = torch.ops.aten.squeeze.dim(permute_167, 0);  permute_167 = None
        clone_31: "f16[1024, s77, 3, 768]" = torch.ops.aten.clone.default(squeeze_12, memory_format = torch.contiguous_format);  squeeze_12 = None
        view_222: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(clone_31, [1024, primals_1, 2304]);  clone_31 = None
        sum_10: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_222, [0, 1], True, dtype = torch.float32)
        view_223: "f32[2304]" = torch.ops.aten.view.default(sum_10, [2304]);  sum_10 = None
        view_224: "f16[1024*s77, 2304]" = torch.ops.aten.view.default(view_222, [mul_15, 2304]);  view_222 = None
        permute_168: "f16[2304, 1024*s77]" = torch.ops.aten.permute.default(view_224, [1, 0])
        mm_23: "f16[2304, 768]" = torch.ops.aten.mm.default(permute_168, view_168);  permute_168 = view_168 = None
        permute_170: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
        mm_24: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_224, permute_170);  view_224 = permute_170 = None
        view_225: "f16[1024, s77, 768]" = torch.ops.aten.view.default(mm_24, [1024, primals_1, 768]);  mm_24 = None
        convert_element_type_350: "f32[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(view_225, torch.float32);  view_225 = None
        convert_element_type_351: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_default_45: "f32[2304]" = torch.ops.prims.convert_element_type.default(view_223, torch.float32);  view_223 = None
        permute_172: "f32[s77, 1024, 768]" = torch.ops.aten.permute.default(convert_element_type_350, [1, 0, 2]);  convert_element_type_350 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        mul_2427: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_172, primals_138);  primals_138 = None
        mul_2428: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2427, 768)
        sum_11: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2427, [2], True)
        mul_2429: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2427, mul_2136);  mul_2427 = None
        sum_12: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2429, [2], True);  mul_2429 = None
        mul_2430: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2136, sum_12);  sum_12 = None
        sub_672: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2428, sum_11);  mul_2428 = sum_11 = None
        sub_673: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_672, mul_2430);  sub_672 = mul_2430 = None
        div_1: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
        mul_2431: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_1, sub_673);  div_1 = sub_673 = None
        mul_2432: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_172, mul_2136);  mul_2136 = None
        sum_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2432, [0, 1]);  mul_2432 = None
        sum_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_172, [0, 1]);  permute_172 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        add_2680: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2677, mul_2431);  add_2677 = mul_2431 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        convert_element_type_353: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2680, torch.float16)
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_1: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        convert_element_type_default_53: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.float16);  inductor_random_default_1 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_47: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_53, 0.2);  convert_element_type_default_53 = None
        convert_element_type_354: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_47, torch.float16);  gt_47 = None
        mul_2433: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_354, 1.25);  convert_element_type_354 = None
        mul_2434: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_353, mul_2433);  convert_element_type_353 = mul_2433 = None
        view_226: "f16[1024*s77, 768]" = torch.ops.aten.view.default(mul_2434, [mul_15, 768]);  mul_2434 = None
        convert_element_type_254: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_136, torch.float16);  primals_136 = None
        permute_123: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_254, [1, 0]);  convert_element_type_254 = None
        permute_173: "f16[768, 1536]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        mm_25: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_226, permute_173);  permute_173 = None
        permute_174: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_226, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_21 = torch.ops.aten.var_mean.correction(add_2265, [2], correction = 0, keepdim = True)
        getitem_141: "f32[s77, 1024, 1]" = var_mean_21[0]
        getitem_142: "f32[s77, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
        add_2270: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_141, 1e-05);  getitem_141 = None
        rsqrt_21: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_2270);  add_2270 = None
        sub_566: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_2265, getitem_142);  add_2265 = getitem_142 = None
        mul_2090: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_566, rsqrt_21);  sub_566 = None
        mul_2091: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2090, primals_132)
        add_2271: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_2091, primals_133);  mul_2091 = primals_133 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_245: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_135, torch.float16);  primals_135 = None
        convert_element_type_246: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_134, torch.float16);  primals_134 = None
        convert_element_type_247: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2271, torch.float16);  add_2271 = None
        view_164: "f16[1024*s77, 768]" = torch.ops.aten.view.default(convert_element_type_247, [mul_15, 768]);  convert_element_type_247 = None
        permute_122: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_246, [1, 0]);  convert_element_type_246 = None
        addmm_31: "f16[1024*s77, 1536]" = torch.ops.aten.addmm.default(convert_element_type_245, view_164, permute_122);  convert_element_type_245 = None
        view_165: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(addmm_31, [primals_1, 1024, 1536]);  addmm_31 = None
        convert_element_type_251: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_165, torch.float32);  view_165 = None
        mul_2111: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_251, 0.5)
        mul_2112: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_251, 0.7071067811865476)
        erf_10: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_2112);  mul_2112 = None
        add_2298: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_2113: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_2111, add_2298);  mul_2111 = None
        convert_element_type_252: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_2113, torch.float16);  mul_2113 = None
        view_166: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_252, [mul_15, 1536]);  convert_element_type_252 = None
        mm_26: "f16[768, 1536]" = torch.ops.aten.mm.default(permute_174, view_166);  permute_174 = view_166 = None
        sum_15: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_226, [0], True, dtype = torch.float32);  view_226 = None
        view_227: "f32[768]" = torch.ops.aten.view.default(sum_15, [768]);  sum_15 = None
        view_228: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(mm_25, [primals_1, 1024, 1536]);  mm_25 = None
        convert_element_type_360: "f32[768, 1536]" = torch.ops.prims.convert_element_type.default(mm_26, torch.float32);  mm_26 = None
        convert_element_type_default_44: "f32[768]" = torch.ops.prims.convert_element_type.default(view_227, torch.float32);  view_227 = None
        convert_element_type_362: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_228, torch.float32);  view_228 = None
        mul_2436: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(add_2298, 0.5);  add_2298 = None
        mul_2437: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_251, convert_element_type_251)
        mul_2438: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_2437, -0.5);  mul_2437 = None
        exp_1: "f32[s77, 1024, 1536]" = torch.ops.aten.exp.default(mul_2438);  mul_2438 = None
        mul_2439: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_2440: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_251, mul_2439);  convert_element_type_251 = mul_2439 = None
        add_2682: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(mul_2436, mul_2440);  mul_2436 = mul_2440 = None
        mul_2441: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_362, add_2682);  convert_element_type_362 = add_2682 = None
        convert_element_type_364: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_2441, torch.float16);  mul_2441 = None
        view_229: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_364, [mul_15, 1536]);  convert_element_type_364 = None
        permute_177: "f16[1536, 768]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
        mm_27: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_229, permute_177);  permute_177 = None
        permute_178: "f16[1536, 1024*s77]" = torch.ops.aten.permute.default(view_229, [1, 0])
        mm_28: "f16[1536, 768]" = torch.ops.aten.mm.default(permute_178, view_164);  permute_178 = view_164 = None
        sum_16: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_229, [0], True, dtype = torch.float32);  view_229 = None
        view_230: "f32[1536]" = torch.ops.aten.view.default(sum_16, [1536]);  sum_16 = None
        view_231: "f16[s77, 1024, 768]" = torch.ops.aten.view.default(mm_27, [primals_1, 1024, 768]);  mm_27 = None
        convert_element_type_370: "f32[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_231, torch.float32);  view_231 = None
        convert_element_type_371: "f32[1536, 768]" = torch.ops.prims.convert_element_type.default(mm_28, torch.float32);  mm_28 = None
        convert_element_type_default_43: "f32[1536]" = torch.ops.prims.convert_element_type.default(view_230, torch.float32);  view_230 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        mul_2443: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_370, primals_132);  primals_132 = None
        mul_2444: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2443, 768)
        sum_17: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2443, [2], True)
        mul_2445: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2443, mul_2090);  mul_2443 = None
        sum_18: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2445, [2], True);  mul_2445 = None
        mul_2446: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2090, sum_18);  sum_18 = None
        sub_675: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2444, sum_17);  mul_2444 = sum_17 = None
        sub_676: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_675, mul_2446);  sub_675 = mul_2446 = None
        div_2: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
        mul_2447: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_676);  div_2 = sub_676 = None
        mul_2448: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_370, mul_2090);  mul_2090 = None
        sum_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2448, [0, 1]);  mul_2448 = None
        sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_370, [0, 1]);  convert_element_type_370 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_2683: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2680, mul_2447);  add_2680 = mul_2447 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        convert_element_type_373: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2683, torch.float16)
        permute_181: "f16[1024, s77, 768]" = torch.ops.aten.permute.default(convert_element_type_373, [1, 0, 2]);  convert_element_type_373 = None
        clone_33: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
        view_232: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_33, [mul_15, 768]);  clone_33 = None
        convert_element_type_241: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_130, torch.float16);  primals_130 = None
        permute_120: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_241, [1, 0]);  convert_element_type_241 = None
        permute_182: "f16[768, 768]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        mm_29: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_232, permute_182);  permute_182 = None
        permute_183: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_232, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_2113, [2], correction = 0, keepdim = True)
        getitem_130: "f32[s77, 1024, 1]" = var_mean_20[0]
        getitem_131: "f32[s77, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
        add_2118: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_20: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_2118);  add_2118 = None
        sub_529: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_2113, getitem_131);  add_2113 = getitem_131 = None
        mul_1946: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_529, rsqrt_20);  sub_529 = None
        mul_1947: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1946, primals_126)
        add_2119: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1947, primals_127);  mul_1947 = primals_127 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_113: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_2119, [1, 0, 2]);  add_2119 = None
        convert_element_type_235: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_128, torch.float16);  primals_128 = None
        convert_element_type_236: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_129, torch.float16);  primals_129 = None
        convert_element_type_237: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_113, torch.float16);  permute_113 = None
        permute_114: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_236, [1, 0]);  convert_element_type_236 = None
        clone_22: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_237, memory_format = torch.contiguous_format);  convert_element_type_237 = None
        view_153: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_22, [mul_15, 768]);  clone_22 = None
        mm_11: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_153, permute_114)
        view_154: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(mm_11, [1024, primals_1, 2304]);  mm_11 = None
        add_2152: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_154, convert_element_type_235);  view_154 = convert_element_type_235 = None
        view_155: "f16[1024, s77, 3, 768]" = torch.ops.aten.view.default(add_2152, [1024, primals_1, 3, 768]);  add_2152 = None
        unsqueeze_16: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_155, 0);  view_155 = None
        permute_115: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_16, [3, 1, 2, 0, 4]);  unsqueeze_16 = None
        squeeze_10: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_115, -2);  permute_115 = None
        clone_23: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_10, memory_format = torch.contiguous_format);  squeeze_10 = None
        select_30: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_23, 0, 0)
        select_31: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_23, 0, 1)
        select_32: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_23, 0, 2);  clone_23 = None
        view_156: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_30, [1024, mul_127, 96]);  select_30 = None
        permute_116: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_156, [1, 0, 2]);  view_156 = None
        view_157: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_31, [1024, mul_127, 96]);  select_31 = None
        permute_117: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_157, [1, 0, 2]);  view_157 = None
        view_158: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_32, [1024, mul_127, 96]);  select_32 = None
        permute_118: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_158, [1, 0, 2]);  view_158 = None
        view_159: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_116, [primals_1, 8, 1024, 96]);  permute_116 = None
        view_160: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_117, [primals_1, 8, 1024, 96]);  permute_117 = None
        view_161: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_118, [primals_1, 8, 1024, 96]);  permute_118 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_10 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_159, view_160, view_161, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_10);  bwd_rng_state_10 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_132: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_10[0]
        getitem_133: "f32[s77, 8, 1024]" = graphsafe_run_with_rng_state_10[1]
        getitem_138: "u64[2]" = graphsafe_run_with_rng_state_10[6]
        getitem_139: "u64[]" = graphsafe_run_with_rng_state_10[7];  graphsafe_run_with_rng_state_10 = None
        permute_119: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_132, [2, 0, 1, 3])
        view_162: "f16[1024*s77, 768]" = torch.ops.aten.view.default(permute_119, [mul_15, 768]);  permute_119 = None
        mm_30: "f16[768, 768]" = torch.ops.aten.mm.default(permute_183, view_162);  permute_183 = view_162 = None
        sum_21: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_232, [0], True, dtype = torch.float32);  view_232 = None
        view_233: "f32[768]" = torch.ops.aten.view.default(sum_21, [768]);  sum_21 = None
        convert_element_type_379: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_30, torch.float32);  mm_30 = None
        convert_element_type_default_42: "f32[768]" = torch.ops.prims.convert_element_type.default(view_233, torch.float32);  view_233 = None
        view_234: "f16[1024, s77, 8, 96]" = torch.ops.aten.view.default(mm_29, [1024, primals_1, 8, 96]);  mm_29 = None
        permute_186: "f16[s77, 8, 1024, 96]" = torch.ops.aten.permute.default(view_234, [1, 2, 0, 3]);  view_234 = None
        _scaled_dot_product_flash_attention_backward_1 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_186, view_159, view_160, view_161, getitem_132, getitem_133, None, None, 1024, 1024, 0.2, False, getitem_138, getitem_139, scale = 0.10206207261596577);  permute_186 = view_159 = view_160 = view_161 = getitem_132 = getitem_133 = getitem_138 = getitem_139 = None
        getitem_159: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_1[0]
        getitem_160: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_1[1]
        getitem_161: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_1[2];  _scaled_dot_product_flash_attention_backward_1 = None
        view_235: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_161, [mul_127, 1024, 96]);  getitem_161 = None
        view_236: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_160, [mul_127, 1024, 96]);  getitem_160 = None
        view_237: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_159, [mul_127, 1024, 96]);  getitem_159 = None
        permute_187: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_235, [1, 0, 2]);  view_235 = None
        view_238: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_187, [1024, primals_1, 768]);  permute_187 = None
        permute_188: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_236, [1, 0, 2]);  view_236 = None
        view_239: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_188, [1024, primals_1, 768]);  permute_188 = None
        permute_189: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_237, [1, 0, 2]);  view_237 = None
        view_240: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_189, [1024, primals_1, 768]);  permute_189 = None
        select_scatter_3: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_238, 0, 2);  view_238 = None
        select_scatter_4: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_239, 0, 1);  view_239 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2684: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(select_scatter_3, select_scatter_4);  select_scatter_3 = select_scatter_4 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        select_scatter_5: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_240, 0, 0);  view_240 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2685: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(add_2684, select_scatter_5);  add_2684 = select_scatter_5 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        unsqueeze_31: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.unsqueeze.default(add_2685, 3);  add_2685 = None
        permute_190: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_31, [3, 1, 2, 0, 4]);  unsqueeze_31 = None
        squeeze_13: "f16[1024, s77, 3, 768]" = torch.ops.aten.squeeze.dim(permute_190, 0);  permute_190 = None
        clone_34: "f16[1024, s77, 3, 768]" = torch.ops.aten.clone.default(squeeze_13, memory_format = torch.contiguous_format);  squeeze_13 = None
        view_241: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(clone_34, [1024, primals_1, 2304]);  clone_34 = None
        sum_22: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_241, [0, 1], True, dtype = torch.float32)
        view_242: "f32[2304]" = torch.ops.aten.view.default(sum_22, [2304]);  sum_22 = None
        view_243: "f16[1024*s77, 2304]" = torch.ops.aten.view.default(view_241, [mul_15, 2304]);  view_241 = None
        permute_191: "f16[2304, 1024*s77]" = torch.ops.aten.permute.default(view_243, [1, 0])
        mm_31: "f16[2304, 768]" = torch.ops.aten.mm.default(permute_191, view_153);  permute_191 = view_153 = None
        permute_193: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
        mm_32: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_243, permute_193);  view_243 = permute_193 = None
        view_244: "f16[1024, s77, 768]" = torch.ops.aten.view.default(mm_32, [1024, primals_1, 768]);  mm_32 = None
        convert_element_type_386: "f32[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(view_244, torch.float32);  view_244 = None
        convert_element_type_387: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_default_41: "f32[2304]" = torch.ops.prims.convert_element_type.default(view_242, torch.float32);  view_242 = None
        permute_195: "f32[s77, 1024, 768]" = torch.ops.aten.permute.default(convert_element_type_386, [1, 0, 2]);  convert_element_type_386 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        mul_2450: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_195, primals_126);  primals_126 = None
        mul_2451: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2450, 768)
        sum_23: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2450, [2], True)
        mul_2452: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2450, mul_1946);  mul_2450 = None
        sum_24: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2452, [2], True);  mul_2452 = None
        mul_2453: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1946, sum_24);  sum_24 = None
        sub_678: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2451, sum_23);  mul_2451 = sum_23 = None
        sub_679: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_678, mul_2453);  sub_678 = mul_2453 = None
        div_3: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
        mul_2454: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_3, sub_679);  div_3 = sub_679 = None
        mul_2455: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_195, mul_1946);  mul_1946 = None
        sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2455, [0, 1]);  mul_2455 = None
        sum_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_195, [0, 1]);  permute_195 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        add_2686: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2683, mul_2454);  add_2683 = mul_2454 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        convert_element_type_389: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2686, torch.float16)
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_2: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        convert_element_type_default_54: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_2, torch.float16);  inductor_random_default_2 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_43: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_54, 0.2);  convert_element_type_default_54 = None
        convert_element_type_390: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_43, torch.float16);  gt_43 = None
        mul_2456: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_390, 1.25);  convert_element_type_390 = None
        mul_2457: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_389, mul_2456);  convert_element_type_389 = mul_2456 = None
        view_245: "f16[1024*s77, 768]" = torch.ops.aten.view.default(mul_2457, [mul_15, 768]);  mul_2457 = None
        convert_element_type_231: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_124, torch.float16);  primals_124 = None
        permute_112: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_231, [1, 0]);  convert_element_type_231 = None
        permute_196: "f16[768, 1536]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        mm_33: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_245, permute_196);  permute_196 = None
        permute_197: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_245, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_19 = torch.ops.aten.var_mean.correction(add_2057, [2], correction = 0, keepdim = True)
        getitem_128: "f32[s77, 1024, 1]" = var_mean_19[0]
        getitem_129: "f32[s77, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
        add_2062: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_19: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_2062);  add_2062 = None
        sub_514: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_2057, getitem_129);  add_2057 = getitem_129 = None
        mul_1900: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_514, rsqrt_19);  sub_514 = None
        mul_1901: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1900, primals_120)
        add_2063: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1901, primals_121);  mul_1901 = primals_121 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_222: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_123, torch.float16);  primals_123 = None
        convert_element_type_223: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_122, torch.float16);  primals_122 = None
        convert_element_type_224: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2063, torch.float16);  add_2063 = None
        view_149: "f16[1024*s77, 768]" = torch.ops.aten.view.default(convert_element_type_224, [mul_15, 768]);  convert_element_type_224 = None
        permute_111: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_223, [1, 0]);  convert_element_type_223 = None
        addmm_28: "f16[1024*s77, 1536]" = torch.ops.aten.addmm.default(convert_element_type_222, view_149, permute_111);  convert_element_type_222 = None
        view_150: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(addmm_28, [primals_1, 1024, 1536]);  addmm_28 = None
        convert_element_type_228: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_150, torch.float32);  view_150 = None
        mul_1921: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_228, 0.5)
        mul_1922: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_228, 0.7071067811865476)
        erf_9: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_1922);  mul_1922 = None
        add_2090: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_1923: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_1921, add_2090);  mul_1921 = None
        convert_element_type_229: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_1923, torch.float16);  mul_1923 = None
        view_151: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_229, [mul_15, 1536]);  convert_element_type_229 = None
        mm_34: "f16[768, 1536]" = torch.ops.aten.mm.default(permute_197, view_151);  permute_197 = view_151 = None
        sum_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_245, [0], True, dtype = torch.float32);  view_245 = None
        view_246: "f32[768]" = torch.ops.aten.view.default(sum_27, [768]);  sum_27 = None
        view_247: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(mm_33, [primals_1, 1024, 1536]);  mm_33 = None
        convert_element_type_396: "f32[768, 1536]" = torch.ops.prims.convert_element_type.default(mm_34, torch.float32);  mm_34 = None
        convert_element_type_default_40: "f32[768]" = torch.ops.prims.convert_element_type.default(view_246, torch.float32);  view_246 = None
        convert_element_type_398: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_247, torch.float32);  view_247 = None
        mul_2459: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(add_2090, 0.5);  add_2090 = None
        mul_2460: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_228, convert_element_type_228)
        mul_2461: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_2460, -0.5);  mul_2460 = None
        exp_2: "f32[s77, 1024, 1536]" = torch.ops.aten.exp.default(mul_2461);  mul_2461 = None
        mul_2462: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_2463: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_228, mul_2462);  convert_element_type_228 = mul_2462 = None
        add_2688: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(mul_2459, mul_2463);  mul_2459 = mul_2463 = None
        mul_2464: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_398, add_2688);  convert_element_type_398 = add_2688 = None
        convert_element_type_400: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_2464, torch.float16);  mul_2464 = None
        view_248: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_400, [mul_15, 1536]);  convert_element_type_400 = None
        permute_200: "f16[1536, 768]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        mm_35: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_248, permute_200);  permute_200 = None
        permute_201: "f16[1536, 1024*s77]" = torch.ops.aten.permute.default(view_248, [1, 0])
        mm_36: "f16[1536, 768]" = torch.ops.aten.mm.default(permute_201, view_149);  permute_201 = view_149 = None
        sum_28: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_248, [0], True, dtype = torch.float32);  view_248 = None
        view_249: "f32[1536]" = torch.ops.aten.view.default(sum_28, [1536]);  sum_28 = None
        view_250: "f16[s77, 1024, 768]" = torch.ops.aten.view.default(mm_35, [primals_1, 1024, 768]);  mm_35 = None
        convert_element_type_406: "f32[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_250, torch.float32);  view_250 = None
        convert_element_type_407: "f32[1536, 768]" = torch.ops.prims.convert_element_type.default(mm_36, torch.float32);  mm_36 = None
        convert_element_type_default_39: "f32[1536]" = torch.ops.prims.convert_element_type.default(view_249, torch.float32);  view_249 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        mul_2466: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_406, primals_120);  primals_120 = None
        mul_2467: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2466, 768)
        sum_29: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2466, [2], True)
        mul_2468: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2466, mul_1900);  mul_2466 = None
        sum_30: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2468, [2], True);  mul_2468 = None
        mul_2469: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1900, sum_30);  sum_30 = None
        sub_681: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2467, sum_29);  mul_2467 = sum_29 = None
        sub_682: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_681, mul_2469);  sub_681 = mul_2469 = None
        div_4: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
        mul_2470: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_4, sub_682);  div_4 = sub_682 = None
        mul_2471: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_406, mul_1900);  mul_1900 = None
        sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2471, [0, 1]);  mul_2471 = None
        sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_406, [0, 1]);  convert_element_type_406 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_2689: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2686, mul_2470);  add_2686 = mul_2470 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        convert_element_type_409: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2689, torch.float16)
        permute_204: "f16[1024, s77, 768]" = torch.ops.aten.permute.default(convert_element_type_409, [1, 0, 2]);  convert_element_type_409 = None
        clone_36: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_251: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_36, [mul_15, 768]);  clone_36 = None
        convert_element_type_218: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_118, torch.float16);  primals_118 = None
        permute_109: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_218, [1, 0]);  convert_element_type_218 = None
        permute_205: "f16[768, 768]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        mm_37: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_251, permute_205);  permute_205 = None
        permute_206: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_251, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_1905, [2], correction = 0, keepdim = True)
        getitem_117: "f32[s77, 1024, 1]" = var_mean_18[0]
        getitem_118: "f32[s77, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
        add_1910: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_117, 1e-05);  getitem_117 = None
        rsqrt_18: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1910);  add_1910 = None
        sub_477: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1905, getitem_118);  add_1905 = getitem_118 = None
        mul_1756: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_477, rsqrt_18);  sub_477 = None
        mul_1757: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1756, primals_114)
        add_1911: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1757, primals_115);  mul_1757 = primals_115 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_102: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_1911, [1, 0, 2]);  add_1911 = None
        convert_element_type_212: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_116, torch.float16);  primals_116 = None
        convert_element_type_213: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_117, torch.float16);  primals_117 = None
        convert_element_type_214: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_102, torch.float16);  permute_102 = None
        permute_103: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_213, [1, 0]);  convert_element_type_213 = None
        clone_20: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_214, memory_format = torch.contiguous_format);  convert_element_type_214 = None
        view_138: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_20, [mul_15, 768]);  clone_20 = None
        mm_10: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_138, permute_103)
        view_139: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(mm_10, [1024, primals_1, 2304]);  mm_10 = None
        add_1944: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_139, convert_element_type_212);  view_139 = convert_element_type_212 = None
        view_140: "f16[1024, s77, 3, 768]" = torch.ops.aten.view.default(add_1944, [1024, primals_1, 3, 768]);  add_1944 = None
        unsqueeze_15: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_140, 0);  view_140 = None
        permute_104: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_15, [3, 1, 2, 0, 4]);  unsqueeze_15 = None
        squeeze_9: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_104, -2);  permute_104 = None
        clone_21: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_9, memory_format = torch.contiguous_format);  squeeze_9 = None
        select_27: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_21, 0, 0)
        select_28: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_21, 0, 1)
        select_29: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_21, 0, 2);  clone_21 = None
        view_141: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_27, [1024, mul_127, 96]);  select_27 = None
        permute_105: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_141, [1, 0, 2]);  view_141 = None
        view_142: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_28, [1024, mul_127, 96]);  select_28 = None
        permute_106: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_142, [1, 0, 2]);  view_142 = None
        view_143: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_29, [1024, mul_127, 96]);  select_29 = None
        permute_107: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_143, [1, 0, 2]);  view_143 = None
        view_144: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_105, [primals_1, 8, 1024, 96]);  permute_105 = None
        view_145: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_106, [primals_1, 8, 1024, 96]);  permute_106 = None
        view_146: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_107, [primals_1, 8, 1024, 96]);  permute_107 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_9 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_144, view_145, view_146, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_9);  bwd_rng_state_9 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_119: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_9[0]
        getitem_120: "f32[s77, 8, 1024]" = graphsafe_run_with_rng_state_9[1]
        getitem_125: "u64[2]" = graphsafe_run_with_rng_state_9[6]
        getitem_126: "u64[]" = graphsafe_run_with_rng_state_9[7];  graphsafe_run_with_rng_state_9 = None
        permute_108: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_119, [2, 0, 1, 3])
        view_147: "f16[1024*s77, 768]" = torch.ops.aten.view.default(permute_108, [mul_15, 768]);  permute_108 = None
        mm_38: "f16[768, 768]" = torch.ops.aten.mm.default(permute_206, view_147);  permute_206 = view_147 = None
        sum_33: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_251, [0], True, dtype = torch.float32);  view_251 = None
        view_252: "f32[768]" = torch.ops.aten.view.default(sum_33, [768]);  sum_33 = None
        convert_element_type_415: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_38, torch.float32);  mm_38 = None
        convert_element_type_default_38: "f32[768]" = torch.ops.prims.convert_element_type.default(view_252, torch.float32);  view_252 = None
        view_253: "f16[1024, s77, 8, 96]" = torch.ops.aten.view.default(mm_37, [1024, primals_1, 8, 96]);  mm_37 = None
        permute_209: "f16[s77, 8, 1024, 96]" = torch.ops.aten.permute.default(view_253, [1, 2, 0, 3]);  view_253 = None
        _scaled_dot_product_flash_attention_backward_2 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_209, view_144, view_145, view_146, getitem_119, getitem_120, None, None, 1024, 1024, 0.2, False, getitem_125, getitem_126, scale = 0.10206207261596577);  permute_209 = view_144 = view_145 = view_146 = getitem_119 = getitem_120 = getitem_125 = getitem_126 = None
        getitem_162: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_2[0]
        getitem_163: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_2[1]
        getitem_164: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_2[2];  _scaled_dot_product_flash_attention_backward_2 = None
        view_254: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_164, [mul_127, 1024, 96]);  getitem_164 = None
        view_255: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_163, [mul_127, 1024, 96]);  getitem_163 = None
        view_256: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_162, [mul_127, 1024, 96]);  getitem_162 = None
        permute_210: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_254, [1, 0, 2]);  view_254 = None
        view_257: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_210, [1024, primals_1, 768]);  permute_210 = None
        permute_211: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_255, [1, 0, 2]);  view_255 = None
        view_258: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_211, [1024, primals_1, 768]);  permute_211 = None
        permute_212: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_256, [1, 0, 2]);  view_256 = None
        view_259: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_212, [1024, primals_1, 768]);  permute_212 = None
        select_scatter_6: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_257, 0, 2);  view_257 = None
        select_scatter_7: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_258, 0, 1);  view_258 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2690: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(select_scatter_6, select_scatter_7);  select_scatter_6 = select_scatter_7 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        select_scatter_8: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_259, 0, 0);  view_259 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2691: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(add_2690, select_scatter_8);  add_2690 = select_scatter_8 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        unsqueeze_32: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.unsqueeze.default(add_2691, 3);  add_2691 = None
        permute_213: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_32, [3, 1, 2, 0, 4]);  unsqueeze_32 = None
        squeeze_14: "f16[1024, s77, 3, 768]" = torch.ops.aten.squeeze.dim(permute_213, 0);  permute_213 = None
        clone_37: "f16[1024, s77, 3, 768]" = torch.ops.aten.clone.default(squeeze_14, memory_format = torch.contiguous_format);  squeeze_14 = None
        view_260: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(clone_37, [1024, primals_1, 2304]);  clone_37 = None
        sum_34: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_260, [0, 1], True, dtype = torch.float32)
        view_261: "f32[2304]" = torch.ops.aten.view.default(sum_34, [2304]);  sum_34 = None
        view_262: "f16[1024*s77, 2304]" = torch.ops.aten.view.default(view_260, [mul_15, 2304]);  view_260 = None
        permute_214: "f16[2304, 1024*s77]" = torch.ops.aten.permute.default(view_262, [1, 0])
        mm_39: "f16[2304, 768]" = torch.ops.aten.mm.default(permute_214, view_138);  permute_214 = view_138 = None
        permute_216: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
        mm_40: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_262, permute_216);  view_262 = permute_216 = None
        view_263: "f16[1024, s77, 768]" = torch.ops.aten.view.default(mm_40, [1024, primals_1, 768]);  mm_40 = None
        convert_element_type_422: "f32[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(view_263, torch.float32);  view_263 = None
        convert_element_type_423: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_default_37: "f32[2304]" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        permute_218: "f32[s77, 1024, 768]" = torch.ops.aten.permute.default(convert_element_type_422, [1, 0, 2]);  convert_element_type_422 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        mul_2473: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_218, primals_114);  primals_114 = None
        mul_2474: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2473, 768)
        sum_35: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2473, [2], True)
        mul_2475: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2473, mul_1756);  mul_2473 = None
        sum_36: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2475, [2], True);  mul_2475 = None
        mul_2476: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1756, sum_36);  sum_36 = None
        sub_684: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2474, sum_35);  mul_2474 = sum_35 = None
        sub_685: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_684, mul_2476);  sub_684 = mul_2476 = None
        div_5: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
        mul_2477: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_5, sub_685);  div_5 = sub_685 = None
        mul_2478: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_218, mul_1756);  mul_1756 = None
        sum_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2478, [0, 1]);  mul_2478 = None
        sum_38: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_218, [0, 1]);  permute_218 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        add_2692: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2689, mul_2477);  add_2689 = mul_2477 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        convert_element_type_425: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2692, torch.float16)
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_3: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        convert_element_type_default_55: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_3, torch.float16);  inductor_random_default_3 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_39: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_55, 0.2);  convert_element_type_default_55 = None
        convert_element_type_426: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_39, torch.float16);  gt_39 = None
        mul_2479: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_426, 1.25);  convert_element_type_426 = None
        mul_2480: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_425, mul_2479);  convert_element_type_425 = mul_2479 = None
        view_264: "f16[1024*s77, 768]" = torch.ops.aten.view.default(mul_2480, [mul_15, 768]);  mul_2480 = None
        convert_element_type_208: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_112, torch.float16);  primals_112 = None
        permute_101: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_208, [1, 0]);  convert_element_type_208 = None
        permute_219: "f16[768, 1536]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        mm_41: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_264, permute_219);  permute_219 = None
        permute_220: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_264, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_17 = torch.ops.aten.var_mean.correction(add_1849, [2], correction = 0, keepdim = True)
        getitem_115: "f32[s77, 1024, 1]" = var_mean_17[0]
        getitem_116: "f32[s77, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
        add_1854: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_115, 1e-05);  getitem_115 = None
        rsqrt_17: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1854);  add_1854 = None
        sub_462: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1849, getitem_116);  add_1849 = getitem_116 = None
        mul_1710: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_462, rsqrt_17);  sub_462 = None
        mul_1711: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1710, primals_108)
        add_1855: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1711, primals_109);  mul_1711 = primals_109 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_199: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_111, torch.float16);  primals_111 = None
        convert_element_type_200: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_110, torch.float16);  primals_110 = None
        convert_element_type_201: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_1855, torch.float16);  add_1855 = None
        view_134: "f16[1024*s77, 768]" = torch.ops.aten.view.default(convert_element_type_201, [mul_15, 768]);  convert_element_type_201 = None
        permute_100: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_200, [1, 0]);  convert_element_type_200 = None
        addmm_25: "f16[1024*s77, 1536]" = torch.ops.aten.addmm.default(convert_element_type_199, view_134, permute_100);  convert_element_type_199 = None
        view_135: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(addmm_25, [primals_1, 1024, 1536]);  addmm_25 = None
        convert_element_type_205: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_135, torch.float32);  view_135 = None
        mul_1731: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_205, 0.5)
        mul_1732: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_205, 0.7071067811865476)
        erf_8: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_1732);  mul_1732 = None
        add_1882: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_1733: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_1731, add_1882);  mul_1731 = None
        convert_element_type_206: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_1733, torch.float16);  mul_1733 = None
        view_136: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_206, [mul_15, 1536]);  convert_element_type_206 = None
        mm_42: "f16[768, 1536]" = torch.ops.aten.mm.default(permute_220, view_136);  permute_220 = view_136 = None
        sum_39: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_264, [0], True, dtype = torch.float32);  view_264 = None
        view_265: "f32[768]" = torch.ops.aten.view.default(sum_39, [768]);  sum_39 = None
        view_266: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(mm_41, [primals_1, 1024, 1536]);  mm_41 = None
        convert_element_type_432: "f32[768, 1536]" = torch.ops.prims.convert_element_type.default(mm_42, torch.float32);  mm_42 = None
        convert_element_type_default_36: "f32[768]" = torch.ops.prims.convert_element_type.default(view_265, torch.float32);  view_265 = None
        convert_element_type_434: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_266, torch.float32);  view_266 = None
        mul_2482: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(add_1882, 0.5);  add_1882 = None
        mul_2483: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_205, convert_element_type_205)
        mul_2484: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_2483, -0.5);  mul_2483 = None
        exp_3: "f32[s77, 1024, 1536]" = torch.ops.aten.exp.default(mul_2484);  mul_2484 = None
        mul_2485: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
        mul_2486: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_205, mul_2485);  convert_element_type_205 = mul_2485 = None
        add_2694: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(mul_2482, mul_2486);  mul_2482 = mul_2486 = None
        mul_2487: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_434, add_2694);  convert_element_type_434 = add_2694 = None
        convert_element_type_436: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_2487, torch.float16);  mul_2487 = None
        view_267: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_436, [mul_15, 1536]);  convert_element_type_436 = None
        permute_223: "f16[1536, 768]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
        mm_43: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_267, permute_223);  permute_223 = None
        permute_224: "f16[1536, 1024*s77]" = torch.ops.aten.permute.default(view_267, [1, 0])
        mm_44: "f16[1536, 768]" = torch.ops.aten.mm.default(permute_224, view_134);  permute_224 = view_134 = None
        sum_40: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_267, [0], True, dtype = torch.float32);  view_267 = None
        view_268: "f32[1536]" = torch.ops.aten.view.default(sum_40, [1536]);  sum_40 = None
        view_269: "f16[s77, 1024, 768]" = torch.ops.aten.view.default(mm_43, [primals_1, 1024, 768]);  mm_43 = None
        convert_element_type_442: "f32[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_269, torch.float32);  view_269 = None
        convert_element_type_443: "f32[1536, 768]" = torch.ops.prims.convert_element_type.default(mm_44, torch.float32);  mm_44 = None
        convert_element_type_default_35: "f32[1536]" = torch.ops.prims.convert_element_type.default(view_268, torch.float32);  view_268 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        mul_2489: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_442, primals_108);  primals_108 = None
        mul_2490: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2489, 768)
        sum_41: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2489, [2], True)
        mul_2491: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2489, mul_1710);  mul_2489 = None
        sum_42: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2491, [2], True);  mul_2491 = None
        mul_2492: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1710, sum_42);  sum_42 = None
        sub_687: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2490, sum_41);  mul_2490 = sum_41 = None
        sub_688: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_687, mul_2492);  sub_687 = mul_2492 = None
        div_6: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
        mul_2493: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_6, sub_688);  div_6 = sub_688 = None
        mul_2494: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_442, mul_1710);  mul_1710 = None
        sum_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2494, [0, 1]);  mul_2494 = None
        sum_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_442, [0, 1]);  convert_element_type_442 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_2695: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2692, mul_2493);  add_2692 = mul_2493 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        convert_element_type_445: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2695, torch.float16)
        permute_227: "f16[1024, s77, 768]" = torch.ops.aten.permute.default(convert_element_type_445, [1, 0, 2]);  convert_element_type_445 = None
        clone_39: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        view_270: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_39, [mul_15, 768]);  clone_39 = None
        convert_element_type_195: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_106, torch.float16);  primals_106 = None
        permute_98: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_195, [1, 0]);  convert_element_type_195 = None
        permute_228: "f16[768, 768]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        mm_45: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_270, permute_228);  permute_228 = None
        permute_229: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_270, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_1697, [2], correction = 0, keepdim = True)
        getitem_104: "f32[s77, 1024, 1]" = var_mean_16[0]
        getitem_105: "f32[s77, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
        add_1702: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_16: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1702);  add_1702 = None
        sub_425: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1697, getitem_105);  add_1697 = getitem_105 = None
        mul_1566: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_425, rsqrt_16);  sub_425 = None
        mul_1567: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1566, primals_102)
        add_1703: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1567, primals_103);  mul_1567 = primals_103 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_91: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_1703, [1, 0, 2]);  add_1703 = None
        convert_element_type_189: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_104, torch.float16);  primals_104 = None
        convert_element_type_190: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_105, torch.float16);  primals_105 = None
        convert_element_type_191: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_91, torch.float16);  permute_91 = None
        permute_92: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_190, [1, 0]);  convert_element_type_190 = None
        clone_18: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_191, memory_format = torch.contiguous_format);  convert_element_type_191 = None
        view_123: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_18, [mul_15, 768]);  clone_18 = None
        mm_9: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_123, permute_92)
        view_124: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(mm_9, [1024, primals_1, 2304]);  mm_9 = None
        add_1736: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_124, convert_element_type_189);  view_124 = convert_element_type_189 = None
        view_125: "f16[1024, s77, 3, 768]" = torch.ops.aten.view.default(add_1736, [1024, primals_1, 3, 768]);  add_1736 = None
        unsqueeze_14: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_125, 0);  view_125 = None
        permute_93: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_14, [3, 1, 2, 0, 4]);  unsqueeze_14 = None
        squeeze_8: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_93, -2);  permute_93 = None
        clone_19: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_8, memory_format = torch.contiguous_format);  squeeze_8 = None
        select_24: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_19, 0, 0)
        select_25: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_19, 0, 1)
        select_26: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_19, 0, 2);  clone_19 = None
        view_126: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_24, [1024, mul_127, 96]);  select_24 = None
        permute_94: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_126, [1, 0, 2]);  view_126 = None
        view_127: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_25, [1024, mul_127, 96]);  select_25 = None
        permute_95: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_127, [1, 0, 2]);  view_127 = None
        view_128: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_26, [1024, mul_127, 96]);  select_26 = None
        permute_96: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_128, [1, 0, 2]);  view_128 = None
        view_129: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_94, [primals_1, 8, 1024, 96]);  permute_94 = None
        view_130: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_95, [primals_1, 8, 1024, 96]);  permute_95 = None
        view_131: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_96, [primals_1, 8, 1024, 96]);  permute_96 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_8 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_129, view_130, view_131, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_8);  bwd_rng_state_8 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_106: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_8[0]
        getitem_107: "f32[s77, 8, 1024]" = graphsafe_run_with_rng_state_8[1]
        getitem_112: "u64[2]" = graphsafe_run_with_rng_state_8[6]
        getitem_113: "u64[]" = graphsafe_run_with_rng_state_8[7];  graphsafe_run_with_rng_state_8 = None
        permute_97: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_106, [2, 0, 1, 3])
        view_132: "f16[1024*s77, 768]" = torch.ops.aten.view.default(permute_97, [mul_15, 768]);  permute_97 = None
        mm_46: "f16[768, 768]" = torch.ops.aten.mm.default(permute_229, view_132);  permute_229 = view_132 = None
        sum_45: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_270, [0], True, dtype = torch.float32);  view_270 = None
        view_271: "f32[768]" = torch.ops.aten.view.default(sum_45, [768]);  sum_45 = None
        convert_element_type_451: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_46, torch.float32);  mm_46 = None
        convert_element_type_default_34: "f32[768]" = torch.ops.prims.convert_element_type.default(view_271, torch.float32);  view_271 = None
        view_272: "f16[1024, s77, 8, 96]" = torch.ops.aten.view.default(mm_45, [1024, primals_1, 8, 96]);  mm_45 = None
        permute_232: "f16[s77, 8, 1024, 96]" = torch.ops.aten.permute.default(view_272, [1, 2, 0, 3]);  view_272 = None
        _scaled_dot_product_flash_attention_backward_3 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_232, view_129, view_130, view_131, getitem_106, getitem_107, None, None, 1024, 1024, 0.2, False, getitem_112, getitem_113, scale = 0.10206207261596577);  permute_232 = view_129 = view_130 = view_131 = getitem_106 = getitem_107 = getitem_112 = getitem_113 = None
        getitem_165: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_3[0]
        getitem_166: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_3[1]
        getitem_167: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_3[2];  _scaled_dot_product_flash_attention_backward_3 = None
        view_273: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_167, [mul_127, 1024, 96]);  getitem_167 = None
        view_274: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_166, [mul_127, 1024, 96]);  getitem_166 = None
        view_275: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_165, [mul_127, 1024, 96]);  getitem_165 = None
        permute_233: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_273, [1, 0, 2]);  view_273 = None
        view_276: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_233, [1024, primals_1, 768]);  permute_233 = None
        permute_234: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_274, [1, 0, 2]);  view_274 = None
        view_277: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_234, [1024, primals_1, 768]);  permute_234 = None
        permute_235: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_275, [1, 0, 2]);  view_275 = None
        view_278: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_235, [1024, primals_1, 768]);  permute_235 = None
        select_scatter_9: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_276, 0, 2);  view_276 = None
        select_scatter_10: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_277, 0, 1);  view_277 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2696: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(select_scatter_9, select_scatter_10);  select_scatter_9 = select_scatter_10 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        select_scatter_11: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_278, 0, 0);  view_278 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2697: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(add_2696, select_scatter_11);  add_2696 = select_scatter_11 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        unsqueeze_33: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.unsqueeze.default(add_2697, 3);  add_2697 = None
        permute_236: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_33, [3, 1, 2, 0, 4]);  unsqueeze_33 = None
        squeeze_15: "f16[1024, s77, 3, 768]" = torch.ops.aten.squeeze.dim(permute_236, 0);  permute_236 = None
        clone_40: "f16[1024, s77, 3, 768]" = torch.ops.aten.clone.default(squeeze_15, memory_format = torch.contiguous_format);  squeeze_15 = None
        view_279: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(clone_40, [1024, primals_1, 2304]);  clone_40 = None
        sum_46: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_279, [0, 1], True, dtype = torch.float32)
        view_280: "f32[2304]" = torch.ops.aten.view.default(sum_46, [2304]);  sum_46 = None
        view_281: "f16[1024*s77, 2304]" = torch.ops.aten.view.default(view_279, [mul_15, 2304]);  view_279 = None
        permute_237: "f16[2304, 1024*s77]" = torch.ops.aten.permute.default(view_281, [1, 0])
        mm_47: "f16[2304, 768]" = torch.ops.aten.mm.default(permute_237, view_123);  permute_237 = view_123 = None
        permute_239: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
        mm_48: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_281, permute_239);  view_281 = permute_239 = None
        view_282: "f16[1024, s77, 768]" = torch.ops.aten.view.default(mm_48, [1024, primals_1, 768]);  mm_48 = None
        convert_element_type_458: "f32[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(view_282, torch.float32);  view_282 = None
        convert_element_type_459: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_default_33: "f32[2304]" = torch.ops.prims.convert_element_type.default(view_280, torch.float32);  view_280 = None
        permute_241: "f32[s77, 1024, 768]" = torch.ops.aten.permute.default(convert_element_type_458, [1, 0, 2]);  convert_element_type_458 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        mul_2496: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_241, primals_102);  primals_102 = None
        mul_2497: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2496, 768)
        sum_47: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2496, [2], True)
        mul_2498: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2496, mul_1566);  mul_2496 = None
        sum_48: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2498, [2], True);  mul_2498 = None
        mul_2499: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1566, sum_48);  sum_48 = None
        sub_690: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2497, sum_47);  mul_2497 = sum_47 = None
        sub_691: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_690, mul_2499);  sub_690 = mul_2499 = None
        div_7: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
        mul_2500: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_7, sub_691);  div_7 = sub_691 = None
        mul_2501: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_241, mul_1566);  mul_1566 = None
        sum_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2501, [0, 1]);  mul_2501 = None
        sum_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_241, [0, 1]);  permute_241 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        add_2698: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2695, mul_2500);  add_2695 = mul_2500 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        convert_element_type_461: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2698, torch.float16)
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_4: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        convert_element_type_default_56: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_4, torch.float16);  inductor_random_default_4 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_35: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_56, 0.2);  convert_element_type_default_56 = None
        convert_element_type_462: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_35, torch.float16);  gt_35 = None
        mul_2502: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_462, 1.25);  convert_element_type_462 = None
        mul_2503: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_461, mul_2502);  convert_element_type_461 = mul_2502 = None
        view_283: "f16[1024*s77, 768]" = torch.ops.aten.view.default(mul_2503, [mul_15, 768]);  mul_2503 = None
        convert_element_type_185: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_100, torch.float16);  primals_100 = None
        permute_90: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_185, [1, 0]);  convert_element_type_185 = None
        permute_242: "f16[768, 1536]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        mm_49: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_283, permute_242);  permute_242 = None
        permute_243: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_283, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_15 = torch.ops.aten.var_mean.correction(add_1641, [2], correction = 0, keepdim = True)
        getitem_102: "f32[s77, 1024, 1]" = var_mean_15[0]
        getitem_103: "f32[s77, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
        add_1646: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
        rsqrt_15: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1646);  add_1646 = None
        sub_410: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1641, getitem_103);  add_1641 = getitem_103 = None
        mul_1520: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_410, rsqrt_15);  sub_410 = None
        mul_1521: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1520, primals_96)
        add_1647: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1521, primals_97);  mul_1521 = primals_97 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_176: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_99, torch.float16);  primals_99 = None
        convert_element_type_177: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_98, torch.float16);  primals_98 = None
        convert_element_type_178: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_1647, torch.float16);  add_1647 = None
        view_119: "f16[1024*s77, 768]" = torch.ops.aten.view.default(convert_element_type_178, [mul_15, 768]);  convert_element_type_178 = None
        permute_89: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_177, [1, 0]);  convert_element_type_177 = None
        addmm_22: "f16[1024*s77, 1536]" = torch.ops.aten.addmm.default(convert_element_type_176, view_119, permute_89);  convert_element_type_176 = None
        view_120: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(addmm_22, [primals_1, 1024, 1536]);  addmm_22 = None
        convert_element_type_182: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_120, torch.float32);  view_120 = None
        mul_1541: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_182, 0.5)
        mul_1542: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_182, 0.7071067811865476)
        erf_7: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_1542);  mul_1542 = None
        add_1674: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_1543: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_1541, add_1674);  mul_1541 = None
        convert_element_type_183: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_1543, torch.float16);  mul_1543 = None
        view_121: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_183, [mul_15, 1536]);  convert_element_type_183 = None
        mm_50: "f16[768, 1536]" = torch.ops.aten.mm.default(permute_243, view_121);  permute_243 = view_121 = None
        sum_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_283, [0], True, dtype = torch.float32);  view_283 = None
        view_284: "f32[768]" = torch.ops.aten.view.default(sum_51, [768]);  sum_51 = None
        view_285: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(mm_49, [primals_1, 1024, 1536]);  mm_49 = None
        convert_element_type_468: "f32[768, 1536]" = torch.ops.prims.convert_element_type.default(mm_50, torch.float32);  mm_50 = None
        convert_element_type_default_32: "f32[768]" = torch.ops.prims.convert_element_type.default(view_284, torch.float32);  view_284 = None
        convert_element_type_470: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_285, torch.float32);  view_285 = None
        mul_2505: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(add_1674, 0.5);  add_1674 = None
        mul_2506: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_182, convert_element_type_182)
        mul_2507: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_2506, -0.5);  mul_2506 = None
        exp_4: "f32[s77, 1024, 1536]" = torch.ops.aten.exp.default(mul_2507);  mul_2507 = None
        mul_2508: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
        mul_2509: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_182, mul_2508);  convert_element_type_182 = mul_2508 = None
        add_2700: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(mul_2505, mul_2509);  mul_2505 = mul_2509 = None
        mul_2510: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_470, add_2700);  convert_element_type_470 = add_2700 = None
        convert_element_type_472: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_2510, torch.float16);  mul_2510 = None
        view_286: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_472, [mul_15, 1536]);  convert_element_type_472 = None
        permute_246: "f16[1536, 768]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        mm_51: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_286, permute_246);  permute_246 = None
        permute_247: "f16[1536, 1024*s77]" = torch.ops.aten.permute.default(view_286, [1, 0])
        mm_52: "f16[1536, 768]" = torch.ops.aten.mm.default(permute_247, view_119);  permute_247 = view_119 = None
        sum_52: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_286, [0], True, dtype = torch.float32);  view_286 = None
        view_287: "f32[1536]" = torch.ops.aten.view.default(sum_52, [1536]);  sum_52 = None
        view_288: "f16[s77, 1024, 768]" = torch.ops.aten.view.default(mm_51, [primals_1, 1024, 768]);  mm_51 = None
        convert_element_type_478: "f32[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_288, torch.float32);  view_288 = None
        convert_element_type_479: "f32[1536, 768]" = torch.ops.prims.convert_element_type.default(mm_52, torch.float32);  mm_52 = None
        convert_element_type_default_31: "f32[1536]" = torch.ops.prims.convert_element_type.default(view_287, torch.float32);  view_287 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        mul_2512: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_478, primals_96);  primals_96 = None
        mul_2513: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2512, 768)
        sum_53: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2512, [2], True)
        mul_2514: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2512, mul_1520);  mul_2512 = None
        sum_54: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2514, [2], True);  mul_2514 = None
        mul_2515: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1520, sum_54);  sum_54 = None
        sub_693: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2513, sum_53);  mul_2513 = sum_53 = None
        sub_694: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_693, mul_2515);  sub_693 = mul_2515 = None
        div_8: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
        mul_2516: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_8, sub_694);  div_8 = sub_694 = None
        mul_2517: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_478, mul_1520);  mul_1520 = None
        sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2517, [0, 1]);  mul_2517 = None
        sum_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_478, [0, 1]);  convert_element_type_478 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_2701: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2698, mul_2516);  add_2698 = mul_2516 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        convert_element_type_481: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2701, torch.float16)
        permute_250: "f16[1024, s77, 768]" = torch.ops.aten.permute.default(convert_element_type_481, [1, 0, 2]);  convert_element_type_481 = None
        clone_42: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None
        view_289: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_42, [mul_15, 768]);  clone_42 = None
        convert_element_type_172: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_94, torch.float16);  primals_94 = None
        permute_87: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_172, [1, 0]);  convert_element_type_172 = None
        permute_251: "f16[768, 768]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        mm_53: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_289, permute_251);  permute_251 = None
        permute_252: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_289, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_1489, [2], correction = 0, keepdim = True)
        getitem_91: "f32[s77, 1024, 1]" = var_mean_14[0]
        getitem_92: "f32[s77, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
        add_1494: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_91, 1e-05);  getitem_91 = None
        rsqrt_14: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1494);  add_1494 = None
        sub_373: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1489, getitem_92);  add_1489 = getitem_92 = None
        mul_1376: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_373, rsqrt_14);  sub_373 = None
        mul_1377: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1376, primals_90)
        add_1495: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1377, primals_91);  mul_1377 = primals_91 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_80: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_1495, [1, 0, 2]);  add_1495 = None
        convert_element_type_166: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_92, torch.float16);  primals_92 = None
        convert_element_type_167: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_93, torch.float16);  primals_93 = None
        convert_element_type_168: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_80, torch.float16);  permute_80 = None
        permute_81: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_167, [1, 0]);  convert_element_type_167 = None
        clone_16: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_168, memory_format = torch.contiguous_format);  convert_element_type_168 = None
        view_108: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_16, [mul_15, 768]);  clone_16 = None
        mm_8: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_108, permute_81)
        view_109: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(mm_8, [1024, primals_1, 2304]);  mm_8 = None
        add_1528: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_109, convert_element_type_166);  view_109 = convert_element_type_166 = None
        view_110: "f16[1024, s77, 3, 768]" = torch.ops.aten.view.default(add_1528, [1024, primals_1, 3, 768]);  add_1528 = None
        unsqueeze_13: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_110, 0);  view_110 = None
        permute_82: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_13, [3, 1, 2, 0, 4]);  unsqueeze_13 = None
        squeeze_7: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_82, -2);  permute_82 = None
        clone_17: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_7, memory_format = torch.contiguous_format);  squeeze_7 = None
        select_21: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_17, 0, 0)
        select_22: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_17, 0, 1)
        select_23: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_17, 0, 2);  clone_17 = None
        view_111: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_21, [1024, mul_127, 96]);  select_21 = None
        permute_83: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_111, [1, 0, 2]);  view_111 = None
        view_112: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_22, [1024, mul_127, 96]);  select_22 = None
        permute_84: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_112, [1, 0, 2]);  view_112 = None
        view_113: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_23, [1024, mul_127, 96]);  select_23 = None
        permute_85: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_113, [1, 0, 2]);  view_113 = None
        view_114: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_83, [primals_1, 8, 1024, 96]);  permute_83 = None
        view_115: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_84, [primals_1, 8, 1024, 96]);  permute_84 = None
        view_116: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_85, [primals_1, 8, 1024, 96]);  permute_85 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_7 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_114, view_115, view_116, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_7);  bwd_rng_state_7 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_93: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_7[0]
        getitem_94: "f32[s77, 8, 1024]" = graphsafe_run_with_rng_state_7[1]
        getitem_99: "u64[2]" = graphsafe_run_with_rng_state_7[6]
        getitem_100: "u64[]" = graphsafe_run_with_rng_state_7[7];  graphsafe_run_with_rng_state_7 = None
        permute_86: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_93, [2, 0, 1, 3])
        view_117: "f16[1024*s77, 768]" = torch.ops.aten.view.default(permute_86, [mul_15, 768]);  permute_86 = None
        mm_54: "f16[768, 768]" = torch.ops.aten.mm.default(permute_252, view_117);  permute_252 = view_117 = None
        sum_57: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_289, [0], True, dtype = torch.float32);  view_289 = None
        view_290: "f32[768]" = torch.ops.aten.view.default(sum_57, [768]);  sum_57 = None
        convert_element_type_487: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_54, torch.float32);  mm_54 = None
        convert_element_type_default_30: "f32[768]" = torch.ops.prims.convert_element_type.default(view_290, torch.float32);  view_290 = None
        view_291: "f16[1024, s77, 8, 96]" = torch.ops.aten.view.default(mm_53, [1024, primals_1, 8, 96]);  mm_53 = None
        permute_255: "f16[s77, 8, 1024, 96]" = torch.ops.aten.permute.default(view_291, [1, 2, 0, 3]);  view_291 = None
        _scaled_dot_product_flash_attention_backward_4 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_255, view_114, view_115, view_116, getitem_93, getitem_94, None, None, 1024, 1024, 0.2, False, getitem_99, getitem_100, scale = 0.10206207261596577);  permute_255 = view_114 = view_115 = view_116 = getitem_93 = getitem_94 = getitem_99 = getitem_100 = None
        getitem_168: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_4[0]
        getitem_169: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_4[1]
        getitem_170: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_4[2];  _scaled_dot_product_flash_attention_backward_4 = None
        view_292: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_170, [mul_127, 1024, 96]);  getitem_170 = None
        view_293: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_169, [mul_127, 1024, 96]);  getitem_169 = None
        view_294: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_168, [mul_127, 1024, 96]);  getitem_168 = None
        permute_256: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_292, [1, 0, 2]);  view_292 = None
        view_295: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_256, [1024, primals_1, 768]);  permute_256 = None
        permute_257: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_293, [1, 0, 2]);  view_293 = None
        view_296: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_257, [1024, primals_1, 768]);  permute_257 = None
        permute_258: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_294, [1, 0, 2]);  view_294 = None
        view_297: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_258, [1024, primals_1, 768]);  permute_258 = None
        select_scatter_12: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_295, 0, 2);  view_295 = None
        select_scatter_13: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_296, 0, 1);  view_296 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2702: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(select_scatter_12, select_scatter_13);  select_scatter_12 = select_scatter_13 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        select_scatter_14: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_297, 0, 0);  view_297 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2703: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(add_2702, select_scatter_14);  add_2702 = select_scatter_14 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        unsqueeze_34: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.unsqueeze.default(add_2703, 3);  add_2703 = None
        permute_259: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_34, [3, 1, 2, 0, 4]);  unsqueeze_34 = None
        squeeze_16: "f16[1024, s77, 3, 768]" = torch.ops.aten.squeeze.dim(permute_259, 0);  permute_259 = None
        clone_43: "f16[1024, s77, 3, 768]" = torch.ops.aten.clone.default(squeeze_16, memory_format = torch.contiguous_format);  squeeze_16 = None
        view_298: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(clone_43, [1024, primals_1, 2304]);  clone_43 = None
        sum_58: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_298, [0, 1], True, dtype = torch.float32)
        view_299: "f32[2304]" = torch.ops.aten.view.default(sum_58, [2304]);  sum_58 = None
        view_300: "f16[1024*s77, 2304]" = torch.ops.aten.view.default(view_298, [mul_15, 2304]);  view_298 = None
        permute_260: "f16[2304, 1024*s77]" = torch.ops.aten.permute.default(view_300, [1, 0])
        mm_55: "f16[2304, 768]" = torch.ops.aten.mm.default(permute_260, view_108);  permute_260 = view_108 = None
        permute_262: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        mm_56: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_300, permute_262);  view_300 = permute_262 = None
        view_301: "f16[1024, s77, 768]" = torch.ops.aten.view.default(mm_56, [1024, primals_1, 768]);  mm_56 = None
        convert_element_type_494: "f32[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(view_301, torch.float32);  view_301 = None
        convert_element_type_495: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_default_29: "f32[2304]" = torch.ops.prims.convert_element_type.default(view_299, torch.float32);  view_299 = None
        permute_264: "f32[s77, 1024, 768]" = torch.ops.aten.permute.default(convert_element_type_494, [1, 0, 2]);  convert_element_type_494 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        mul_2519: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_264, primals_90);  primals_90 = None
        mul_2520: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2519, 768)
        sum_59: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2519, [2], True)
        mul_2521: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2519, mul_1376);  mul_2519 = None
        sum_60: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2521, [2], True);  mul_2521 = None
        mul_2522: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1376, sum_60);  sum_60 = None
        sub_696: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2520, sum_59);  mul_2520 = sum_59 = None
        sub_697: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_696, mul_2522);  sub_696 = mul_2522 = None
        div_9: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
        mul_2523: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_9, sub_697);  div_9 = sub_697 = None
        mul_2524: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_264, mul_1376);  mul_1376 = None
        sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2524, [0, 1]);  mul_2524 = None
        sum_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_264, [0, 1]);  permute_264 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        add_2704: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2701, mul_2523);  add_2701 = mul_2523 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        convert_element_type_497: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2704, torch.float16)
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_5: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        convert_element_type_default_57: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_5, torch.float16);  inductor_random_default_5 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_31: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_57, 0.2);  convert_element_type_default_57 = None
        convert_element_type_498: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_31, torch.float16);  gt_31 = None
        mul_2525: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_498, 1.25);  convert_element_type_498 = None
        mul_2526: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_497, mul_2525);  convert_element_type_497 = mul_2525 = None
        view_302: "f16[1024*s77, 768]" = torch.ops.aten.view.default(mul_2526, [mul_15, 768]);  mul_2526 = None
        convert_element_type_162: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_88, torch.float16);  primals_88 = None
        permute_79: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_162, [1, 0]);  convert_element_type_162 = None
        permute_265: "f16[768, 1536]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        mm_57: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_302, permute_265);  permute_265 = None
        permute_266: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_302, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_13 = torch.ops.aten.var_mean.correction(add_1433, [2], correction = 0, keepdim = True)
        getitem_89: "f32[s77, 1024, 1]" = var_mean_13[0]
        getitem_90: "f32[s77, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
        add_1438: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_89, 1e-05);  getitem_89 = None
        rsqrt_13: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1438);  add_1438 = None
        sub_358: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1433, getitem_90);  add_1433 = getitem_90 = None
        mul_1330: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_358, rsqrt_13);  sub_358 = None
        mul_1331: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1330, primals_84)
        add_1439: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1331, primals_85);  mul_1331 = primals_85 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_153: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_87, torch.float16);  primals_87 = None
        convert_element_type_154: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_86, torch.float16);  primals_86 = None
        convert_element_type_155: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_1439, torch.float16);  add_1439 = None
        view_104: "f16[1024*s77, 768]" = torch.ops.aten.view.default(convert_element_type_155, [mul_15, 768]);  convert_element_type_155 = None
        permute_78: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_154, [1, 0]);  convert_element_type_154 = None
        addmm_19: "f16[1024*s77, 1536]" = torch.ops.aten.addmm.default(convert_element_type_153, view_104, permute_78);  convert_element_type_153 = None
        view_105: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(addmm_19, [primals_1, 1024, 1536]);  addmm_19 = None
        convert_element_type_159: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None
        mul_1351: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_159, 0.5)
        mul_1352: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_159, 0.7071067811865476)
        erf_6: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_1352);  mul_1352 = None
        add_1466: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_1353: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_1351, add_1466);  mul_1351 = None
        convert_element_type_160: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_1353, torch.float16);  mul_1353 = None
        view_106: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_160, [mul_15, 1536]);  convert_element_type_160 = None
        mm_58: "f16[768, 1536]" = torch.ops.aten.mm.default(permute_266, view_106);  permute_266 = view_106 = None
        sum_63: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_302, [0], True, dtype = torch.float32);  view_302 = None
        view_303: "f32[768]" = torch.ops.aten.view.default(sum_63, [768]);  sum_63 = None
        view_304: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(mm_57, [primals_1, 1024, 1536]);  mm_57 = None
        convert_element_type_504: "f32[768, 1536]" = torch.ops.prims.convert_element_type.default(mm_58, torch.float32);  mm_58 = None
        convert_element_type_default_28: "f32[768]" = torch.ops.prims.convert_element_type.default(view_303, torch.float32);  view_303 = None
        convert_element_type_506: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_304, torch.float32);  view_304 = None
        mul_2528: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(add_1466, 0.5);  add_1466 = None
        mul_2529: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_159, convert_element_type_159)
        mul_2530: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_2529, -0.5);  mul_2529 = None
        exp_5: "f32[s77, 1024, 1536]" = torch.ops.aten.exp.default(mul_2530);  mul_2530 = None
        mul_2531: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
        mul_2532: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_159, mul_2531);  convert_element_type_159 = mul_2531 = None
        add_2706: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(mul_2528, mul_2532);  mul_2528 = mul_2532 = None
        mul_2533: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_506, add_2706);  convert_element_type_506 = add_2706 = None
        convert_element_type_508: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_2533, torch.float16);  mul_2533 = None
        view_305: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_508, [mul_15, 1536]);  convert_element_type_508 = None
        permute_269: "f16[1536, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        mm_59: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_305, permute_269);  permute_269 = None
        permute_270: "f16[1536, 1024*s77]" = torch.ops.aten.permute.default(view_305, [1, 0])
        mm_60: "f16[1536, 768]" = torch.ops.aten.mm.default(permute_270, view_104);  permute_270 = view_104 = None
        sum_64: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True, dtype = torch.float32);  view_305 = None
        view_306: "f32[1536]" = torch.ops.aten.view.default(sum_64, [1536]);  sum_64 = None
        view_307: "f16[s77, 1024, 768]" = torch.ops.aten.view.default(mm_59, [primals_1, 1024, 768]);  mm_59 = None
        convert_element_type_514: "f32[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_307, torch.float32);  view_307 = None
        convert_element_type_515: "f32[1536, 768]" = torch.ops.prims.convert_element_type.default(mm_60, torch.float32);  mm_60 = None
        convert_element_type_default_27: "f32[1536]" = torch.ops.prims.convert_element_type.default(view_306, torch.float32);  view_306 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        mul_2535: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_514, primals_84);  primals_84 = None
        mul_2536: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2535, 768)
        sum_65: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2535, [2], True)
        mul_2537: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2535, mul_1330);  mul_2535 = None
        sum_66: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2537, [2], True);  mul_2537 = None
        mul_2538: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1330, sum_66);  sum_66 = None
        sub_699: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2536, sum_65);  mul_2536 = sum_65 = None
        sub_700: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_699, mul_2538);  sub_699 = mul_2538 = None
        div_10: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        mul_2539: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_10, sub_700);  div_10 = sub_700 = None
        mul_2540: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_514, mul_1330);  mul_1330 = None
        sum_67: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2540, [0, 1]);  mul_2540 = None
        sum_68: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_514, [0, 1]);  convert_element_type_514 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_2707: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2704, mul_2539);  add_2704 = mul_2539 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        convert_element_type_517: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2707, torch.float16)
        permute_273: "f16[1024, s77, 768]" = torch.ops.aten.permute.default(convert_element_type_517, [1, 0, 2]);  convert_element_type_517 = None
        clone_45: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(permute_273, memory_format = torch.contiguous_format);  permute_273 = None
        view_308: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_45, [mul_15, 768]);  clone_45 = None
        convert_element_type_149: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_82, torch.float16);  primals_82 = None
        permute_76: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_149, [1, 0]);  convert_element_type_149 = None
        permute_274: "f16[768, 768]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        mm_61: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_308, permute_274);  permute_274 = None
        permute_275: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_308, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_1281, [2], correction = 0, keepdim = True)
        getitem_78: "f32[s77, 1024, 1]" = var_mean_12[0]
        getitem_79: "f32[s77, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
        add_1286: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_12: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1286);  add_1286 = None
        sub_321: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1281, getitem_79);  add_1281 = getitem_79 = None
        mul_1186: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_321, rsqrt_12);  sub_321 = None
        mul_1187: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1186, primals_78)
        add_1287: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1187, primals_79);  mul_1187 = primals_79 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_69: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_1287, [1, 0, 2]);  add_1287 = None
        convert_element_type_143: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_80, torch.float16);  primals_80 = None
        convert_element_type_144: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_81, torch.float16);  primals_81 = None
        convert_element_type_145: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_69, torch.float16);  permute_69 = None
        permute_70: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_144, [1, 0]);  convert_element_type_144 = None
        clone_14: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_145, memory_format = torch.contiguous_format);  convert_element_type_145 = None
        view_93: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_14, [mul_15, 768]);  clone_14 = None
        mm_7: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_93, permute_70)
        view_94: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(mm_7, [1024, primals_1, 2304]);  mm_7 = None
        add_1320: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_94, convert_element_type_143);  view_94 = convert_element_type_143 = None
        view_95: "f16[1024, s77, 3, 768]" = torch.ops.aten.view.default(add_1320, [1024, primals_1, 3, 768]);  add_1320 = None
        unsqueeze_12: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_95, 0);  view_95 = None
        permute_71: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_12, [3, 1, 2, 0, 4]);  unsqueeze_12 = None
        squeeze_6: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_71, -2);  permute_71 = None
        clone_15: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_6, memory_format = torch.contiguous_format);  squeeze_6 = None
        select_18: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_15, 0, 0)
        select_19: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_15, 0, 1)
        select_20: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_15, 0, 2);  clone_15 = None
        view_96: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_18, [1024, mul_127, 96]);  select_18 = None
        permute_72: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_96, [1, 0, 2]);  view_96 = None
        view_97: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_19, [1024, mul_127, 96]);  select_19 = None
        permute_73: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_97, [1, 0, 2]);  view_97 = None
        view_98: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_20, [1024, mul_127, 96]);  select_20 = None
        permute_74: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_98, [1, 0, 2]);  view_98 = None
        view_99: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_72, [primals_1, 8, 1024, 96]);  permute_72 = None
        view_100: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_73, [primals_1, 8, 1024, 96]);  permute_73 = None
        view_101: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_74, [primals_1, 8, 1024, 96]);  permute_74 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_6 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_99, view_100, view_101, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_6);  bwd_rng_state_6 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_80: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_6[0]
        getitem_81: "f32[s77, 8, 1024]" = graphsafe_run_with_rng_state_6[1]
        getitem_86: "u64[2]" = graphsafe_run_with_rng_state_6[6]
        getitem_87: "u64[]" = graphsafe_run_with_rng_state_6[7];  graphsafe_run_with_rng_state_6 = None
        permute_75: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_80, [2, 0, 1, 3])
        view_102: "f16[1024*s77, 768]" = torch.ops.aten.view.default(permute_75, [mul_15, 768]);  permute_75 = None
        mm_62: "f16[768, 768]" = torch.ops.aten.mm.default(permute_275, view_102);  permute_275 = view_102 = None
        sum_69: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_308, [0], True, dtype = torch.float32);  view_308 = None
        view_309: "f32[768]" = torch.ops.aten.view.default(sum_69, [768]);  sum_69 = None
        convert_element_type_523: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_62, torch.float32);  mm_62 = None
        convert_element_type_default_26: "f32[768]" = torch.ops.prims.convert_element_type.default(view_309, torch.float32);  view_309 = None
        view_310: "f16[1024, s77, 8, 96]" = torch.ops.aten.view.default(mm_61, [1024, primals_1, 8, 96]);  mm_61 = None
        permute_278: "f16[s77, 8, 1024, 96]" = torch.ops.aten.permute.default(view_310, [1, 2, 0, 3]);  view_310 = None
        _scaled_dot_product_flash_attention_backward_5 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_278, view_99, view_100, view_101, getitem_80, getitem_81, None, None, 1024, 1024, 0.2, False, getitem_86, getitem_87, scale = 0.10206207261596577);  permute_278 = view_99 = view_100 = view_101 = getitem_80 = getitem_81 = getitem_86 = getitem_87 = None
        getitem_171: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_5[0]
        getitem_172: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_5[1]
        getitem_173: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_5[2];  _scaled_dot_product_flash_attention_backward_5 = None
        view_311: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_173, [mul_127, 1024, 96]);  getitem_173 = None
        view_312: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_172, [mul_127, 1024, 96]);  getitem_172 = None
        view_313: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_171, [mul_127, 1024, 96]);  getitem_171 = None
        permute_279: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_311, [1, 0, 2]);  view_311 = None
        view_314: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_279, [1024, primals_1, 768]);  permute_279 = None
        permute_280: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_312, [1, 0, 2]);  view_312 = None
        view_315: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_280, [1024, primals_1, 768]);  permute_280 = None
        permute_281: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_313, [1, 0, 2]);  view_313 = None
        view_316: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_281, [1024, primals_1, 768]);  permute_281 = None
        select_scatter_15: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_314, 0, 2);  view_314 = None
        select_scatter_16: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_315, 0, 1);  view_315 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2708: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(select_scatter_15, select_scatter_16);  select_scatter_15 = select_scatter_16 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        select_scatter_17: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_316, 0, 0);  view_316 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2709: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(add_2708, select_scatter_17);  add_2708 = select_scatter_17 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        unsqueeze_35: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.unsqueeze.default(add_2709, 3);  add_2709 = None
        permute_282: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_35, [3, 1, 2, 0, 4]);  unsqueeze_35 = None
        squeeze_17: "f16[1024, s77, 3, 768]" = torch.ops.aten.squeeze.dim(permute_282, 0);  permute_282 = None
        clone_46: "f16[1024, s77, 3, 768]" = torch.ops.aten.clone.default(squeeze_17, memory_format = torch.contiguous_format);  squeeze_17 = None
        view_317: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(clone_46, [1024, primals_1, 2304]);  clone_46 = None
        sum_70: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_317, [0, 1], True, dtype = torch.float32)
        view_318: "f32[2304]" = torch.ops.aten.view.default(sum_70, [2304]);  sum_70 = None
        view_319: "f16[1024*s77, 2304]" = torch.ops.aten.view.default(view_317, [mul_15, 2304]);  view_317 = None
        permute_283: "f16[2304, 1024*s77]" = torch.ops.aten.permute.default(view_319, [1, 0])
        mm_63: "f16[2304, 768]" = torch.ops.aten.mm.default(permute_283, view_93);  permute_283 = view_93 = None
        permute_285: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        mm_64: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_319, permute_285);  view_319 = permute_285 = None
        view_320: "f16[1024, s77, 768]" = torch.ops.aten.view.default(mm_64, [1024, primals_1, 768]);  mm_64 = None
        convert_element_type_530: "f32[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(view_320, torch.float32);  view_320 = None
        convert_element_type_531: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_default_25: "f32[2304]" = torch.ops.prims.convert_element_type.default(view_318, torch.float32);  view_318 = None
        permute_287: "f32[s77, 1024, 768]" = torch.ops.aten.permute.default(convert_element_type_530, [1, 0, 2]);  convert_element_type_530 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        mul_2542: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_287, primals_78);  primals_78 = None
        mul_2543: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2542, 768)
        sum_71: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2542, [2], True)
        mul_2544: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2542, mul_1186);  mul_2542 = None
        sum_72: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2544, [2], True);  mul_2544 = None
        mul_2545: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1186, sum_72);  sum_72 = None
        sub_702: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2543, sum_71);  mul_2543 = sum_71 = None
        sub_703: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_702, mul_2545);  sub_702 = mul_2545 = None
        div_11: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        mul_2546: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_11, sub_703);  div_11 = sub_703 = None
        mul_2547: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_287, mul_1186);  mul_1186 = None
        sum_73: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2547, [0, 1]);  mul_2547 = None
        sum_74: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_287, [0, 1]);  permute_287 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        add_2710: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2707, mul_2546);  add_2707 = mul_2546 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        convert_element_type_533: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2710, torch.float16)
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_6: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        convert_element_type_default_58: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_6, torch.float16);  inductor_random_default_6 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_27: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_58, 0.2);  convert_element_type_default_58 = None
        convert_element_type_534: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_27, torch.float16);  gt_27 = None
        mul_2548: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_534, 1.25);  convert_element_type_534 = None
        mul_2549: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_533, mul_2548);  convert_element_type_533 = mul_2548 = None
        view_321: "f16[1024*s77, 768]" = torch.ops.aten.view.default(mul_2549, [mul_15, 768]);  mul_2549 = None
        convert_element_type_139: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_76, torch.float16);  primals_76 = None
        permute_68: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_139, [1, 0]);  convert_element_type_139 = None
        permute_288: "f16[768, 1536]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        mm_65: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_321, permute_288);  permute_288 = None
        permute_289: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_321, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_11 = torch.ops.aten.var_mean.correction(add_1225, [2], correction = 0, keepdim = True)
        getitem_76: "f32[s77, 1024, 1]" = var_mean_11[0]
        getitem_77: "f32[s77, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
        add_1230: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_11: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1230);  add_1230 = None
        sub_306: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1225, getitem_77);  add_1225 = getitem_77 = None
        mul_1140: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_306, rsqrt_11);  sub_306 = None
        mul_1141: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1140, primals_72)
        add_1231: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1141, primals_73);  mul_1141 = primals_73 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_130: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_75, torch.float16);  primals_75 = None
        convert_element_type_131: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_74, torch.float16);  primals_74 = None
        convert_element_type_132: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_1231, torch.float16);  add_1231 = None
        view_89: "f16[1024*s77, 768]" = torch.ops.aten.view.default(convert_element_type_132, [mul_15, 768]);  convert_element_type_132 = None
        permute_67: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_131, [1, 0]);  convert_element_type_131 = None
        addmm_16: "f16[1024*s77, 1536]" = torch.ops.aten.addmm.default(convert_element_type_130, view_89, permute_67);  convert_element_type_130 = None
        view_90: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(addmm_16, [primals_1, 1024, 1536]);  addmm_16 = None
        convert_element_type_136: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_90, torch.float32);  view_90 = None
        mul_1161: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_136, 0.5)
        mul_1162: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_136, 0.7071067811865476)
        erf_5: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_1162);  mul_1162 = None
        add_1258: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_1163: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_1161, add_1258);  mul_1161 = None
        convert_element_type_137: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_1163, torch.float16);  mul_1163 = None
        view_91: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_137, [mul_15, 1536]);  convert_element_type_137 = None
        mm_66: "f16[768, 1536]" = torch.ops.aten.mm.default(permute_289, view_91);  permute_289 = view_91 = None
        sum_75: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_321, [0], True, dtype = torch.float32);  view_321 = None
        view_322: "f32[768]" = torch.ops.aten.view.default(sum_75, [768]);  sum_75 = None
        view_323: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(mm_65, [primals_1, 1024, 1536]);  mm_65 = None
        convert_element_type_540: "f32[768, 1536]" = torch.ops.prims.convert_element_type.default(mm_66, torch.float32);  mm_66 = None
        convert_element_type_default_24: "f32[768]" = torch.ops.prims.convert_element_type.default(view_322, torch.float32);  view_322 = None
        convert_element_type_542: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_323, torch.float32);  view_323 = None
        mul_2551: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(add_1258, 0.5);  add_1258 = None
        mul_2552: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_136, convert_element_type_136)
        mul_2553: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_2552, -0.5);  mul_2552 = None
        exp_6: "f32[s77, 1024, 1536]" = torch.ops.aten.exp.default(mul_2553);  mul_2553 = None
        mul_2554: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
        mul_2555: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_136, mul_2554);  convert_element_type_136 = mul_2554 = None
        add_2712: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(mul_2551, mul_2555);  mul_2551 = mul_2555 = None
        mul_2556: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_542, add_2712);  convert_element_type_542 = add_2712 = None
        convert_element_type_544: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_2556, torch.float16);  mul_2556 = None
        view_324: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_544, [mul_15, 1536]);  convert_element_type_544 = None
        permute_292: "f16[1536, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        mm_67: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_324, permute_292);  permute_292 = None
        permute_293: "f16[1536, 1024*s77]" = torch.ops.aten.permute.default(view_324, [1, 0])
        mm_68: "f16[1536, 768]" = torch.ops.aten.mm.default(permute_293, view_89);  permute_293 = view_89 = None
        sum_76: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_324, [0], True, dtype = torch.float32);  view_324 = None
        view_325: "f32[1536]" = torch.ops.aten.view.default(sum_76, [1536]);  sum_76 = None
        view_326: "f16[s77, 1024, 768]" = torch.ops.aten.view.default(mm_67, [primals_1, 1024, 768]);  mm_67 = None
        convert_element_type_550: "f32[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_326, torch.float32);  view_326 = None
        convert_element_type_551: "f32[1536, 768]" = torch.ops.prims.convert_element_type.default(mm_68, torch.float32);  mm_68 = None
        convert_element_type_default_23: "f32[1536]" = torch.ops.prims.convert_element_type.default(view_325, torch.float32);  view_325 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        mul_2558: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_550, primals_72);  primals_72 = None
        mul_2559: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2558, 768)
        sum_77: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2558, [2], True)
        mul_2560: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2558, mul_1140);  mul_2558 = None
        sum_78: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2560, [2], True);  mul_2560 = None
        mul_2561: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1140, sum_78);  sum_78 = None
        sub_705: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2559, sum_77);  mul_2559 = sum_77 = None
        sub_706: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_705, mul_2561);  sub_705 = mul_2561 = None
        div_12: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        mul_2562: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_12, sub_706);  div_12 = sub_706 = None
        mul_2563: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_550, mul_1140);  mul_1140 = None
        sum_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2563, [0, 1]);  mul_2563 = None
        sum_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_550, [0, 1]);  convert_element_type_550 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_2713: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2710, mul_2562);  add_2710 = mul_2562 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        convert_element_type_553: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2713, torch.float16)
        permute_296: "f16[1024, s77, 768]" = torch.ops.aten.permute.default(convert_element_type_553, [1, 0, 2]);  convert_element_type_553 = None
        clone_48: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(permute_296, memory_format = torch.contiguous_format);  permute_296 = None
        view_327: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_48, [mul_15, 768]);  clone_48 = None
        convert_element_type_126: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_70, torch.float16);  primals_70 = None
        permute_65: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_126, [1, 0]);  convert_element_type_126 = None
        permute_297: "f16[768, 768]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_69: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_327, permute_297);  permute_297 = None
        permute_298: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_327, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_1073, [2], correction = 0, keepdim = True)
        getitem_65: "f32[s77, 1024, 1]" = var_mean_10[0]
        getitem_66: "f32[s77, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
        add_1078: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_65, 1e-05);  getitem_65 = None
        rsqrt_10: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1078);  add_1078 = None
        sub_269: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1073, getitem_66);  add_1073 = getitem_66 = None
        mul_996: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_269, rsqrt_10);  sub_269 = None
        mul_997: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_996, primals_66)
        add_1079: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_997, primals_67);  mul_997 = primals_67 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_58: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_1079, [1, 0, 2]);  add_1079 = None
        convert_element_type_120: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_68, torch.float16);  primals_68 = None
        convert_element_type_121: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_69, torch.float16);  primals_69 = None
        convert_element_type_122: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_58, torch.float16);  permute_58 = None
        permute_59: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_121, [1, 0]);  convert_element_type_121 = None
        clone_12: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_122, memory_format = torch.contiguous_format);  convert_element_type_122 = None
        view_78: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_12, [mul_15, 768]);  clone_12 = None
        mm_6: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_78, permute_59)
        view_79: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(mm_6, [1024, primals_1, 2304]);  mm_6 = None
        add_1112: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_79, convert_element_type_120);  view_79 = convert_element_type_120 = None
        view_80: "f16[1024, s77, 3, 768]" = torch.ops.aten.view.default(add_1112, [1024, primals_1, 3, 768]);  add_1112 = None
        unsqueeze_11: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_80, 0);  view_80 = None
        permute_60: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_11, [3, 1, 2, 0, 4]);  unsqueeze_11 = None
        squeeze_5: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_60, -2);  permute_60 = None
        clone_13: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_5, memory_format = torch.contiguous_format);  squeeze_5 = None
        select_15: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_13, 0, 0)
        select_16: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_13, 0, 1)
        select_17: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_13, 0, 2);  clone_13 = None
        view_81: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_15, [1024, mul_127, 96]);  select_15 = None
        permute_61: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_81, [1, 0, 2]);  view_81 = None
        view_82: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_16, [1024, mul_127, 96]);  select_16 = None
        permute_62: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_82, [1, 0, 2]);  view_82 = None
        view_83: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_17, [1024, mul_127, 96]);  select_17 = None
        permute_63: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_83, [1, 0, 2]);  view_83 = None
        view_84: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_61, [primals_1, 8, 1024, 96]);  permute_61 = None
        view_85: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_62, [primals_1, 8, 1024, 96]);  permute_62 = None
        view_86: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_63, [primals_1, 8, 1024, 96]);  permute_63 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_5 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_84, view_85, view_86, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_5);  bwd_rng_state_5 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_67: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_5[0]
        getitem_68: "f32[s77, 8, 1024]" = graphsafe_run_with_rng_state_5[1]
        getitem_73: "u64[2]" = graphsafe_run_with_rng_state_5[6]
        getitem_74: "u64[]" = graphsafe_run_with_rng_state_5[7];  graphsafe_run_with_rng_state_5 = None
        permute_64: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_67, [2, 0, 1, 3])
        view_87: "f16[1024*s77, 768]" = torch.ops.aten.view.default(permute_64, [mul_15, 768]);  permute_64 = None
        mm_70: "f16[768, 768]" = torch.ops.aten.mm.default(permute_298, view_87);  permute_298 = view_87 = None
        sum_81: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True, dtype = torch.float32);  view_327 = None
        view_328: "f32[768]" = torch.ops.aten.view.default(sum_81, [768]);  sum_81 = None
        convert_element_type_559: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_70, torch.float32);  mm_70 = None
        convert_element_type_default_22: "f32[768]" = torch.ops.prims.convert_element_type.default(view_328, torch.float32);  view_328 = None
        view_329: "f16[1024, s77, 8, 96]" = torch.ops.aten.view.default(mm_69, [1024, primals_1, 8, 96]);  mm_69 = None
        permute_301: "f16[s77, 8, 1024, 96]" = torch.ops.aten.permute.default(view_329, [1, 2, 0, 3]);  view_329 = None
        _scaled_dot_product_flash_attention_backward_6 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_301, view_84, view_85, view_86, getitem_67, getitem_68, None, None, 1024, 1024, 0.2, False, getitem_73, getitem_74, scale = 0.10206207261596577);  permute_301 = view_84 = view_85 = view_86 = getitem_67 = getitem_68 = getitem_73 = getitem_74 = None
        getitem_174: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_6[0]
        getitem_175: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_6[1]
        getitem_176: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_6[2];  _scaled_dot_product_flash_attention_backward_6 = None
        view_330: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_176, [mul_127, 1024, 96]);  getitem_176 = None
        view_331: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_175, [mul_127, 1024, 96]);  getitem_175 = None
        view_332: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_174, [mul_127, 1024, 96]);  getitem_174 = None
        permute_302: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_330, [1, 0, 2]);  view_330 = None
        view_333: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_302, [1024, primals_1, 768]);  permute_302 = None
        permute_303: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_331, [1, 0, 2]);  view_331 = None
        view_334: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_303, [1024, primals_1, 768]);  permute_303 = None
        permute_304: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_332, [1, 0, 2]);  view_332 = None
        view_335: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_304, [1024, primals_1, 768]);  permute_304 = None
        select_scatter_18: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_333, 0, 2);  view_333 = None
        select_scatter_19: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_334, 0, 1);  view_334 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2714: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(select_scatter_18, select_scatter_19);  select_scatter_18 = select_scatter_19 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        select_scatter_20: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_335, 0, 0);  view_335 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2715: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(add_2714, select_scatter_20);  add_2714 = select_scatter_20 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        unsqueeze_36: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.unsqueeze.default(add_2715, 3);  add_2715 = None
        permute_305: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_36, [3, 1, 2, 0, 4]);  unsqueeze_36 = None
        squeeze_18: "f16[1024, s77, 3, 768]" = torch.ops.aten.squeeze.dim(permute_305, 0);  permute_305 = None
        clone_49: "f16[1024, s77, 3, 768]" = torch.ops.aten.clone.default(squeeze_18, memory_format = torch.contiguous_format);  squeeze_18 = None
        view_336: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(clone_49, [1024, primals_1, 2304]);  clone_49 = None
        sum_82: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_336, [0, 1], True, dtype = torch.float32)
        view_337: "f32[2304]" = torch.ops.aten.view.default(sum_82, [2304]);  sum_82 = None
        view_338: "f16[1024*s77, 2304]" = torch.ops.aten.view.default(view_336, [mul_15, 2304]);  view_336 = None
        permute_306: "f16[2304, 1024*s77]" = torch.ops.aten.permute.default(view_338, [1, 0])
        mm_71: "f16[2304, 768]" = torch.ops.aten.mm.default(permute_306, view_78);  permute_306 = view_78 = None
        permute_308: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        mm_72: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_338, permute_308);  view_338 = permute_308 = None
        view_339: "f16[1024, s77, 768]" = torch.ops.aten.view.default(mm_72, [1024, primals_1, 768]);  mm_72 = None
        convert_element_type_566: "f32[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(view_339, torch.float32);  view_339 = None
        convert_element_type_567: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_default_21: "f32[2304]" = torch.ops.prims.convert_element_type.default(view_337, torch.float32);  view_337 = None
        permute_310: "f32[s77, 1024, 768]" = torch.ops.aten.permute.default(convert_element_type_566, [1, 0, 2]);  convert_element_type_566 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        mul_2565: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_310, primals_66);  primals_66 = None
        mul_2566: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2565, 768)
        sum_83: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2565, [2], True)
        mul_2567: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2565, mul_996);  mul_2565 = None
        sum_84: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2567, [2], True);  mul_2567 = None
        mul_2568: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_996, sum_84);  sum_84 = None
        sub_708: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2566, sum_83);  mul_2566 = sum_83 = None
        sub_709: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_708, mul_2568);  sub_708 = mul_2568 = None
        div_13: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        mul_2569: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_13, sub_709);  div_13 = sub_709 = None
        mul_2570: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_310, mul_996);  mul_996 = None
        sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2570, [0, 1]);  mul_2570 = None
        sum_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_310, [0, 1]);  permute_310 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        add_2716: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2713, mul_2569);  add_2713 = mul_2569 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        convert_element_type_569: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2716, torch.float16)
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_7: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        convert_element_type_default_59: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_7, torch.float16);  inductor_random_default_7 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_23: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_59, 0.2);  convert_element_type_default_59 = None
        convert_element_type_570: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_23, torch.float16);  gt_23 = None
        mul_2571: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_570, 1.25);  convert_element_type_570 = None
        mul_2572: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_569, mul_2571);  convert_element_type_569 = mul_2571 = None
        view_340: "f16[1024*s77, 768]" = torch.ops.aten.view.default(mul_2572, [mul_15, 768]);  mul_2572 = None
        convert_element_type_116: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_64, torch.float16);  primals_64 = None
        permute_57: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_116, [1, 0]);  convert_element_type_116 = None
        permute_311: "f16[768, 1536]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        mm_73: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_340, permute_311);  permute_311 = None
        permute_312: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_340, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_9 = torch.ops.aten.var_mean.correction(add_1017, [2], correction = 0, keepdim = True)
        getitem_63: "f32[s77, 1024, 1]" = var_mean_9[0]
        getitem_64: "f32[s77, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
        add_1022: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_63, 1e-05);  getitem_63 = None
        rsqrt_9: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1022);  add_1022 = None
        sub_254: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1017, getitem_64);  add_1017 = getitem_64 = None
        mul_950: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_254, rsqrt_9);  sub_254 = None
        mul_951: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_950, primals_60)
        add_1023: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_951, primals_61);  mul_951 = primals_61 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_107: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_63, torch.float16);  primals_63 = None
        convert_element_type_108: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_62, torch.float16);  primals_62 = None
        convert_element_type_109: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_1023, torch.float16);  add_1023 = None
        view_74: "f16[1024*s77, 768]" = torch.ops.aten.view.default(convert_element_type_109, [mul_15, 768]);  convert_element_type_109 = None
        permute_56: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_108, [1, 0]);  convert_element_type_108 = None
        addmm_13: "f16[1024*s77, 1536]" = torch.ops.aten.addmm.default(convert_element_type_107, view_74, permute_56);  convert_element_type_107 = None
        view_75: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(addmm_13, [primals_1, 1024, 1536]);  addmm_13 = None
        convert_element_type_113: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_75, torch.float32);  view_75 = None
        mul_971: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_113, 0.5)
        mul_972: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_113, 0.7071067811865476)
        erf_4: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_972);  mul_972 = None
        add_1050: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_973: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_971, add_1050);  mul_971 = None
        convert_element_type_114: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_973, torch.float16);  mul_973 = None
        view_76: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_114, [mul_15, 1536]);  convert_element_type_114 = None
        mm_74: "f16[768, 1536]" = torch.ops.aten.mm.default(permute_312, view_76);  permute_312 = view_76 = None
        sum_87: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_340, [0], True, dtype = torch.float32);  view_340 = None
        view_341: "f32[768]" = torch.ops.aten.view.default(sum_87, [768]);  sum_87 = None
        view_342: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(mm_73, [primals_1, 1024, 1536]);  mm_73 = None
        convert_element_type_576: "f32[768, 1536]" = torch.ops.prims.convert_element_type.default(mm_74, torch.float32);  mm_74 = None
        convert_element_type_default_20: "f32[768]" = torch.ops.prims.convert_element_type.default(view_341, torch.float32);  view_341 = None
        convert_element_type_578: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_342, torch.float32);  view_342 = None
        mul_2574: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(add_1050, 0.5);  add_1050 = None
        mul_2575: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_113, convert_element_type_113)
        mul_2576: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_2575, -0.5);  mul_2575 = None
        exp_7: "f32[s77, 1024, 1536]" = torch.ops.aten.exp.default(mul_2576);  mul_2576 = None
        mul_2577: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
        mul_2578: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_113, mul_2577);  convert_element_type_113 = mul_2577 = None
        add_2718: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(mul_2574, mul_2578);  mul_2574 = mul_2578 = None
        mul_2579: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_578, add_2718);  convert_element_type_578 = add_2718 = None
        convert_element_type_580: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_2579, torch.float16);  mul_2579 = None
        view_343: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_580, [mul_15, 1536]);  convert_element_type_580 = None
        permute_315: "f16[1536, 768]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        mm_75: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_343, permute_315);  permute_315 = None
        permute_316: "f16[1536, 1024*s77]" = torch.ops.aten.permute.default(view_343, [1, 0])
        mm_76: "f16[1536, 768]" = torch.ops.aten.mm.default(permute_316, view_74);  permute_316 = view_74 = None
        sum_88: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_343, [0], True, dtype = torch.float32);  view_343 = None
        view_344: "f32[1536]" = torch.ops.aten.view.default(sum_88, [1536]);  sum_88 = None
        view_345: "f16[s77, 1024, 768]" = torch.ops.aten.view.default(mm_75, [primals_1, 1024, 768]);  mm_75 = None
        convert_element_type_586: "f32[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_345, torch.float32);  view_345 = None
        convert_element_type_587: "f32[1536, 768]" = torch.ops.prims.convert_element_type.default(mm_76, torch.float32);  mm_76 = None
        convert_element_type_default_19: "f32[1536]" = torch.ops.prims.convert_element_type.default(view_344, torch.float32);  view_344 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        mul_2581: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_586, primals_60);  primals_60 = None
        mul_2582: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2581, 768)
        sum_89: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2581, [2], True)
        mul_2583: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2581, mul_950);  mul_2581 = None
        sum_90: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2583, [2], True);  mul_2583 = None
        mul_2584: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_950, sum_90);  sum_90 = None
        sub_711: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2582, sum_89);  mul_2582 = sum_89 = None
        sub_712: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_711, mul_2584);  sub_711 = mul_2584 = None
        div_14: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        mul_2585: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_712);  div_14 = sub_712 = None
        mul_2586: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_586, mul_950);  mul_950 = None
        sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2586, [0, 1]);  mul_2586 = None
        sum_92: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_586, [0, 1]);  convert_element_type_586 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_2719: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2716, mul_2585);  add_2716 = mul_2585 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        convert_element_type_589: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2719, torch.float16)
        permute_319: "f16[1024, s77, 768]" = torch.ops.aten.permute.default(convert_element_type_589, [1, 0, 2]);  convert_element_type_589 = None
        clone_51: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
        view_346: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_51, [mul_15, 768]);  clone_51 = None
        convert_element_type_103: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_58, torch.float16);  primals_58 = None
        permute_54: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_103, [1, 0]);  convert_element_type_103 = None
        permute_320: "f16[768, 768]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_77: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_346, permute_320);  permute_320 = None
        permute_321: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_346, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_865, [2], correction = 0, keepdim = True)
        getitem_52: "f32[s77, 1024, 1]" = var_mean_8[0]
        getitem_53: "f32[s77, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
        add_870: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_8: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_870);  add_870 = None
        sub_217: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_865, getitem_53);  add_865 = getitem_53 = None
        mul_806: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_217, rsqrt_8);  sub_217 = None
        mul_807: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_806, primals_54)
        add_871: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_807, primals_55);  mul_807 = primals_55 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_47: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_871, [1, 0, 2]);  add_871 = None
        convert_element_type_97: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_56, torch.float16);  primals_56 = None
        convert_element_type_98: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_57, torch.float16);  primals_57 = None
        convert_element_type_99: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_47, torch.float16);  permute_47 = None
        permute_48: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_98, [1, 0]);  convert_element_type_98 = None
        clone_10: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_99, memory_format = torch.contiguous_format);  convert_element_type_99 = None
        view_63: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_10, [mul_15, 768]);  clone_10 = None
        mm_5: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_63, permute_48)
        view_64: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(mm_5, [1024, primals_1, 2304]);  mm_5 = None
        add_904: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_64, convert_element_type_97);  view_64 = convert_element_type_97 = None
        view_65: "f16[1024, s77, 3, 768]" = torch.ops.aten.view.default(add_904, [1024, primals_1, 3, 768]);  add_904 = None
        unsqueeze_10: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_65, 0);  view_65 = None
        permute_49: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_10, [3, 1, 2, 0, 4]);  unsqueeze_10 = None
        squeeze_4: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_49, -2);  permute_49 = None
        clone_11: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_4, memory_format = torch.contiguous_format);  squeeze_4 = None
        select_12: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_11, 0, 0)
        select_13: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_11, 0, 1)
        select_14: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_11, 0, 2);  clone_11 = None
        view_66: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_12, [1024, mul_127, 96]);  select_12 = None
        permute_50: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_66, [1, 0, 2]);  view_66 = None
        view_67: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_13, [1024, mul_127, 96]);  select_13 = None
        permute_51: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_67, [1, 0, 2]);  view_67 = None
        view_68: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_14, [1024, mul_127, 96]);  select_14 = None
        permute_52: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_68, [1, 0, 2]);  view_68 = None
        view_69: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_50, [primals_1, 8, 1024, 96]);  permute_50 = None
        view_70: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_51, [primals_1, 8, 1024, 96]);  permute_51 = None
        view_71: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_52, [primals_1, 8, 1024, 96]);  permute_52 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_4 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_69, view_70, view_71, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_4);  bwd_rng_state_4 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_54: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_4[0]
        getitem_55: "f32[s77, 8, 1024]" = graphsafe_run_with_rng_state_4[1]
        getitem_60: "u64[2]" = graphsafe_run_with_rng_state_4[6]
        getitem_61: "u64[]" = graphsafe_run_with_rng_state_4[7];  graphsafe_run_with_rng_state_4 = None
        permute_53: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_54, [2, 0, 1, 3])
        view_72: "f16[1024*s77, 768]" = torch.ops.aten.view.default(permute_53, [mul_15, 768]);  permute_53 = None
        mm_78: "f16[768, 768]" = torch.ops.aten.mm.default(permute_321, view_72);  permute_321 = view_72 = None
        sum_93: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_346, [0], True, dtype = torch.float32);  view_346 = None
        view_347: "f32[768]" = torch.ops.aten.view.default(sum_93, [768]);  sum_93 = None
        convert_element_type_595: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_78, torch.float32);  mm_78 = None
        convert_element_type_default_18: "f32[768]" = torch.ops.prims.convert_element_type.default(view_347, torch.float32);  view_347 = None
        view_348: "f16[1024, s77, 8, 96]" = torch.ops.aten.view.default(mm_77, [1024, primals_1, 8, 96]);  mm_77 = None
        permute_324: "f16[s77, 8, 1024, 96]" = torch.ops.aten.permute.default(view_348, [1, 2, 0, 3]);  view_348 = None
        _scaled_dot_product_flash_attention_backward_7 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_324, view_69, view_70, view_71, getitem_54, getitem_55, None, None, 1024, 1024, 0.2, False, getitem_60, getitem_61, scale = 0.10206207261596577);  permute_324 = view_69 = view_70 = view_71 = getitem_54 = getitem_55 = getitem_60 = getitem_61 = None
        getitem_177: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_7[0]
        getitem_178: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_7[1]
        getitem_179: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_7[2];  _scaled_dot_product_flash_attention_backward_7 = None
        view_349: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_179, [mul_127, 1024, 96]);  getitem_179 = None
        view_350: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_178, [mul_127, 1024, 96]);  getitem_178 = None
        view_351: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_177, [mul_127, 1024, 96]);  getitem_177 = None
        permute_325: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_349, [1, 0, 2]);  view_349 = None
        view_352: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_325, [1024, primals_1, 768]);  permute_325 = None
        permute_326: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_350, [1, 0, 2]);  view_350 = None
        view_353: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_326, [1024, primals_1, 768]);  permute_326 = None
        permute_327: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_351, [1, 0, 2]);  view_351 = None
        view_354: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_327, [1024, primals_1, 768]);  permute_327 = None
        select_scatter_21: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_352, 0, 2);  view_352 = None
        select_scatter_22: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_353, 0, 1);  view_353 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2720: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(select_scatter_21, select_scatter_22);  select_scatter_21 = select_scatter_22 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        select_scatter_23: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_354, 0, 0);  view_354 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2721: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(add_2720, select_scatter_23);  add_2720 = select_scatter_23 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        unsqueeze_37: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.unsqueeze.default(add_2721, 3);  add_2721 = None
        permute_328: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_37, [3, 1, 2, 0, 4]);  unsqueeze_37 = None
        squeeze_19: "f16[1024, s77, 3, 768]" = torch.ops.aten.squeeze.dim(permute_328, 0);  permute_328 = None
        clone_52: "f16[1024, s77, 3, 768]" = torch.ops.aten.clone.default(squeeze_19, memory_format = torch.contiguous_format);  squeeze_19 = None
        view_355: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(clone_52, [1024, primals_1, 2304]);  clone_52 = None
        sum_94: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_355, [0, 1], True, dtype = torch.float32)
        view_356: "f32[2304]" = torch.ops.aten.view.default(sum_94, [2304]);  sum_94 = None
        view_357: "f16[1024*s77, 2304]" = torch.ops.aten.view.default(view_355, [mul_15, 2304]);  view_355 = None
        permute_329: "f16[2304, 1024*s77]" = torch.ops.aten.permute.default(view_357, [1, 0])
        mm_79: "f16[2304, 768]" = torch.ops.aten.mm.default(permute_329, view_63);  permute_329 = view_63 = None
        permute_331: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        mm_80: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_357, permute_331);  view_357 = permute_331 = None
        view_358: "f16[1024, s77, 768]" = torch.ops.aten.view.default(mm_80, [1024, primals_1, 768]);  mm_80 = None
        convert_element_type_602: "f32[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(view_358, torch.float32);  view_358 = None
        convert_element_type_603: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_default_17: "f32[2304]" = torch.ops.prims.convert_element_type.default(view_356, torch.float32);  view_356 = None
        permute_333: "f32[s77, 1024, 768]" = torch.ops.aten.permute.default(convert_element_type_602, [1, 0, 2]);  convert_element_type_602 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        mul_2588: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_333, primals_54);  primals_54 = None
        mul_2589: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2588, 768)
        sum_95: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2588, [2], True)
        mul_2590: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2588, mul_806);  mul_2588 = None
        sum_96: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2590, [2], True);  mul_2590 = None
        mul_2591: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_806, sum_96);  sum_96 = None
        sub_714: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2589, sum_95);  mul_2589 = sum_95 = None
        sub_715: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_714, mul_2591);  sub_714 = mul_2591 = None
        div_15: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        mul_2592: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_715);  div_15 = sub_715 = None
        mul_2593: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_333, mul_806);  mul_806 = None
        sum_97: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2593, [0, 1]);  mul_2593 = None
        sum_98: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_333, [0, 1]);  permute_333 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        add_2722: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2719, mul_2592);  add_2719 = mul_2592 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        convert_element_type_605: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2722, torch.float16)
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_8: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        convert_element_type_default_60: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_8, torch.float16);  inductor_random_default_8 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_19: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_60, 0.2);  convert_element_type_default_60 = None
        convert_element_type_606: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_19, torch.float16);  gt_19 = None
        mul_2594: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_606, 1.25);  convert_element_type_606 = None
        mul_2595: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_605, mul_2594);  convert_element_type_605 = mul_2594 = None
        view_359: "f16[1024*s77, 768]" = torch.ops.aten.view.default(mul_2595, [mul_15, 768]);  mul_2595 = None
        convert_element_type_93: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_52, torch.float16);  primals_52 = None
        permute_46: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_93, [1, 0]);  convert_element_type_93 = None
        permute_334: "f16[768, 1536]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_81: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_359, permute_334);  permute_334 = None
        permute_335: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_359, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_7 = torch.ops.aten.var_mean.correction(add_809, [2], correction = 0, keepdim = True)
        getitem_50: "f32[s77, 1024, 1]" = var_mean_7[0]
        getitem_51: "f32[s77, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
        add_814: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_7: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_814);  add_814 = None
        sub_202: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_809, getitem_51);  add_809 = getitem_51 = None
        mul_760: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_202, rsqrt_7);  sub_202 = None
        mul_761: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_760, primals_48)
        add_815: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_761, primals_49);  mul_761 = primals_49 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_84: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_51, torch.float16);  primals_51 = None
        convert_element_type_85: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_50, torch.float16);  primals_50 = None
        convert_element_type_86: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_815, torch.float16);  add_815 = None
        view_59: "f16[1024*s77, 768]" = torch.ops.aten.view.default(convert_element_type_86, [mul_15, 768]);  convert_element_type_86 = None
        permute_45: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_85, [1, 0]);  convert_element_type_85 = None
        addmm_10: "f16[1024*s77, 1536]" = torch.ops.aten.addmm.default(convert_element_type_84, view_59, permute_45);  convert_element_type_84 = None
        view_60: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(addmm_10, [primals_1, 1024, 1536]);  addmm_10 = None
        convert_element_type_90: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_60, torch.float32);  view_60 = None
        mul_781: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_90, 0.5)
        mul_782: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_90, 0.7071067811865476)
        erf_3: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_782);  mul_782 = None
        add_842: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_783: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_781, add_842);  mul_781 = None
        convert_element_type_91: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_783, torch.float16);  mul_783 = None
        view_61: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_91, [mul_15, 1536]);  convert_element_type_91 = None
        mm_82: "f16[768, 1536]" = torch.ops.aten.mm.default(permute_335, view_61);  permute_335 = view_61 = None
        sum_99: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True, dtype = torch.float32);  view_359 = None
        view_360: "f32[768]" = torch.ops.aten.view.default(sum_99, [768]);  sum_99 = None
        view_361: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(mm_81, [primals_1, 1024, 1536]);  mm_81 = None
        convert_element_type_612: "f32[768, 1536]" = torch.ops.prims.convert_element_type.default(mm_82, torch.float32);  mm_82 = None
        convert_element_type_default_16: "f32[768]" = torch.ops.prims.convert_element_type.default(view_360, torch.float32);  view_360 = None
        convert_element_type_614: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_361, torch.float32);  view_361 = None
        mul_2597: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(add_842, 0.5);  add_842 = None
        mul_2598: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_90, convert_element_type_90)
        mul_2599: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_2598, -0.5);  mul_2598 = None
        exp_8: "f32[s77, 1024, 1536]" = torch.ops.aten.exp.default(mul_2599);  mul_2599 = None
        mul_2600: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
        mul_2601: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_90, mul_2600);  convert_element_type_90 = mul_2600 = None
        add_2724: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(mul_2597, mul_2601);  mul_2597 = mul_2601 = None
        mul_2602: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_614, add_2724);  convert_element_type_614 = add_2724 = None
        convert_element_type_616: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_2602, torch.float16);  mul_2602 = None
        view_362: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_616, [mul_15, 1536]);  convert_element_type_616 = None
        permute_338: "f16[1536, 768]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        mm_83: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_362, permute_338);  permute_338 = None
        permute_339: "f16[1536, 1024*s77]" = torch.ops.aten.permute.default(view_362, [1, 0])
        mm_84: "f16[1536, 768]" = torch.ops.aten.mm.default(permute_339, view_59);  permute_339 = view_59 = None
        sum_100: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_362, [0], True, dtype = torch.float32);  view_362 = None
        view_363: "f32[1536]" = torch.ops.aten.view.default(sum_100, [1536]);  sum_100 = None
        view_364: "f16[s77, 1024, 768]" = torch.ops.aten.view.default(mm_83, [primals_1, 1024, 768]);  mm_83 = None
        convert_element_type_622: "f32[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_364, torch.float32);  view_364 = None
        convert_element_type_623: "f32[1536, 768]" = torch.ops.prims.convert_element_type.default(mm_84, torch.float32);  mm_84 = None
        convert_element_type_default_15: "f32[1536]" = torch.ops.prims.convert_element_type.default(view_363, torch.float32);  view_363 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        mul_2604: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_622, primals_48);  primals_48 = None
        mul_2605: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2604, 768)
        sum_101: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2604, [2], True)
        mul_2606: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2604, mul_760);  mul_2604 = None
        sum_102: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2606, [2], True);  mul_2606 = None
        mul_2607: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_760, sum_102);  sum_102 = None
        sub_717: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2605, sum_101);  mul_2605 = sum_101 = None
        sub_718: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_717, mul_2607);  sub_717 = mul_2607 = None
        div_16: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        mul_2608: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_718);  div_16 = sub_718 = None
        mul_2609: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_622, mul_760);  mul_760 = None
        sum_103: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2609, [0, 1]);  mul_2609 = None
        sum_104: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_622, [0, 1]);  convert_element_type_622 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_2725: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2722, mul_2608);  add_2722 = mul_2608 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        convert_element_type_625: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2725, torch.float16)
        permute_342: "f16[1024, s77, 768]" = torch.ops.aten.permute.default(convert_element_type_625, [1, 0, 2]);  convert_element_type_625 = None
        clone_54: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(permute_342, memory_format = torch.contiguous_format);  permute_342 = None
        view_365: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_54, [mul_15, 768]);  clone_54 = None
        convert_element_type_80: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_46, torch.float16);  primals_46 = None
        permute_43: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_80, [1, 0]);  convert_element_type_80 = None
        permute_343: "f16[768, 768]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_85: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_365, permute_343);  permute_343 = None
        permute_344: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_365, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_657, [2], correction = 0, keepdim = True)
        getitem_39: "f32[s77, 1024, 1]" = var_mean_6[0]
        getitem_40: "f32[s77, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
        add_662: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_39, 1e-05);  getitem_39 = None
        rsqrt_6: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_662);  add_662 = None
        sub_165: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_657, getitem_40);  add_657 = getitem_40 = None
        mul_616: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_165, rsqrt_6);  sub_165 = None
        mul_617: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_616, primals_42)
        add_663: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_617, primals_43);  mul_617 = primals_43 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_36: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_663, [1, 0, 2]);  add_663 = None
        convert_element_type_74: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_44, torch.float16);  primals_44 = None
        convert_element_type_75: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_45, torch.float16);  primals_45 = None
        convert_element_type_76: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_36, torch.float16);  permute_36 = None
        permute_37: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_75, [1, 0]);  convert_element_type_75 = None
        clone_8: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_76, memory_format = torch.contiguous_format);  convert_element_type_76 = None
        view_48: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_8, [mul_15, 768]);  clone_8 = None
        mm_4: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_48, permute_37)
        view_49: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(mm_4, [1024, primals_1, 2304]);  mm_4 = None
        add_696: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_49, convert_element_type_74);  view_49 = convert_element_type_74 = None
        view_50: "f16[1024, s77, 3, 768]" = torch.ops.aten.view.default(add_696, [1024, primals_1, 3, 768]);  add_696 = None
        unsqueeze_9: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_50, 0);  view_50 = None
        permute_38: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_9, [3, 1, 2, 0, 4]);  unsqueeze_9 = None
        squeeze_3: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_38, -2);  permute_38 = None
        clone_9: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_3, memory_format = torch.contiguous_format);  squeeze_3 = None
        select_9: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_9, 0, 0)
        select_10: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_9, 0, 1)
        select_11: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_9, 0, 2);  clone_9 = None
        view_51: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_9, [1024, mul_127, 96]);  select_9 = None
        permute_39: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_51, [1, 0, 2]);  view_51 = None
        view_52: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_10, [1024, mul_127, 96]);  select_10 = None
        permute_40: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_52, [1, 0, 2]);  view_52 = None
        view_53: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_11, [1024, mul_127, 96]);  select_11 = None
        permute_41: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_53, [1, 0, 2]);  view_53 = None
        view_54: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_39, [primals_1, 8, 1024, 96]);  permute_39 = None
        view_55: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_40, [primals_1, 8, 1024, 96]);  permute_40 = None
        view_56: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_41, [primals_1, 8, 1024, 96]);  permute_41 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_3 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_54, view_55, view_56, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_3);  bwd_rng_state_3 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_41: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_3[0]
        getitem_42: "f32[s77, 8, 1024]" = graphsafe_run_with_rng_state_3[1]
        getitem_47: "u64[2]" = graphsafe_run_with_rng_state_3[6]
        getitem_48: "u64[]" = graphsafe_run_with_rng_state_3[7];  graphsafe_run_with_rng_state_3 = None
        permute_42: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_41, [2, 0, 1, 3])
        view_57: "f16[1024*s77, 768]" = torch.ops.aten.view.default(permute_42, [mul_15, 768]);  permute_42 = None
        mm_86: "f16[768, 768]" = torch.ops.aten.mm.default(permute_344, view_57);  permute_344 = view_57 = None
        sum_105: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_365, [0], True, dtype = torch.float32);  view_365 = None
        view_366: "f32[768]" = torch.ops.aten.view.default(sum_105, [768]);  sum_105 = None
        convert_element_type_631: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_86, torch.float32);  mm_86 = None
        convert_element_type_default_14: "f32[768]" = torch.ops.prims.convert_element_type.default(view_366, torch.float32);  view_366 = None
        view_367: "f16[1024, s77, 8, 96]" = torch.ops.aten.view.default(mm_85, [1024, primals_1, 8, 96]);  mm_85 = None
        permute_347: "f16[s77, 8, 1024, 96]" = torch.ops.aten.permute.default(view_367, [1, 2, 0, 3]);  view_367 = None
        _scaled_dot_product_flash_attention_backward_8 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_347, view_54, view_55, view_56, getitem_41, getitem_42, None, None, 1024, 1024, 0.2, False, getitem_47, getitem_48, scale = 0.10206207261596577);  permute_347 = view_54 = view_55 = view_56 = getitem_41 = getitem_42 = getitem_47 = getitem_48 = None
        getitem_180: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_8[0]
        getitem_181: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_8[1]
        getitem_182: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_8[2];  _scaled_dot_product_flash_attention_backward_8 = None
        view_368: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_182, [mul_127, 1024, 96]);  getitem_182 = None
        view_369: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_181, [mul_127, 1024, 96]);  getitem_181 = None
        view_370: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_180, [mul_127, 1024, 96]);  getitem_180 = None
        permute_348: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_368, [1, 0, 2]);  view_368 = None
        view_371: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_348, [1024, primals_1, 768]);  permute_348 = None
        permute_349: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_369, [1, 0, 2]);  view_369 = None
        view_372: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_349, [1024, primals_1, 768]);  permute_349 = None
        permute_350: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_370, [1, 0, 2]);  view_370 = None
        view_373: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_350, [1024, primals_1, 768]);  permute_350 = None
        select_scatter_24: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_371, 0, 2);  view_371 = None
        select_scatter_25: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_372, 0, 1);  view_372 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2726: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(select_scatter_24, select_scatter_25);  select_scatter_24 = select_scatter_25 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        select_scatter_26: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_373, 0, 0);  view_373 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2727: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(add_2726, select_scatter_26);  add_2726 = select_scatter_26 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        unsqueeze_38: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.unsqueeze.default(add_2727, 3);  add_2727 = None
        permute_351: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_38, [3, 1, 2, 0, 4]);  unsqueeze_38 = None
        squeeze_20: "f16[1024, s77, 3, 768]" = torch.ops.aten.squeeze.dim(permute_351, 0);  permute_351 = None
        clone_55: "f16[1024, s77, 3, 768]" = torch.ops.aten.clone.default(squeeze_20, memory_format = torch.contiguous_format);  squeeze_20 = None
        view_374: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(clone_55, [1024, primals_1, 2304]);  clone_55 = None
        sum_106: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_374, [0, 1], True, dtype = torch.float32)
        view_375: "f32[2304]" = torch.ops.aten.view.default(sum_106, [2304]);  sum_106 = None
        view_376: "f16[1024*s77, 2304]" = torch.ops.aten.view.default(view_374, [mul_15, 2304]);  view_374 = None
        permute_352: "f16[2304, 1024*s77]" = torch.ops.aten.permute.default(view_376, [1, 0])
        mm_87: "f16[2304, 768]" = torch.ops.aten.mm.default(permute_352, view_48);  permute_352 = view_48 = None
        permute_354: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        mm_88: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_376, permute_354);  view_376 = permute_354 = None
        view_377: "f16[1024, s77, 768]" = torch.ops.aten.view.default(mm_88, [1024, primals_1, 768]);  mm_88 = None
        convert_element_type_638: "f32[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(view_377, torch.float32);  view_377 = None
        convert_element_type_639: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_default_13: "f32[2304]" = torch.ops.prims.convert_element_type.default(view_375, torch.float32);  view_375 = None
        permute_356: "f32[s77, 1024, 768]" = torch.ops.aten.permute.default(convert_element_type_638, [1, 0, 2]);  convert_element_type_638 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        mul_2611: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_356, primals_42);  primals_42 = None
        mul_2612: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2611, 768)
        sum_107: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2611, [2], True)
        mul_2613: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2611, mul_616);  mul_2611 = None
        sum_108: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2613, [2], True);  mul_2613 = None
        mul_2614: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_616, sum_108);  sum_108 = None
        sub_720: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2612, sum_107);  mul_2612 = sum_107 = None
        sub_721: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_720, mul_2614);  sub_720 = mul_2614 = None
        div_17: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        mul_2615: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_721);  div_17 = sub_721 = None
        mul_2616: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_356, mul_616);  mul_616 = None
        sum_109: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2616, [0, 1]);  mul_2616 = None
        sum_110: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_356, [0, 1]);  permute_356 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        add_2728: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2725, mul_2615);  add_2725 = mul_2615 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        convert_element_type_641: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2728, torch.float16)
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_9: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        convert_element_type_default_61: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_9, torch.float16);  inductor_random_default_9 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_15: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_61, 0.2);  convert_element_type_default_61 = None
        convert_element_type_642: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_15, torch.float16);  gt_15 = None
        mul_2617: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_642, 1.25);  convert_element_type_642 = None
        mul_2618: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_641, mul_2617);  convert_element_type_641 = mul_2617 = None
        view_378: "f16[1024*s77, 768]" = torch.ops.aten.view.default(mul_2618, [mul_15, 768]);  mul_2618 = None
        convert_element_type_70: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_40, torch.float16);  primals_40 = None
        permute_35: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_70, [1, 0]);  convert_element_type_70 = None
        permute_357: "f16[768, 1536]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_89: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_378, permute_357);  permute_357 = None
        permute_358: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_378, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_5 = torch.ops.aten.var_mean.correction(add_601, [2], correction = 0, keepdim = True)
        getitem_37: "f32[s77, 1024, 1]" = var_mean_5[0]
        getitem_38: "f32[s77, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
        add_606: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_37, 1e-05);  getitem_37 = None
        rsqrt_5: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_606);  add_606 = None
        sub_150: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_601, getitem_38);  add_601 = getitem_38 = None
        mul_570: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_150, rsqrt_5);  sub_150 = None
        mul_571: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_570, primals_36)
        add_607: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_571, primals_37);  mul_571 = primals_37 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_61: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_39, torch.float16);  primals_39 = None
        convert_element_type_62: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_38, torch.float16);  primals_38 = None
        convert_element_type_63: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_607, torch.float16);  add_607 = None
        view_44: "f16[1024*s77, 768]" = torch.ops.aten.view.default(convert_element_type_63, [mul_15, 768]);  convert_element_type_63 = None
        permute_34: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_62, [1, 0]);  convert_element_type_62 = None
        addmm_7: "f16[1024*s77, 1536]" = torch.ops.aten.addmm.default(convert_element_type_61, view_44, permute_34);  convert_element_type_61 = None
        view_45: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(addmm_7, [primals_1, 1024, 1536]);  addmm_7 = None
        convert_element_type_67: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        mul_591: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.5)
        mul_592: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.7071067811865476)
        erf_2: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_592);  mul_592 = None
        add_634: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_593: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_591, add_634);  mul_591 = None
        convert_element_type_68: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_593, torch.float16);  mul_593 = None
        view_46: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_68, [mul_15, 1536]);  convert_element_type_68 = None
        mm_90: "f16[768, 1536]" = torch.ops.aten.mm.default(permute_358, view_46);  permute_358 = view_46 = None
        sum_111: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_378, [0], True, dtype = torch.float32);  view_378 = None
        view_379: "f32[768]" = torch.ops.aten.view.default(sum_111, [768]);  sum_111 = None
        view_380: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(mm_89, [primals_1, 1024, 1536]);  mm_89 = None
        convert_element_type_648: "f32[768, 1536]" = torch.ops.prims.convert_element_type.default(mm_90, torch.float32);  mm_90 = None
        convert_element_type_default_12: "f32[768]" = torch.ops.prims.convert_element_type.default(view_379, torch.float32);  view_379 = None
        convert_element_type_650: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_380, torch.float32);  view_380 = None
        mul_2620: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(add_634, 0.5);  add_634 = None
        mul_2621: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_67, convert_element_type_67)
        mul_2622: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_2621, -0.5);  mul_2621 = None
        exp_9: "f32[s77, 1024, 1536]" = torch.ops.aten.exp.default(mul_2622);  mul_2622 = None
        mul_2623: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
        mul_2624: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_67, mul_2623);  convert_element_type_67 = mul_2623 = None
        add_2730: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(mul_2620, mul_2624);  mul_2620 = mul_2624 = None
        mul_2625: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_650, add_2730);  convert_element_type_650 = add_2730 = None
        convert_element_type_652: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_2625, torch.float16);  mul_2625 = None
        view_381: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_652, [mul_15, 1536]);  convert_element_type_652 = None
        permute_361: "f16[1536, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        mm_91: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_381, permute_361);  permute_361 = None
        permute_362: "f16[1536, 1024*s77]" = torch.ops.aten.permute.default(view_381, [1, 0])
        mm_92: "f16[1536, 768]" = torch.ops.aten.mm.default(permute_362, view_44);  permute_362 = view_44 = None
        sum_112: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_381, [0], True, dtype = torch.float32);  view_381 = None
        view_382: "f32[1536]" = torch.ops.aten.view.default(sum_112, [1536]);  sum_112 = None
        view_383: "f16[s77, 1024, 768]" = torch.ops.aten.view.default(mm_91, [primals_1, 1024, 768]);  mm_91 = None
        convert_element_type_658: "f32[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_383, torch.float32);  view_383 = None
        convert_element_type_659: "f32[1536, 768]" = torch.ops.prims.convert_element_type.default(mm_92, torch.float32);  mm_92 = None
        convert_element_type_default_11: "f32[1536]" = torch.ops.prims.convert_element_type.default(view_382, torch.float32);  view_382 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        mul_2627: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_658, primals_36);  primals_36 = None
        mul_2628: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2627, 768)
        sum_113: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2627, [2], True)
        mul_2629: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2627, mul_570);  mul_2627 = None
        sum_114: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2629, [2], True);  mul_2629 = None
        mul_2630: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_570, sum_114);  sum_114 = None
        sub_723: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2628, sum_113);  mul_2628 = sum_113 = None
        sub_724: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_723, mul_2630);  sub_723 = mul_2630 = None
        div_18: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        mul_2631: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_724);  div_18 = sub_724 = None
        mul_2632: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_658, mul_570);  mul_570 = None
        sum_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2632, [0, 1]);  mul_2632 = None
        sum_116: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_658, [0, 1]);  convert_element_type_658 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_2731: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2728, mul_2631);  add_2728 = mul_2631 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        convert_element_type_661: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2731, torch.float16)
        permute_365: "f16[1024, s77, 768]" = torch.ops.aten.permute.default(convert_element_type_661, [1, 0, 2]);  convert_element_type_661 = None
        clone_57: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(permute_365, memory_format = torch.contiguous_format);  permute_365 = None
        view_384: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_57, [mul_15, 768]);  clone_57 = None
        convert_element_type_57: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_34, torch.float16);  primals_34 = None
        permute_32: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_57, [1, 0]);  convert_element_type_57 = None
        permute_366: "f16[768, 768]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_93: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_384, permute_366);  permute_366 = None
        permute_367: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_384, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_449, [2], correction = 0, keepdim = True)
        getitem_26: "f32[s77, 1024, 1]" = var_mean_4[0]
        getitem_27: "f32[s77, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
        add_454: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_4: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_454);  add_454 = None
        sub_113: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_449, getitem_27);  add_449 = getitem_27 = None
        mul_426: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_4);  sub_113 = None
        mul_427: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_426, primals_30)
        add_455: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_427, primals_31);  mul_427 = primals_31 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_25: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_455, [1, 0, 2]);  add_455 = None
        convert_element_type_51: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_32, torch.float16);  primals_32 = None
        convert_element_type_52: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_33, torch.float16);  primals_33 = None
        convert_element_type_53: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_25, torch.float16);  permute_25 = None
        permute_26: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_52, [1, 0]);  convert_element_type_52 = None
        clone_6: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_53, memory_format = torch.contiguous_format);  convert_element_type_53 = None
        view_33: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_6, [mul_15, 768]);  clone_6 = None
        mm_3: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_33, permute_26)
        view_34: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(mm_3, [1024, primals_1, 2304]);  mm_3 = None
        add_488: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_34, convert_element_type_51);  view_34 = convert_element_type_51 = None
        view_35: "f16[1024, s77, 3, 768]" = torch.ops.aten.view.default(add_488, [1024, primals_1, 3, 768]);  add_488 = None
        unsqueeze_8: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_35, 0);  view_35 = None
        permute_27: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_8, [3, 1, 2, 0, 4]);  unsqueeze_8 = None
        squeeze_2: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_27, -2);  permute_27 = None
        clone_7: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
        select_6: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_7, 0, 0)
        select_7: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_7, 0, 1)
        select_8: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_7, 0, 2);  clone_7 = None
        view_36: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_6, [1024, mul_127, 96]);  select_6 = None
        permute_28: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_36, [1, 0, 2]);  view_36 = None
        view_37: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_7, [1024, mul_127, 96]);  select_7 = None
        permute_29: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_37, [1, 0, 2]);  view_37 = None
        view_38: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_8, [1024, mul_127, 96]);  select_8 = None
        permute_30: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_38, [1, 0, 2]);  view_38 = None
        view_39: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_28, [primals_1, 8, 1024, 96]);  permute_28 = None
        view_40: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_29, [primals_1, 8, 1024, 96]);  permute_29 = None
        view_41: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_30, [primals_1, 8, 1024, 96]);  permute_30 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_2 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_39, view_40, view_41, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_2);  bwd_rng_state_2 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_28: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_2[0]
        getitem_29: "f32[s77, 8, 1024]" = graphsafe_run_with_rng_state_2[1]
        getitem_34: "u64[2]" = graphsafe_run_with_rng_state_2[6]
        getitem_35: "u64[]" = graphsafe_run_with_rng_state_2[7];  graphsafe_run_with_rng_state_2 = None
        permute_31: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_28, [2, 0, 1, 3])
        view_42: "f16[1024*s77, 768]" = torch.ops.aten.view.default(permute_31, [mul_15, 768]);  permute_31 = None
        mm_94: "f16[768, 768]" = torch.ops.aten.mm.default(permute_367, view_42);  permute_367 = view_42 = None
        sum_117: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_384, [0], True, dtype = torch.float32);  view_384 = None
        view_385: "f32[768]" = torch.ops.aten.view.default(sum_117, [768]);  sum_117 = None
        convert_element_type_667: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_94, torch.float32);  mm_94 = None
        convert_element_type_default_10: "f32[768]" = torch.ops.prims.convert_element_type.default(view_385, torch.float32);  view_385 = None
        view_386: "f16[1024, s77, 8, 96]" = torch.ops.aten.view.default(mm_93, [1024, primals_1, 8, 96]);  mm_93 = None
        permute_370: "f16[s77, 8, 1024, 96]" = torch.ops.aten.permute.default(view_386, [1, 2, 0, 3]);  view_386 = None
        _scaled_dot_product_flash_attention_backward_9 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_370, view_39, view_40, view_41, getitem_28, getitem_29, None, None, 1024, 1024, 0.2, False, getitem_34, getitem_35, scale = 0.10206207261596577);  permute_370 = view_39 = view_40 = view_41 = getitem_28 = getitem_29 = getitem_34 = getitem_35 = None
        getitem_183: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_9[0]
        getitem_184: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_9[1]
        getitem_185: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_9[2];  _scaled_dot_product_flash_attention_backward_9 = None
        view_387: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_185, [mul_127, 1024, 96]);  getitem_185 = None
        view_388: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_184, [mul_127, 1024, 96]);  getitem_184 = None
        view_389: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_183, [mul_127, 1024, 96]);  getitem_183 = None
        permute_371: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_387, [1, 0, 2]);  view_387 = None
        view_390: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_371, [1024, primals_1, 768]);  permute_371 = None
        permute_372: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_388, [1, 0, 2]);  view_388 = None
        view_391: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_372, [1024, primals_1, 768]);  permute_372 = None
        permute_373: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_389, [1, 0, 2]);  view_389 = None
        view_392: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_373, [1024, primals_1, 768]);  permute_373 = None
        select_scatter_27: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_390, 0, 2);  view_390 = None
        select_scatter_28: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_391, 0, 1);  view_391 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2732: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(select_scatter_27, select_scatter_28);  select_scatter_27 = select_scatter_28 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        select_scatter_29: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_392, 0, 0);  view_392 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2733: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(add_2732, select_scatter_29);  add_2732 = select_scatter_29 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        unsqueeze_39: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.unsqueeze.default(add_2733, 3);  add_2733 = None
        permute_374: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_39, [3, 1, 2, 0, 4]);  unsqueeze_39 = None
        squeeze_21: "f16[1024, s77, 3, 768]" = torch.ops.aten.squeeze.dim(permute_374, 0);  permute_374 = None
        clone_58: "f16[1024, s77, 3, 768]" = torch.ops.aten.clone.default(squeeze_21, memory_format = torch.contiguous_format);  squeeze_21 = None
        view_393: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(clone_58, [1024, primals_1, 2304]);  clone_58 = None
        sum_118: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_393, [0, 1], True, dtype = torch.float32)
        view_394: "f32[2304]" = torch.ops.aten.view.default(sum_118, [2304]);  sum_118 = None
        view_395: "f16[1024*s77, 2304]" = torch.ops.aten.view.default(view_393, [mul_15, 2304]);  view_393 = None
        permute_375: "f16[2304, 1024*s77]" = torch.ops.aten.permute.default(view_395, [1, 0])
        mm_95: "f16[2304, 768]" = torch.ops.aten.mm.default(permute_375, view_33);  permute_375 = view_33 = None
        permute_377: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        mm_96: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_395, permute_377);  view_395 = permute_377 = None
        view_396: "f16[1024, s77, 768]" = torch.ops.aten.view.default(mm_96, [1024, primals_1, 768]);  mm_96 = None
        convert_element_type_674: "f32[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(view_396, torch.float32);  view_396 = None
        convert_element_type_675: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_default_9: "f32[2304]" = torch.ops.prims.convert_element_type.default(view_394, torch.float32);  view_394 = None
        permute_379: "f32[s77, 1024, 768]" = torch.ops.aten.permute.default(convert_element_type_674, [1, 0, 2]);  convert_element_type_674 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        mul_2634: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_379, primals_30);  primals_30 = None
        mul_2635: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2634, 768)
        sum_119: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2634, [2], True)
        mul_2636: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2634, mul_426);  mul_2634 = None
        sum_120: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2636, [2], True);  mul_2636 = None
        mul_2637: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_426, sum_120);  sum_120 = None
        sub_726: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2635, sum_119);  mul_2635 = sum_119 = None
        sub_727: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_726, mul_2637);  sub_726 = mul_2637 = None
        div_19: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        mul_2638: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_727);  div_19 = sub_727 = None
        mul_2639: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_379, mul_426);  mul_426 = None
        sum_121: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2639, [0, 1]);  mul_2639 = None
        sum_122: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_379, [0, 1]);  permute_379 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        add_2734: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2731, mul_2638);  add_2731 = mul_2638 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        convert_element_type_677: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2734, torch.float16)
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_10: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default_62: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_10, torch.float16);  inductor_random_default_10 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_11: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_62, 0.2);  convert_element_type_default_62 = None
        convert_element_type_678: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_11, torch.float16);  gt_11 = None
        mul_2640: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_678, 1.25);  convert_element_type_678 = None
        mul_2641: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_677, mul_2640);  convert_element_type_677 = mul_2640 = None
        view_397: "f16[1024*s77, 768]" = torch.ops.aten.view.default(mul_2641, [mul_15, 768]);  mul_2641 = None
        convert_element_type_47: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_28, torch.float16);  primals_28 = None
        permute_24: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_47, [1, 0]);  convert_element_type_47 = None
        permute_380: "f16[768, 1536]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_97: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_397, permute_380);  permute_380 = None
        permute_381: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_397, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_3 = torch.ops.aten.var_mean.correction(add_393, [2], correction = 0, keepdim = True)
        getitem_24: "f32[s77, 1024, 1]" = var_mean_3[0]
        getitem_25: "f32[s77, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
        add_398: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_3: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_398);  add_398 = None
        sub_98: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_393, getitem_25);  add_393 = getitem_25 = None
        mul_380: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_3);  sub_98 = None
        mul_381: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_380, primals_24)
        add_399: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_381, primals_25);  mul_381 = primals_25 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_38: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_27, torch.float16);  primals_27 = None
        convert_element_type_39: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_26, torch.float16);  primals_26 = None
        convert_element_type_40: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_399, torch.float16);  add_399 = None
        view_29: "f16[1024*s77, 768]" = torch.ops.aten.view.default(convert_element_type_40, [mul_15, 768]);  convert_element_type_40 = None
        permute_23: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_39, [1, 0]);  convert_element_type_39 = None
        addmm_4: "f16[1024*s77, 1536]" = torch.ops.aten.addmm.default(convert_element_type_38, view_29, permute_23);  convert_element_type_38 = None
        view_30: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(addmm_4, [primals_1, 1024, 1536]);  addmm_4 = None
        convert_element_type_44: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_30, torch.float32);  view_30 = None
        mul_401: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 0.5)
        mul_402: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 0.7071067811865476)
        erf_1: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_402);  mul_402 = None
        add_426: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_403: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_401, add_426);  mul_401 = None
        convert_element_type_45: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_403, torch.float16);  mul_403 = None
        view_31: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_45, [mul_15, 1536]);  convert_element_type_45 = None
        mm_98: "f16[768, 1536]" = torch.ops.aten.mm.default(permute_381, view_31);  permute_381 = view_31 = None
        sum_123: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_397, [0], True, dtype = torch.float32);  view_397 = None
        view_398: "f32[768]" = torch.ops.aten.view.default(sum_123, [768]);  sum_123 = None
        view_399: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(mm_97, [primals_1, 1024, 1536]);  mm_97 = None
        convert_element_type_684: "f32[768, 1536]" = torch.ops.prims.convert_element_type.default(mm_98, torch.float32);  mm_98 = None
        convert_element_type_default_8: "f32[768]" = torch.ops.prims.convert_element_type.default(view_398, torch.float32);  view_398 = None
        convert_element_type_686: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_399, torch.float32);  view_399 = None
        mul_2643: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(add_426, 0.5);  add_426 = None
        mul_2644: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_44, convert_element_type_44)
        mul_2645: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_2644, -0.5);  mul_2644 = None
        exp_10: "f32[s77, 1024, 1536]" = torch.ops.aten.exp.default(mul_2645);  mul_2645 = None
        mul_2646: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
        mul_2647: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_44, mul_2646);  convert_element_type_44 = mul_2646 = None
        add_2736: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(mul_2643, mul_2647);  mul_2643 = mul_2647 = None
        mul_2648: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_686, add_2736);  convert_element_type_686 = add_2736 = None
        convert_element_type_688: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_2648, torch.float16);  mul_2648 = None
        view_400: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_688, [mul_15, 1536]);  convert_element_type_688 = None
        permute_384: "f16[1536, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_99: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_400, permute_384);  permute_384 = None
        permute_385: "f16[1536, 1024*s77]" = torch.ops.aten.permute.default(view_400, [1, 0])
        mm_100: "f16[1536, 768]" = torch.ops.aten.mm.default(permute_385, view_29);  permute_385 = view_29 = None
        sum_124: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_400, [0], True, dtype = torch.float32);  view_400 = None
        view_401: "f32[1536]" = torch.ops.aten.view.default(sum_124, [1536]);  sum_124 = None
        view_402: "f16[s77, 1024, 768]" = torch.ops.aten.view.default(mm_99, [primals_1, 1024, 768]);  mm_99 = None
        convert_element_type_694: "f32[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_402, torch.float32);  view_402 = None
        convert_element_type_695: "f32[1536, 768]" = torch.ops.prims.convert_element_type.default(mm_100, torch.float32);  mm_100 = None
        convert_element_type_default_7: "f32[1536]" = torch.ops.prims.convert_element_type.default(view_401, torch.float32);  view_401 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        mul_2650: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_694, primals_24);  primals_24 = None
        mul_2651: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2650, 768)
        sum_125: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2650, [2], True)
        mul_2652: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2650, mul_380);  mul_2650 = None
        sum_126: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2652, [2], True);  mul_2652 = None
        mul_2653: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_380, sum_126);  sum_126 = None
        sub_729: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2651, sum_125);  mul_2651 = sum_125 = None
        sub_730: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_729, mul_2653);  sub_729 = mul_2653 = None
        div_20: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        mul_2654: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_730);  div_20 = sub_730 = None
        mul_2655: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_694, mul_380);  mul_380 = None
        sum_127: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2655, [0, 1]);  mul_2655 = None
        sum_128: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_694, [0, 1]);  convert_element_type_694 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_2737: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2734, mul_2654);  add_2734 = mul_2654 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        convert_element_type_697: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2737, torch.float16)
        permute_388: "f16[1024, s77, 768]" = torch.ops.aten.permute.default(convert_element_type_697, [1, 0, 2]);  convert_element_type_697 = None
        clone_60: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(permute_388, memory_format = torch.contiguous_format);  permute_388 = None
        view_403: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_60, [mul_15, 768]);  clone_60 = None
        convert_element_type_34: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_22, torch.float16);  primals_22 = None
        permute_21: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_34, [1, 0]);  convert_element_type_34 = None
        permute_389: "f16[768, 768]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_101: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_403, permute_389);  permute_389 = None
        permute_390: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_403, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_241, [2], correction = 0, keepdim = True)
        getitem_13: "f32[s77, 1024, 1]" = var_mean_2[0]
        getitem_14: "f32[s77, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
        add_246: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_13, 1e-05);  getitem_13 = None
        rsqrt_2: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_246);  add_246 = None
        sub_61: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_241, getitem_14);  add_241 = getitem_14 = None
        mul_236: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_2);  sub_61 = None
        mul_237: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_236, primals_18)
        add_247: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_237, primals_19);  mul_237 = primals_19 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_14: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_247, [1, 0, 2]);  add_247 = None
        convert_element_type_28: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_20, torch.float16);  primals_20 = None
        convert_element_type_29: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_21, torch.float16);  primals_21 = None
        convert_element_type_30: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_14, torch.float16);  permute_14 = None
        permute_15: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_29, [1, 0]);  convert_element_type_29 = None
        clone_4: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_30, memory_format = torch.contiguous_format);  convert_element_type_30 = None
        view_18: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_4, [mul_15, 768]);  clone_4 = None
        mm_2: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_18, permute_15)
        view_19: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(mm_2, [1024, primals_1, 2304]);  mm_2 = None
        add_280: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_19, convert_element_type_28);  view_19 = convert_element_type_28 = None
        view_20: "f16[1024, s77, 3, 768]" = torch.ops.aten.view.default(add_280, [1024, primals_1, 3, 768]);  add_280 = None
        unsqueeze_7: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_20, 0);  view_20 = None
        permute_16: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_7, [3, 1, 2, 0, 4]);  unsqueeze_7 = None
        squeeze_1: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_16, -2);  permute_16 = None
        clone_5: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        select_3: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_5, 0, 0)
        select_4: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_5, 0, 1)
        select_5: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_5, 0, 2);  clone_5 = None
        view_21: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_3, [1024, mul_127, 96]);  select_3 = None
        permute_17: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_21, [1, 0, 2]);  view_21 = None
        view_22: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_4, [1024, mul_127, 96]);  select_4 = None
        permute_18: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_22, [1, 0, 2]);  view_22 = None
        view_23: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_5, [1024, mul_127, 96]);  select_5 = None
        permute_19: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_23, [1, 0, 2]);  view_23 = None
        view_24: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_17, [primals_1, 8, 1024, 96]);  permute_17 = None
        view_25: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_18, [primals_1, 8, 1024, 96]);  permute_18 = None
        view_26: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_19, [primals_1, 8, 1024, 96]);  permute_19 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_1 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_24, view_25, view_26, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_1);  bwd_rng_state_1 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_15: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_1[0]
        getitem_16: "f32[s77, 8, 1024]" = graphsafe_run_with_rng_state_1[1]
        getitem_21: "u64[2]" = graphsafe_run_with_rng_state_1[6]
        getitem_22: "u64[]" = graphsafe_run_with_rng_state_1[7];  graphsafe_run_with_rng_state_1 = None
        permute_20: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_15, [2, 0, 1, 3])
        view_27: "f16[1024*s77, 768]" = torch.ops.aten.view.default(permute_20, [mul_15, 768]);  permute_20 = None
        mm_102: "f16[768, 768]" = torch.ops.aten.mm.default(permute_390, view_27);  permute_390 = view_27 = None
        sum_129: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_403, [0], True, dtype = torch.float32);  view_403 = None
        view_404: "f32[768]" = torch.ops.aten.view.default(sum_129, [768]);  sum_129 = None
        convert_element_type_703: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_102, torch.float32);  mm_102 = None
        convert_element_type_default_6: "f32[768]" = torch.ops.prims.convert_element_type.default(view_404, torch.float32);  view_404 = None
        view_405: "f16[1024, s77, 8, 96]" = torch.ops.aten.view.default(mm_101, [1024, primals_1, 8, 96]);  mm_101 = None
        permute_393: "f16[s77, 8, 1024, 96]" = torch.ops.aten.permute.default(view_405, [1, 2, 0, 3]);  view_405 = None
        _scaled_dot_product_flash_attention_backward_10 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_393, view_24, view_25, view_26, getitem_15, getitem_16, None, None, 1024, 1024, 0.2, False, getitem_21, getitem_22, scale = 0.10206207261596577);  permute_393 = view_24 = view_25 = view_26 = getitem_15 = getitem_16 = getitem_21 = getitem_22 = None
        getitem_186: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_10[0]
        getitem_187: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_10[1]
        getitem_188: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_10[2];  _scaled_dot_product_flash_attention_backward_10 = None
        view_406: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_188, [mul_127, 1024, 96]);  getitem_188 = None
        view_407: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_187, [mul_127, 1024, 96]);  getitem_187 = None
        view_408: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_186, [mul_127, 1024, 96]);  getitem_186 = None
        permute_394: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_406, [1, 0, 2]);  view_406 = None
        view_409: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_394, [1024, primals_1, 768]);  permute_394 = None
        permute_395: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_407, [1, 0, 2]);  view_407 = None
        view_410: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_395, [1024, primals_1, 768]);  permute_395 = None
        permute_396: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_408, [1, 0, 2]);  view_408 = None
        view_411: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_396, [1024, primals_1, 768]);  permute_396 = None
        select_scatter_30: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_409, 0, 2);  view_409 = None
        select_scatter_31: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_410, 0, 1);  view_410 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2738: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(select_scatter_30, select_scatter_31);  select_scatter_30 = select_scatter_31 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        select_scatter_32: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_411, 0, 0);  view_411 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2739: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(add_2738, select_scatter_32);  add_2738 = select_scatter_32 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        unsqueeze_40: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.unsqueeze.default(add_2739, 3);  add_2739 = None
        permute_397: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_40, [3, 1, 2, 0, 4]);  unsqueeze_40 = None
        squeeze_22: "f16[1024, s77, 3, 768]" = torch.ops.aten.squeeze.dim(permute_397, 0);  permute_397 = None
        clone_61: "f16[1024, s77, 3, 768]" = torch.ops.aten.clone.default(squeeze_22, memory_format = torch.contiguous_format);  squeeze_22 = None
        view_412: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(clone_61, [1024, primals_1, 2304]);  clone_61 = None
        sum_130: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_412, [0, 1], True, dtype = torch.float32)
        view_413: "f32[2304]" = torch.ops.aten.view.default(sum_130, [2304]);  sum_130 = None
        view_414: "f16[1024*s77, 2304]" = torch.ops.aten.view.default(view_412, [mul_15, 2304]);  view_412 = None
        permute_398: "f16[2304, 1024*s77]" = torch.ops.aten.permute.default(view_414, [1, 0])
        mm_103: "f16[2304, 768]" = torch.ops.aten.mm.default(permute_398, view_18);  permute_398 = view_18 = None
        permute_400: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        mm_104: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_414, permute_400);  view_414 = permute_400 = None
        view_415: "f16[1024, s77, 768]" = torch.ops.aten.view.default(mm_104, [1024, primals_1, 768]);  mm_104 = None
        convert_element_type_710: "f32[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(view_415, torch.float32);  view_415 = None
        convert_element_type_711: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        convert_element_type_default_5: "f32[2304]" = torch.ops.prims.convert_element_type.default(view_413, torch.float32);  view_413 = None
        permute_402: "f32[s77, 1024, 768]" = torch.ops.aten.permute.default(convert_element_type_710, [1, 0, 2]);  convert_element_type_710 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        mul_2657: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_402, primals_18);  primals_18 = None
        mul_2658: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2657, 768)
        sum_131: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2657, [2], True)
        mul_2659: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2657, mul_236);  mul_2657 = None
        sum_132: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2659, [2], True);  mul_2659 = None
        mul_2660: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_236, sum_132);  sum_132 = None
        sub_732: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2658, sum_131);  mul_2658 = sum_131 = None
        sub_733: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_732, mul_2660);  sub_732 = mul_2660 = None
        div_21: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        mul_2661: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_733);  div_21 = sub_733 = None
        mul_2662: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_402, mul_236);  mul_236 = None
        sum_133: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2662, [0, 1]);  mul_2662 = None
        sum_134: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_402, [0, 1]);  permute_402 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        add_2740: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2737, mul_2661);  add_2737 = mul_2661 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        convert_element_type_713: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2740, torch.float16)
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default_11: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default_63: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_11, torch.float16);  inductor_random_default_11 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_7: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_63, 0.2);  convert_element_type_default_63 = None
        convert_element_type_714: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_7, torch.float16);  gt_7 = None
        mul_2663: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_714, 1.25);  convert_element_type_714 = None
        mul_2664: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_713, mul_2663);  convert_element_type_713 = mul_2663 = None
        view_416: "f16[1024*s77, 768]" = torch.ops.aten.view.default(mul_2664, [mul_15, 768]);  mul_2664 = None
        convert_element_type_24: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_16, torch.float16);  primals_16 = None
        permute_13: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_24, [1, 0]);  convert_element_type_24 = None
        permute_403: "f16[768, 1536]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_105: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_416, permute_403);  permute_403 = None
        permute_404: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_416, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_1 = torch.ops.aten.var_mean.correction(add_185, [2], correction = 0, keepdim = True)
        getitem_11: "f32[s77, 1024, 1]" = var_mean_1[0]
        getitem_12: "f32[s77, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
        add_190: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_1: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
        sub_46: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_185, getitem_12);  add_185 = getitem_12 = None
        mul_190: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_1);  sub_46 = None
        mul_191: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_190, primals_12)
        add_191: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_191, primals_13);  mul_191 = primals_13 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_15: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_15, torch.float16);  primals_15 = None
        convert_element_type_16: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_14, torch.float16);  primals_14 = None
        convert_element_type_17: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_191, torch.float16);  add_191 = None
        view_14: "f16[1024*s77, 768]" = torch.ops.aten.view.default(convert_element_type_17, [mul_15, 768]);  convert_element_type_17 = None
        permute_12: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_16, [1, 0]);  convert_element_type_16 = None
        addmm_1: "f16[1024*s77, 1536]" = torch.ops.aten.addmm.default(convert_element_type_15, view_14, permute_12);  convert_element_type_15 = None
        view_15: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(addmm_1, [primals_1, 1024, 1536]);  addmm_1 = None
        convert_element_type_21: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_15, torch.float32);  view_15 = None
        mul_211: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.5)
        mul_212: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.7071067811865476)
        erf: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_212);  mul_212 = None
        add_218: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_213: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_211, add_218);  mul_211 = None
        convert_element_type_22: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_213, torch.float16);  mul_213 = None
        view_16: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_22, [mul_15, 1536]);  convert_element_type_22 = None
        mm_106: "f16[768, 1536]" = torch.ops.aten.mm.default(permute_404, view_16);  permute_404 = view_16 = None
        sum_135: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_416, [0], True, dtype = torch.float32);  view_416 = None
        view_417: "f32[768]" = torch.ops.aten.view.default(sum_135, [768]);  sum_135 = None
        view_418: "f16[s77, 1024, 1536]" = torch.ops.aten.view.default(mm_105, [primals_1, 1024, 1536]);  mm_105 = None
        convert_element_type_720: "f32[768, 1536]" = torch.ops.prims.convert_element_type.default(mm_106, torch.float32);  mm_106 = None
        convert_element_type_default_4: "f32[768]" = torch.ops.prims.convert_element_type.default(view_417, torch.float32);  view_417 = None
        convert_element_type_722: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_418, torch.float32);  view_418 = None
        mul_2666: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(add_218, 0.5);  add_218 = None
        mul_2667: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_21, convert_element_type_21)
        mul_2668: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_2667, -0.5);  mul_2667 = None
        exp_11: "f32[s77, 1024, 1536]" = torch.ops.aten.exp.default(mul_2668);  mul_2668 = None
        mul_2669: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
        mul_2670: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_21, mul_2669);  convert_element_type_21 = mul_2669 = None
        add_2742: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(mul_2666, mul_2670);  mul_2666 = mul_2670 = None
        mul_2671: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_722, add_2742);  convert_element_type_722 = add_2742 = None
        convert_element_type_724: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_2671, torch.float16);  mul_2671 = None
        view_419: "f16[1024*s77, 1536]" = torch.ops.aten.view.default(convert_element_type_724, [mul_15, 1536]);  convert_element_type_724 = None
        permute_407: "f16[1536, 768]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_107: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_419, permute_407);  permute_407 = None
        permute_408: "f16[1536, 1024*s77]" = torch.ops.aten.permute.default(view_419, [1, 0])
        mm_108: "f16[1536, 768]" = torch.ops.aten.mm.default(permute_408, view_14);  permute_408 = view_14 = None
        sum_136: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_419, [0], True, dtype = torch.float32);  view_419 = None
        view_420: "f32[1536]" = torch.ops.aten.view.default(sum_136, [1536]);  sum_136 = None
        view_421: "f16[s77, 1024, 768]" = torch.ops.aten.view.default(mm_107, [primals_1, 1024, 768]);  mm_107 = None
        convert_element_type_730: "f32[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_421, torch.float32);  view_421 = None
        convert_element_type_731: "f32[1536, 768]" = torch.ops.prims.convert_element_type.default(mm_108, torch.float32);  mm_108 = None
        convert_element_type_default_3: "f32[1536]" = torch.ops.prims.convert_element_type.default(view_420, torch.float32);  view_420 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        mul_2673: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_730, primals_12);  primals_12 = None
        mul_2674: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2673, 768)
        sum_137: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2673, [2], True)
        mul_2675: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2673, mul_190);  mul_2673 = None
        sum_138: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2675, [2], True);  mul_2675 = None
        mul_2676: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_190, sum_138);  sum_138 = None
        sub_735: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2674, sum_137);  mul_2674 = sum_137 = None
        sub_736: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_735, mul_2676);  sub_735 = mul_2676 = None
        div_22: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        mul_2677: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_736);  div_22 = sub_736 = None
        mul_2678: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_730, mul_190);  mul_190 = None
        sum_139: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2678, [0, 1]);  mul_2678 = None
        sum_140: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_730, [0, 1]);  convert_element_type_730 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_2743: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2740, mul_2677);  add_2740 = mul_2677 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        convert_element_type_733: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2743, torch.float16)
        permute_411: "f16[1024, s77, 768]" = torch.ops.aten.permute.default(convert_element_type_733, [1, 0, 2]);  convert_element_type_733 = None
        clone_63: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(permute_411, memory_format = torch.contiguous_format);  permute_411 = None
        view_422: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_63, [mul_15, 768]);  clone_63 = None
        convert_element_type_11: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_10, torch.float16);  primals_10 = None
        permute_10: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_11, [1, 0]);  convert_element_type_11 = None
        permute_412: "f16[768, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_109: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_422, permute_412);  permute_412 = None
        permute_413: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_422, [1, 0])
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:41 in forward, code: x = self.embed_layer(x)
        convert_element_type: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_4, torch.float16);  primals_4 = None
        view_2: "f16[s77, 1024, 768]" = torch.ops.aten.view.default(mm, [primals_1, 1024, 768]);  mm = None
        add_28: "f16[s77, 1024, 768]" = torch.ops.aten.add.Tensor(view_2, convert_element_type);  view_2 = convert_element_type = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:74 in forward, code: x = x + self.position_embed
        add_33: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_28, primals_5);  add_28 = primals_5 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem: "f32[s77, 1024, 1]" = var_mean[0]
        getitem_1: "f32[s77, 1024, 1]" = var_mean[1];  var_mean = None
        add_38: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_9: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_33, getitem_1);  add_33 = getitem_1 = None
        mul_46: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt);  sub_9 = None
        mul_47: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_46, primals_6)
        add_39: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_47, primals_7);  mul_47 = primals_7 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_3: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_39, [1, 0, 2]);  add_39 = None
        convert_element_type_5: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_8, torch.float16);  primals_8 = None
        convert_element_type_6: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_9, torch.float16);  primals_9 = None
        convert_element_type_7: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_3, torch.float16);  permute_3 = None
        permute_4: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_6, [1, 0]);  convert_element_type_6 = None
        clone_2: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_7, memory_format = torch.contiguous_format);  convert_element_type_7 = None
        view_3: "f16[1024*s77, 768]" = torch.ops.aten.view.default(clone_2, [mul_15, 768]);  clone_2 = None
        mm_1: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_3, permute_4)
        view_4: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(mm_1, [1024, primals_1, 2304]);  mm_1 = None
        add_72: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_4, convert_element_type_5);  view_4 = convert_element_type_5 = None
        view_5: "f16[1024, s77, 3, 768]" = torch.ops.aten.view.default(add_72, [1024, primals_1, 3, 768]);  add_72 = None
        unsqueeze_6: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_5, 0);  view_5 = None
        permute_5: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_6, [3, 1, 2, 0, 4]);  unsqueeze_6 = None
        squeeze: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_5, -2);  permute_5 = None
        clone_3: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        select: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_3, 0, 0)
        select_1: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_3, 0, 1)
        select_2: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_3, 0, 2);  clone_3 = None
        view_6: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select, [1024, mul_127, 96]);  select = None
        permute_6: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_6, [1, 0, 2]);  view_6 = None
        view_7: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_1, [1024, mul_127, 96]);  select_1 = None
        permute_7: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_7, [1, 0, 2]);  view_7 = None
        view_8: "f16[1024, 8*s77, 96]" = torch.ops.aten.view.default(select_2, [1024, mul_127, 96]);  select_2 = None
        permute_8: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_8, [1, 0, 2]);  view_8 = None
        view_9: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_6, [primals_1, 8, 1024, 96]);  permute_6 = None
        view_10: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_7, [primals_1, 8, 1024, 96]);  permute_7 = None
        view_11: "f16[s77, 8, 1024, 96]" = torch.ops.aten.view.default(permute_8, [primals_1, 8, 1024, 96]);  permute_8 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_9, view_10, view_11, 0.2, scale = 0.10206207261596577, rng_state = bwd_rng_state_0);  bwd_rng_state_0 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_2: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state[0]
        getitem_3: "f32[s77, 8, 1024]" = graphsafe_run_with_rng_state[1]
        getitem_8: "u64[2]" = graphsafe_run_with_rng_state[6]
        getitem_9: "u64[]" = graphsafe_run_with_rng_state[7];  graphsafe_run_with_rng_state = None
        permute_9: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_2, [2, 0, 1, 3])
        view_12: "f16[1024*s77, 768]" = torch.ops.aten.view.default(permute_9, [mul_15, 768]);  permute_9 = None
        mm_110: "f16[768, 768]" = torch.ops.aten.mm.default(permute_413, view_12);  permute_413 = view_12 = None
        sum_141: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True, dtype = torch.float32);  view_422 = None
        view_423: "f32[768]" = torch.ops.aten.view.default(sum_141, [768]);  sum_141 = None
        convert_element_type_739: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_110, torch.float32);  mm_110 = None
        convert_element_type_default_2: "f32[768]" = torch.ops.prims.convert_element_type.default(view_423, torch.float32);  view_423 = None
        view_424: "f16[1024, s77, 8, 96]" = torch.ops.aten.view.default(mm_109, [1024, primals_1, 8, 96]);  mm_109 = None
        permute_416: "f16[s77, 8, 1024, 96]" = torch.ops.aten.permute.default(view_424, [1, 2, 0, 3]);  view_424 = None
        _scaled_dot_product_flash_attention_backward_11 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_416, view_9, view_10, view_11, getitem_2, getitem_3, None, None, 1024, 1024, 0.2, False, getitem_8, getitem_9, scale = 0.10206207261596577);  permute_416 = view_9 = view_10 = view_11 = getitem_2 = getitem_3 = getitem_8 = getitem_9 = None
        getitem_189: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_11[0]
        getitem_190: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_11[1]
        getitem_191: "f16[s77, 8, 1024, 96]" = _scaled_dot_product_flash_attention_backward_11[2];  _scaled_dot_product_flash_attention_backward_11 = None
        view_425: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_191, [mul_127, 1024, 96]);  getitem_191 = None
        view_426: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_190, [mul_127, 1024, 96]);  getitem_190 = None
        view_427: "f16[8*s77, 1024, 96]" = torch.ops.aten.view.default(getitem_189, [mul_127, 1024, 96]);  getitem_189 = mul_127 = None
        permute_417: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_425, [1, 0, 2]);  view_425 = None
        view_428: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_417, [1024, primals_1, 768]);  permute_417 = None
        permute_418: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_426, [1, 0, 2]);  view_426 = None
        view_429: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_418, [1024, primals_1, 768]);  permute_418 = None
        permute_419: "f16[1024, 8*s77, 96]" = torch.ops.aten.permute.default(view_427, [1, 0, 2]);  view_427 = None
        view_430: "f16[1024, s77, 768]" = torch.ops.aten.view.default(permute_419, [1024, primals_1, 768]);  permute_419 = None
        select_scatter_33: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_428, 0, 2);  view_428 = None
        select_scatter_34: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_429, 0, 1);  view_429 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2744: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(select_scatter_33, select_scatter_34);  select_scatter_33 = select_scatter_34 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        select_scatter_35: "f16[3, 1024, s77, 768]" = torch.ops.aten.select_scatter.default(full_default_6, view_430, 0, 0);  full_default_6 = view_430 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        add_2745: "f16[3, 1024, s77, 768]" = torch.ops.aten.add.Tensor(add_2744, select_scatter_35);  add_2744 = select_scatter_35 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        unsqueeze_41: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.unsqueeze.default(add_2745, 3);  add_2745 = None
        permute_420: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.permute.default(unsqueeze_41, [3, 1, 2, 0, 4]);  unsqueeze_41 = None
        squeeze_23: "f16[1024, s77, 3, 768]" = torch.ops.aten.squeeze.dim(permute_420, 0);  permute_420 = None
        clone_64: "f16[1024, s77, 3, 768]" = torch.ops.aten.clone.default(squeeze_23, memory_format = torch.contiguous_format);  squeeze_23 = None
        view_431: "f16[1024, s77, 2304]" = torch.ops.aten.view.default(clone_64, [1024, primals_1, 2304]);  clone_64 = None
        sum_142: "f32[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_431, [0, 1], True, dtype = torch.float32)
        view_432: "f32[2304]" = torch.ops.aten.view.default(sum_142, [2304]);  sum_142 = None
        view_433: "f16[1024*s77, 2304]" = torch.ops.aten.view.default(view_431, [mul_15, 2304]);  view_431 = None
        permute_421: "f16[2304, 1024*s77]" = torch.ops.aten.permute.default(view_433, [1, 0])
        mm_111: "f16[2304, 768]" = torch.ops.aten.mm.default(permute_421, view_3);  permute_421 = view_3 = None
        permute_423: "f16[2304, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_112: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_433, permute_423);  view_433 = permute_423 = None
        view_434: "f16[1024, s77, 768]" = torch.ops.aten.view.default(mm_112, [1024, primals_1, 768]);  mm_112 = primals_1 = None
        convert_element_type_746: "f32[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(view_434, torch.float32);  view_434 = None
        convert_element_type_747: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_default_1: "f32[2304]" = torch.ops.prims.convert_element_type.default(view_432, torch.float32);  view_432 = None
        permute_425: "f32[s77, 1024, 768]" = torch.ops.aten.permute.default(convert_element_type_746, [1, 0, 2]);  convert_element_type_746 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        mul_2680: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_425, primals_6);  primals_6 = None
        mul_2681: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2680, 768)
        sum_143: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2680, [2], True)
        mul_2682: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2680, mul_46);  mul_2680 = None
        sum_144: "f32[s77, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2682, [2], True);  mul_2682 = None
        mul_2683: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_46, sum_144);  sum_144 = None
        sub_738: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2681, sum_143);  mul_2681 = sum_143 = None
        sub_739: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_738, mul_2683);  sub_738 = mul_2683 = None
        div_23: "f32[s77, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_2684: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_739);  div_23 = sub_739 = None
        mul_2685: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(permute_425, mul_46);  mul_46 = None
        sum_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2685, [0, 1]);  mul_2685 = None
        sum_146: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_425, [0, 1]);  permute_425 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        add_2746: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2743, mul_2684);  add_2743 = mul_2684 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:74 in forward, code: x = x + self.position_embed
        convert_element_type_749: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2746, torch.float16)
        sum_147: "f32[1, 1024, 768]" = torch.ops.aten.sum.dim_IntList(add_2746, [0], True, dtype = torch.float32);  add_2746 = None
        view_435: "f32[1024, 768]" = torch.ops.aten.view.default(sum_147, [1024, 768]);  sum_147 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:41 in forward, code: x = self.embed_layer(x)
        sum_148: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_749, [0, 1], True, dtype = torch.float32)
        view_436: "f32[768]" = torch.ops.aten.view.default(sum_148, [768]);  sum_148 = None
        view_437: "f16[1024*s77, 768]" = torch.ops.aten.view.default(convert_element_type_749, [mul_15, 768]);  convert_element_type_749 = mul_15 = None
        permute_426: "f16[768, 1024*s77]" = torch.ops.aten.permute.default(view_437, [1, 0]);  view_437 = None
        mm_113: "f16[768, 16]" = torch.ops.aten.mm.default(permute_426, view_1);  permute_426 = view_1 = None
        convert_element_type_753: "f32[768, 16]" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_default: "f32[768]" = torch.ops.prims.convert_element_type.default(view_436, torch.float32);  view_436 = None
        return (None, None, convert_element_type_753, convert_element_type_default, view_435, sum_145, sum_146, convert_element_type_default_1, convert_element_type_747, convert_element_type_739, convert_element_type_default_2, sum_139, sum_140, convert_element_type_731, convert_element_type_default_3, convert_element_type_720, convert_element_type_default_4, sum_133, sum_134, convert_element_type_default_5, convert_element_type_711, convert_element_type_703, convert_element_type_default_6, sum_127, sum_128, convert_element_type_695, convert_element_type_default_7, convert_element_type_684, convert_element_type_default_8, sum_121, sum_122, convert_element_type_default_9, convert_element_type_675, convert_element_type_667, convert_element_type_default_10, sum_115, sum_116, convert_element_type_659, convert_element_type_default_11, convert_element_type_648, convert_element_type_default_12, sum_109, sum_110, convert_element_type_default_13, convert_element_type_639, convert_element_type_631, convert_element_type_default_14, sum_103, sum_104, convert_element_type_623, convert_element_type_default_15, convert_element_type_612, convert_element_type_default_16, sum_97, sum_98, convert_element_type_default_17, convert_element_type_603, convert_element_type_595, convert_element_type_default_18, sum_91, sum_92, convert_element_type_587, convert_element_type_default_19, convert_element_type_576, convert_element_type_default_20, sum_85, sum_86, convert_element_type_default_21, convert_element_type_567, convert_element_type_559, convert_element_type_default_22, sum_79, sum_80, convert_element_type_551, convert_element_type_default_23, convert_element_type_540, convert_element_type_default_24, sum_73, sum_74, convert_element_type_default_25, convert_element_type_531, convert_element_type_523, convert_element_type_default_26, sum_67, sum_68, convert_element_type_515, convert_element_type_default_27, convert_element_type_504, convert_element_type_default_28, sum_61, sum_62, convert_element_type_default_29, convert_element_type_495, convert_element_type_487, convert_element_type_default_30, sum_55, sum_56, convert_element_type_479, convert_element_type_default_31, convert_element_type_468, convert_element_type_default_32, sum_49, sum_50, convert_element_type_default_33, convert_element_type_459, convert_element_type_451, convert_element_type_default_34, sum_43, sum_44, convert_element_type_443, convert_element_type_default_35, convert_element_type_432, convert_element_type_default_36, sum_37, sum_38, convert_element_type_default_37, convert_element_type_423, convert_element_type_415, convert_element_type_default_38, sum_31, sum_32, convert_element_type_407, convert_element_type_default_39, convert_element_type_396, convert_element_type_default_40, sum_25, sum_26, convert_element_type_default_41, convert_element_type_387, convert_element_type_379, convert_element_type_default_42, sum_19, sum_20, convert_element_type_371, convert_element_type_default_43, convert_element_type_360, convert_element_type_default_44, sum_13, sum_14, convert_element_type_default_45, convert_element_type_351, convert_element_type_343, convert_element_type_default_46, sum_7, sum_8, convert_element_type_335, convert_element_type_default_47, convert_element_type_324, convert_element_type_default_48, convert_element_type_315, convert_element_type_default_49, convert_element_type_307, convert_element_type_default_50)
        