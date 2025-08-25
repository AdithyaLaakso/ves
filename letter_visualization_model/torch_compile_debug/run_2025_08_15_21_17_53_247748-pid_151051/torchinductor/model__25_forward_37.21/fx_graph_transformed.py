class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(s77)", primals_2: "f32[s77, 1, 128, 128]", primals_3: "f32[768, 16]", primals_4: "f32[768]", primals_5: "f32[1024, 768]", primals_6: "f32[768]", primals_7: "f32[768]", primals_8: "f32[2304]", primals_9: "f32[2304, 768]", primals_10: "f32[768, 768]", primals_11: "f32[768]", primals_12: "f32[768]", primals_13: "f32[768]", primals_14: "f32[1536, 768]", primals_15: "f32[1536]", primals_16: "f32[768, 1536]", primals_17: "f32[768]", primals_18: "f32[768]", primals_19: "f32[768]", primals_20: "f32[2304]", primals_21: "f32[2304, 768]", primals_22: "f32[768, 768]", primals_23: "f32[768]", primals_24: "f32[768]", primals_25: "f32[768]", primals_26: "f32[1536, 768]", primals_27: "f32[1536]", primals_28: "f32[768, 1536]", primals_29: "f32[768]", primals_30: "f32[768]", primals_31: "f32[768]", primals_32: "f32[2304]", primals_33: "f32[2304, 768]", primals_34: "f32[768, 768]", primals_35: "f32[768]", primals_36: "f32[768]", primals_37: "f32[768]", primals_38: "f32[1536, 768]", primals_39: "f32[1536]", primals_40: "f32[768, 1536]", primals_41: "f32[768]", primals_42: "f32[768]", primals_43: "f32[768]", primals_44: "f32[2304]", primals_45: "f32[2304, 768]", primals_46: "f32[768, 768]", primals_47: "f32[768]", primals_48: "f32[768]", primals_49: "f32[768]", primals_50: "f32[1536, 768]", primals_51: "f32[1536]", primals_52: "f32[768, 1536]", primals_53: "f32[768]", primals_54: "f32[768]", primals_55: "f32[768]", primals_56: "f32[2304]", primals_57: "f32[2304, 768]", primals_58: "f32[768, 768]", primals_59: "f32[768]", primals_60: "f32[768]", primals_61: "f32[768]", primals_62: "f32[1536, 768]", primals_63: "f32[1536]", primals_64: "f32[768, 1536]", primals_65: "f32[768]", primals_66: "f32[768]", primals_67: "f32[768]", primals_68: "f32[2304]", primals_69: "f32[2304, 768]", primals_70: "f32[768, 768]", primals_71: "f32[768]", primals_72: "f32[768]", primals_73: "f32[768]", primals_74: "f32[1536, 768]", primals_75: "f32[1536]", primals_76: "f32[768, 1536]", primals_77: "f32[768]", primals_78: "f32[768]", primals_79: "f32[768]", primals_80: "f32[2304]", primals_81: "f32[2304, 768]", primals_82: "f32[768, 768]", primals_83: "f32[768]", primals_84: "f32[768]", primals_85: "f32[768]", primals_86: "f32[1536, 768]", primals_87: "f32[1536]", primals_88: "f32[768, 1536]", primals_89: "f32[768]", primals_90: "f32[768]", primals_91: "f32[768]", primals_92: "f32[2304]", primals_93: "f32[2304, 768]", primals_94: "f32[768, 768]", primals_95: "f32[768]", primals_96: "f32[768]", primals_97: "f32[768]", primals_98: "f32[1536, 768]", primals_99: "f32[1536]", primals_100: "f32[768, 1536]", primals_101: "f32[768]", primals_102: "f32[768]", primals_103: "f32[768]", primals_104: "f32[2304]", primals_105: "f32[2304, 768]", primals_106: "f32[768, 768]", primals_107: "f32[768]", primals_108: "f32[768]", primals_109: "f32[768]", primals_110: "f32[1536, 768]", primals_111: "f32[1536]", primals_112: "f32[768, 1536]", primals_113: "f32[768]", primals_114: "f32[768]", primals_115: "f32[768]", primals_116: "f32[2304]", primals_117: "f32[2304, 768]", primals_118: "f32[768, 768]", primals_119: "f32[768]", primals_120: "f32[768]", primals_121: "f32[768]", primals_122: "f32[1536, 768]", primals_123: "f32[1536]", primals_124: "f32[768, 1536]", primals_125: "f32[768]", primals_126: "f32[768]", primals_127: "f32[768]", primals_128: "f32[2304]", primals_129: "f32[2304, 768]", primals_130: "f32[768, 768]", primals_131: "f32[768]", primals_132: "f32[768]", primals_133: "f32[768]", primals_134: "f32[1536, 768]", primals_135: "f32[1536]", primals_136: "f32[768, 1536]", primals_137: "f32[768]", primals_138: "f32[768]", primals_139: "f32[768]", primals_140: "f32[2304]", primals_141: "f32[2304, 768]", primals_142: "f32[768, 768]", primals_143: "f32[768]", primals_144: "f32[768]", primals_145: "f32[768]", primals_146: "f32[1536, 768]", primals_147: "f32[1536]", primals_148: "f32[768, 1536]", primals_149: "f32[768]", primals_150: "f32[128, 768]", primals_151: "f32[128]", primals_152: "f32[16, 128]", primals_153: "f32[16]", fwd_rng_state_0, fwd_rng_state_1, fwd_rng_state_2, fwd_rng_state_3, fwd_rng_state_4, fwd_rng_state_5, fwd_rng_state_6, fwd_rng_state_7, fwd_rng_state_8, fwd_rng_state_9, fwd_rng_state_10, fwd_rng_state_11):
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:22 in forward, code: y = self.unfold(x)
        iota: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 4, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[1, 32]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        iota_1: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_1: "i64[4, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        add: "i64[4, 32]" = torch.ops.aten.add.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        unsqueeze_4: "i64[4, 32, 1]" = torch.ops.aten.unsqueeze.default(add, -1)
        unsqueeze_5: "i64[4, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        index: "f32[s77, 1, 4, 32, 4, 32]" = torch.ops.aten.index.Tensor(primals_2, [None, None, unsqueeze_5, add]);  primals_2 = None
        permute: "f32[s77, 1, 4, 4, 32, 32]" = torch.ops.aten.permute.default(index, [0, 1, 2, 4, 3, 5]);  index = None
        clone: "f32[s77, 1, 4, 4, 32, 32]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view: "f32[s77, 16, 1024]" = torch.ops.aten.reshape.default(clone, [primals_1, 16, 1024]);  clone = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:23 in forward, code: y = y.permute(0, 2, 1)
        permute_1: "f32[s77, 1024, 16]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:41 in forward, code: x = self.embed_layer(x)
        convert_element_type: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_4, torch.float16)
        convert_element_type_1: "f16[768, 16]" = torch.ops.prims.convert_element_type.default(primals_3, torch.float16);  primals_3 = None
        convert_element_type_2: "f16[s77, 1024, 16]" = torch.ops.prims.convert_element_type.default(permute_1, torch.float16);  permute_1 = None
        permute_2: "f16[16, 768]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        mul_15: "Sym(1024*s77)" = primals_1 * 1024
        clone_1: "f16[s77, 1024, 16]" = torch.ops.aten.clone.default(convert_element_type_2, memory_format = torch.contiguous_format);  convert_element_type_2 = None
        view_1: "f16[1024*s77, 16]" = torch.ops.aten.reshape.default(clone_1, [mul_15, 16]);  clone_1 = None
        mm: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_1, permute_2);  permute_2 = None
        view_2: "f16[s77, 1024, 768]" = torch.ops.aten.reshape.default(mm, [primals_1, 1024, 768])
        add_28: "f16[s77, 1024, 768]" = torch.ops.aten.add.Tensor(view_2, convert_element_type);  view_2 = convert_element_type = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:74 in forward, code: x = x + self.position_embed
        add_33: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_28, primals_5);  add_28 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem: "f32[s77, 1024, 1]" = var_mean[0]
        getitem_1: "f32[s77, 1024, 1]" = var_mean[1];  var_mean = None
        add_38: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_9: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_33, getitem_1);  getitem_1 = None
        mul_46: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt);  sub_9 = rsqrt = None
        mul_47: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_46, primals_6);  mul_46 = None
        add_39: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_47, primals_7);  mul_47 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_3: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_39, [1, 0, 2]);  add_39 = None
        convert_element_type_5: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_8, torch.float16)
        convert_element_type_6: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_9, torch.float16)
        convert_element_type_7: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_3, torch.float16);  permute_3 = None
        permute_4: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_6, [1, 0]);  convert_element_type_6 = None
        clone_2: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_7, memory_format = torch.contiguous_format);  convert_element_type_7 = None
        view_3: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(clone_2, [mul_15, 768]);  clone_2 = None
        mm_1: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_3, permute_4);  view_3 = permute_4 = None
        view_4: "f16[1024, s77, 2304]" = torch.ops.aten.reshape.default(mm_1, [1024, primals_1, 2304]);  mm_1 = None
        add_72: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_4, convert_element_type_5);  view_4 = convert_element_type_5 = None
        view_5: "f16[1024, s77, 3, 768]" = torch.ops.aten.reshape.default(add_72, [1024, primals_1, 3, 768]);  add_72 = None
        unsqueeze_6: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_5, 0);  view_5 = None
        permute_5: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_6, [3, 1, 2, 0, 4]);  unsqueeze_6 = None
        squeeze: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_5, -2);  permute_5 = None
        clone_3: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        select: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_3, 0, 0)
        select_1: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_3, 0, 1)
        select_2: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_3, 0, 2);  clone_3 = None
        mul_127: "Sym(8*s77)" = primals_1 * 8
        view_6: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select, [1024, mul_127, 96]);  select = None
        permute_6: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_6, [1, 0, 2]);  view_6 = None
        view_7: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_1, [1024, mul_127, 96]);  select_1 = None
        permute_7: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_7, [1, 0, 2]);  view_7 = None
        view_8: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_2, [1024, mul_127, 96]);  select_2 = None
        permute_8: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_8, [1, 0, 2]);  view_8 = None
        view_9: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_6, [primals_1, 8, 1024, 96]);  permute_6 = None
        view_10: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_7, [primals_1, 8, 1024, 96]);  permute_7 = None
        view_11: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_8, [primals_1, 8, 1024, 96]);  permute_8 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_9, view_10, view_11, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_0);  view_9 = view_10 = view_11 = fwd_rng_state_0 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_2: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state[0];  graphsafe_run_with_rng_state = None
        permute_9: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_2, [2, 0, 1, 3]);  getitem_2 = None
        view_12: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(permute_9, [mul_15, 768]);  permute_9 = None
        convert_element_type_10: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_11, torch.float16);  primals_11 = None
        convert_element_type_11: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_10, torch.float16)
        permute_10: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_11, [1, 0]);  convert_element_type_11 = None
        
        # No stacktrace found for following nodes
        mm_default_37: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_12, permute_10);  view_12 = permute_10 = None
        add_tensor_37: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_37, convert_element_type_10);  mm_default_37 = convert_element_type_10 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        view_13: "f16[1024, s77, 768]" = torch.ops.aten.reshape.default(add_tensor_37, [1024, primals_1, 768]);  add_tensor_37 = None
        permute_11: "f16[s77, 1024, 768]" = torch.ops.aten.permute.default(view_13, [1, 0, 2]);  view_13 = None
        add_185: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_33, permute_11);  add_33 = permute_11 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_1 = torch.ops.aten.var_mean.correction(add_185, [2], correction = 0, keepdim = True)
        getitem_11: "f32[s77, 1024, 1]" = var_mean_1[0]
        getitem_12: "f32[s77, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
        add_190: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_1: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
        sub_46: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_185, getitem_12);  getitem_12 = None
        mul_190: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_1);  sub_46 = rsqrt_1 = None
        mul_191: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_190, primals_12);  mul_190 = None
        add_191: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_191, primals_13);  mul_191 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_15: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_15, torch.float16)
        convert_element_type_16: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_14, torch.float16)
        convert_element_type_17: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_191, torch.float16);  add_191 = None
        view_14: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(convert_element_type_17, [mul_15, 768]);  convert_element_type_17 = None
        permute_12: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_16, [1, 0]);  convert_element_type_16 = None
        
        # No stacktrace found for following nodes
        mm_default_36: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_14, permute_12);  view_14 = permute_12 = None
        add_tensor_36: "f16[1024*s77, 1536]" = torch.ops.aten.add.Tensor(mm_default_36, convert_element_type_15);  mm_default_36 = convert_element_type_15 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_15: "f16[s77, 1024, 1536]" = torch.ops.aten.reshape.default(add_tensor_36, [primals_1, 1024, 1536]);  add_tensor_36 = None
        convert_element_type_21: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_15, torch.float32);  view_15 = None
        mul_211: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.5)
        mul_212: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.7071067811865476);  convert_element_type_21 = None
        erf: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_212);  mul_212 = None
        add_218: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_213: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_211, add_218);  mul_211 = add_218 = None
        convert_element_type_22: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_213, torch.float16);  mul_213 = None
        convert_element_type_23: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_17, torch.float16);  primals_17 = None
        convert_element_type_24: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_16, torch.float16)
        view_16: "f16[1024*s77, 1536]" = torch.ops.aten.reshape.default(convert_element_type_22, [mul_15, 1536]);  convert_element_type_22 = None
        permute_13: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_24, [1, 0]);  convert_element_type_24 = None
        
        # No stacktrace found for following nodes
        mm_default_35: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_16, permute_13);  view_16 = permute_13 = None
        add_tensor_35: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_35, convert_element_type_23);  mm_default_35 = convert_element_type_23 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_17: "f16[s77, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_35, [primals_1, 1024, 768]);  add_tensor_35 = None
        
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[12]" = torch.ops.prims.inductor_seeds.default(12, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_11: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default_63: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_11, torch.float16);  inductor_random_default_11 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_7: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_63, 0.2);  convert_element_type_default_63 = None
        mul_227: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_7, view_17);  gt_7 = view_17 = None
        mul_228: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_227, 1.25);  mul_227 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_241: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_185, mul_228);  mul_228 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_241, [2], correction = 0, keepdim = True)
        getitem_13: "f32[s77, 1024, 1]" = var_mean_2[0]
        getitem_14: "f32[s77, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
        add_246: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_13, 1e-05);  getitem_13 = None
        rsqrt_2: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_246);  add_246 = None
        sub_61: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_241, getitem_14);  getitem_14 = None
        mul_236: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_2);  sub_61 = rsqrt_2 = None
        mul_237: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_236, primals_18);  mul_236 = None
        add_247: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_237, primals_19);  mul_237 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_14: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_247, [1, 0, 2]);  add_247 = None
        convert_element_type_28: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_20, torch.float16)
        convert_element_type_29: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_21, torch.float16)
        convert_element_type_30: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_14, torch.float16);  permute_14 = None
        permute_15: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_29, [1, 0]);  convert_element_type_29 = None
        clone_4: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_30, memory_format = torch.contiguous_format);  convert_element_type_30 = None
        view_18: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(clone_4, [mul_15, 768]);  clone_4 = None
        mm_2: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_18, permute_15);  view_18 = permute_15 = None
        view_19: "f16[1024, s77, 2304]" = torch.ops.aten.reshape.default(mm_2, [1024, primals_1, 2304]);  mm_2 = None
        add_280: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_19, convert_element_type_28);  view_19 = convert_element_type_28 = None
        view_20: "f16[1024, s77, 3, 768]" = torch.ops.aten.reshape.default(add_280, [1024, primals_1, 3, 768]);  add_280 = None
        unsqueeze_7: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_20, 0);  view_20 = None
        permute_16: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_7, [3, 1, 2, 0, 4]);  unsqueeze_7 = None
        squeeze_1: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_16, -2);  permute_16 = None
        clone_5: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        select_3: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_5, 0, 0)
        select_4: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_5, 0, 1)
        select_5: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_5, 0, 2);  clone_5 = None
        view_21: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_3, [1024, mul_127, 96]);  select_3 = None
        permute_17: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_21, [1, 0, 2]);  view_21 = None
        view_22: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_4, [1024, mul_127, 96]);  select_4 = None
        permute_18: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_22, [1, 0, 2]);  view_22 = None
        view_23: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_5, [1024, mul_127, 96]);  select_5 = None
        permute_19: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_23, [1, 0, 2]);  view_23 = None
        view_24: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_17, [primals_1, 8, 1024, 96]);  permute_17 = None
        view_25: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_18, [primals_1, 8, 1024, 96]);  permute_18 = None
        view_26: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_19, [primals_1, 8, 1024, 96]);  permute_19 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_1 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_24, view_25, view_26, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_1);  view_24 = view_25 = view_26 = fwd_rng_state_1 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_15: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_1[0];  graphsafe_run_with_rng_state_1 = None
        permute_20: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_15, [2, 0, 1, 3]);  getitem_15 = None
        view_27: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(permute_20, [mul_15, 768]);  permute_20 = None
        convert_element_type_33: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_23, torch.float16);  primals_23 = None
        convert_element_type_34: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_22, torch.float16)
        permute_21: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_34, [1, 0]);  convert_element_type_34 = None
        
        # No stacktrace found for following nodes
        mm_default_34: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_27, permute_21);  view_27 = permute_21 = None
        add_tensor_34: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_34, convert_element_type_33);  mm_default_34 = convert_element_type_33 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        view_28: "f16[1024, s77, 768]" = torch.ops.aten.reshape.default(add_tensor_34, [1024, primals_1, 768]);  add_tensor_34 = None
        permute_22: "f16[s77, 1024, 768]" = torch.ops.aten.permute.default(view_28, [1, 0, 2]);  view_28 = None
        add_393: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_241, permute_22);  permute_22 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_3 = torch.ops.aten.var_mean.correction(add_393, [2], correction = 0, keepdim = True)
        getitem_24: "f32[s77, 1024, 1]" = var_mean_3[0]
        getitem_25: "f32[s77, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
        add_398: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_3: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_398);  add_398 = None
        sub_98: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_393, getitem_25);  getitem_25 = None
        mul_380: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_3);  sub_98 = rsqrt_3 = None
        mul_381: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_380, primals_24);  mul_380 = None
        add_399: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_381, primals_25);  mul_381 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_38: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_27, torch.float16)
        convert_element_type_39: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_26, torch.float16)
        convert_element_type_40: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_399, torch.float16);  add_399 = None
        view_29: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(convert_element_type_40, [mul_15, 768]);  convert_element_type_40 = None
        permute_23: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_39, [1, 0]);  convert_element_type_39 = None
        
        # No stacktrace found for following nodes
        mm_default_33: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_29, permute_23);  view_29 = permute_23 = None
        add_tensor_33: "f16[1024*s77, 1536]" = torch.ops.aten.add.Tensor(mm_default_33, convert_element_type_38);  mm_default_33 = convert_element_type_38 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_30: "f16[s77, 1024, 1536]" = torch.ops.aten.reshape.default(add_tensor_33, [primals_1, 1024, 1536]);  add_tensor_33 = None
        convert_element_type_44: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_30, torch.float32);  view_30 = None
        mul_401: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 0.5)
        mul_402: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 0.7071067811865476);  convert_element_type_44 = None
        erf_1: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_402);  mul_402 = None
        add_426: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_403: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_401, add_426);  mul_401 = add_426 = None
        convert_element_type_45: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_403, torch.float16);  mul_403 = None
        convert_element_type_46: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_29, torch.float16);  primals_29 = None
        convert_element_type_47: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_28, torch.float16)
        view_31: "f16[1024*s77, 1536]" = torch.ops.aten.reshape.default(convert_element_type_45, [mul_15, 1536]);  convert_element_type_45 = None
        permute_24: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_47, [1, 0]);  convert_element_type_47 = None
        
        # No stacktrace found for following nodes
        mm_default_32: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_31, permute_24);  view_31 = permute_24 = None
        add_tensor_32: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_32, convert_element_type_46);  mm_default_32 = convert_element_type_46 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_32: "f16[s77, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_32, [primals_1, 1024, 768]);  add_tensor_32 = None
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_10: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default_62: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_10, torch.float16);  inductor_random_default_10 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_11: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_62, 0.2);  convert_element_type_default_62 = None
        mul_417: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_11, view_32);  gt_11 = view_32 = None
        mul_418: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_417, 1.25);  mul_417 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_449: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_393, mul_418);  mul_418 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_449, [2], correction = 0, keepdim = True)
        getitem_26: "f32[s77, 1024, 1]" = var_mean_4[0]
        getitem_27: "f32[s77, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
        add_454: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_4: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_454);  add_454 = None
        sub_113: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_449, getitem_27);  getitem_27 = None
        mul_426: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_4);  sub_113 = rsqrt_4 = None
        mul_427: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_426, primals_30);  mul_426 = None
        add_455: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_427, primals_31);  mul_427 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_25: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_455, [1, 0, 2]);  add_455 = None
        convert_element_type_51: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_32, torch.float16)
        convert_element_type_52: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_33, torch.float16)
        convert_element_type_53: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_25, torch.float16);  permute_25 = None
        permute_26: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_52, [1, 0]);  convert_element_type_52 = None
        clone_6: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_53, memory_format = torch.contiguous_format);  convert_element_type_53 = None
        view_33: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(clone_6, [mul_15, 768]);  clone_6 = None
        mm_3: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_33, permute_26);  view_33 = permute_26 = None
        view_34: "f16[1024, s77, 2304]" = torch.ops.aten.reshape.default(mm_3, [1024, primals_1, 2304]);  mm_3 = None
        add_488: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_34, convert_element_type_51);  view_34 = convert_element_type_51 = None
        view_35: "f16[1024, s77, 3, 768]" = torch.ops.aten.reshape.default(add_488, [1024, primals_1, 3, 768]);  add_488 = None
        unsqueeze_8: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_35, 0);  view_35 = None
        permute_27: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_8, [3, 1, 2, 0, 4]);  unsqueeze_8 = None
        squeeze_2: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_27, -2);  permute_27 = None
        clone_7: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
        select_6: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_7, 0, 0)
        select_7: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_7, 0, 1)
        select_8: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_7, 0, 2);  clone_7 = None
        view_36: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_6, [1024, mul_127, 96]);  select_6 = None
        permute_28: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_36, [1, 0, 2]);  view_36 = None
        view_37: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_7, [1024, mul_127, 96]);  select_7 = None
        permute_29: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_37, [1, 0, 2]);  view_37 = None
        view_38: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_8, [1024, mul_127, 96]);  select_8 = None
        permute_30: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_38, [1, 0, 2]);  view_38 = None
        view_39: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_28, [primals_1, 8, 1024, 96]);  permute_28 = None
        view_40: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_29, [primals_1, 8, 1024, 96]);  permute_29 = None
        view_41: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_30, [primals_1, 8, 1024, 96]);  permute_30 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_2 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_39, view_40, view_41, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_2);  view_39 = view_40 = view_41 = fwd_rng_state_2 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_28: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_2[0];  graphsafe_run_with_rng_state_2 = None
        permute_31: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_28, [2, 0, 1, 3]);  getitem_28 = None
        view_42: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(permute_31, [mul_15, 768]);  permute_31 = None
        convert_element_type_56: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_35, torch.float16);  primals_35 = None
        convert_element_type_57: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_34, torch.float16)
        permute_32: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_57, [1, 0]);  convert_element_type_57 = None
        
        # No stacktrace found for following nodes
        mm_default_31: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_42, permute_32);  view_42 = permute_32 = None
        add_tensor_31: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_31, convert_element_type_56);  mm_default_31 = convert_element_type_56 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        view_43: "f16[1024, s77, 768]" = torch.ops.aten.reshape.default(add_tensor_31, [1024, primals_1, 768]);  add_tensor_31 = None
        permute_33: "f16[s77, 1024, 768]" = torch.ops.aten.permute.default(view_43, [1, 0, 2]);  view_43 = None
        add_601: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_449, permute_33);  permute_33 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_5 = torch.ops.aten.var_mean.correction(add_601, [2], correction = 0, keepdim = True)
        getitem_37: "f32[s77, 1024, 1]" = var_mean_5[0]
        getitem_38: "f32[s77, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
        add_606: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_37, 1e-05);  getitem_37 = None
        rsqrt_5: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_606);  add_606 = None
        sub_150: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_601, getitem_38);  getitem_38 = None
        mul_570: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_150, rsqrt_5);  sub_150 = rsqrt_5 = None
        mul_571: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_570, primals_36);  mul_570 = None
        add_607: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_571, primals_37);  mul_571 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_61: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_39, torch.float16)
        convert_element_type_62: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_38, torch.float16)
        convert_element_type_63: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_607, torch.float16);  add_607 = None
        view_44: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(convert_element_type_63, [mul_15, 768]);  convert_element_type_63 = None
        permute_34: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_62, [1, 0]);  convert_element_type_62 = None
        
        # No stacktrace found for following nodes
        mm_default_30: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_44, permute_34);  view_44 = permute_34 = None
        add_tensor_30: "f16[1024*s77, 1536]" = torch.ops.aten.add.Tensor(mm_default_30, convert_element_type_61);  mm_default_30 = convert_element_type_61 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_45: "f16[s77, 1024, 1536]" = torch.ops.aten.reshape.default(add_tensor_30, [primals_1, 1024, 1536]);  add_tensor_30 = None
        convert_element_type_67: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        mul_591: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.5)
        mul_592: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.7071067811865476);  convert_element_type_67 = None
        erf_2: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_592);  mul_592 = None
        add_634: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_593: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_591, add_634);  mul_591 = add_634 = None
        convert_element_type_68: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_593, torch.float16);  mul_593 = None
        convert_element_type_69: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_41, torch.float16);  primals_41 = None
        convert_element_type_70: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_40, torch.float16)
        view_46: "f16[1024*s77, 1536]" = torch.ops.aten.reshape.default(convert_element_type_68, [mul_15, 1536]);  convert_element_type_68 = None
        permute_35: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_70, [1, 0]);  convert_element_type_70 = None
        
        # No stacktrace found for following nodes
        mm_default_29: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_46, permute_35);  view_46 = permute_35 = None
        add_tensor_29: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_29, convert_element_type_69);  mm_default_29 = convert_element_type_69 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_47: "f16[s77, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_29, [primals_1, 1024, 768]);  add_tensor_29 = None
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_9: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        convert_element_type_default_61: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_9, torch.float16);  inductor_random_default_9 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_15: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_61, 0.2);  convert_element_type_default_61 = None
        mul_607: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_15, view_47);  gt_15 = view_47 = None
        mul_608: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_607, 1.25);  mul_607 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_657: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_601, mul_608);  mul_608 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_657, [2], correction = 0, keepdim = True)
        getitem_39: "f32[s77, 1024, 1]" = var_mean_6[0]
        getitem_40: "f32[s77, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
        add_662: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_39, 1e-05);  getitem_39 = None
        rsqrt_6: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_662);  add_662 = None
        sub_165: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_657, getitem_40);  getitem_40 = None
        mul_616: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_165, rsqrt_6);  sub_165 = rsqrt_6 = None
        mul_617: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_616, primals_42);  mul_616 = None
        add_663: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_617, primals_43);  mul_617 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_36: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_663, [1, 0, 2]);  add_663 = None
        convert_element_type_74: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_44, torch.float16)
        convert_element_type_75: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_45, torch.float16)
        convert_element_type_76: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_36, torch.float16);  permute_36 = None
        permute_37: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_75, [1, 0]);  convert_element_type_75 = None
        clone_8: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_76, memory_format = torch.contiguous_format);  convert_element_type_76 = None
        view_48: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(clone_8, [mul_15, 768]);  clone_8 = None
        mm_4: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_48, permute_37);  view_48 = permute_37 = None
        view_49: "f16[1024, s77, 2304]" = torch.ops.aten.reshape.default(mm_4, [1024, primals_1, 2304]);  mm_4 = None
        add_696: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_49, convert_element_type_74);  view_49 = convert_element_type_74 = None
        view_50: "f16[1024, s77, 3, 768]" = torch.ops.aten.reshape.default(add_696, [1024, primals_1, 3, 768]);  add_696 = None
        unsqueeze_9: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_50, 0);  view_50 = None
        permute_38: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_9, [3, 1, 2, 0, 4]);  unsqueeze_9 = None
        squeeze_3: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_38, -2);  permute_38 = None
        clone_9: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_3, memory_format = torch.contiguous_format);  squeeze_3 = None
        select_9: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_9, 0, 0)
        select_10: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_9, 0, 1)
        select_11: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_9, 0, 2);  clone_9 = None
        view_51: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_9, [1024, mul_127, 96]);  select_9 = None
        permute_39: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_51, [1, 0, 2]);  view_51 = None
        view_52: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_10, [1024, mul_127, 96]);  select_10 = None
        permute_40: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_52, [1, 0, 2]);  view_52 = None
        view_53: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_11, [1024, mul_127, 96]);  select_11 = None
        permute_41: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_53, [1, 0, 2]);  view_53 = None
        view_54: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_39, [primals_1, 8, 1024, 96]);  permute_39 = None
        view_55: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_40, [primals_1, 8, 1024, 96]);  permute_40 = None
        view_56: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_41, [primals_1, 8, 1024, 96]);  permute_41 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_3 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_54, view_55, view_56, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_3);  view_54 = view_55 = view_56 = fwd_rng_state_3 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_41: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_3[0];  graphsafe_run_with_rng_state_3 = None
        permute_42: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_41, [2, 0, 1, 3]);  getitem_41 = None
        view_57: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(permute_42, [mul_15, 768]);  permute_42 = None
        convert_element_type_79: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_47, torch.float16);  primals_47 = None
        convert_element_type_80: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_46, torch.float16)
        permute_43: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_80, [1, 0]);  convert_element_type_80 = None
        
        # No stacktrace found for following nodes
        mm_default_28: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_57, permute_43);  view_57 = permute_43 = None
        add_tensor_28: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_28, convert_element_type_79);  mm_default_28 = convert_element_type_79 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        view_58: "f16[1024, s77, 768]" = torch.ops.aten.reshape.default(add_tensor_28, [1024, primals_1, 768]);  add_tensor_28 = None
        permute_44: "f16[s77, 1024, 768]" = torch.ops.aten.permute.default(view_58, [1, 0, 2]);  view_58 = None
        add_809: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_657, permute_44);  permute_44 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_7 = torch.ops.aten.var_mean.correction(add_809, [2], correction = 0, keepdim = True)
        getitem_50: "f32[s77, 1024, 1]" = var_mean_7[0]
        getitem_51: "f32[s77, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
        add_814: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_7: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_814);  add_814 = None
        sub_202: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_809, getitem_51);  getitem_51 = None
        mul_760: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_202, rsqrt_7);  sub_202 = rsqrt_7 = None
        mul_761: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_760, primals_48);  mul_760 = None
        add_815: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_761, primals_49);  mul_761 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_84: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_51, torch.float16)
        convert_element_type_85: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_50, torch.float16)
        convert_element_type_86: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_815, torch.float16);  add_815 = None
        view_59: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(convert_element_type_86, [mul_15, 768]);  convert_element_type_86 = None
        permute_45: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_85, [1, 0]);  convert_element_type_85 = None
        
        # No stacktrace found for following nodes
        mm_default_27: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_59, permute_45);  view_59 = permute_45 = None
        add_tensor_27: "f16[1024*s77, 1536]" = torch.ops.aten.add.Tensor(mm_default_27, convert_element_type_84);  mm_default_27 = convert_element_type_84 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_60: "f16[s77, 1024, 1536]" = torch.ops.aten.reshape.default(add_tensor_27, [primals_1, 1024, 1536]);  add_tensor_27 = None
        convert_element_type_90: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_60, torch.float32);  view_60 = None
        mul_781: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_90, 0.5)
        mul_782: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_90, 0.7071067811865476);  convert_element_type_90 = None
        erf_3: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_782);  mul_782 = None
        add_842: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_783: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_781, add_842);  mul_781 = add_842 = None
        convert_element_type_91: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_783, torch.float16);  mul_783 = None
        convert_element_type_92: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_53, torch.float16);  primals_53 = None
        convert_element_type_93: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_52, torch.float16)
        view_61: "f16[1024*s77, 1536]" = torch.ops.aten.reshape.default(convert_element_type_91, [mul_15, 1536]);  convert_element_type_91 = None
        permute_46: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_93, [1, 0]);  convert_element_type_93 = None
        
        # No stacktrace found for following nodes
        mm_default_26: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_61, permute_46);  view_61 = permute_46 = None
        add_tensor_26: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_26, convert_element_type_92);  mm_default_26 = convert_element_type_92 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_62: "f16[s77, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_26, [primals_1, 1024, 768]);  add_tensor_26 = None
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_8: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        convert_element_type_default_60: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_8, torch.float16);  inductor_random_default_8 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_19: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_60, 0.2);  convert_element_type_default_60 = None
        mul_797: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_19, view_62);  gt_19 = view_62 = None
        mul_798: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_797, 1.25);  mul_797 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_865: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_809, mul_798);  mul_798 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_865, [2], correction = 0, keepdim = True)
        getitem_52: "f32[s77, 1024, 1]" = var_mean_8[0]
        getitem_53: "f32[s77, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
        add_870: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_8: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_870);  add_870 = None
        sub_217: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_865, getitem_53);  getitem_53 = None
        mul_806: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_217, rsqrt_8);  sub_217 = rsqrt_8 = None
        mul_807: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_806, primals_54);  mul_806 = None
        add_871: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_807, primals_55);  mul_807 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_47: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_871, [1, 0, 2]);  add_871 = None
        convert_element_type_97: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_56, torch.float16)
        convert_element_type_98: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_57, torch.float16)
        convert_element_type_99: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_47, torch.float16);  permute_47 = None
        permute_48: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_98, [1, 0]);  convert_element_type_98 = None
        clone_10: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_99, memory_format = torch.contiguous_format);  convert_element_type_99 = None
        view_63: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(clone_10, [mul_15, 768]);  clone_10 = None
        mm_5: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_63, permute_48);  view_63 = permute_48 = None
        view_64: "f16[1024, s77, 2304]" = torch.ops.aten.reshape.default(mm_5, [1024, primals_1, 2304]);  mm_5 = None
        add_904: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_64, convert_element_type_97);  view_64 = convert_element_type_97 = None
        view_65: "f16[1024, s77, 3, 768]" = torch.ops.aten.reshape.default(add_904, [1024, primals_1, 3, 768]);  add_904 = None
        unsqueeze_10: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_65, 0);  view_65 = None
        permute_49: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_10, [3, 1, 2, 0, 4]);  unsqueeze_10 = None
        squeeze_4: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_49, -2);  permute_49 = None
        clone_11: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_4, memory_format = torch.contiguous_format);  squeeze_4 = None
        select_12: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_11, 0, 0)
        select_13: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_11, 0, 1)
        select_14: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_11, 0, 2);  clone_11 = None
        view_66: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_12, [1024, mul_127, 96]);  select_12 = None
        permute_50: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_66, [1, 0, 2]);  view_66 = None
        view_67: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_13, [1024, mul_127, 96]);  select_13 = None
        permute_51: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_67, [1, 0, 2]);  view_67 = None
        view_68: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_14, [1024, mul_127, 96]);  select_14 = None
        permute_52: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_68, [1, 0, 2]);  view_68 = None
        view_69: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_50, [primals_1, 8, 1024, 96]);  permute_50 = None
        view_70: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_51, [primals_1, 8, 1024, 96]);  permute_51 = None
        view_71: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_52, [primals_1, 8, 1024, 96]);  permute_52 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_4 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_69, view_70, view_71, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_4);  view_69 = view_70 = view_71 = fwd_rng_state_4 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_54: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_4[0];  graphsafe_run_with_rng_state_4 = None
        permute_53: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_54, [2, 0, 1, 3]);  getitem_54 = None
        view_72: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(permute_53, [mul_15, 768]);  permute_53 = None
        convert_element_type_102: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_59, torch.float16);  primals_59 = None
        convert_element_type_103: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_58, torch.float16)
        permute_54: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_103, [1, 0]);  convert_element_type_103 = None
        
        # No stacktrace found for following nodes
        mm_default_25: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_72, permute_54);  view_72 = permute_54 = None
        add_tensor_25: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_25, convert_element_type_102);  mm_default_25 = convert_element_type_102 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        view_73: "f16[1024, s77, 768]" = torch.ops.aten.reshape.default(add_tensor_25, [1024, primals_1, 768]);  add_tensor_25 = None
        permute_55: "f16[s77, 1024, 768]" = torch.ops.aten.permute.default(view_73, [1, 0, 2]);  view_73 = None
        add_1017: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_865, permute_55);  permute_55 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_9 = torch.ops.aten.var_mean.correction(add_1017, [2], correction = 0, keepdim = True)
        getitem_63: "f32[s77, 1024, 1]" = var_mean_9[0]
        getitem_64: "f32[s77, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
        add_1022: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_63, 1e-05);  getitem_63 = None
        rsqrt_9: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1022);  add_1022 = None
        sub_254: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1017, getitem_64);  getitem_64 = None
        mul_950: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_254, rsqrt_9);  sub_254 = rsqrt_9 = None
        mul_951: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_950, primals_60);  mul_950 = None
        add_1023: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_951, primals_61);  mul_951 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_107: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_63, torch.float16)
        convert_element_type_108: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_62, torch.float16)
        convert_element_type_109: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_1023, torch.float16);  add_1023 = None
        view_74: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(convert_element_type_109, [mul_15, 768]);  convert_element_type_109 = None
        permute_56: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_108, [1, 0]);  convert_element_type_108 = None
        
        # No stacktrace found for following nodes
        mm_default_24: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_74, permute_56);  view_74 = permute_56 = None
        add_tensor_24: "f16[1024*s77, 1536]" = torch.ops.aten.add.Tensor(mm_default_24, convert_element_type_107);  mm_default_24 = convert_element_type_107 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_75: "f16[s77, 1024, 1536]" = torch.ops.aten.reshape.default(add_tensor_24, [primals_1, 1024, 1536]);  add_tensor_24 = None
        convert_element_type_113: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_75, torch.float32);  view_75 = None
        mul_971: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_113, 0.5)
        mul_972: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_113, 0.7071067811865476);  convert_element_type_113 = None
        erf_4: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_972);  mul_972 = None
        add_1050: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_973: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_971, add_1050);  mul_971 = add_1050 = None
        convert_element_type_114: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_973, torch.float16);  mul_973 = None
        convert_element_type_115: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_65, torch.float16);  primals_65 = None
        convert_element_type_116: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_64, torch.float16)
        view_76: "f16[1024*s77, 1536]" = torch.ops.aten.reshape.default(convert_element_type_114, [mul_15, 1536]);  convert_element_type_114 = None
        permute_57: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_116, [1, 0]);  convert_element_type_116 = None
        
        # No stacktrace found for following nodes
        mm_default_23: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_76, permute_57);  view_76 = permute_57 = None
        add_tensor_23: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_23, convert_element_type_115);  mm_default_23 = convert_element_type_115 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_77: "f16[s77, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_23, [primals_1, 1024, 768]);  add_tensor_23 = None
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_7: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        convert_element_type_default_59: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_7, torch.float16);  inductor_random_default_7 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_23: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_59, 0.2);  convert_element_type_default_59 = None
        mul_987: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_23, view_77);  gt_23 = view_77 = None
        mul_988: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_987, 1.25);  mul_987 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_1073: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_1017, mul_988);  mul_988 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_1073, [2], correction = 0, keepdim = True)
        getitem_65: "f32[s77, 1024, 1]" = var_mean_10[0]
        getitem_66: "f32[s77, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
        add_1078: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_65, 1e-05);  getitem_65 = None
        rsqrt_10: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1078);  add_1078 = None
        sub_269: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1073, getitem_66);  getitem_66 = None
        mul_996: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_269, rsqrt_10);  sub_269 = rsqrt_10 = None
        mul_997: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_996, primals_66);  mul_996 = None
        add_1079: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_997, primals_67);  mul_997 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_58: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_1079, [1, 0, 2]);  add_1079 = None
        convert_element_type_120: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_68, torch.float16)
        convert_element_type_121: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_69, torch.float16)
        convert_element_type_122: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_58, torch.float16);  permute_58 = None
        permute_59: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_121, [1, 0]);  convert_element_type_121 = None
        clone_12: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_122, memory_format = torch.contiguous_format);  convert_element_type_122 = None
        view_78: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(clone_12, [mul_15, 768]);  clone_12 = None
        mm_6: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_78, permute_59);  view_78 = permute_59 = None
        view_79: "f16[1024, s77, 2304]" = torch.ops.aten.reshape.default(mm_6, [1024, primals_1, 2304]);  mm_6 = None
        add_1112: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_79, convert_element_type_120);  view_79 = convert_element_type_120 = None
        view_80: "f16[1024, s77, 3, 768]" = torch.ops.aten.reshape.default(add_1112, [1024, primals_1, 3, 768]);  add_1112 = None
        unsqueeze_11: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_80, 0);  view_80 = None
        permute_60: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_11, [3, 1, 2, 0, 4]);  unsqueeze_11 = None
        squeeze_5: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_60, -2);  permute_60 = None
        clone_13: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_5, memory_format = torch.contiguous_format);  squeeze_5 = None
        select_15: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_13, 0, 0)
        select_16: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_13, 0, 1)
        select_17: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_13, 0, 2);  clone_13 = None
        view_81: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_15, [1024, mul_127, 96]);  select_15 = None
        permute_61: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_81, [1, 0, 2]);  view_81 = None
        view_82: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_16, [1024, mul_127, 96]);  select_16 = None
        permute_62: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_82, [1, 0, 2]);  view_82 = None
        view_83: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_17, [1024, mul_127, 96]);  select_17 = None
        permute_63: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_83, [1, 0, 2]);  view_83 = None
        view_84: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_61, [primals_1, 8, 1024, 96]);  permute_61 = None
        view_85: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_62, [primals_1, 8, 1024, 96]);  permute_62 = None
        view_86: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_63, [primals_1, 8, 1024, 96]);  permute_63 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_5 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_84, view_85, view_86, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_5);  view_84 = view_85 = view_86 = fwd_rng_state_5 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_67: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_5[0];  graphsafe_run_with_rng_state_5 = None
        permute_64: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_67, [2, 0, 1, 3]);  getitem_67 = None
        view_87: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(permute_64, [mul_15, 768]);  permute_64 = None
        convert_element_type_125: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_71, torch.float16);  primals_71 = None
        convert_element_type_126: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_70, torch.float16)
        permute_65: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_126, [1, 0]);  convert_element_type_126 = None
        
        # No stacktrace found for following nodes
        mm_default_22: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_87, permute_65);  view_87 = permute_65 = None
        add_tensor_22: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_22, convert_element_type_125);  mm_default_22 = convert_element_type_125 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        view_88: "f16[1024, s77, 768]" = torch.ops.aten.reshape.default(add_tensor_22, [1024, primals_1, 768]);  add_tensor_22 = None
        permute_66: "f16[s77, 1024, 768]" = torch.ops.aten.permute.default(view_88, [1, 0, 2]);  view_88 = None
        add_1225: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_1073, permute_66);  permute_66 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_11 = torch.ops.aten.var_mean.correction(add_1225, [2], correction = 0, keepdim = True)
        getitem_76: "f32[s77, 1024, 1]" = var_mean_11[0]
        getitem_77: "f32[s77, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
        add_1230: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_11: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1230);  add_1230 = None
        sub_306: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1225, getitem_77);  getitem_77 = None
        mul_1140: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_306, rsqrt_11);  sub_306 = rsqrt_11 = None
        mul_1141: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1140, primals_72);  mul_1140 = None
        add_1231: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1141, primals_73);  mul_1141 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_130: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_75, torch.float16)
        convert_element_type_131: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_74, torch.float16)
        convert_element_type_132: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_1231, torch.float16);  add_1231 = None
        view_89: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(convert_element_type_132, [mul_15, 768]);  convert_element_type_132 = None
        permute_67: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_131, [1, 0]);  convert_element_type_131 = None
        
        # No stacktrace found for following nodes
        mm_default_21: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_89, permute_67);  view_89 = permute_67 = None
        add_tensor_21: "f16[1024*s77, 1536]" = torch.ops.aten.add.Tensor(mm_default_21, convert_element_type_130);  mm_default_21 = convert_element_type_130 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_90: "f16[s77, 1024, 1536]" = torch.ops.aten.reshape.default(add_tensor_21, [primals_1, 1024, 1536]);  add_tensor_21 = None
        convert_element_type_136: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_90, torch.float32);  view_90 = None
        mul_1161: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_136, 0.5)
        mul_1162: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_136, 0.7071067811865476);  convert_element_type_136 = None
        erf_5: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_1162);  mul_1162 = None
        add_1258: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_1163: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_1161, add_1258);  mul_1161 = add_1258 = None
        convert_element_type_137: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_1163, torch.float16);  mul_1163 = None
        convert_element_type_138: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_77, torch.float16);  primals_77 = None
        convert_element_type_139: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_76, torch.float16)
        view_91: "f16[1024*s77, 1536]" = torch.ops.aten.reshape.default(convert_element_type_137, [mul_15, 1536]);  convert_element_type_137 = None
        permute_68: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_139, [1, 0]);  convert_element_type_139 = None
        
        # No stacktrace found for following nodes
        mm_default_20: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_91, permute_68);  view_91 = permute_68 = None
        add_tensor_20: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_20, convert_element_type_138);  mm_default_20 = convert_element_type_138 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_92: "f16[s77, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_20, [primals_1, 1024, 768]);  add_tensor_20 = None
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_6: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        convert_element_type_default_58: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_6, torch.float16);  inductor_random_default_6 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_27: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_58, 0.2);  convert_element_type_default_58 = None
        mul_1177: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_27, view_92);  gt_27 = view_92 = None
        mul_1178: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1177, 1.25);  mul_1177 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_1281: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_1225, mul_1178);  mul_1178 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_1281, [2], correction = 0, keepdim = True)
        getitem_78: "f32[s77, 1024, 1]" = var_mean_12[0]
        getitem_79: "f32[s77, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
        add_1286: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_12: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1286);  add_1286 = None
        sub_321: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1281, getitem_79);  getitem_79 = None
        mul_1186: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_321, rsqrt_12);  sub_321 = rsqrt_12 = None
        mul_1187: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1186, primals_78);  mul_1186 = None
        add_1287: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1187, primals_79);  mul_1187 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_69: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_1287, [1, 0, 2]);  add_1287 = None
        convert_element_type_143: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_80, torch.float16)
        convert_element_type_144: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_81, torch.float16)
        convert_element_type_145: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_69, torch.float16);  permute_69 = None
        permute_70: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_144, [1, 0]);  convert_element_type_144 = None
        clone_14: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_145, memory_format = torch.contiguous_format);  convert_element_type_145 = None
        view_93: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(clone_14, [mul_15, 768]);  clone_14 = None
        mm_7: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_93, permute_70);  view_93 = permute_70 = None
        view_94: "f16[1024, s77, 2304]" = torch.ops.aten.reshape.default(mm_7, [1024, primals_1, 2304]);  mm_7 = None
        add_1320: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_94, convert_element_type_143);  view_94 = convert_element_type_143 = None
        view_95: "f16[1024, s77, 3, 768]" = torch.ops.aten.reshape.default(add_1320, [1024, primals_1, 3, 768]);  add_1320 = None
        unsqueeze_12: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_95, 0);  view_95 = None
        permute_71: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_12, [3, 1, 2, 0, 4]);  unsqueeze_12 = None
        squeeze_6: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_71, -2);  permute_71 = None
        clone_15: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_6, memory_format = torch.contiguous_format);  squeeze_6 = None
        select_18: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_15, 0, 0)
        select_19: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_15, 0, 1)
        select_20: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_15, 0, 2);  clone_15 = None
        view_96: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_18, [1024, mul_127, 96]);  select_18 = None
        permute_72: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_96, [1, 0, 2]);  view_96 = None
        view_97: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_19, [1024, mul_127, 96]);  select_19 = None
        permute_73: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_97, [1, 0, 2]);  view_97 = None
        view_98: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_20, [1024, mul_127, 96]);  select_20 = None
        permute_74: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_98, [1, 0, 2]);  view_98 = None
        view_99: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_72, [primals_1, 8, 1024, 96]);  permute_72 = None
        view_100: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_73, [primals_1, 8, 1024, 96]);  permute_73 = None
        view_101: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_74, [primals_1, 8, 1024, 96]);  permute_74 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_6 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_99, view_100, view_101, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_6);  view_99 = view_100 = view_101 = fwd_rng_state_6 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_80: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_6[0];  graphsafe_run_with_rng_state_6 = None
        permute_75: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_80, [2, 0, 1, 3]);  getitem_80 = None
        view_102: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(permute_75, [mul_15, 768]);  permute_75 = None
        convert_element_type_148: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_83, torch.float16);  primals_83 = None
        convert_element_type_149: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_82, torch.float16)
        permute_76: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_149, [1, 0]);  convert_element_type_149 = None
        
        # No stacktrace found for following nodes
        mm_default_19: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_102, permute_76);  view_102 = permute_76 = None
        add_tensor_19: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_19, convert_element_type_148);  mm_default_19 = convert_element_type_148 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        view_103: "f16[1024, s77, 768]" = torch.ops.aten.reshape.default(add_tensor_19, [1024, primals_1, 768]);  add_tensor_19 = None
        permute_77: "f16[s77, 1024, 768]" = torch.ops.aten.permute.default(view_103, [1, 0, 2]);  view_103 = None
        add_1433: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_1281, permute_77);  permute_77 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_13 = torch.ops.aten.var_mean.correction(add_1433, [2], correction = 0, keepdim = True)
        getitem_89: "f32[s77, 1024, 1]" = var_mean_13[0]
        getitem_90: "f32[s77, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
        add_1438: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_89, 1e-05);  getitem_89 = None
        rsqrt_13: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1438);  add_1438 = None
        sub_358: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1433, getitem_90);  getitem_90 = None
        mul_1330: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_358, rsqrt_13);  sub_358 = rsqrt_13 = None
        mul_1331: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1330, primals_84);  mul_1330 = None
        add_1439: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1331, primals_85);  mul_1331 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_153: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_87, torch.float16)
        convert_element_type_154: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_86, torch.float16)
        convert_element_type_155: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_1439, torch.float16);  add_1439 = None
        view_104: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(convert_element_type_155, [mul_15, 768]);  convert_element_type_155 = None
        permute_78: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_154, [1, 0]);  convert_element_type_154 = None
        
        # No stacktrace found for following nodes
        mm_default_18: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_104, permute_78);  view_104 = permute_78 = None
        add_tensor_18: "f16[1024*s77, 1536]" = torch.ops.aten.add.Tensor(mm_default_18, convert_element_type_153);  mm_default_18 = convert_element_type_153 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_105: "f16[s77, 1024, 1536]" = torch.ops.aten.reshape.default(add_tensor_18, [primals_1, 1024, 1536]);  add_tensor_18 = None
        convert_element_type_159: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None
        mul_1351: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_159, 0.5)
        mul_1352: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_159, 0.7071067811865476);  convert_element_type_159 = None
        erf_6: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_1352);  mul_1352 = None
        add_1466: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_1353: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_1351, add_1466);  mul_1351 = add_1466 = None
        convert_element_type_160: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_1353, torch.float16);  mul_1353 = None
        convert_element_type_161: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_89, torch.float16);  primals_89 = None
        convert_element_type_162: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_88, torch.float16)
        view_106: "f16[1024*s77, 1536]" = torch.ops.aten.reshape.default(convert_element_type_160, [mul_15, 1536]);  convert_element_type_160 = None
        permute_79: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_162, [1, 0]);  convert_element_type_162 = None
        
        # No stacktrace found for following nodes
        mm_default_17: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_106, permute_79);  view_106 = permute_79 = None
        add_tensor_17: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_17, convert_element_type_161);  mm_default_17 = convert_element_type_161 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_107: "f16[s77, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_17, [primals_1, 1024, 768]);  add_tensor_17 = None
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_5: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        convert_element_type_default_57: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_5, torch.float16);  inductor_random_default_5 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_31: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_57, 0.2);  convert_element_type_default_57 = None
        mul_1367: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_31, view_107);  gt_31 = view_107 = None
        mul_1368: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1367, 1.25);  mul_1367 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_1489: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_1433, mul_1368);  mul_1368 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_1489, [2], correction = 0, keepdim = True)
        getitem_91: "f32[s77, 1024, 1]" = var_mean_14[0]
        getitem_92: "f32[s77, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
        add_1494: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_91, 1e-05);  getitem_91 = None
        rsqrt_14: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1494);  add_1494 = None
        sub_373: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1489, getitem_92);  getitem_92 = None
        mul_1376: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_373, rsqrt_14);  sub_373 = rsqrt_14 = None
        mul_1377: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1376, primals_90);  mul_1376 = None
        add_1495: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1377, primals_91);  mul_1377 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_80: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_1495, [1, 0, 2]);  add_1495 = None
        convert_element_type_166: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_92, torch.float16)
        convert_element_type_167: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_93, torch.float16)
        convert_element_type_168: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_80, torch.float16);  permute_80 = None
        permute_81: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_167, [1, 0]);  convert_element_type_167 = None
        clone_16: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_168, memory_format = torch.contiguous_format);  convert_element_type_168 = None
        view_108: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(clone_16, [mul_15, 768]);  clone_16 = None
        mm_8: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_108, permute_81);  view_108 = permute_81 = None
        view_109: "f16[1024, s77, 2304]" = torch.ops.aten.reshape.default(mm_8, [1024, primals_1, 2304]);  mm_8 = None
        add_1528: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_109, convert_element_type_166);  view_109 = convert_element_type_166 = None
        view_110: "f16[1024, s77, 3, 768]" = torch.ops.aten.reshape.default(add_1528, [1024, primals_1, 3, 768]);  add_1528 = None
        unsqueeze_13: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_110, 0);  view_110 = None
        permute_82: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_13, [3, 1, 2, 0, 4]);  unsqueeze_13 = None
        squeeze_7: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_82, -2);  permute_82 = None
        clone_17: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_7, memory_format = torch.contiguous_format);  squeeze_7 = None
        select_21: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_17, 0, 0)
        select_22: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_17, 0, 1)
        select_23: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_17, 0, 2);  clone_17 = None
        view_111: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_21, [1024, mul_127, 96]);  select_21 = None
        permute_83: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_111, [1, 0, 2]);  view_111 = None
        view_112: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_22, [1024, mul_127, 96]);  select_22 = None
        permute_84: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_112, [1, 0, 2]);  view_112 = None
        view_113: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_23, [1024, mul_127, 96]);  select_23 = None
        permute_85: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_113, [1, 0, 2]);  view_113 = None
        view_114: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_83, [primals_1, 8, 1024, 96]);  permute_83 = None
        view_115: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_84, [primals_1, 8, 1024, 96]);  permute_84 = None
        view_116: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_85, [primals_1, 8, 1024, 96]);  permute_85 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_7 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_114, view_115, view_116, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_7);  view_114 = view_115 = view_116 = fwd_rng_state_7 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_93: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_7[0];  graphsafe_run_with_rng_state_7 = None
        permute_86: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_93, [2, 0, 1, 3]);  getitem_93 = None
        view_117: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(permute_86, [mul_15, 768]);  permute_86 = None
        convert_element_type_171: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_95, torch.float16);  primals_95 = None
        convert_element_type_172: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_94, torch.float16)
        permute_87: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_172, [1, 0]);  convert_element_type_172 = None
        
        # No stacktrace found for following nodes
        mm_default_16: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_117, permute_87);  view_117 = permute_87 = None
        add_tensor_16: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_16, convert_element_type_171);  mm_default_16 = convert_element_type_171 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        view_118: "f16[1024, s77, 768]" = torch.ops.aten.reshape.default(add_tensor_16, [1024, primals_1, 768]);  add_tensor_16 = None
        permute_88: "f16[s77, 1024, 768]" = torch.ops.aten.permute.default(view_118, [1, 0, 2]);  view_118 = None
        add_1641: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_1489, permute_88);  permute_88 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_15 = torch.ops.aten.var_mean.correction(add_1641, [2], correction = 0, keepdim = True)
        getitem_102: "f32[s77, 1024, 1]" = var_mean_15[0]
        getitem_103: "f32[s77, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
        add_1646: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
        rsqrt_15: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1646);  add_1646 = None
        sub_410: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1641, getitem_103);  getitem_103 = None
        mul_1520: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_410, rsqrt_15);  sub_410 = rsqrt_15 = None
        mul_1521: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1520, primals_96);  mul_1520 = None
        add_1647: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1521, primals_97);  mul_1521 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_176: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_99, torch.float16)
        convert_element_type_177: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_98, torch.float16)
        convert_element_type_178: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_1647, torch.float16);  add_1647 = None
        view_119: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(convert_element_type_178, [mul_15, 768]);  convert_element_type_178 = None
        permute_89: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_177, [1, 0]);  convert_element_type_177 = None
        
        # No stacktrace found for following nodes
        mm_default_15: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_119, permute_89);  view_119 = permute_89 = None
        add_tensor_15: "f16[1024*s77, 1536]" = torch.ops.aten.add.Tensor(mm_default_15, convert_element_type_176);  mm_default_15 = convert_element_type_176 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_120: "f16[s77, 1024, 1536]" = torch.ops.aten.reshape.default(add_tensor_15, [primals_1, 1024, 1536]);  add_tensor_15 = None
        convert_element_type_182: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_120, torch.float32);  view_120 = None
        mul_1541: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_182, 0.5)
        mul_1542: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_182, 0.7071067811865476);  convert_element_type_182 = None
        erf_7: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_1542);  mul_1542 = None
        add_1674: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_1543: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_1541, add_1674);  mul_1541 = add_1674 = None
        convert_element_type_183: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_1543, torch.float16);  mul_1543 = None
        convert_element_type_184: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_101, torch.float16);  primals_101 = None
        convert_element_type_185: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_100, torch.float16)
        view_121: "f16[1024*s77, 1536]" = torch.ops.aten.reshape.default(convert_element_type_183, [mul_15, 1536]);  convert_element_type_183 = None
        permute_90: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_185, [1, 0]);  convert_element_type_185 = None
        
        # No stacktrace found for following nodes
        mm_default_14: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_121, permute_90);  view_121 = permute_90 = None
        add_tensor_14: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_14, convert_element_type_184);  mm_default_14 = convert_element_type_184 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_122: "f16[s77, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_14, [primals_1, 1024, 768]);  add_tensor_14 = None
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_4: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        convert_element_type_default_56: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_4, torch.float16);  inductor_random_default_4 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_35: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_56, 0.2);  convert_element_type_default_56 = None
        mul_1557: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_35, view_122);  gt_35 = view_122 = None
        mul_1558: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1557, 1.25);  mul_1557 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_1697: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_1641, mul_1558);  mul_1558 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_1697, [2], correction = 0, keepdim = True)
        getitem_104: "f32[s77, 1024, 1]" = var_mean_16[0]
        getitem_105: "f32[s77, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
        add_1702: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_16: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1702);  add_1702 = None
        sub_425: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1697, getitem_105);  getitem_105 = None
        mul_1566: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_425, rsqrt_16);  sub_425 = rsqrt_16 = None
        mul_1567: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1566, primals_102);  mul_1566 = None
        add_1703: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1567, primals_103);  mul_1567 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_91: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_1703, [1, 0, 2]);  add_1703 = None
        convert_element_type_189: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_104, torch.float16)
        convert_element_type_190: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_105, torch.float16)
        convert_element_type_191: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_91, torch.float16);  permute_91 = None
        permute_92: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_190, [1, 0]);  convert_element_type_190 = None
        clone_18: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_191, memory_format = torch.contiguous_format);  convert_element_type_191 = None
        view_123: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(clone_18, [mul_15, 768]);  clone_18 = None
        mm_9: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_123, permute_92);  view_123 = permute_92 = None
        view_124: "f16[1024, s77, 2304]" = torch.ops.aten.reshape.default(mm_9, [1024, primals_1, 2304]);  mm_9 = None
        add_1736: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_124, convert_element_type_189);  view_124 = convert_element_type_189 = None
        view_125: "f16[1024, s77, 3, 768]" = torch.ops.aten.reshape.default(add_1736, [1024, primals_1, 3, 768]);  add_1736 = None
        unsqueeze_14: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_125, 0);  view_125 = None
        permute_93: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_14, [3, 1, 2, 0, 4]);  unsqueeze_14 = None
        squeeze_8: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_93, -2);  permute_93 = None
        clone_19: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_8, memory_format = torch.contiguous_format);  squeeze_8 = None
        select_24: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_19, 0, 0)
        select_25: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_19, 0, 1)
        select_26: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_19, 0, 2);  clone_19 = None
        view_126: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_24, [1024, mul_127, 96]);  select_24 = None
        permute_94: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_126, [1, 0, 2]);  view_126 = None
        view_127: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_25, [1024, mul_127, 96]);  select_25 = None
        permute_95: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_127, [1, 0, 2]);  view_127 = None
        view_128: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_26, [1024, mul_127, 96]);  select_26 = None
        permute_96: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_128, [1, 0, 2]);  view_128 = None
        view_129: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_94, [primals_1, 8, 1024, 96]);  permute_94 = None
        view_130: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_95, [primals_1, 8, 1024, 96]);  permute_95 = None
        view_131: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_96, [primals_1, 8, 1024, 96]);  permute_96 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_8 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_129, view_130, view_131, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_8);  view_129 = view_130 = view_131 = fwd_rng_state_8 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_106: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_8[0];  graphsafe_run_with_rng_state_8 = None
        permute_97: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_106, [2, 0, 1, 3]);  getitem_106 = None
        view_132: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(permute_97, [mul_15, 768]);  permute_97 = None
        convert_element_type_194: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_107, torch.float16);  primals_107 = None
        convert_element_type_195: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_106, torch.float16)
        permute_98: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_195, [1, 0]);  convert_element_type_195 = None
        
        # No stacktrace found for following nodes
        mm_default_13: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_132, permute_98);  view_132 = permute_98 = None
        add_tensor_13: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_13, convert_element_type_194);  mm_default_13 = convert_element_type_194 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        view_133: "f16[1024, s77, 768]" = torch.ops.aten.reshape.default(add_tensor_13, [1024, primals_1, 768]);  add_tensor_13 = None
        permute_99: "f16[s77, 1024, 768]" = torch.ops.aten.permute.default(view_133, [1, 0, 2]);  view_133 = None
        add_1849: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_1697, permute_99);  permute_99 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_17 = torch.ops.aten.var_mean.correction(add_1849, [2], correction = 0, keepdim = True)
        getitem_115: "f32[s77, 1024, 1]" = var_mean_17[0]
        getitem_116: "f32[s77, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
        add_1854: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_115, 1e-05);  getitem_115 = None
        rsqrt_17: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1854);  add_1854 = None
        sub_462: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1849, getitem_116);  getitem_116 = None
        mul_1710: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_462, rsqrt_17);  sub_462 = rsqrt_17 = None
        mul_1711: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1710, primals_108);  mul_1710 = None
        add_1855: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1711, primals_109);  mul_1711 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_199: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_111, torch.float16)
        convert_element_type_200: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_110, torch.float16)
        convert_element_type_201: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_1855, torch.float16);  add_1855 = None
        view_134: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(convert_element_type_201, [mul_15, 768]);  convert_element_type_201 = None
        permute_100: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_200, [1, 0]);  convert_element_type_200 = None
        
        # No stacktrace found for following nodes
        mm_default_12: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_134, permute_100);  view_134 = permute_100 = None
        add_tensor_12: "f16[1024*s77, 1536]" = torch.ops.aten.add.Tensor(mm_default_12, convert_element_type_199);  mm_default_12 = convert_element_type_199 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_135: "f16[s77, 1024, 1536]" = torch.ops.aten.reshape.default(add_tensor_12, [primals_1, 1024, 1536]);  add_tensor_12 = None
        convert_element_type_205: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_135, torch.float32);  view_135 = None
        mul_1731: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_205, 0.5)
        mul_1732: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_205, 0.7071067811865476);  convert_element_type_205 = None
        erf_8: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_1732);  mul_1732 = None
        add_1882: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_1733: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_1731, add_1882);  mul_1731 = add_1882 = None
        convert_element_type_206: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_1733, torch.float16);  mul_1733 = None
        convert_element_type_207: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_113, torch.float16);  primals_113 = None
        convert_element_type_208: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_112, torch.float16)
        view_136: "f16[1024*s77, 1536]" = torch.ops.aten.reshape.default(convert_element_type_206, [mul_15, 1536]);  convert_element_type_206 = None
        permute_101: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_208, [1, 0]);  convert_element_type_208 = None
        
        # No stacktrace found for following nodes
        mm_default_11: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_136, permute_101);  view_136 = permute_101 = None
        add_tensor_11: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_11, convert_element_type_207);  mm_default_11 = convert_element_type_207 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_137: "f16[s77, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_11, [primals_1, 1024, 768]);  add_tensor_11 = None
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_3: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        convert_element_type_default_55: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_3, torch.float16);  inductor_random_default_3 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_39: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_55, 0.2);  convert_element_type_default_55 = None
        mul_1747: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_39, view_137);  gt_39 = view_137 = None
        mul_1748: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1747, 1.25);  mul_1747 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_1905: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_1849, mul_1748);  mul_1748 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_1905, [2], correction = 0, keepdim = True)
        getitem_117: "f32[s77, 1024, 1]" = var_mean_18[0]
        getitem_118: "f32[s77, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
        add_1910: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_117, 1e-05);  getitem_117 = None
        rsqrt_18: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1910);  add_1910 = None
        sub_477: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1905, getitem_118);  getitem_118 = None
        mul_1756: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_477, rsqrt_18);  sub_477 = rsqrt_18 = None
        mul_1757: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1756, primals_114);  mul_1756 = None
        add_1911: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1757, primals_115);  mul_1757 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_102: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_1911, [1, 0, 2]);  add_1911 = None
        convert_element_type_212: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_116, torch.float16)
        convert_element_type_213: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_117, torch.float16)
        convert_element_type_214: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_102, torch.float16);  permute_102 = None
        permute_103: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_213, [1, 0]);  convert_element_type_213 = None
        clone_20: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_214, memory_format = torch.contiguous_format);  convert_element_type_214 = None
        view_138: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(clone_20, [mul_15, 768]);  clone_20 = None
        mm_10: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_138, permute_103);  view_138 = permute_103 = None
        view_139: "f16[1024, s77, 2304]" = torch.ops.aten.reshape.default(mm_10, [1024, primals_1, 2304]);  mm_10 = None
        add_1944: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_139, convert_element_type_212);  view_139 = convert_element_type_212 = None
        view_140: "f16[1024, s77, 3, 768]" = torch.ops.aten.reshape.default(add_1944, [1024, primals_1, 3, 768]);  add_1944 = None
        unsqueeze_15: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_140, 0);  view_140 = None
        permute_104: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_15, [3, 1, 2, 0, 4]);  unsqueeze_15 = None
        squeeze_9: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_104, -2);  permute_104 = None
        clone_21: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_9, memory_format = torch.contiguous_format);  squeeze_9 = None
        select_27: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_21, 0, 0)
        select_28: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_21, 0, 1)
        select_29: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_21, 0, 2);  clone_21 = None
        view_141: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_27, [1024, mul_127, 96]);  select_27 = None
        permute_105: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_141, [1, 0, 2]);  view_141 = None
        view_142: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_28, [1024, mul_127, 96]);  select_28 = None
        permute_106: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_142, [1, 0, 2]);  view_142 = None
        view_143: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_29, [1024, mul_127, 96]);  select_29 = None
        permute_107: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_143, [1, 0, 2]);  view_143 = None
        view_144: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_105, [primals_1, 8, 1024, 96]);  permute_105 = None
        view_145: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_106, [primals_1, 8, 1024, 96]);  permute_106 = None
        view_146: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_107, [primals_1, 8, 1024, 96]);  permute_107 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_9 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_144, view_145, view_146, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_9);  view_144 = view_145 = view_146 = fwd_rng_state_9 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_119: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_9[0];  graphsafe_run_with_rng_state_9 = None
        permute_108: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_119, [2, 0, 1, 3]);  getitem_119 = None
        view_147: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(permute_108, [mul_15, 768]);  permute_108 = None
        convert_element_type_217: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_119, torch.float16);  primals_119 = None
        convert_element_type_218: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_118, torch.float16)
        permute_109: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_218, [1, 0]);  convert_element_type_218 = None
        
        # No stacktrace found for following nodes
        mm_default_10: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_147, permute_109);  view_147 = permute_109 = None
        add_tensor_10: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_10, convert_element_type_217);  mm_default_10 = convert_element_type_217 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        view_148: "f16[1024, s77, 768]" = torch.ops.aten.reshape.default(add_tensor_10, [1024, primals_1, 768]);  add_tensor_10 = None
        permute_110: "f16[s77, 1024, 768]" = torch.ops.aten.permute.default(view_148, [1, 0, 2]);  view_148 = None
        add_2057: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_1905, permute_110);  permute_110 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_19 = torch.ops.aten.var_mean.correction(add_2057, [2], correction = 0, keepdim = True)
        getitem_128: "f32[s77, 1024, 1]" = var_mean_19[0]
        getitem_129: "f32[s77, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
        add_2062: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_19: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_2062);  add_2062 = None
        sub_514: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_2057, getitem_129);  getitem_129 = None
        mul_1900: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_514, rsqrt_19);  sub_514 = rsqrt_19 = None
        mul_1901: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1900, primals_120);  mul_1900 = None
        add_2063: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1901, primals_121);  mul_1901 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_222: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_123, torch.float16)
        convert_element_type_223: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_122, torch.float16)
        convert_element_type_224: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2063, torch.float16);  add_2063 = None
        view_149: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(convert_element_type_224, [mul_15, 768]);  convert_element_type_224 = None
        permute_111: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_223, [1, 0]);  convert_element_type_223 = None
        
        # No stacktrace found for following nodes
        mm_default_9: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_149, permute_111);  view_149 = permute_111 = None
        add_tensor_9: "f16[1024*s77, 1536]" = torch.ops.aten.add.Tensor(mm_default_9, convert_element_type_222);  mm_default_9 = convert_element_type_222 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_150: "f16[s77, 1024, 1536]" = torch.ops.aten.reshape.default(add_tensor_9, [primals_1, 1024, 1536]);  add_tensor_9 = None
        convert_element_type_228: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_150, torch.float32);  view_150 = None
        mul_1921: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_228, 0.5)
        mul_1922: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_228, 0.7071067811865476);  convert_element_type_228 = None
        erf_9: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_1922);  mul_1922 = None
        add_2090: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_1923: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_1921, add_2090);  mul_1921 = add_2090 = None
        convert_element_type_229: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_1923, torch.float16);  mul_1923 = None
        convert_element_type_230: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_125, torch.float16);  primals_125 = None
        convert_element_type_231: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_124, torch.float16)
        view_151: "f16[1024*s77, 1536]" = torch.ops.aten.reshape.default(convert_element_type_229, [mul_15, 1536]);  convert_element_type_229 = None
        permute_112: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_231, [1, 0]);  convert_element_type_231 = None
        
        # No stacktrace found for following nodes
        mm_default_8: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_151, permute_112);  view_151 = permute_112 = None
        add_tensor_8: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_8, convert_element_type_230);  mm_default_8 = convert_element_type_230 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_152: "f16[s77, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_8, [primals_1, 1024, 768]);  add_tensor_8 = None
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_2: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        convert_element_type_default_54: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_2, torch.float16);  inductor_random_default_2 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_43: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_54, 0.2);  convert_element_type_default_54 = None
        mul_1937: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_43, view_152);  gt_43 = view_152 = None
        mul_1938: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1937, 1.25);  mul_1937 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_2113: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2057, mul_1938);  mul_1938 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_2113, [2], correction = 0, keepdim = True)
        getitem_130: "f32[s77, 1024, 1]" = var_mean_20[0]
        getitem_131: "f32[s77, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
        add_2118: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_20: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_2118);  add_2118 = None
        sub_529: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_2113, getitem_131);  getitem_131 = None
        mul_1946: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_529, rsqrt_20);  sub_529 = rsqrt_20 = None
        mul_1947: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1946, primals_126);  mul_1946 = None
        add_2119: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1947, primals_127);  mul_1947 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_113: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_2119, [1, 0, 2]);  add_2119 = None
        convert_element_type_235: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_128, torch.float16)
        convert_element_type_236: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_129, torch.float16)
        convert_element_type_237: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_113, torch.float16);  permute_113 = None
        permute_114: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_236, [1, 0]);  convert_element_type_236 = None
        clone_22: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_237, memory_format = torch.contiguous_format);  convert_element_type_237 = None
        view_153: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(clone_22, [mul_15, 768]);  clone_22 = None
        mm_11: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_153, permute_114);  view_153 = permute_114 = None
        view_154: "f16[1024, s77, 2304]" = torch.ops.aten.reshape.default(mm_11, [1024, primals_1, 2304]);  mm_11 = None
        add_2152: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_154, convert_element_type_235);  view_154 = convert_element_type_235 = None
        view_155: "f16[1024, s77, 3, 768]" = torch.ops.aten.reshape.default(add_2152, [1024, primals_1, 3, 768]);  add_2152 = None
        unsqueeze_16: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_155, 0);  view_155 = None
        permute_115: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_16, [3, 1, 2, 0, 4]);  unsqueeze_16 = None
        squeeze_10: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_115, -2);  permute_115 = None
        clone_23: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_10, memory_format = torch.contiguous_format);  squeeze_10 = None
        select_30: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_23, 0, 0)
        select_31: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_23, 0, 1)
        select_32: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_23, 0, 2);  clone_23 = None
        view_156: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_30, [1024, mul_127, 96]);  select_30 = None
        permute_116: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_156, [1, 0, 2]);  view_156 = None
        view_157: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_31, [1024, mul_127, 96]);  select_31 = None
        permute_117: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_157, [1, 0, 2]);  view_157 = None
        view_158: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_32, [1024, mul_127, 96]);  select_32 = None
        permute_118: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_158, [1, 0, 2]);  view_158 = None
        view_159: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_116, [primals_1, 8, 1024, 96]);  permute_116 = None
        view_160: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_117, [primals_1, 8, 1024, 96]);  permute_117 = None
        view_161: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_118, [primals_1, 8, 1024, 96]);  permute_118 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_10 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_159, view_160, view_161, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_10);  view_159 = view_160 = view_161 = fwd_rng_state_10 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_132: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_10[0];  graphsafe_run_with_rng_state_10 = None
        permute_119: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_132, [2, 0, 1, 3]);  getitem_132 = None
        view_162: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(permute_119, [mul_15, 768]);  permute_119 = None
        convert_element_type_240: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_131, torch.float16);  primals_131 = None
        convert_element_type_241: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_130, torch.float16)
        permute_120: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_241, [1, 0]);  convert_element_type_241 = None
        
        # No stacktrace found for following nodes
        mm_default_7: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_162, permute_120);  view_162 = permute_120 = None
        add_tensor_7: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_7, convert_element_type_240);  mm_default_7 = convert_element_type_240 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        view_163: "f16[1024, s77, 768]" = torch.ops.aten.reshape.default(add_tensor_7, [1024, primals_1, 768]);  add_tensor_7 = None
        permute_121: "f16[s77, 1024, 768]" = torch.ops.aten.permute.default(view_163, [1, 0, 2]);  view_163 = None
        add_2265: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2113, permute_121);  permute_121 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_21 = torch.ops.aten.var_mean.correction(add_2265, [2], correction = 0, keepdim = True)
        getitem_141: "f32[s77, 1024, 1]" = var_mean_21[0]
        getitem_142: "f32[s77, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
        add_2270: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_141, 1e-05);  getitem_141 = None
        rsqrt_21: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_2270);  add_2270 = None
        sub_566: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_2265, getitem_142);  getitem_142 = None
        mul_2090: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_566, rsqrt_21);  sub_566 = rsqrt_21 = None
        mul_2091: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2090, primals_132);  mul_2090 = None
        add_2271: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_2091, primals_133);  mul_2091 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_245: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_135, torch.float16)
        convert_element_type_246: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_134, torch.float16)
        convert_element_type_247: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2271, torch.float16);  add_2271 = None
        view_164: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(convert_element_type_247, [mul_15, 768]);  convert_element_type_247 = None
        permute_122: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_246, [1, 0]);  convert_element_type_246 = None
        
        # No stacktrace found for following nodes
        mm_default_6: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_164, permute_122);  view_164 = permute_122 = None
        add_tensor_6: "f16[1024*s77, 1536]" = torch.ops.aten.add.Tensor(mm_default_6, convert_element_type_245);  mm_default_6 = convert_element_type_245 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_165: "f16[s77, 1024, 1536]" = torch.ops.aten.reshape.default(add_tensor_6, [primals_1, 1024, 1536]);  add_tensor_6 = None
        convert_element_type_251: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_165, torch.float32);  view_165 = None
        mul_2111: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_251, 0.5)
        mul_2112: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_251, 0.7071067811865476);  convert_element_type_251 = None
        erf_10: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_2112);  mul_2112 = None
        add_2298: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_2113: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_2111, add_2298);  mul_2111 = add_2298 = None
        convert_element_type_252: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_2113, torch.float16);  mul_2113 = None
        convert_element_type_253: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_137, torch.float16);  primals_137 = None
        convert_element_type_254: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_136, torch.float16)
        view_166: "f16[1024*s77, 1536]" = torch.ops.aten.reshape.default(convert_element_type_252, [mul_15, 1536]);  convert_element_type_252 = None
        permute_123: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_254, [1, 0]);  convert_element_type_254 = None
        
        # No stacktrace found for following nodes
        mm_default_5: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_166, permute_123);  view_166 = permute_123 = None
        add_tensor_5: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_5, convert_element_type_253);  mm_default_5 = convert_element_type_253 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_167: "f16[s77, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_5, [primals_1, 1024, 768]);  add_tensor_5 = None
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_1: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        convert_element_type_default_53: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.float16);  inductor_random_default_1 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_47: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_53, 0.2);  convert_element_type_default_53 = None
        mul_2127: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_47, view_167);  gt_47 = view_167 = None
        mul_2128: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2127, 1.25);  mul_2127 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_2321: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2265, mul_2128);  mul_2128 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:121 in _attention_forward, code: y = self.ln1(x)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_2321, [2], correction = 0, keepdim = True)
        getitem_143: "f32[s77, 1024, 1]" = var_mean_22[0]
        getitem_144: "f32[s77, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
        add_2326: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_143, 1e-05);  getitem_143 = None
        rsqrt_22: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_2326);  add_2326 = None
        sub_581: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_2321, getitem_144);  getitem_144 = None
        mul_2136: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_581, rsqrt_22);  sub_581 = rsqrt_22 = None
        mul_2137: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2136, primals_138);  mul_2136 = None
        add_2327: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_2137, primals_139);  mul_2137 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        permute_124: "f32[1024, s77, 768]" = torch.ops.aten.permute.default(add_2327, [1, 0, 2]);  add_2327 = None
        convert_element_type_258: "f16[2304]" = torch.ops.prims.convert_element_type.default(primals_140, torch.float16)
        convert_element_type_259: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(primals_141, torch.float16)
        convert_element_type_260: "f16[1024, s77, 768]" = torch.ops.prims.convert_element_type.default(permute_124, torch.float16);  permute_124 = None
        permute_125: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_259, [1, 0]);  convert_element_type_259 = None
        clone_24: "f16[1024, s77, 768]" = torch.ops.aten.clone.default(convert_element_type_260, memory_format = torch.contiguous_format);  convert_element_type_260 = None
        view_168: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(clone_24, [mul_15, 768]);  clone_24 = None
        mm_12: "f16[1024*s77, 2304]" = torch.ops.aten.mm.default(view_168, permute_125);  view_168 = permute_125 = None
        view_169: "f16[1024, s77, 2304]" = torch.ops.aten.reshape.default(mm_12, [1024, primals_1, 2304]);  mm_12 = None
        add_2360: "f16[1024, s77, 2304]" = torch.ops.aten.add.Tensor(view_169, convert_element_type_258);  view_169 = convert_element_type_258 = None
        view_170: "f16[1024, s77, 3, 768]" = torch.ops.aten.reshape.default(add_2360, [1024, primals_1, 3, 768]);  add_2360 = None
        unsqueeze_17: "f16[1, 1024, s77, 3, 768]" = torch.ops.aten.unsqueeze.default(view_170, 0);  view_170 = None
        permute_126: "f16[3, 1024, s77, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_17, [3, 1, 2, 0, 4]);  unsqueeze_17 = None
        squeeze_11: "f16[3, 1024, s77, 768]" = torch.ops.aten.squeeze.dim(permute_126, -2);  permute_126 = None
        clone_25: "f16[3, 1024, s77, 768]" = torch.ops.aten.clone.default(squeeze_11, memory_format = torch.contiguous_format);  squeeze_11 = None
        select_33: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_25, 0, 0)
        select_34: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_25, 0, 1)
        select_35: "f16[1024, s77, 768]" = torch.ops.aten.select.int(clone_25, 0, 2);  clone_25 = None
        view_171: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_33, [1024, mul_127, 96]);  select_33 = None
        permute_127: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_171, [1, 0, 2]);  view_171 = None
        view_172: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_34, [1024, mul_127, 96]);  select_34 = None
        permute_128: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_172, [1, 0, 2]);  view_172 = None
        view_173: "f16[1024, 8*s77, 96]" = torch.ops.aten.reshape.default(select_35, [1024, mul_127, 96]);  select_35 = None
        permute_129: "f16[8*s77, 1024, 96]" = torch.ops.aten.permute.default(view_173, [1, 0, 2]);  view_173 = None
        view_174: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_127, [primals_1, 8, 1024, 96]);  permute_127 = None
        view_175: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_128, [primals_1, 8, 1024, 96]);  permute_128 = None
        view_176: "f16[s77, 8, 1024, 96]" = torch.ops.aten.reshape.default(permute_129, [primals_1, 8, 1024, 96]);  permute_129 = None
        
        # No stacktrace found for following nodes
        graphsafe_run_with_rng_state_11 = torch.ops.higher_order.graphsafe_run_with_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, view_174, view_175, view_176, 0.2, scale = 0.10206207261596577, rng_state = fwd_rng_state_11);  view_174 = view_175 = view_176 = fwd_rng_state_11 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        getitem_145: "f16[s77, 8, 1024, 96]" = graphsafe_run_with_rng_state_11[0];  graphsafe_run_with_rng_state_11 = None
        permute_130: "f16[1024, s77, 8, 96]" = torch.ops.aten.permute.default(getitem_145, [2, 0, 1, 3]);  getitem_145 = None
        view_177: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(permute_130, [mul_15, 768]);  permute_130 = None
        convert_element_type_263: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_143, torch.float16);  primals_143 = None
        convert_element_type_264: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(primals_142, torch.float16)
        permute_131: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_264, [1, 0]);  convert_element_type_264 = None
        
        # No stacktrace found for following nodes
        mm_default_4: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_177, permute_131);  view_177 = permute_131 = None
        add_tensor_4: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_4, convert_element_type_263);  mm_default_4 = convert_element_type_263 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:122 in _attention_forward, code: return x + self.mha(y, y, y, need_weights=False)[0]
        view_178: "f16[1024, s77, 768]" = torch.ops.aten.reshape.default(add_tensor_4, [1024, primals_1, 768]);  add_tensor_4 = None
        permute_132: "f16[s77, 1024, 768]" = torch.ops.aten.permute.default(view_178, [1, 0, 2]);  view_178 = None
        add_2473: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2321, permute_132);  permute_132 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        var_mean_23 = torch.ops.aten.var_mean.correction(add_2473, [2], correction = 0, keepdim = True)
        getitem_154: "f32[s77, 1024, 1]" = var_mean_23[0]
        getitem_155: "f32[s77, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
        add_2478: "f32[s77, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
        rsqrt_23: "f32[s77, 1024, 1]" = torch.ops.aten.rsqrt.default(add_2478);  add_2478 = None
        sub_618: "f32[s77, 1024, 768]" = torch.ops.aten.sub.Tensor(add_2473, getitem_155);  getitem_155 = None
        mul_2280: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_618, rsqrt_23);  sub_618 = rsqrt_23 = None
        mul_2281: "f32[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2280, primals_144);  mul_2280 = None
        add_2479: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(mul_2281, primals_145);  mul_2281 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        convert_element_type_268: "f16[1536]" = torch.ops.prims.convert_element_type.default(primals_147, torch.float16)
        convert_element_type_269: "f16[1536, 768]" = torch.ops.prims.convert_element_type.default(primals_146, torch.float16)
        convert_element_type_270: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2479, torch.float16);  add_2479 = None
        view_179: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(convert_element_type_270, [mul_15, 768]);  convert_element_type_270 = None
        permute_133: "f16[768, 1536]" = torch.ops.aten.permute.default(convert_element_type_269, [1, 0]);  convert_element_type_269 = None
        
        # No stacktrace found for following nodes
        mm_default_3: "f16[1024*s77, 1536]" = torch.ops.aten.mm.default(view_179, permute_133);  view_179 = permute_133 = None
        add_tensor_3: "f16[1024*s77, 1536]" = torch.ops.aten.add.Tensor(mm_default_3, convert_element_type_268);  mm_default_3 = convert_element_type_268 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_180: "f16[s77, 1024, 1536]" = torch.ops.aten.reshape.default(add_tensor_3, [primals_1, 1024, 1536]);  add_tensor_3 = None
        convert_element_type_274: "f32[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(view_180, torch.float32);  view_180 = None
        mul_2301: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_274, 0.5)
        mul_2302: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_274, 0.7071067811865476);  convert_element_type_274 = None
        erf_11: "f32[s77, 1024, 1536]" = torch.ops.aten.erf.default(mul_2302);  mul_2302 = None
        add_2506: "f32[s77, 1024, 1536]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_2303: "f32[s77, 1024, 1536]" = torch.ops.aten.mul.Tensor(mul_2301, add_2506);  mul_2301 = add_2506 = None
        convert_element_type_275: "f16[s77, 1024, 1536]" = torch.ops.prims.convert_element_type.default(mul_2303, torch.float16);  mul_2303 = None
        convert_element_type_276: "f16[768]" = torch.ops.prims.convert_element_type.default(primals_149, torch.float16);  primals_149 = None
        convert_element_type_277: "f16[768, 1536]" = torch.ops.prims.convert_element_type.default(primals_148, torch.float16)
        view_181: "f16[1024*s77, 1536]" = torch.ops.aten.reshape.default(convert_element_type_275, [mul_15, 1536]);  convert_element_type_275 = None
        permute_134: "f16[1536, 768]" = torch.ops.aten.permute.default(convert_element_type_277, [1, 0]);  convert_element_type_277 = None
        
        # No stacktrace found for following nodes
        mm_default_2: "f16[1024*s77, 768]" = torch.ops.aten.mm.default(view_181, permute_134);  view_181 = permute_134 = None
        add_tensor_2: "f16[1024*s77, 768]" = torch.ops.aten.add.Tensor(mm_default_2, convert_element_type_276);  mm_default_2 = convert_element_type_276 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        view_182: "f16[s77, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_2, [primals_1, 1024, 768]);  add_tensor_2 = None
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default: "f32[s77, 1024, 768]" = torch.ops.prims.inductor_random.default([primals_1, 1024, 768], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        convert_element_type_default_52: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:96 in forward, code: return self.layers(x)
        gt_51: "b8[s77, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_52, 0.2);  convert_element_type_default_52 = None
        mul_2317: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_51, view_182);  gt_51 = view_182 = None
        mul_2318: "f16[s77, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2317, 1.25);  mul_2317 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:125 in _mlp_forward, code: return x + self.mlp(self.ln2(x))
        add_2529: "f32[s77, 1024, 768]" = torch.ops.aten.add.Tensor(add_2473, mul_2318);  mul_2318 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:165 in forward, code: x = self.progressive_projection(x)
        convert_element_type_281: "f16[128]" = torch.ops.prims.convert_element_type.default(primals_151, torch.float16);  primals_151 = None
        convert_element_type_282: "f16[128, 768]" = torch.ops.prims.convert_element_type.default(primals_150, torch.float16);  primals_150 = None
        convert_element_type_283: "f16[s77, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2529, torch.float16);  add_2529 = None
        view_183: "f16[1024*s77, 768]" = torch.ops.aten.reshape.default(convert_element_type_283, [mul_15, 768]);  convert_element_type_283 = None
        permute_135: "f16[768, 128]" = torch.ops.aten.permute.default(convert_element_type_282, [1, 0]);  convert_element_type_282 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "f16[1024*s77, 128]" = torch.ops.aten.mm.default(view_183, permute_135)
        add_tensor_1: "f16[1024*s77, 128]" = torch.ops.aten.add.Tensor(mm_default_1, convert_element_type_281);  mm_default_1 = convert_element_type_281 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:165 in forward, code: x = self.progressive_projection(x)
        view_184: "f16[s77, 1024, 128]" = torch.ops.aten.reshape.default(add_tensor_1, [primals_1, 1024, 128]);  add_tensor_1 = None
        relu: "f16[s77, 1024, 128]" = torch.ops.aten.relu.default(view_184);  view_184 = None
        view_185: "f16[1024*s77, 128]" = torch.ops.aten.reshape.default(relu, [mul_15, 128]);  relu = None
        convert_element_type_287: "f16[16]" = torch.ops.prims.convert_element_type.default(primals_153, torch.float16);  primals_153 = None
        convert_element_type_288: "f16[16, 128]" = torch.ops.prims.convert_element_type.default(primals_152, torch.float16);  primals_152 = None
        permute_136: "f16[128, 16]" = torch.ops.aten.permute.default(convert_element_type_288, [1, 0]);  convert_element_type_288 = None
        
        # No stacktrace found for following nodes
        mm_default: "f16[1024*s77, 16]" = torch.ops.aten.mm.default(view_185, permute_136)
        add_tensor: "f16[1024*s77, 16]" = torch.ops.aten.add.Tensor(mm_default, convert_element_type_287);  mm_default = convert_element_type_287 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:165 in forward, code: x = self.progressive_projection(x)
        view_191: "f16[s77, 1024, 16]" = torch.ops.aten.reshape.default(add_tensor, [primals_1, 1024, 16]);  add_tensor = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:174 in forward, code: x = x.permute(0, 2, 1)
        permute_137: "f16[s77, 16, 1024]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:175 in forward, code: x = self.fold(x)
        convert_element_type_292: "f32[s77, 16, 1024]" = torch.ops.prims.convert_element_type.default(permute_137, torch.float32);  permute_137 = None
        view_192: "f32[s77, 1, 4, 4, 32, 32]" = torch.ops.aten.reshape.default(convert_element_type_292, [primals_1, 1, 4, 4, 32, 32]);  convert_element_type_292 = None
        permute_138: "f32[s77, 1, 4, 32, 4, 32]" = torch.ops.aten.permute.default(view_192, [0, 1, 2, 4, 3, 5]);  view_192 = None
        full_default: "f32[s77, 1, 128, 128]" = torch.ops.aten.full.default([primals_1, 1, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[s77, 1, 128, 128]" = torch.ops.aten.index_put.default(full_default, [None, None, unsqueeze_5, add], permute_138, True);  unsqueeze_5 = add = permute_138 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:179 in forward, code: x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        iota_8: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_295: "f32[32]" = torch.ops.prims.convert_element_type.default(iota_8, torch.float32);  iota_8 = None
        add_2590: "f32[32]" = torch.ops.aten.add.Tensor(convert_element_type_295, 0.5);  convert_element_type_295 = None
        mul_2370: "f32[32]" = torch.ops.aten.mul.Tensor(add_2590, 4.0);  add_2590 = None
        sub_647: "f32[32]" = torch.ops.aten.sub.Tensor(mul_2370, 0.5);  mul_2370 = None
        clamp_min: "f32[32]" = torch.ops.aten.clamp_min.default(sub_647, 0.0);  sub_647 = None
        view_193: "f32[32, 1]" = torch.ops.aten.reshape.default(clamp_min, [32, 1])
        convert_element_type_296: "i64[32, 1]" = torch.ops.prims.convert_element_type.default(view_193, torch.int64)
        add_2591: "i64[32, 1]" = torch.ops.aten.add.Tensor(convert_element_type_296, 1)
        clamp_max: "i64[32, 1]" = torch.ops.aten.clamp_max.default(add_2591, 127);  add_2591 = None
        convert_element_type_298: "i64[32]" = torch.ops.prims.convert_element_type.default(clamp_min, torch.int64)
        add_2593: "i64[32]" = torch.ops.aten.add.Tensor(convert_element_type_298, 1)
        clamp_max_1: "i64[32]" = torch.ops.aten.clamp_max.default(add_2593, 127);  add_2593 = None
        _unsafe_index: "f32[s77, 1, 32, 32]" = torch.ops.aten._unsafe_index.Tensor(index_put, [None, None, convert_element_type_296, convert_element_type_298])
        _unsafe_index_1: "f32[s77, 1, 32, 32]" = torch.ops.aten._unsafe_index.Tensor(index_put, [None, None, convert_element_type_296, clamp_max_1])
        _unsafe_index_2: "f32[s77, 1, 32, 32]" = torch.ops.aten._unsafe_index.Tensor(index_put, [None, None, clamp_max, convert_element_type_298])
        _unsafe_index_3: "f32[s77, 1, 32, 32]" = torch.ops.aten._unsafe_index.Tensor(index_put, [None, None, clamp_max, clamp_max_1]);  index_put = None
        sub_653: "f32[32]" = torch.ops.aten.sub.Tensor(clamp_min, convert_element_type_298);  clamp_min = None
        clamp_min_2: "f32[32]" = torch.ops.aten.clamp_min.default(sub_653, 0.0);  sub_653 = None
        clamp_max_2: "f32[32]" = torch.ops.aten.clamp_max.default(clamp_min_2, 1.0);  clamp_min_2 = None
        sub_654: "f32[s77, 1, 32, 32]" = torch.ops.aten.sub.Tensor(_unsafe_index_1, _unsafe_index);  _unsafe_index_1 = None
        mul_2382: "f32[s77, 1, 32, 32]" = torch.ops.aten.mul.Tensor(sub_654, clamp_max_2);  sub_654 = None
        add_2624: "f32[s77, 1, 32, 32]" = torch.ops.aten.add.Tensor(_unsafe_index, mul_2382);  _unsafe_index = mul_2382 = None
        sub_658: "f32[s77, 1, 32, 32]" = torch.ops.aten.sub.Tensor(_unsafe_index_3, _unsafe_index_2);  _unsafe_index_3 = None
        mul_2389: "f32[s77, 1, 32, 32]" = torch.ops.aten.mul.Tensor(sub_658, clamp_max_2);  sub_658 = None
        add_2640: "f32[s77, 1, 32, 32]" = torch.ops.aten.add.Tensor(_unsafe_index_2, mul_2389);  _unsafe_index_2 = mul_2389 = None
        sub_662: "f32[32, 1]" = torch.ops.aten.sub.Tensor(view_193, convert_element_type_296);  view_193 = None
        clamp_min_3: "f32[32, 1]" = torch.ops.aten.clamp_min.default(sub_662, 0.0);  sub_662 = None
        clamp_max_3: "f32[32, 1]" = torch.ops.aten.clamp_max.default(clamp_min_3, 1.0);  clamp_min_3 = None
        sub_663: "f32[s77, 1, 32, 32]" = torch.ops.aten.sub.Tensor(add_2640, add_2624);  add_2640 = None
        mul_2396: "f32[s77, 1, 32, 32]" = torch.ops.aten.mul.Tensor(sub_663, clamp_max_3);  sub_663 = None
        add_2656: "f32[s77, 1, 32, 32]" = torch.ops.aten.add.Tensor(add_2624, mul_2396);  add_2624 = mul_2396 = None
        convert_element_type_299: "f16[s77, 1, 32, 32]" = torch.ops.prims.convert_element_type.default(add_2656, torch.float16);  add_2656 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/model.py:165 in forward, code: x = self.progressive_projection(x)
        permute_141: "f16[16, 128]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
        permute_145: "f16[128, 768]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
        return (convert_element_type_299, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, primals_16, primals_18, primals_19, primals_20, primals_21, primals_22, primals_24, primals_25, primals_26, primals_27, primals_28, primals_30, primals_31, primals_32, primals_33, primals_34, primals_36, primals_37, primals_38, primals_39, primals_40, primals_42, primals_43, primals_44, primals_45, primals_46, primals_48, primals_49, primals_50, primals_51, primals_52, primals_54, primals_55, primals_56, primals_57, primals_58, primals_60, primals_61, primals_62, primals_63, primals_64, primals_66, primals_67, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_75, primals_76, primals_78, primals_79, primals_80, primals_81, primals_82, primals_84, primals_85, primals_86, primals_87, primals_88, primals_90, primals_91, primals_92, primals_93, primals_94, primals_96, primals_97, primals_98, primals_99, primals_100, primals_102, primals_103, primals_104, primals_105, primals_106, primals_108, primals_109, primals_110, primals_111, primals_112, primals_114, primals_115, primals_116, primals_117, primals_118, primals_120, primals_121, primals_122, primals_123, primals_124, primals_126, primals_127, primals_128, primals_129, primals_130, primals_132, primals_133, primals_134, primals_135, primals_136, primals_138, primals_139, primals_140, primals_141, primals_142, primals_144, primals_145, primals_146, primals_147, primals_148, view_1, mm, add_185, inductor_seeds_default, add_241, add_393, add_449, add_601, add_657, add_809, add_865, add_1017, add_1073, add_1225, add_1281, add_1433, add_1489, add_1641, add_1697, add_1849, add_1905, add_2057, add_2113, add_2265, add_2321, add_2473, view_183, view_185, full_default, convert_element_type_296, clamp_max, convert_element_type_298, clamp_max_1, clamp_max_2, clamp_max_3, permute_141, permute_145, primals_1, mul_15, mul_127)
        