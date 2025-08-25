class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[15799]", arg1_1: "f32[20, 1, 32, 32]", arg2_1: "b8[20, 1, 32, 32]"):
         # File: /home/Adithya/Documents/ves/letter_visualization_model/loss.py:142 in torch_dynamo_resume_in_compute_signed_distance_map_gpu_at_142, code: phi_G[gt_bool] = -dist_inside[gt_bool]
        neg: "f32[15799]" = torch.ops.aten.neg.default(arg0_1);  arg0_1 = None
        index_put: "f32[20, 1, 32, 32]" = torch.ops.aten.index_put.default(arg1_1, [arg2_1], neg);  arg2_1 = neg = None
        copy_: "f32[20, 1, 32, 32]" = torch.ops.aten.copy_.default(arg1_1, index_put);  arg1_1 = index_put = None
        return (copy_,)
        