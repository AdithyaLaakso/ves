class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[9, 1, 32, 32]", primals_2: "f32[9, 1, 32, 32]"):
         # File: /home/Adithya/Documents/ves/letter_visualization_model/loss.py:149 in torch_dynamo_resume_in_boundary_loss_at_148, code: return (phi_G * s_theta).mean()
        mul: "f32[9, 1, 32, 32]" = torch.ops.aten.mul.Tensor(primals_1, primals_2);  primals_2 = None
        mean: "f32[]" = torch.ops.aten.mean.default(mul);  mul = None
        return (mean, primals_1)
        