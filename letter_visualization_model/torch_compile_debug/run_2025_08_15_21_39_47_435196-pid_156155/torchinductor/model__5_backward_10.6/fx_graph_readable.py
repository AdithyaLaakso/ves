class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[9, 1, 32, 32]", tangents_1: "f32[]"):
         # File: /home/Adithya/Documents/ves/letter_visualization_model/loss.py:149 in torch_dynamo_resume_in_boundary_loss_at_148, code: return (phi_G * s_theta).mean()
        expand: "f32[9, 1, 32, 32]" = torch.ops.aten.expand.default(tangents_1, [9, 1, 32, 32]);  tangents_1 = None
        div: "f32[9, 1, 32, 32]" = torch.ops.aten.div.Scalar(expand, 9216);  expand = None
        mul_1: "f32[9, 1, 32, 32]" = torch.ops.aten.mul.Tensor(div, primals_1);  div = primals_1 = None
        return (None, mul_1)
        