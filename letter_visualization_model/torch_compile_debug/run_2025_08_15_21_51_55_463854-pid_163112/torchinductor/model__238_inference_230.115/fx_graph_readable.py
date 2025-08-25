class <lambda>(torch.nn.Module):
    def forward(self):
         # File: /home/Adithya/Documents/ves/letter_visualization_model/loss.py:52 in torch_dynamo_resume_in_forward_at_51, code: self.running_b_loss += (self.boundary_weight * boundary_val.item())
        full_default: "f64[]" = torch.ops.aten.full.default([], 164.4388611614704, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        return (full_default,)
        