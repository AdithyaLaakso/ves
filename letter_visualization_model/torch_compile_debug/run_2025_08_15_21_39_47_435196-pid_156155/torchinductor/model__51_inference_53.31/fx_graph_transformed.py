class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s37)"):
         # File: /home/Adithya/Documents/ves/letter_visualization_model/loss.py:51 in torch_dynamo_resume_in_forward_at_51, code: self.running_dice_loss += self.dice_weight * dice_val.item()
        sym_float: "Sym(ToFloat(s37))" = torch.sym_float(arg0_1);  arg0_1 = None
        add: "Sym(ToFloat(s37) + 2.02343240380287)" = sym_float + 2.0234324038028717;  sym_float = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/loss.py:52 in torch_dynamo_resume_in_forward_at_51, code: self.running_b_loss += (self.boundary_weight * boundary_val.item())
        scalar_tensor: "f64[]" = torch.ops.aten.scalar_tensor.default(add, dtype = torch.float64, device = device(type='cpu'), pin_memory = False);  add = None
        return (scalar_tensor,)
        