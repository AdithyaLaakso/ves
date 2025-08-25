class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[9, 1, 32, 32]", primals_2: "f32[9, 1, 32, 32]"):
         # File: /home/Adithya/Documents/ves/letter_visualization_model/loss.py:33 in forward, code: pred_probs = torch.sigmoid(pred_masks)
        sigmoid: "f32[9, 1, 32, 32]" = torch.ops.aten.sigmoid.default(primals_2);  primals_2 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/loss.py:83 in dice_loss, code: intersection = (pred * target).sum(dim=(2, 3))
        mul: "f32[9, 1, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid, primals_1)
        sum_1: "f32[9, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2, 3], dtype = torch.float32);  mul = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/loss.py:84 in dice_loss, code: union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        sum_2: "f32[9, 1]" = torch.ops.aten.sum.dim_IntList(sigmoid, [2, 3], dtype = torch.float32)
        sum_3: "f32[9, 1]" = torch.ops.aten.sum.dim_IntList(primals_1, [2, 3], dtype = torch.float32)
        add: "f32[9, 1]" = torch.ops.aten.add.Tensor(sum_2, sum_3);  sum_2 = sum_3 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/loss.py:85 in dice_loss, code: dice = (2. * intersection + epsilon) / (union + epsilon)
        mul_1: "f32[9, 1]" = torch.ops.aten.mul.Tensor(sum_1, 2.0);  sum_1 = None
        add_1: "f32[9, 1]" = torch.ops.aten.add.Tensor(mul_1, 1e-06);  mul_1 = None
        add_2: "f32[9, 1]" = torch.ops.aten.add.Tensor(add, 1e-06);  add = None
        div: "f32[9, 1]" = torch.ops.aten.div.Tensor(add_1, add_2);  add_1 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/loss.py:86 in dice_loss, code: return 1 - dice.mean()
        mean: "f32[]" = torch.ops.aten.mean.default(div)
        sub: "f32[]" = torch.ops.aten.sub.Tensor(1, mean);  mean = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/loss.py:85 in dice_loss, code: dice = (2. * intersection + epsilon) / (union + epsilon)
        div_3: "f32[9, 1]" = torch.ops.aten.div.Tensor(div, add_2);  div = None
        return (sigmoid, primals_1, sub, primals_1, sigmoid, add_2, div_3)
        