class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[20, 1, 32, 32]", sigmoid: "f32[20, 1, 32, 32]", add_2: "f32[20, 1]", div_3: "f32[20, 1]", tangents_1: "f32[20, 1, 32, 32]", tangents_2: "f32[]"):
         # File: /home/Adithya/Documents/ves/letter_visualization_model/loss.py:86 in dice_loss, code: return 1 - dice.mean()
        neg: "f32[]" = torch.ops.aten.neg.default(tangents_2);  tangents_2 = None
        expand: "f32[20, 1]" = torch.ops.aten.expand.default(neg, [20, 1]);  neg = None
        div_1: "f32[20, 1]" = torch.ops.aten.div.Scalar(expand, 20);  expand = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/loss.py:85 in dice_loss, code: dice = (2. * intersection + epsilon) / (union + epsilon)
        neg_1: "f32[20, 1]" = torch.ops.aten.neg.default(div_1)
        mul_2: "f32[20, 1]" = torch.ops.aten.mul.Tensor(neg_1, div_3);  neg_1 = div_3 = None
        div_4: "f32[20, 1]" = torch.ops.aten.div.Tensor(div_1, add_2);  div_1 = add_2 = None
        mul_3: "f32[20, 1]" = torch.ops.aten.mul.Tensor(div_4, 2.0);  div_4 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/loss.py:84 in dice_loss, code: union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        unsqueeze: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_2, 2);  mul_2 = None
        unsqueeze_1: "f32[20, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 3);  unsqueeze = None
        expand_1: "f32[20, 1, 32, 32]" = torch.ops.aten.expand.default(unsqueeze_1, [20, 1, 32, 32]);  unsqueeze_1 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/loss.py:84 in dice_loss, code: union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        add_3: "f32[20, 1, 32, 32]" = torch.ops.aten.add.Tensor(tangents_1, expand_1);  tangents_1 = expand_1 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/loss.py:83 in dice_loss, code: intersection = (pred * target).sum(dim=(2, 3))
        unsqueeze_2: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(mul_3, 2);  mul_3 = None
        unsqueeze_3: "f32[20, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None
        expand_2: "f32[20, 1, 32, 32]" = torch.ops.aten.expand.default(unsqueeze_3, [20, 1, 32, 32]);  unsqueeze_3 = None
        mul_4: "f32[20, 1, 32, 32]" = torch.ops.aten.mul.Tensor(expand_2, primals_1);  expand_2 = primals_1 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/loss.py:83 in dice_loss, code: intersection = (pred * target).sum(dim=(2, 3))
        add_4: "f32[20, 1, 32, 32]" = torch.ops.aten.add.Tensor(add_3, mul_4);  add_3 = mul_4 = None
        
         # File: /home/Adithya/Documents/ves/letter_visualization_model/loss.py:33 in forward, code: pred_probs = torch.sigmoid(pred_masks)
        sub_1: "f32[20, 1, 32, 32]" = torch.ops.aten.sub.Tensor(1, sigmoid)
        mul_5: "f32[20, 1, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid, sub_1);  sigmoid = sub_1 = None
        mul_6: "f32[20, 1, 32, 32]" = torch.ops.aten.mul.Tensor(add_4, mul_5);  add_4 = mul_5 = None
        return (None, mul_6)
        