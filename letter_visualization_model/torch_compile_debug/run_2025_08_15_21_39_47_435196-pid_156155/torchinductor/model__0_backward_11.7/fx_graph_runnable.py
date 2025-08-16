
import os
os.environ['TORCHDYNAMO_VERBOSE'] = '1'
os.environ['TORCH_TRACE'] = './logs.txt'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_LOGS'] = 'recompiles,+dynamo'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_Adithya'
os.environ['TRITON_CACHE_DIR'] = '/tmp/torchinductor_Adithya/triton/0'

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
torch._dynamo.config.recompile_limit = 2048
torch._dynamo.config.accumulated_recompile_limit = 2048
torch._inductor.config.triton.store_cubin = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = False



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

    
    
    def forward(self, primals_1, sigmoid, add_2, div_3, tangents_1, tangents_2):
        neg = torch.ops.aten.neg.default(tangents_2);  tangents_2 = None
        expand = torch.ops.aten.expand.default(neg, [9, 1]);  neg = None
        div_1 = torch.ops.aten.div.Scalar(expand, 9);  expand = None
        neg_1 = torch.ops.aten.neg.default(div_1)
        mul_2 = torch.ops.aten.mul.Tensor(neg_1, div_3);  neg_1 = div_3 = None
        div_4 = torch.ops.aten.div.Tensor(div_1, add_2);  div_1 = add_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(div_4, 2.0);  div_4 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(mul_2, 2);  mul_2 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 3);  unsqueeze = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_1, [9, 1, 32, 32]);  unsqueeze_1 = None
        add_3 = torch.ops.aten.add.Tensor(tangents_1, expand_1);  tangents_1 = expand_1 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(mul_3, 2);  mul_3 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None
        expand_2 = torch.ops.aten.expand.default(unsqueeze_3, [9, 1, 32, 32]);  unsqueeze_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(expand_2, primals_1);  expand_2 = primals_1 = None
        add_4 = torch.ops.aten.add.Tensor(add_3, mul_4);  add_3 = mul_4 = None
        sub_1 = torch.ops.aten.sub.Tensor(1, sigmoid)
        mul_5 = torch.ops.aten.mul.Tensor(sigmoid, sub_1);  sigmoid = sub_1 = None
        mul_6 = torch.ops.aten.mul.Tensor(add_4, mul_5);  add_4 = mul_5 = None
        return (None, mul_6)
        
def load_args(reader):
    buf0 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf0, (9, 1, 32, 32), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf1, (9, 1, 32, 32), is_leaf=True)  # sigmoid
    buf2 = reader.storage(None, 36, device=device(type='cuda', index=0))
    reader.tensor(buf2, (9, 1), is_leaf=True)  # add_2
    buf3 = reader.storage(None, 36, device=device(type='cuda', index=0))
    reader.tensor(buf3, (9, 1), is_leaf=True)  # div_3
    buf4 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf4, (9, 1, 32, 32), is_leaf=True)  # tangents_1
    buf5 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf5, (), is_leaf=True)  # tangents_2
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)