
import os
os.environ['TORCHDYNAMO_VERBOSE'] = '1'
os.environ['TORCH_TRACE'] = './logs.txt'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_LOGS'] = 'recompiles,+dynamo'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_Adithya'

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
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



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

    
    
    def forward(self, primals_1, primals_2):
        sigmoid = torch.ops.aten.sigmoid.default(primals_2);  primals_2 = None
        mul = torch.ops.aten.mul.Tensor(sigmoid, primals_1)
        sum_1 = torch.ops.aten.sum.dim_IntList(mul, [2, 3], dtype = torch.float32);  mul = None
        sum_2 = torch.ops.aten.sum.dim_IntList(sigmoid, [2, 3], dtype = torch.float32)
        sum_3 = torch.ops.aten.sum.dim_IntList(primals_1, [2, 3], dtype = torch.float32)
        add = torch.ops.aten.add.Tensor(sum_2, sum_3);  sum_2 = sum_3 = None
        mul_1 = torch.ops.aten.mul.Tensor(sum_1, 2.0);  sum_1 = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, 1e-06);  mul_1 = None
        add_2 = torch.ops.aten.add.Tensor(add, 1e-06);  add = None
        div = torch.ops.aten.div.Tensor(add_1, add_2);  add_1 = None
        mean = torch.ops.aten.mean.default(div)
        sub = torch.ops.aten.sub.Tensor(1, mean);  mean = None
        div_3 = torch.ops.aten.div.Tensor(div, add_2);  div = None
        return (sigmoid, primals_1, sub, primals_1, sigmoid, add_2, div_3)
        
def load_args(reader):
    buf0 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf0, (9, 1, 32, 32), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf1, (9, 1, 32, 32), is_leaf=True)  # primals_2
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)