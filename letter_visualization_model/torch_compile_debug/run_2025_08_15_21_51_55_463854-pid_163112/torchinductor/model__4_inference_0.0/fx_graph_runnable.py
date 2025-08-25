
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1):
        neg = torch.ops.aten.neg.default(arg0_1);  arg0_1 = None
        index_put = torch.ops.aten.index_put.default(arg1_1, [arg2_1], neg);  arg2_1 = neg = None
        copy_ = torch.ops.aten.copy_.default(arg1_1, index_put);  arg1_1 = index_put = None
        return (copy_,)
        
def load_args(reader):
    buf0 = reader.storage(None, 63196, device=device(type='cuda', index=0))
    reader.tensor(buf0, (15799,), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 81920, device=device(type='cuda', index=0))
    reader.tensor(buf1, (20, 1, 32, 32), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 20480, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf2, (20, 1, 32, 32), dtype=torch.bool, is_leaf=True)  # arg2_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)