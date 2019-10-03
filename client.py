import torch
import torch.distributed as dist

def rpc_return_rref(dst):
    #return dist.remote(dst, torch.add, args=(torch.ones(2, 2), 1))
    return dst

dist.init_process_group("gloo", init_method='tcp://127.0.0.1:12356', world_size=2, rank=1)
print("=== done init process group")
dist.init_model_parallel("worker1")
print("== done init rpc")
res = dist.rpc("worker0", rpc_return_rref, args=("worker1",))
print("== got res", res)
dist.join_rpc()
