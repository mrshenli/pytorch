import torch.distributed as dist
print(dir(dist))

def rpc_return_rref(dst):
    #return dist.remote(dst, torch.add, args=(torch.ones(2, 2), 1))
    return dst

dist.init_process_group("gloo", init_method='tcp://127.0.0.1:12356', world_size=2, rank=0)
dist.init_model_parallel("worker0")
dist.join_rpc()
