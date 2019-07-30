import torch
import torch.distributed as dist

dist.init_process_group("gloo", init_method='tcp://127.0.0.1:12356', world_size=2, rank=1)
print("=== done init process group")
dist.init_rpc("worker1")
print("== done init rpc")
fut = dist.rpc_sync("worker0", "aten::add", torch.ones(2, 2), torch.ones(2, 2))



def callback(ret):
    print ("--- in callback, ret is ", ret)
    results.append(ret.clone())
    print ("--- done with callback")
    return ret


fut = dist.rpc_async('worker0', 'aten::add', torch.ones(2, 2), torch.ones(2, 2))
results = []

fut.then(callback)

fut.wait()
print("--- wait done\n")
ret = fut.get()
print("--- get done\n")
