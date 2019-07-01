import torch
import torch.distributed as dist

dist.init_process_group("gloo", init_method='tcp://127.0.0.1:12356', world_size=2, rank=1)
print("=== done init process group")
dist.init_rpc("worker1")
print("== done init rpc")
res = dist.rpc_sync("worker0", "aten::add", torch.ones(2, 2), torch.ones(2, 2))
print("== got res", res)
