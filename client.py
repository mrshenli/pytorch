import torch
import torch.distributed as dist

dist.init_process_group("gloo", init_method='tcp://127.0.0.1:12356', world_size=2, rank=0)
dist.init_rpc()
res = dist.rpc(1, "aten::add", torch.ones(2, 2), torch.ones(2, 2))
res.get()
