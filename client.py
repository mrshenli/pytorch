import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

dist.init_process_group("gloo", init_method='tcp://127.0.0.1:12356', world_size=2, rank=0)
transport = dist.ProcessGroupTransport(c10d._default_pg)
client = dist.Client(transport, 0)
res = dist.rpc(client, "aten::add", 1, torch.ones(2, 2), torch.ones(2, 2))
res.get()
