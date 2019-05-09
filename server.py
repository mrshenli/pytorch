import torch.distributed as dist

print(dir(dist))

dist.init_process_group("gloo", init_method='tcp://127.0.0.1:12356', world_size=2, rank=1)
dist.init_rpc()
input()
