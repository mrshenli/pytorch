import torch
import torch.distributed.rpc as rpc
import torch.nn.parallel.DistributedDataParallel as DDP


N_BUCKETS = 10
N_FEATURES = 1000
BATCH_SIZE = 100

@rpc.function.async_execution
def average_gradients(ps_rref, bucket, id):
    return ps_rref.local_value().add_grad_bucket(bucket, id)


def ps_hook(state, bucket):
    pass


class Trainer:
    def __init__(self, ps_rref, hook=None):
        self.ps_rref = ps_rref
        self.model = torch.nn.Sequential(
            *[nn.Linear(N_FEATURES, N_FEATURES, bias=False) for _ in range(N_BUCKETS)]
        ).cuda(0)
        self.ddp = DDP(model, device_ids=[0], bucket_cap_mb=4 * N_FEATURES * N_FEATURES / 1000000)
        if hook is not None:
            self.ddp.register_comm_hook(None, hook)


    def run(self):
        inputs = torch.zeros(BATCH_SIZE, N_FEATURES).cuda(0)
        # warmup
        for _ in range(20):
            self.ddp(inputs).sum().backward()
