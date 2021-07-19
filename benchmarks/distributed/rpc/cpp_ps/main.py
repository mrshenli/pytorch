import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import time


N_BUCKETS = 10
N_FEATURES = 1024
BATCH_SIZE = 64

@rpc.functions.async_execution
def average_gradients(ps_rref, bucket, bucket_id):
    def func(fut):
        print(f"grad {bucket_id} is ready", flush=True)
        return fut.wait()

    return ps_rref.local_value().add_grad_bucket(bucket, bucket_id).then(func)


def ps_hook(ps_rref, bucket):
    bucket_tensor = bucket.get_tensor()
    bucket_id = bucket.get_index()

    print(f"in bucket {bucket_id}", flush=True)

    def func(fut):
        print(f"bucket {bucket_id} is ready", flush=True)
        return [fut.wait()]

    return rpc.rpc_async(
        ps_rref.owner(),
        average_gradients,
        args=(ps_rref, bucket_tensor, bucket_id)
    ).then(func)  # DDP hook expects a list of tensors


class PyParameterServer(rpc.ParameterServer):
    def __init__(self, num_trainers, num_buckets):
        super().__init__(num_trainers, num_buckets)

    def __getstate__(self):
        return {}


class Trainer:
    def __init__(self, ps_rref, hook=None):
        self.ps_rref = ps_rref
        model = nn.Sequential(
            *[nn.Linear(N_FEATURES, N_FEATURES, bias=False) for _ in range(N_BUCKETS)]
        ).cuda(0)
        # CUDA_VISIBLE_DEVICES is set
        self.ddp = DDP(model, device_ids=[0], bucket_cap_mb=4 * N_FEATURES * N_FEATURES / (1024 * 1024))
        if hook is not None:
            self.ddp.register_comm_hook(ps_rref, hook)


    def run(self):
        inputs = torch.zeros(BATCH_SIZE, N_FEATURES).cuda(0)
        # warmup
        for _ in range(20):
            self.ddp(inputs).sum().backward()

        # measure
        torch.cuda.current_stream(0).synchronize()
        tik = time.time()
        for _ in range(20):
            self.ddp(inputs).sum().backward()

        torch.cuda.current_stream(0).synchronize()
        tok = time.time()

        print(f"{rpc.get_worker_info().name} delay: {tok - tik}")


def run(rank, world_size, num_gpu_per_node=8):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{rank % num_gpu_per_node}"
    assert torch.cuda.device_count() == 1

    # init RPC, worker0 hosts PS and serves as a coordinator
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)
    for r in range(world_size):
        options.set_device_map(f"worker{r}", {0: 0})

    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options
    )

    if rank != 0:
        # create process group across DDP processes
        torch.distributed.init_process_group(
            "nccl",
            init_method="tcp://localhost:29501",
            rank=rank - 1,
            world_size=world_size - 1
        )

    rpc.api._barrier([f"worker{r}" for r in range(world_size)])

    print(f"worker{rank} initialized")

    if rank == 0:
        print("00000")
        ps = PyParameterServer(world_size - 1, N_BUCKETS)
        print("01010")
        ps_rref = rpc.RRef(ps)

        print("11111")
        trainers = [
            #rpc.remote(f"worker{i}", Trainer, args=(ps_rref, ps_hook))
            rpc.remote(f"worker{i}", Trainer, args=(ps_rref, ))
            for i in range(1, world_size)
        ]
        print("22222")

        futs = [trainer.rpc_async().run() for trainer in trainers]
        print("33333")
        torch.futures.wait_all(futs)

    print(f"worker{rank} reached shutdown")
    rpc.shutdown()


if __name__=="__main__":
    world_size = 8
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
