import sys
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from common_cuda import TEST_MULTIGPU
from common_utils import TestCase, load_tests, run_tests
from common_utils import NO_MULTIPROCESSING_SPAWN

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if not dist.is_available():
    print('c10d not available, skipping tests')
    sys.exit(0)


if NO_MULTIPROCESSING_SPAWN:
    print('spawn not available, skipping tests')
    sys.exit(0)


class RpcTest(TestCase):

    world_size = 2

    @classmethod
    def opts(cls, threads=2):
        opts = dist.ProcessGroupGloo.Options()
        opts.devices = [dist.ProcessGroupGloo.create_tcp_device(interface="lo")]
        opts.timeout = 5.0
        opts.threads = threads
        return opts

    @classmethod
    def _init_rpc(cls, rank, filename, world_size):
        store = dist.FileStore(filename, world_size)
        dist.init_process_group(
            backend='gloo', rank=rank, world_size=world_size, store=store)
        dist.init_rpc('worker%d' % rank)

    @classmethod
    def _destory_rpc(cls):
        dist.destory_rpc()
        dist.destroy_process_group(dist.group.WORLD)

    def _test_multiprocess(self, f, n_output):
        # file store will delete the test file on destruction
        file = tempfile.NamedTemporaryFile(delete=False)
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(10)
        p2c = ctx.Queue(10)
        ps = []
        for i in range(self.world_size):
            p = ctx.Process(
                target=f,
                args=(i, file.name, c2p, p2c))

            p.start()
            ps.append(p)

        for _ in range(self.world_size * n_output):
            pid, expected, result = c2p.get()
            self.assertEqual(
                expected,
                result,
                (
                    "Expect rank {} to receive tensor {} but got {}."
                ).format(pid, expected, result)
            )

        for _ in range(self.world_size):
            p2c.put(0)

        for p in ps:
            p.join(2)

    # Why classmethod? multiprocessing cannot pickle TestCase subclass when in
    # spawn mode. See https://bugs.python.org/issue33884.
    @classmethod
    def _test_builtin(cls, rank, filename, c2p, p2c):
        RpcTest._init_rpc(rank, filename, 2)

        if rank == 0:
            ret = dist.rpc_sync('worker1', 'aten::add', torch.ones(2, 2), torch.ones(2, 2))
            c2p.put((rank, torch.ones(2, 2) * 2, ret))
        else:
            ret = dist.rpc_sync('worker0', 'aten::add', torch.ones(3, 3), torch.zeros(3, 3))
            c2p.put((rank, torch.ones(3, 3), ret))
        p2c.get()

        RpcTest._destory_rpc()

    @unittest.skipIf(not TEST_MULTIGPU, "At least 2 CUDA GPUS needed")
    def test_builtin(self):
        self._test_multiprocess(RpcTest._test_builtin, 1)


if __name__ == '__main__':
    run_tests()
