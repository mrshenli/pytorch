from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
)

from torch.distributed.spmd import (
    AllReduceComm,
    DefaultBucketer,
    DefaultTrigger,
    Engine,
)

import torch
import torch.nn as nn
import torch.distributed as c10d

import os

class EngineTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._fork_processes()

    def tearDown(self):
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return 2

    def test_engine(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        net = nn.Linear(10, 10)

        engine = Engine(
            [DefaultTrigger(), DefaultBucketer(), AllReduceComm(pg)]
        )

        engine.prepare_module(list(net.parameters()))
        print("before iteration")
        net(torch.zeros(10, 10)).sum().backward()
        print("after iteration")
        print(net.bias.grad)
