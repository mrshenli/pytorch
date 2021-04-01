from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
)

from torch.distributed.spmd import (
    AllReduceComm,
    DefaultBucketAllocator,
    DefaultBucketIndexer,
    DefaultTrigger,
    Engine,
)

import torch
import torch.nn as nn

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
        net = nn.Linear(10, 10)

        engine = Engine([DefaultTrigger()])

        engine.prepare_module(list(net.parameters()))
