Distributed Pipeline Parallelism Using RPC
==========================================
**Author**: `Shen Li <https://mrshenli.github.io/>`_



`Getting Started with Distributed RPC Framework <rpc_tutorial.html>`__ tutorial
shows how to use `torch.distributed.rpc <https://pytorch.org/docs/master/rpc.html>`__
to implement distributed model parallelism. This post extends that to pipeline
parallelism. This is also the distributed counterpart of the multi-GPU pipeline 
parallelism discussed in `Single-Machine Model Parallel Best Practices <model_parallel_tutorial.html>`__
