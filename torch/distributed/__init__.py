import torch


def is_available():
    print("=== checking is_available")
    return hasattr(torch._C, "_c10d_init") and hasattr(torch._C, "_rpc_init")

print("=== is available?? ", is_available())

if is_available() and not (torch._C._c10d_init() and torch._C._rpc_init()):
    raise RuntimeError("Failed to initialize PyTorch distributed support")


if is_available():
    from .distributed_c10d import *  # noqa: F401
    # Variables prefixed with underscore are not auto imported
    # See the comment in `distributed_c10d.py` above `_backend` on why we expose
    # this.
    from .distributed_c10d import _backend  # noqa: F401
