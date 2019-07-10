from .distributed_c10d import new_group
from . import invoke_rpc
from . import ProcessGroupAgent

_agent = None

def init_rpc(name, backend='pg'):
    global _agent
    if backend == 'pg':
        from .distributed_c10d import _default_pg
        name_dict = {'worker%d' % r : r for r in range(2)}
        _agent = ProcessGroupAgent(name, name_dict, _default_pg)
    else:
        raise RuntimeError("Unrecognized RPC backend ", backend)

def destory_rpc():
    _agent.shutdown()

def rpc_async(dst, op, *args, **kargs):
    global _agent
    return invoke_rpc(_agent, dst, op, *args, **kargs)

def rpc_sync(dst, op, *args, **kargs):
    future = rpc_async(dst, op, *args, **kargs)
    future.wait()
    return future.get()
