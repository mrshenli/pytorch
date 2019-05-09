from .distributed_c10d import new_group
from . import Client
from . import Server
from . import ProcessGroupTransportFactory

_client = None
_server = None


def init_rpc(backend='pg', factory=None):
    if backend == 'pg':
        from .distributed_c10d import _default_pg
        print (_default_pg)
        factory = ProcessGroupTransportFactory(_default_pg)
        _client = Client(factory)
        _server = Server(factory)
    else:
        raise RuntimeError("Unrecognized RPC backend ", backend)

def rpc_async(dst, op, *args, **kargs):
    return _client.send_request(dst, op, args, kargs)

def rpc(dst, op, *args, **kargs):
    future = rpc_async(dst, op, args, kargs)
    future.wait()
