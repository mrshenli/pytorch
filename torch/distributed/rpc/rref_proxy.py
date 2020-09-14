from functools import partial

from . import functions

def _local_invoke(rref, func_name, args, kwargs):
    return getattr(rref.local_value(), func_name)(*args, **kwargs)

def _invoke_rpc(rref, rpc_api, func_name, *args, **kwargs):
    print("===before", flush=True)
    tyep = rref._get_type()
    print("===after", flush=True)
    func = getattr(globals()[type.__name__], func_name)

    if hasattr(func, "_wrapped_async_rpc_function"):
        return rpc_api(
            rref.owner(),
            functions.async_execution(_local_invoke),
            args=(rref, func_name, args, kwargs)
        )
    else:
        return rpc_api(
            rref.owner(),
            _local_invoke,
            args=(rref, func_name, args, kwargs)
        )


class RRefProxy:
    def __init__(self, rref, rpc_api):
        self.rref = rref
        self.rpc_api = rpc_api

    def __getattr__(self, func_name):
        return partial(_invoke_rpc, self.rref, self.rpc_api, func_name)
