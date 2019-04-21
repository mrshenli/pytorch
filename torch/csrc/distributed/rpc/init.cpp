#include <torch/csrc/python_headers.h>



#include <torch/csrc/distributed/rpc/client.h>


namespace torch {
namespace distributed {
namespace rpc {

namespace {

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

PyObject* rpc_init(PyObject* _unused) {
  auto rpc_module = THPObjectPtr(PyImport_ImportModule("torch.distributed"));
  if (!rpc_module) {
    throw python_error();
  }

  auto module = py::handle(rpc_module).cast<py::module>();

  module.def("rpc", &::rpc::call_rpc);

  Py_RETURN_TRUE;
}

} // namespace

// c10d methods on torch._C
static PyMethodDef methods[] = {
    {"_rpc_init", (PyCFunction)rpc_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  std::cout << "=== calling rpc init functions with methods of size " << (sizeof(methods)/sizeof(*methods)) << std::endl;
  return methods;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
