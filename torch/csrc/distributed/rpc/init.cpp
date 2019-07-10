#include <torch/csrc/python_headers.h>

#include <torch/csrc/distributed/rpc/Future.h>
#include <torch/csrc/distributed/rpc/functions.h>
#include <torch/csrc/distributed/rpc/python_functions.h>
#include <torch/csrc/distributed/rpc/RpcAgent.h>
#include <torch/csrc/distributed/rpc/ProcessGroupAgent.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

PyObject* rpc_init(PyObject* _unused) {
  auto dist_module = THPObjectPtr(PyImport_ImportModule("torch.distributed"));
  if (!dist_module) {
    throw python_error();
  }

  auto module = py::handle(dist_module).cast<py::module>();

  auto rpcAgent = shared_ptr_class_<RpcAgent>(module, "RpcAgent")
      .def("shutdown",
           &RpcAgent::shutdown,
           py::call_guard<py::gil_scoped_release>());

  auto future = shared_ptr_class_<Future>(module, "Future")
      .def("wait",
          &Future::wait,
          py::call_guard<py::gil_scoped_release>())
      .def("get",
          [&](Future& fut) {
            return to_py_obj(fut.message());
          },
          py::call_guard<py::gil_scoped_release>());

  auto processGroupAgent =
      shared_ptr_class_<ProcessGroupAgent>(
          module, "ProcessGroupAgent", rpcAgent)
          .def(py::init<std::string,
                        std::unordered_map<std::string, int>,
                        ::c10d::ProcessGroup&>())
          .def("shutdown",
               &ProcessGroupAgent::shutdown,
               py::call_guard<py::gil_scoped_release>());

  module.def("invoke_rpc", [](
      RpcAgent& agent,
      std::string dstName,
      std::string opName,
      py::args args,
      py::kwargs kwargs) {
    return py_rpc(agent, std::move(dstName), std::move(opName), args, kwargs);
  });

  Py_RETURN_TRUE;
}

} // namespace

static PyMethodDef methods[] = {
    {"_rpc_init", (PyCFunction)rpc_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
