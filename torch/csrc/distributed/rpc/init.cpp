#include <torch/csrc/python_headers.h>



#include <torch/csrc/distributed/rpc/client.h>
#include <torch/csrc/distributed/rpc/server.h>
#include <torch/csrc/distributed/rpc/Transport.h>
#include <torch/csrc/distributed/rpc/ProcessGroupTransport.h>


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

  module.def("rpc", &::rpc::invoke);

  auto rpcWork = shared_ptr_class_<::rpc::RpcWork>(module, "RpcWork")
      .def(
          "wait",
          &::rpc::RpcWork::wait,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get",
          &::rpc::RpcWork::get_py_obj,
          py::call_guard<py::gil_scoped_release>());

  auto transport =
    shared_ptr_class_<::rpc::Transport>(module, "Transport");


  auto processGroupTransport =
    shared_ptr_class_<::rpc::ProcessGroupTransport>(
      module, "ProcessGroupTransport", transport)
    .def(py::init<::c10d::ProcessGroup&>());

  auto client =
    shared_ptr_class_<::rpc::Client>(module, "Client")
    .def(py::init<std::shared_ptr<::rpc::Transport>, int64_t>());

  auto server =
    shared_ptr_class_<::rpc::Server>(module, "Server")
    .def(py::init<std::shared_ptr<::rpc::Transport>, int64_t>());

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
