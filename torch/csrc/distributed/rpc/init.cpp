#include <torch/csrc/python_headers.h>

#include <torch/csrc/distributed/rpc/client.h>
#include <torch/csrc/distributed/rpc/server.h>
#include <torch/csrc/distributed/rpc/Transport.h>
#include <torch/csrc/distributed/rpc/ProcessGroupTransport.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

std::shared_ptr<RpcFuture> py_rpc(Client& client,
                                  std::string name,
                                  int64_t dst,
                                  py::args args,
                                  py::kwargs kwargs) {
  Symbol symbol = Symbol::fromQualString(name);
  for (auto op: torch::jit::getAllOperatorsFor(symbol)) {
    try {
      Stack stack =
          torch::jit::createStackForSchema(op->schema(), args, kwargs);
      return client.sendRequest(dst, op, stack);
    } catch (std::runtime_error) {}
  }
  throw std::runtime_error("unrecognized function " + name + " with args ...");
}

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

PyObject* rpc_init(PyObject* _unused) {
  auto rpc_module = THPObjectPtr(PyImport_ImportModule("torch.distributed"));
  if (!rpc_module) {
    throw python_error();
  }

  auto module = py::handle(rpc_module).cast<py::module>();


  auto rpcFuture = shared_ptr_class_<RpcFuture>(module, "RpcFuture")
      .def(
          "wait",
          &RpcFuture::wait,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get",
          &RpcFuture::get_py_obj,
          py::call_guard<py::gil_scoped_release>());


  auto transportFactory =
      shared_ptr_class_<TransportFactory>(module, "TransportFactory");
      //.def_readwrite("id_", &::rpc::TransportFactory::id_);

  auto processGroupTransportFactory =
      shared_ptr_class_<ProcessGroupTransportFactory>(
          module, "ProcessGroupTransportFactory", transportFactory)
          .def(py::init<::c10d::ProcessGroup&>());

  auto client =
      shared_ptr_class_<Client>(module, "Client")
      .def(py::init<std::shared_ptr<TransportFactory>>());

  auto server =
      shared_ptr_class_<Server>(module, "Server")
      .def(py::init<std::shared_ptr<TransportFactory>>());

  module.def("rpc", &py_rpc);

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
