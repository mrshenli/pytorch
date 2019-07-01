#include <torch/csrc/python_headers.h>

#include <torch/csrc/distributed/rpc/MapNameStore.h>
#include <torch/csrc/distributed/rpc/ProcessGroupAgent.h>
#include <torch/csrc/distributed/rpc/RpcAgent.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

std::shared_ptr<Future> invoke_rpc(RpcAgent& agent,
                               std::string dstName,
                               std::string opName,
                               py::args args,
                               py::kwargs kwargs) {
  std::cout<< "== invoking rpc\n" << std::flush;
  Symbol symbol = Symbol::fromQualString(opName);
  for (auto op: torch::jit::getAllOperatorsFor(symbol)) {
    try {
      Stack stack =
          torch::jit::createStackForSchema(op->schema(), args, kwargs, nullptr);
      return agent.sendRequest(dstName, Request(op, stack));
    } catch (std::runtime_error) {}
  }
  throw std::runtime_error("unrecognized function " + opName);
}

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

PyObject* rpc_init(PyObject* _unused) {
  std::cout << "----- calling rpc_init\n" << std::flush;
  auto dist_module = THPObjectPtr(PyImport_ImportModule("torch.distributed"));
  if (!dist_module) {
    throw python_error();
  }

  auto module = py::handle(dist_module).cast<py::module>();

  auto nameStore = shared_ptr_class_<NameStore>(module, "NameStore");

  auto mapNameStore =
      shared_ptr_class_<MapNameStore>(module, "MapNameStore", nameStore)
      .def(py::init<std::unordered_map<std::string, std::string>>());

  auto rpcAgent =
      shared_ptr_class_<RpcAgent>(module, "RpcAgent");
      //.def_readwrite("id_", &::rpc::TransportFactory::id_);

  auto processGroupAgent =
      shared_ptr_class_<ProcessGroupAgent>(
          module, "ProcessGroupAgent", rpcAgent)
          .def(py::init<std::string,
                        rpc::NameStore&,
                        ::c10d::ProcessGroup&>());

  module.def("invoke_rpc", &invoke_rpc);

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
