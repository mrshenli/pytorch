#include <torch/csrc/python_headers.h>

#include <torch/csrc/distributed/spmd/engine.h>
#include <torch/csrc/utils/pybind.h>


namespace torch {
namespace distributed {
namespace spmd {

namespace {

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

PyObject* spmd_init(PyObject* _unused, PyObject* noargs) {
  auto spmd_module =
      THPObjectPtr(PyImport_ImportModule("torch.distributed.spmd"));
  if (!spmd_module) {
    throw python_error();
  }

  auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
  if (!torch_C_module) {
    throw python_error();
  }

  auto torch_C_m = py::handle(torch_C_module).cast<py::module>();
  auto m =
      torch_C_m.def_submodule("_distributed_spmd", "distributed spmd bindings");

  auto module = py::handle(m).cast<py::module>();


  auto eventHandler = shared_ptr_class_<EventHandler>(module, "EventHandler");

  shared_ptr_class_<DefaultTrigger>(module, "DefaultTrigger", eventHandler)
      .def(py::init<>());

  shared_ptr_class_<DefaultBucketIndexer>(
      module, "DefaultBucketIndexer", eventHandler)
      .def(py::init<>());

  shared_ptr_class_<DefaultBucketAllocator>(
      module, "DefaultBucketAllocator", eventHandler)
      .def(py::init<>());

  shared_ptr_class_<AllReduceComm>(
      module, "AllReduceComm", eventHandler)
      .def(py::init<>());

  shared_ptr_class_<Engine>(module, "Engine")
      .def(
          py::init([](const std::vector<std::shared_ptr<EventHandler>>& handlers) {
            return std::make_shared<Engine>(handlers);
          }),
          py::arg("handlers"));

  Py_RETURN_TRUE;
}

} // namespace

static PyMethodDef methods[] = { // NOLINT
    {"_spmd_init", spmd_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // spmd
} // distributed
} // torch
