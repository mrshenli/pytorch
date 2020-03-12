#pragma once

#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace distributed {
namespace rpc {

class TORCH_API UnpickledPythonCall : public RpcCommandBase {
 public:
  explicit UnpickledPythonCall(const SerializedPyObj& serializedPyObj);

  Message toMessage() && override;
  py::object movePythonUdf() &&;

 private:
  py::object pythonUdf_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
