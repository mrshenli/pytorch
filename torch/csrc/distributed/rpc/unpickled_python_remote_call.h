#pragma once

#include <torch/csrc/distributed/rpc/unpickled_python_call.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace distributed {
namespace rpc {

class TORCH_API UnpickledPythonRemoteCall final : public UnpickledPythonCall {
 public:
  explicit UnpickledPythonRemoteCall(
    const SerializedPyObj& serializedPyObj,
    at::IValue retRRefId,
    at::IValue retForkId);

  const RRefId& rrefId() const;
  const ForkId& forkId() const;

 private:
  RRefId rrefId_;
  ForkId forkId_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
