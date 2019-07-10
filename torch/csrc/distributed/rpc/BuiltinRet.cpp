#include <torch/csrc/distributed/rpc/BuiltinRet.h>

namespace torch {
namespace distributed {
namespace rpc {

std::vector<at::IValue>& BuiltinRet::values() {
  return values_;
}

Message BuiltinRet::toMessage() {
  std::vector<torch::Tensor> tensor_table;
  Pickler pickler(&tensor_table);

  pickler.start();
  pickler.addIValue(values_);
  pickler.finish();

  return Message(pickler.stack(),
                 std::move(tensor_table),
                 MessageType::BUILTIN_RET);
}

BuiltinRet BuiltinRet::fromMessage(Message message) {
  auto data = static_cast<void*>(message.meta().data());
  auto size = message.meta().size();
  Unpickler unpickler(data, size, &message.tensors());

  return BuiltinRet(unpickler.parse_ivalue_list());
}

} // namespace rpc
} // namespace distributed
} // namespace torch
