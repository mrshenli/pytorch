#include <torch/csrc/distributed/rpc/python_functions.h>

namespace torch {
namespace distributed {
namespace rpc {

py::object to_py_obj(const Message& message) {
  switch (message.type()) {
    case MessageType::BUILTIN_RET: {
      BuiltinRet ret = BuiltinRet::fromMessage(message);
      Stack stack = ret.values();
      return torch::jit::createPyObjectForStack(std::move(stack));
    }
    default: {
      AT_ERROR("Unrecognized response message type ", message.type());
    }
  }
}

FutureMessage::Callback wrap_callback(
    const std::function<py::object(py::object)>& py_cb) {
  return [py_cb](const Message& message) -> int{
    std::cout << "--- got message \n" << std::flush;
    auto ret = py_cb(to_py_obj(message));
    std::cout << "--- done wrapper cb\n" << std::flush;
    return 0;
  };
}

std::shared_ptr<FutureMessage> py_rpc(
    RpcAgent& agent,
    const std::string& dstName,
    const std::string& opName,
    const py::args& args,
    const py::kwargs& kwargs) {
  if (opName.rfind("aten", 0) == 0) {
    // builtin operators
    Symbol symbol = Symbol::fromQualString(opName);
    for (const auto& op: torch::jit::getAllOperatorsFor(symbol)) {
      try {
        Stack stack = torch::jit::createStackForSchema(
            op->schema(), args, kwargs, c10::nullopt);
        return agent.send(dstName, BuiltinOp(op, std::move(stack)).toMessage());
      } catch (std::runtime_error) {}
    }
  }

  AT_ERROR("Failed to match operator name ", opName, " and arguments "
      "(args: ", args, ", kwargs: ", kwargs, ") to a builtin operator");
}

}
}
}
