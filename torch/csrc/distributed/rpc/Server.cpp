#include <rpc/Server.hpp>

#include <torch/csrc/jit/operator.h>
#include <ATen/core/stack.h>

namespace rpc {

namespace {
  using Stack = std::vector<at::IValue>;
}

Response Server::processRequest(Request& request) {
  at::Symbol symbol = request.symbol();
  std::vector<at::IValue> args = request.args();

  auto& ops = torch::jit::getAllOperatorsFor(symbol);

  std::cout << "=== got ops \n " << std::flush;

  AT_CHECK(ops.size(), 1);
  auto& op = ops.front();

  std::cout << "==== got op " << op << std::endl << std::flush;

  Stack stack;
  stack.insert(stack.end(), args.begin(), args.end());
  std::cout << "=== before invocation \n" << std::flush;
  op->getOperation()(stack);
  std::cout << "=== after invocation \n" << std::flush;

  Response response(0, stack);
  std::cout << "=== create response \n" << std::flush;

  return response;
}
}
