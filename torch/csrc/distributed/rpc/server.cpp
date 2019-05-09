#include <torch/csrc/distributed/rpc/server.h>

namespace torch {
namespace distributed {
namespace rpc {

Server::Server(std::shared_ptr<TransportFactory> transportFactory) {
  transportFactory->createTransport(
      processRequest,
      std::unique_ptr<RequestDeserializerFactory>(
          new RequestDeserializerFactory()),
      false);
}

std::unique_ptr<Message> Server::processRequest(
    int64_t src,
    std::unique_ptr<Message> msg,
    std::unique_ptr<SendContext> /* unused */) {
  std::unique_ptr<Request> request =
      static_unique_ptr_cast<Message, Request>(std::move(msg));
  std::shared_ptr<Operator> op = request->op();
  std::vector<at::IValue> args = request->args();

  Stack stack;
  stack.insert(stack.end(), args.begin(), args.end());
  op->getOperation()(stack);

  return std::unique_ptr<Response>(new Response(0, stack));
}

}
}
}
