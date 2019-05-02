#include <torch/csrc/distributed/rpc/server.h>

namespace rpc {

Server::Server(std::shared_ptr<Transport> transport, int64_t rank)
  : rank_(rank), transport_(transport) {
  transport->serveRpc(Request::load, [this](std::unique_ptr<Message> msg){
    this->processRequest(std::move(msg));
  });
}

void Server::processRequest(std::shared_ptr<Message> msg) {
  std::shared_ptr<Request> request = std::static_pointer_cast<Request>(msg);
  std::shared_ptr<Operator> op = request->op();
  std::vector<at::IValue> args = request->args();

  Stack stack;
  stack.insert(stack.end(), args.begin(), args.end());
  op->getOperation()(stack);

  transport_->send(std::make_shared<Response>(
    Response(0, stack, request->id, request->dst, request->src)));
}

}
