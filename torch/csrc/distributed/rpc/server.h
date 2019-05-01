#pragma once

#include <torch/csrc/distributed/rpc/Request.h>
#include <torch/csrc/distributed/rpc/Response.h>
#include <torch/csrc/distributed/rpc/Transport.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>

#include <thread>


namespace rpc {

class Server {
 public:

  Server(std::shared_ptr<Transport> transport, int64_t rank)
    : rank_(rank), transport_(transport) {
    transport->serveRpc(Request::load, [this](std::unique_ptr<Message> msg){
      this->processRequest(std::move(msg));
    });
  }


 private:

  void processRequest(std::shared_ptr<Message> msg) {
    std::shared_ptr<Request> request = std::static_pointer_cast<Request>(msg);
    std::shared_ptr<Operator> op = request->op();
    std::vector<at::IValue> args = request->args();

    Stack stack;
    stack.insert(stack.end(), args.begin(), args.end());
    op->getOperation()(stack);

    std::cout << "=== sending response \n" << std::flush;
    transport_->send(std::make_shared<Response>(
      Response(0, stack, request->id, request->dst, request->src)));
  }

  const int64_t rank_;
  std::shared_ptr<Transport> transport_;
};

}
