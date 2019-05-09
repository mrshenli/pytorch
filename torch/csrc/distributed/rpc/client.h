#pragma once
#include <torch/csrc/python_headers.h>

#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/csrc/distributed/rpc/Request.h>
#include <torch/csrc/distributed/rpc/Response.h>
#include <torch/csrc/distributed/rpc/Transport.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>

namespace torch {
namespace distributed {
namespace rpc {

class RpcFuture {
 public:
  RpcFuture() : future_(new Future()) {}
  void wait() {
    future_->wait();
  }
  void notify(std::unique_ptr<Response> response);
  // block and get IValues
  const Stack& get();
  // get IValues and convert it to Python obj
  py::object get_py_obj() {
    Stack copy = get();
    return torch::jit::createPyObjectForStack(std::move(copy));
  }

 private:
  std::unique_ptr<Future> future_;
};

struct ClientSendContext : SendContext {
  ClientSendContext() : future_(std::make_shared<RpcFuture>()) {}
  std::shared_ptr<RpcFuture> future() {
    return future_;
  }
 private:
  std::shared_ptr<RpcFuture> future_;
};

class Client {
 public:
  Client(std::shared_ptr<TransportFactory> transportFactory);
  std::shared_ptr<RpcFuture> sendRequest(
      int64_t dst, std::shared_ptr<Operator> op, std::vector<IValue> args);

 private:
  static std::unique_ptr<Message> processResponse(
      int64_t src,
      std::unique_ptr<Message> msg,
      std::unique_ptr<SendContext> sendCtx);

  std::unique_ptr<Transport> transport_;
};


} // namespace rpc
}
}
