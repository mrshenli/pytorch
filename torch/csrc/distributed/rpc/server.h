#pragma once

#include <torch/csrc/distributed/rpc/Request.h>
#include <torch/csrc/distributed/rpc/Response.h>
#include <torch/csrc/distributed/rpc/Transport.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>

namespace torch {
namespace distributed {
namespace rpc {

class Server {
 public:
  Server(std::shared_ptr<TransportFactory> transportFactory);

 private:
  static std::unique_ptr<Message> processRequest(
      int64_t src,
      std::unique_ptr<Message> msg,
      std::unique_ptr<SendContext> sendCtx);

  std::unique_ptr<Transport> transport_;
};

}
}
}
