#pragma once

#include <torch/csrc/distributed/rpc/Request.h>
#include <torch/csrc/distributed/rpc/Response.h>

namespace rpc {

using RpcCallback = std::function<void(std::unique_ptr<Message>)>;
using MessageDeserializer = std::function<std::unique_ptr<Message>(std::istream& stream)>;

class Transport {
 public:
  virtual void send(std::shared_ptr<Message> msg) = 0;
  virtual void serveRpc(MessageDeserializer md, RpcCallback cb) = 0;
};

}
