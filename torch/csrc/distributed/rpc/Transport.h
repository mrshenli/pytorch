#pragma once

#include <torch/csrc/distributed/rpc/Request.h>
#include <torch/csrc/distributed/rpc/Response.h>

namespace rpc {


class Transport {
 public:
  virtual void send(std::shared_ptr<Message> msg) = 0;
  virtual std::shared_ptr<Response> receiveResponse() = 0;
  virtual std::shared_ptr<Request> receiveRequest() = 0;
};

}
