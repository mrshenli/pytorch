#pragma once

#include <torch/csrc/distributed/rpc/Request.h>
#include <torch/csrc/distributed/rpc/Response.h>
#include <torch/csrc/distributed/rpc/Transport.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>

#include <thread>


namespace rpc {

class Server {
 public:
  Server(std::shared_ptr<Transport> transport, int64_t rank);

 private:
  void processRequest(std::shared_ptr<Message> msg);

  const int64_t rank_;
  std::shared_ptr<Transport> transport_;
};

}
