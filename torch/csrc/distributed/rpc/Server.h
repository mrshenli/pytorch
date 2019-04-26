#pragma once

#include <torch/csrc/distributed/rpc/Request.h>
#include <torch/csrc/distributed/rpc/Response.h>

namespace rpc {

class Server {
 public:
  //virtual void run();

  Response processRequest(Request& request);
};

}
