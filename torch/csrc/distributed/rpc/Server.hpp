#pragma once

#include <rpc/Request.hpp>
#include <rpc/Response.hpp>

namespace rpc {

class Server {
 public:
  //virtual void run();

  Response processRequest(Request& request);
};

}
