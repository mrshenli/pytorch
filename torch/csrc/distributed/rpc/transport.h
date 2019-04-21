#pragma once

#include <rpc/Request.hpp>
#include <rpc/Response.hpp>

namespace rpc {


class Transport {
 public:
  virtual Response sendRequest(Request req) = 0;
  virtual void sendResponse(Response res) = 0;
};

}
