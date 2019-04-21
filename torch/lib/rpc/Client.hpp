#pragma once

#include <rpc/Request.hpp>
#include <rpc/Response.hpp>
#include <torch/csrc/jit/operator.h>

#include <future>

namespace rpc {

using torch::jit::IValue;

class Client {
 public:
  //virtual void run();

  void rpcSync(std::string name, std::vector<IValue> args) {
  }

 private:
  // can call Response.wait()
  // Response sendRequest()
};

}
