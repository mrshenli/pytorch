#pragma once

#include <torch/csrc/python_headers.h>



#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <ATen/core/stack.h>

#include <rpc/Request.hpp>
#include <rpc/Response.hpp>

namespace rpc {

using torch::jit::IValue;
using torch::jit::Symbol;
using torch::jit::Stack;

class Client {
 public:
  //virtual void run();

  void call_rpc(std::string name, std::vector<IValue> args) {
  }

 private:
  // can call Response.wait()
  Response sendRequest(Request req);
};

Response sendRequest(Request req) {

}

void invoke(std::string name, py::args args, py::kwargs kwargs) {
  std::cout << "=== in call_rpc\n";
  Symbol symbol = Symbol::fromQualString(name);
  for (auto op: torch::jit::getAllOperatorsFor(symbol)) {
    try {
      Stack stack = torch::jit::createStackForSchema(op->schema(), args, kwargs);
      std::cout << "=== matching signature is " << op->schema() << std::endl;

      //std::cout << "=== answer is " << op->getOperation()(stack) << std::endl << std::flush;
      auto res = sendRequest(Request(symbol, stack));
      break;
    } catch (std::runtime_error) {}
  }
}

}
