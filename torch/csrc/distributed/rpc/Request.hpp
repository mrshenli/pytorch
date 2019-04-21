#pragma once

#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <vector>

namespace rpc {

class Request {
 public:
  Request(at::Symbol symbol, const std::vector<at::IValue> args)
    : symbol_(symbol), args_(std::move(args)){}

  Request(std::istream& istream);

  at::Symbol symbol() {
    return symbol_;
  }

  std::vector<at::IValue> args() {
    return args_;
  }

  void serialize(std::ostream& stream);
 private:
  at::Symbol symbol_;
  std::vector<at::IValue> args_;
};
}
