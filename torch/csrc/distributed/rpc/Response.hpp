#pragma once

#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <vector>

namespace rpc {

class Response {
 public:

  Response(int code, const std::vector<at::IValue> values)
    : code_(code), values_(std::move(values)){}

  // encode
  // decode
  // wait
  // code
  // values

  const int code_;
  const std::vector<at::IValue> values_;
};

}
