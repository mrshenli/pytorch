#pragma once

#include <torch/csrc/distributed/rpc/Message.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/serialize.h>
#include <vector>

namespace rpc {

class Response : public Message {
 public:
  Response(int64_t code,
           std::vector<at::IValue> values,
           int64_t id,
           int64_t src,
           int64_t dst);

  int code();
  const std::vector<at::IValue> values();
  void save(std::ostream& stream) override;
  static std::unique_ptr<Response> load(std::istream& stream);

 private:
  const int64_t code_;
  const std::vector<at::IValue> values_;
};

}
