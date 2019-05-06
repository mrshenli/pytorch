#pragma once

#include <torch/csrc/distributed/rpc/Message.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/serialize.h>
#include <vector>

namespace rpc {

class Request : public Message {
 public:
  Request(std::shared_ptr<Operator> op,
          const std::vector<at::IValue> args,
          int64_t id,
          int64_t src, // src and dst already available in pg
          int64_t dst);
  at::Symbol symbol();
  std::shared_ptr<Operator> op();
  std::vector<at::IValue> args();
  // how to do streaming for super large request/response?
  // serializer save();
  void save(std::ostream& stream) override;
  // deserializer load();
  static std::unique_ptr<Request> load(std::istream& stream);

 private:
  static std::shared_ptr<Operator> matchOperator(
      at::Symbol symbol, std::string str_schema);

  std::shared_ptr<Operator> op_;
  std::vector<at::IValue> args_;
};
}
