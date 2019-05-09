#pragma once

#include <torch/csrc/distributed/rpc/Message.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/serialize.h>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

static std::shared_ptr<Operator> matchOperator(
    at::Symbol symbol, std::string str_schema);

class RequestSerializer : public MessageSerializer {
 public:
  RequestSerializer(std::shared_ptr<Operator> op,
                    const std::vector<IValue>& values)
      : MessageSerializer(values), op_(op) {}

  int64_t writeNext(std::ostream& os, uint64_t size) override;

 private:
  std::shared_ptr<Operator> op_;
};

class Request : public Message {
 public:
  Request(std::shared_ptr<Operator> op,
          const std::vector<at::IValue> args)
      : Message(std::move(args)), op_(op) {}

  std::shared_ptr<Operator> op();
  std::vector<at::IValue> args();
  std::unique_ptr<MessageSerializer> serializer() override;

 private:
  std::shared_ptr<Operator> op_;
};

class RequestDeserializer : public MessageDeserializer {
 public:
  std::unique_ptr<Message> readNext(std::istream& is, int64_t size) override;
};

class RequestDeserializerFactory : public MessageDeserializerFactory {
 public:
  std::unique_ptr<MessageDeserializer> deserializer() override {
    return std::unique_ptr<RequestDeserializer>(new RequestDeserializer());
  }
};

}
}
}
