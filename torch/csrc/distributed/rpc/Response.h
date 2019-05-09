#pragma once

#include <torch/csrc/distributed/rpc/Message.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/serialize.h>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

class ResponseSerializer : public MessageSerializer {
 public:
  ResponseSerializer(const int64_t code,
                     const std::vector<IValue>& values)
      : MessageSerializer(values), code_(code) {}

  int64_t writeNext(std::ostream& os, uint64_t size) override;

 private:
  const int64_t code_;
};

class Response : public Message {
 public:
  Response(int64_t code,
           std::vector<at::IValue> values)
      : Message(std::move(values)), code_(code) {}

  int64_t code();
  const std::vector<at::IValue> values();
  std::unique_ptr<MessageSerializer> serializer() override;

 private:
  const int64_t code_;
};

class ResponseDeserializer : public MessageDeserializer {
 public:
  std::unique_ptr<Message> readNext(std::istream& is, int64_t size) override;
};

class ResponseDeserializerFactory : public MessageDeserializerFactory {
 public:
  std::unique_ptr<MessageDeserializer> deserializer() override {
    return std::unique_ptr<ResponseDeserializer>(new ResponseDeserializer());
  }
};

}
}
}
