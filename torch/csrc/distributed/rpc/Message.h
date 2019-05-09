#pragma once

#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/serialize.h>

namespace torch {
namespace distributed {
namespace rpc {

class MessageSerializer {
 public:
  MessageSerializer(const std::vector<IValue>& values);

  virtual int64_t writeNext(std::ostream& os, uint64_t size) = 0;
protected:
  const std::vector<IValue>& values_;
};

class Message {
 public:
  Message(const std::vector<IValue> values);
  virtual std::unique_ptr<MessageSerializer> serializer() = 0;

 protected:
  const std::vector<IValue> values_;
};

class MessageDeserializer {
 public:
  virtual std::unique_ptr<Message> readNext(std::istream& is, int64_t size) = 0;
};

class MessageDeserializerFactory {
 public:
  virtual std::unique_ptr<MessageDeserializer> deserializer() = 0;
};

}
}
}
