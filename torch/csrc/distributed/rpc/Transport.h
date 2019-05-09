#pragma once

#include <torch/csrc/distributed/rpc/Message.h>

namespace torch {
namespace distributed {
namespace rpc {

struct SendContext {};

using Callback =
  std::function<std::unique_ptr<Message>(int64_t /* src */,
                                         std::unique_ptr<Message>,
                                         std::unique_ptr<SendContext>)>;

class Transport {
 public:
  Transport(const int64_t id,
            Callback cb,
            std::unique_ptr<MessageDeserializerFactory> mdf)
      : id_(id), cb_(std::move(cb)), mdf_(std::move(mdf)) {}

  virtual void send(int64_t dst,
                    std::unique_ptr<Message> msg,
                    std::unique_ptr<SendContext> ctx) = 0;

 protected:
  const int64_t id_;
  const Callback cb_;
  std::unique_ptr<MessageDeserializerFactory> mdf_;
};

class TransportFactory {
 public:
  virtual std::unique_ptr<Transport> createTransport(
      Callback cb, std::unique_ptr<MessageDeserializerFactory> mdf, bool isClient) = 0;
};

}
}
}
