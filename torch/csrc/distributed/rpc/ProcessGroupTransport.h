#pragma once

#include <c10d/ProcessGroup.hpp>
#include <deque>
#include <thread>
#include <torch/csrc/distributed/rpc/Transport.h>

namespace torch {
namespace distributed {
namespace rpc {

using SendWork=std::pair<std::shared_ptr<std::vector<torch::Tensor>>,
                         std::shared_ptr<c10d::ProcessGroup::Work>>;

struct MessageData {
  int64_t msgId;
  int64_t srcId;
  std::unique_ptr<Message> msg;
};


class ProcessGroupTransportFactory : public TransportFactory {
 public:
  ProcessGroupTransportFactory(c10d::ProcessGroup& pg);

  std::unique_ptr<Transport> createTransport(
      Callback cb,
      std::unique_ptr<MessageDeserializerFactory> mdf,
      bool isClient) override;

 private:
  c10d::ProcessGroup& pg_;
};

class ProcessGroupTransport : public Transport {
 public:
  ProcessGroupTransport(const int64_t id,
                        Callback cb,
                        std::unique_ptr<MessageDeserializerFactory> mdf,
                        c10d::ProcessGroup& pg,
                        bool isClient);

  ~ProcessGroupTransport();
  void send(int64_t dst,
            std::unique_ptr<Message> msg,
            std::unique_ptr<SendContext> ctx) override;

 private:
  void sendTensor(torch::Tensor tensor, int64_t dst);
  MessageData receiveData();
  void enqueue(std::shared_ptr<SendWork> work);

  // making sure tensors are not deleted before send finishes
  void gcLoop();
  void listenerLoop();

  int64_t nextId() {
    std::lock_guard<std::mutex> lock{idMutex_};
    return nextId_++;
  }

  const int sendTag_;
  const int recvTag_;
  bool stop_;
  c10d::ProcessGroup& pg_;
  int64_t nextId_;
  std::deque<std::shared_ptr<SendWork>> sendQueue_;
  std::mutex idMutex_;
  std::mutex sendQueueMutex_;
  std::condition_variable workProduceCV_;
  std::condition_variable workConsumeCV_;
  std::thread gcThread_;
  std::thread listenerThread_;
  std::unique_ptr<MessageDeserializer> md_;
  std::unordered_map<int64_t, std::unique_ptr<SendContext>> ctxs_;
};

}
}
}
