#pragma once

#include <c10d/ProcessGroup.hpp>
#include <deque>
#include <thread>
#include <torch/csrc/distributed/rpc/Transport.h>
#include <streambuf>

namespace rpc {

namespace {

using SendWork=std::pair<std::shared_ptr<std::vector<torch::Tensor>>,
                         std::shared_ptr<c10d::ProcessGroup::Work>>;

}

class ProcessGroupTransport : public Transport {

 public:
  ProcessGroupTransport(c10d::ProcessGroup& pg);
  ~ProcessGroupTransport();
  void send(std::shared_ptr<Message> message) override;
  void serveRpc(MessageDeserializer md, RpcCallback cb) override;

 private:
  void sendTensor(torch::Tensor tensor, int64_t dst);
  torch::Tensor receiveData();
  void enqueue(std::shared_ptr<SendWork> work);

  // making sure tensors are not deleted before send finishes
  void gcLoop();
  void listenerLoop();

  c10d::ProcessGroup& pg_;
  std::deque<std::shared_ptr<SendWork> > sendQueue_;
  std::mutex sendQueueMutex_;
  std::condition_variable workProduceCV_;
  std::condition_variable workConsumeCV_;
  std::thread gcThread_;
  std::thread listenerThread_;
  bool stop_;
  RpcCallback cb_;
  MessageDeserializer md_;
};

}
