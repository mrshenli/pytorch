#pragma once

#include <c10d/ProcessGroup.hpp>
#include <deque>
#include <thread>
#include <torch/csrc/distributed/rpc/RpcAgent.h>
#include <torch/csrc/distributed/rpc/Deserializer.h>
#include <torch/csrc/distributed/rpc/Serializer.h>

namespace torch {
namespace distributed {
namespace rpc {

using SendWork=std::pair<std::shared_ptr<std::vector<torch::Tensor>>,
                         std::shared_ptr<c10d::ProcessGroup::Work>>;

class ProcessGroupAgent final : public RpcAgent {
 public:

  ProcessGroupAgent(std::string workerName,
                    NameStore& ns,
                    c10d::ProcessGroup& pg);

  ~ProcessGroupAgent() override;

  std::shared_ptr<Future> sendRequest(
      std::string dstName, Request request) override;

 protected:
  void sendResponse(std::string dstName,
                    const int64_t requestId,
                    Response response) override;

 private:
  void _send(std::string dstName,
             const int64_t requestId,
             const std::vector<at::IValue> values,
             const int type);
  void sendTensor(torch::Tensor tensor, const int dstRank);
  void enqueue(std::shared_ptr<SendWork> work);

  // making sure tensors are not deleted before send finishes
  void gcLoop();
  void listen();

  int64_t nextId() {
    std::lock_guard<std::mutex> lock{idMutex_};
    return nextId_++;
  }

  const int64_t REQUEST_TYPE = 0;
  const int64_t RESPONSE_TYPE = 1;
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
  std::unordered_map<int64_t, std::shared_ptr<Future>> futures_;
};

}
}
}
