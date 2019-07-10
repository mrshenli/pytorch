#pragma once

#include <c10d/ProcessGroup.hpp>
#include <deque>
#include <thread>
#include <torch/csrc/distributed/rpc/Future.h>
#include <torch/csrc/distributed/rpc/RpcAgent.h>
#include <torch/csrc/distributed/rpc/functions.h>


namespace torch {
namespace distributed {
namespace rpc {

struct SendWork {
  SendWork(const int dstRank,
           Message message)
      : dstRank_(dstRank),
        message_(std::move(message)) {}

  const int dstRank_;
  Message message_;

};

class ProcessGroupAgent : public RpcAgent {
 public:

  ProcessGroupAgent(std::string workerName,
                    std::unordered_map<std::string, int> nameMap,
                    c10d::ProcessGroup& pg);

  ~ProcessGroupAgent() noexcept(false) override;

  std::shared_ptr<Future> send(std::string to, Message message) override;

  void shutdown() override;

 private:
  void _send(std::string dstName,
             const int64_t requestId,
             const std::vector<at::IValue> values,
             const int type);
  void sendTensor(torch::Tensor tensor, const int dstRank);
  void enqueue(SendWork work);

  // making sure tensors are not deleted before send finishes
  void sendLoop();
  void listen();

  int64_t nextId() {
    std::lock_guard<std::mutex> lock{idMutex_};
    return nextId_++;
  }

  const int64_t REQUEST_TYPE = 0;
  const int64_t RESPONSE_TYPE = 1;
  const int64_t SHUTDOWN_TYPE = 2;
  std::unordered_map<std::string, int> nameMap_;
  std::unordered_map<int, std::string> reversedNameMap_;
  bool stop_;
  c10d::ProcessGroup& pg_;
  int64_t nextId_;
  std::deque<SendWork> sendQueue_;
  std::mutex idMutex_;
  std::mutex sendQueueMutex_;
  std::condition_variable workProduceCV_;
  std::condition_variable workConsumeCV_;
  std::thread sendThread_;
  std::thread listenerThread_;
  std::unordered_map<int64_t, std::shared_ptr<Future>> futures_;
};

}
}
}
