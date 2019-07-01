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

struct SendWork {
  SendWork(const int dstRank,
           std::unique_ptr<std::stringstream> data,
           const int64_t requestId,
           const int type)
      : dstRank_(dstRank),
        data_(std::move(data)),
        requestId_(requestId),
        type_(type) {}

  const int dstRank_;
  std::unique_ptr<std::stringstream> data_;
  const int64_t requestId_;
  const int type_;

};

class ProcessGroupAgent : public RpcAgent {
 public:

  ProcessGroupAgent(std::string workerName,
                    std::unordered_map<std::string, int> nameMap,
                    c10d::ProcessGroup& pg);

  ~ProcessGroupAgent() noexcept(false) override;

  c10::intrusive_ptr<c10::ivalue::Future> sendRequest(
      std::string dstName, Request request) override;

  void shutdown() override;

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
  std::unordered_map<int64_t, c10::intrusive_ptr<c10::ivalue::Future>> futures_;
};

}
}
}
