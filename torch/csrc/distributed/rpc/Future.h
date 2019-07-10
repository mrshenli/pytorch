#pragma once

#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/csrc/distributed/rpc/Message.h>


namespace torch {
namespace distributed {
namespace rpc {

struct TORCH_API Future final {

 public:
  void wait();
  void markCompleted(Message message);
  void markCompleted();
  Message& message();
  bool completed();

 private:
  std::mutex mutex_;
  std::atomic_bool completed_ = {false}; // is this future complete
  std::condition_variable finished_cv_;
  Message message_;
};

}
}
}
