#include <torch/csrc/distributed/rpc/Future.h>

namespace torch {
namespace distributed {
namespace rpc {

void Future::wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!completed_) {
    finished_cv_.wait(lock);
  }
}

void Future::markCompleted(Message message) {
  std::unique_lock<std::mutex> lock(mutex_);
  AT_ASSERT(!completed());
  completed_ = true;
  message_ = std::move(message);

  finished_cv_.notify_all();
}

void Future::markCompleted() {
  markCompleted(Message());
}

Message& Future::message() {
  std::unique_lock<std::mutex> lock(mutex_);
  AT_ASSERT(completed());

  return message_;
}

bool Future::completed() {
  return completed_;
}

}
}
}
