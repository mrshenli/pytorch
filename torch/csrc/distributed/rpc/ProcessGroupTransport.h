#pragma once

#include <c10d/ProcessGroup.hpp>
#include <deque>
#include <thread>
#include <torch/csrc/distributed/rpc/Transport.h>

namespace rpc {

class ProcessGroupTransport : public Transport {

 public:

  ProcessGroupTransport(std::shared_ptr<c10d::ProcessGroup> pg)
      : pg_(pg), stop_(false) {
    thread_ = std::thread(&ProcessGroupTransport::dequeueLoop, this);
  }

  ~ProcessGroupTransport() {
    std::unique_lock<std::mutex> lock(sendQueueMutex_);
    workConsumeCV_.wait(lock, [&] { return sendQueue_.empty(); });
    stop_ = true;
    lock.unlock();

    workProduceCV_.notify_all();
    thread_.join();
  }

  void send(std::shared_ptr<Message> message) override {
    std::ostringstream stream;
    message->save(stream);

    const std::string str = stream.str();
    std::vector<torch::Tensor> data =
      {torch::from_blob((void *)str.c_str(), str.length(), {torch::kChar})};
    enqueue(pg_->send(data, message->dst_rank, 0));
  }

  std::shared_ptr<Request> receiveRequest() override {
    torch::Tensor data = receiveData();
    std::istringstream stream(
      std::string(data.storage().data_ptr(), data.numel()));
    return std::make_shared<Request>(Request::load(stream));
  }

  std::shared_ptr<Response> receiveResponse() override {
    torch::Tensor data = receiveData();
    std::istringstream stream(
      std::string(data.storage().data_ptr(), data.numel()));
    return std::make_shared<Response>(Response::load(stream));
  }

 private:

   torch::Tensor receiveData() {
     std::vector<torch::Tensor> data = {torch::empty({0}, {torch::kChar})};
     pg_->recvAnysource(data, 0)->wait();
     return std::move(data.front());
   }

  void enqueue(std::shared_ptr<c10d::ProcessGroup::Work> work) {
    std::unique_lock<std::mutex> lock(sendQueueMutex_);
    sendQueue_.push_back(std::move(work));
    lock.unlock();

    workProduceCV_.notify_one();
  }

  // making sure tensors are not deleted before send finishes
  void dequeueLoop() {
    std::unique_lock<std::mutex> lock(sendQueueMutex_);

    while (!stop_) {
      if (sendQueue_.empty()) {
        workProduceCV_.wait(lock);
        continue;
      }

      auto work = std::move(sendQueue_.front());
      sendQueue_.pop_front();
      lock.unlock();

      workConsumeCV_.notify_one();

      work->wait();
      lock.lock();
    }
  }

  const std::shared_ptr<c10d::ProcessGroup> pg_;
  std::deque<std::shared_ptr<c10d::ProcessGroup::Work>> sendQueue_;
  std::mutex sendQueueMutex_;
  std::condition_variable workProduceCV_;
  std::condition_variable workConsumeCV_;
  std::thread thread_;
  bool stop_;
};

}
