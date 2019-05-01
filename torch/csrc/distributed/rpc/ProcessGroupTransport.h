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

  ProcessGroupTransport(c10d::ProcessGroup& pg)
      : pg_(pg), stop_(false) {
    gcThread_ = std::thread(&ProcessGroupTransport::gcLoop, this);
  }

  ~ProcessGroupTransport() {
    std::unique_lock<std::mutex> lock(sendQueueMutex_);
    workConsumeCV_.wait(lock, [&] { return sendQueue_.empty(); });
    stop_ = true;
    lock.unlock();

    workProduceCV_.notify_all();
    gcThread_.join();
    listenerThread_.join();
  }

  void send(std::shared_ptr<Message> message) override {
    std::stringstream stream;
    message->save(stream);

    const std::string str = stream.str();
    sendTensor(torch::tensor({message->src, (int64_t)str.length()}, {torch::kInt64}), message->dst);
    sendTensor(torch::from_blob((void *)str.c_str(), str.length(), {torch::kChar}), message->dst);
  }

  void sendTensor(torch::Tensor tensor, int64_t dst) {
    std::vector<torch::Tensor> vec = {tensor};
    auto data = std::make_shared<std::vector<torch::Tensor>>(std::move(vec));
    auto work = pg_.send(*data, dst, 0);
    work->wait();
    //enqueue(std::make_shared<SendWork>(data, work));
  }

  void serveRpc(MessageDeserializer md, RpcCallback cb) override {
    cb_ = cb;
    md_ = md;
    listenerThread_ = std::thread(&ProcessGroupTransport::listenerLoop, this);
  }

 private:

  torch::Tensor receiveData() {
    // rank, and tensor size
    std::vector<torch::Tensor> meta = {torch::empty({2}, {torch::kInt64})};
    pg_.recvAnysource(meta, 0)->wait();
    int64_t* meta_items = meta.front().data<int64_t>();
    int64_t src = meta_items[0];
    int64_t size = meta_items[1];
    std::vector<torch::Tensor> data = {torch::empty({size}, {torch::kChar})};
    pg_.recv(data, src, 0)->wait();
    return std::move(data.front());
  }

  void enqueue(std::shared_ptr<SendWork> work) {
    std::unique_lock<std::mutex> lock(sendQueueMutex_);
    sendQueue_.emplace_back(std::move(work));
    lock.unlock();

    workProduceCV_.notify_one();
  }

  // making sure tensors are not deleted before send finishes
  void gcLoop() {
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

      work->second->wait();
      lock.lock();
    }
  }

  void listenerLoop() {
    while (!stop_) {
      torch::Tensor data = receiveData();
      std::istringstream stream(
        std::string((char*)data.storage().data<signed char>(), data.numel()));
      std::unique_ptr<Message> msg = md_(stream);
      cb_(std::move(msg));
    }
  }

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

  std::vector<std::shared_ptr<std::vector<torch::Tensor>>> tensors;
  std::vector<std::shared_ptr<c10d::ProcessGroup::Work>> works;
};

}
