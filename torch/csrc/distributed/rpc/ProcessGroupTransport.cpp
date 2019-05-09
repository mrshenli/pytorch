#include <torch/csrc/distributed/rpc/ProcessGroupTransport.h>

namespace torch {
namespace distributed {
namespace rpc {

// ProcessGroupTransportFactory
ProcessGroupTransportFactory::ProcessGroupTransportFactory(
  c10d::ProcessGroup& pg) : pg_(pg) {}

std::unique_ptr<Transport> ProcessGroupTransportFactory::createTransport(
    Callback cb, std::unique_ptr<MessageDeserializerFactory> mdf, bool isClient) {
  return std::unique_ptr<ProcessGroupTransport>(new ProcessGroupTransport(
    pg_.getRank(), std::move(cb), std::move(mdf), pg_, isClient));
}


ProcessGroupTransport::ProcessGroupTransport(
    const int64_t id,
    Callback cb,
    std::unique_ptr<MessageDeserializerFactory> mdf,
    c10d::ProcessGroup& pg,
    bool isClient)
    : Transport(id, std::move(cb), std::move(mdf)),
      sendTag_(isClient),
      recvTag_(!isClient),
      stop_(false),
      pg_(pg),
      nextId_(0) {
  md_ = mdf_->deserializer();
  gcThread_ = std::thread(&ProcessGroupTransport::gcLoop, this);
  listenerThread_ = std::thread(&ProcessGroupTransport::listenerLoop, this);
}

ProcessGroupTransport::~ProcessGroupTransport() {
  std::unique_lock<std::mutex> lock(sendQueueMutex_);
  workConsumeCV_.wait(lock, [&] { return sendQueue_.empty(); });
  stop_ = true;
  lock.unlock();

  workProduceCV_.notify_all();
  gcThread_.join();
  listenerThread_.join();
}

void ProcessGroupTransport::send(int64_t dst,
                                 std::unique_ptr<Message> msg,
                                 std::unique_ptr<SendContext> ctx) {
  std::stringstream stream;
  msg->serializer()->writeNext(stream, 0);

  auto msgId = nextId();
  const std::string str = stream.str();
  auto metaTensor =
    torch::tensor({msgId, (int64_t)pg_.getRank(), (int64_t)str.length()},
                  {torch::kInt64});
  sendTensor(metaTensor, dst);
  auto dataTensor =
    torch::from_blob((void *)str.c_str(), str.length(), {torch::kChar});
  sendTensor(dataTensor, dst);
  if (ctx) {
    ctxs_.emplace(msgId, std::move(ctx));
  }
}

void ProcessGroupTransport::sendTensor(torch::Tensor tensor, int64_t dst) {
  std::vector<torch::Tensor> vec = {tensor};
  auto data = std::make_shared<std::vector<torch::Tensor>>(std::move(vec));
  auto work = pg_.send(*data, dst, sendTag_);
  work->wait();
  //enqueue(std::make_shared<SendWork>(data, work));
}

MessageData ProcessGroupTransport::receiveData() {
  // rank, tensor size, msg id
  std::vector<torch::Tensor> meta = {torch::empty({3}, {torch::kInt64})};
  pg_.recvAnysource(meta, recvTag_)->wait();
  int64_t* meta_items = meta.front().data<int64_t>();
  MessageData data;
  data.msgId = meta_items[0];
  data.srcId = meta_items[1];
  int64_t size = meta_items[2];
  std::vector<torch::Tensor> tensors = {torch::empty({size}, {torch::kChar})};
  pg_.recv(tensors, data.srcId, recvTag_)->wait();
  std::istringstream stream(std::string(
    (char*)tensors[0].storage().data<signed char>(), tensors[0].numel()));
  data.msg = md_->readNext(stream, 0);
  return data;
}

void ProcessGroupTransport::enqueue(std::shared_ptr<SendWork> work) {
  std::unique_lock<std::mutex> lock(sendQueueMutex_);
  sendQueue_.emplace_back(std::move(work));
  lock.unlock();

  workProduceCV_.notify_one();
}

// making sure tensors are not deleted before send finishes
void ProcessGroupTransport::gcLoop() {
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

void ProcessGroupTransport::listenerLoop() {
  while (!stop_) {
    // TODO: stop receiving data when stop
    MessageData data = receiveData();
    std::unique_ptr<Message> msg =
        cb_(data.srcId, std::move(data.msg), std::move(ctxs_[data.msgId]));
    ctxs_.erase(data.msgId);
    if (msg) {
      send(data.srcId, std::move(msg), nullptr);
    }
  }
}

}
}
}
