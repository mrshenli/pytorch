#include <torch/csrc/distributed/rpc/ProcessGroupAgent.h>

namespace torch {
namespace distributed {
namespace rpc {

ProcessGroupAgent::ProcessGroupAgent(
    std::string workerName,
    NameStore& ns,
    c10d::ProcessGroup& pg)
    : RpcAgent(std::move(workerName), ns),
      stop_(false),
      pg_(pg),
      nextId_(0) {
  std::cout << "-- constructing processgroupagent\n" << std::flush;
  auto resolved = ns_.resolve(workerName_);
  if (!std::all_of(resolved.begin(), resolved.end(), ::isdigit)
      || stoi(resolved) != pg_.getRank()) {
    throw std::runtime_error("NameStore resolved incorrectly.");
  }
  gcThread_ = std::thread(&ProcessGroupAgent::gcLoop, this);
  listenerThread_ = std::thread(&ProcessGroupAgent::listen, this);
}

ProcessGroupAgent::~ProcessGroupAgent() {
  std::unique_lock<std::mutex> lock(sendQueueMutex_);
  workConsumeCV_.wait(lock, [&] { return sendQueue_.empty(); });
  stop_ = true;
  lock.unlock();

  workProduceCV_.notify_all();
  gcThread_.join();
  listenerThread_.join();
}

void ProcessGroupAgent::_send(
    std::string dstName,
    const int64_t requestId,
    const std::vector<at::IValue> values,
    const int type) {
  std::stringstream stream;
  Serializer serializer(values);
  serializer.writeNext(stream, 0);

  const int dst = stoi(ns_.resolve(dstName));
  const std::string str = stream.str();
  auto metaTensor =
    torch::tensor(
      {requestId, (int64_t)pg_.getRank(), (int64_t)str.length(), (int64_t)type},
      {torch::kInt64});
  sendTensor(metaTensor, dst);
  auto dataTensor =
    torch::from_blob((void *)str.c_str(), str.length(), {torch::kChar});
  sendTensor(dataTensor, dst);
}

std::shared_ptr<Future> ProcessGroupAgent::sendRequest(
    std::string dstName, Request request) {
  auto requestId = nextId();
  auto future = std::make_shared<Future>();
  futures_[requestId] = future;
  _send(std::move(dstName), requestId, request.toIValues(), REQUEST_TYPE);
  return future;
}

void ProcessGroupAgent::sendResponse(
    std::string dstName, const int64_t requestId, Response response) {
  _send(std::move(dstName), requestId, response.toIValues(), RESPONSE_TYPE);
}

void ProcessGroupAgent::sendTensor(
    torch::Tensor tensor, const int dstRank) {
  std::vector<torch::Tensor> vec = {tensor};
  auto data = std::make_shared<std::vector<torch::Tensor>>(std::move(vec));
  auto work = pg_.send(*data, dstRank, dstRank /* channelTag */);
  work->wait();
  //enqueue(std::make_shared<SendWork>(data, work));
}

void ProcessGroupAgent::enqueue(std::shared_ptr<SendWork> work) {
  std::unique_lock<std::mutex> lock(sendQueueMutex_);
  sendQueue_.emplace_back(std::move(work));
  lock.unlock();

  workProduceCV_.notify_one();
}

// making sure tensors are not deleted before send finishes
void ProcessGroupAgent::gcLoop() {
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

void ProcessGroupAgent::listen() {
  Deserializer deserializer;
  while (!stop_) {
    // rank, tensor size, requestId
    std::vector<torch::Tensor> meta = {torch::empty({4}, {torch::kInt64})};
    pg_.recvAnysource(meta, pg_.getRank())->wait();
    int64_t* meta_items = meta.front().data<int64_t>();
    auto requestId = meta_items[0];
    auto srcRank = meta_items[1];
    auto size = meta_items[2];
    auto type = meta_items[3];
    std::vector<torch::Tensor> tensors = {torch::empty({size}, {torch::kChar})};
    pg_.recv(tensors, srcRank, pg_.getRank())->wait();
    std::istringstream stream(std::string(
      (char*)tensors[0].storage().data<signed char>(), tensors[0].numel()));

    auto values = deserializer.readNext(stream, 0);
    if (type == REQUEST_TYPE) {
      // TODO: grad a thread from thread pool to process this request
      processRequest(
        Request::fromIValues(std::move(values)),
        RequestContext(
            [this](std::string dstName,
                   const int64_t requestId,
                   Response response) {
              this->sendResponse(
                  std::move(dstName), requestId, std::move(response));
            },
            std::to_string(srcRank),
            requestId));
    } else if (type == RESPONSE_TYPE){
      Response response = Response::fromIValues(std::move(values));
      // TODO: handle errors
      futures_[requestId]->markCompleted(IValue(std::move(response.values())));
      futures_.erase(requestId);
    } else {
      throw std::runtime_error("Unrecognized message type.");
    }
  }
}

}
}
}
