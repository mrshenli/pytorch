#include <torch/csrc/distributed/rpc/ProcessGroupAgent.h>

namespace torch {
namespace distributed {
namespace rpc {

ProcessGroupAgent::ProcessGroupAgent(
    std::string workerName,
    std::unordered_map<std::string, int> nameMap,
    c10d::ProcessGroup& pg)
    : RpcAgent(std::move(workerName)),
      nameMap_(std::move(nameMap)),
      stop_(false),
      pg_(pg),
      nextId_(0) {
  if (nameMap_.find(workerName_) == nameMap_.end()
      || pg_.getRank() != nameMap_[workerName_]) {
    throw std::runtime_error("resolved worker name does not match rank");
  }
  for (auto entry : nameMap_) {
    reversedNameMap_[entry.second] = entry.first;
  }
  sendThread_ = std::thread(&ProcessGroupAgent::sendLoop, this);
  listenerThread_ = std::thread(&ProcessGroupAgent::listen, this);
}

ProcessGroupAgent::~ProcessGroupAgent() noexcept(false) {
  if (!stop_) {
    throw std::runtime_error("ProcessGroupAgent cannot be destroyed before"
      "calling shutdown");
  }
}

void ProcessGroupAgent::shutdown() {
  // cannot put this into the destructor, as it is not safe to call virtual
  // functions in constructor and destructor. We can drop this when we can
  // gracefully abort a recvAnysource.
  std::cout << "=== calling destructor \n" << std::flush;
  int dst = (pg_.getRank() + 1) % pg_.getSize();
  std::unique_ptr<std::stringstream> stream(new std::stringstream);
  *stream << 0;
  enqueue(SendWork(dst, std::move(stream), nextId(), SHUTDOWN_TYPE));
  std::unique_lock<std::mutex> lock(sendQueueMutex_);
  std::cout << "-- after enqueue \n" << std::flush;
  workConsumeCV_.wait(lock, [&] { return sendQueue_.empty(); });
  std::cout << "-- after wait \n" << std::flush;
  stop_ = true;
  lock.unlock();

  workProduceCV_.notify_all();
  std::cout << "-- waiting for gcThread\n" << std::flush;
  sendThread_.join();
  std::cout << "-- waiting for listenerTHread\n" << std::flush;
  listenerThread_.join();
  std::cout << "done destructor\n" << std::flush;
}

void ProcessGroupAgent::_send(
    std::string dstName,
    const int64_t requestId,
    const std::vector<at::IValue> values,
    const int type) {
  // TODO check if dstName is known
  std::cout << "--- in _send() \n" << std::flush;
  std::unique_ptr<std::stringstream> stream(new std::stringstream);
  Serializer serializer(values);
  std::cout << "-- before writeNext \n " << std::flush;
  serializer.writeNext(*stream, 0);

  std::cout << "--- dst name is " << dstName << std::endl << std::flush;
  if (nameMap_.find(dstName) == nameMap_.end()) {
    throw std::runtime_error("unrecoganized destination in _send");
  }
  const int dst = nameMap_[dstName];
  SendWork work(dst, std::move(stream), requestId, type);
  enqueue(std::move(work));
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupAgent::sendRequest(
    std::string dstName, Request request) {
  std::cout << "=== in send request \n" << std::flush;
  auto requestId = nextId();
  auto future = c10::make_intrusive<c10::ivalue::Future>();
  futures_[requestId] = future;
  _send(std::move(dstName), requestId, request.toIValues(), REQUEST_TYPE);
  std::cout << "=== finished sending request \n" << std::flush;
  return future;
}

void ProcessGroupAgent::sendResponse(
    std::string dstName, const int64_t requestId, Response response) {
  _send(std::move(dstName), requestId, response.toIValues(), RESPONSE_TYPE);
}

void ProcessGroupAgent::enqueue(SendWork work) {
//void ProcessGroupAgent::enqueue(const int dst, torch::Tensor tensor) {
  std::unique_lock<std::mutex> lock(sendQueueMutex_);
  sendQueue_.emplace_back(std::move(work));
  lock.unlock();

  workProduceCV_.notify_one();
}

// making sure tensors are not deleted before send finishes
void ProcessGroupAgent::sendLoop() {
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

    std::string str = work.data_->str();

    std::vector<torch::Tensor> metaData = {
      torch::tensor(
        {
          work.requestId_,
          (int64_t)pg_.getRank(),
          (int64_t)str.length(),
          (int64_t)work.type_
        }, {torch::kInt64})
    };
    std::cout << "-- before sendTensor 1\n" << std::flush;
    pg_.send(metaData, work.dstRank_, work.dstRank_ /* channelTag */)->wait();
    std::cout << "-- sending data tensor " << str.length() << ". " << str << std::flush;
    if (SHUTDOWN_TYPE != work.type_) {
      std::vector<torch::Tensor> dataTensor =
          {torch::from_blob((void *)str.c_str(), str.length(), {torch::kChar})};
      pg_.send(dataTensor, work.dstRank_, work.dstRank_ /* channelTag */)->wait();
    }

    lock.lock();
  }
}

void ProcessGroupAgent::listen() {
  Deserializer deserializer;
  while (!stop_) {
    // rank, tensor size, requestId
    std::vector<torch::Tensor> meta = {torch::empty({4}, {torch::kInt64})};
    std::cout << "-- " << pg_.getRank() << " waiting recvAnysource\n" << std::flush;
    pg_.recvAnysource(meta, pg_.getRank())->wait();
    std::cout << "-- " << pg_.getRank() << " got one from recvAnysource " << meta[0].numel() * meta[0].element_size() << std::flush;
    int64_t* meta_items = meta.front().data<int64_t>();
    auto requestId = meta_items[0];
    auto srcRank = meta_items[1];
    auto size = meta_items[2];
    auto type = meta_items[3];

    if (SHUTDOWN_TYPE == type) {
      break;
    }

    std::cout << "--- " << requestId << ", " << srcRank << ", " << size << ", " << type << std::endl << std::flush;
    std::vector<torch::Tensor> tensors = {torch::empty({size}, {torch::kChar})};
    pg_.recv(tensors, srcRank, pg_.getRank())->wait();
    std::cout << "-- " << pg_.getRank() << " got the data tensor " << tensors[0].numel() << std::flush;
    std::istringstream stream(std::string(
      (char*)tensors[0].storage().data<signed char>(), tensors[0].numel()));

    std::cout << "-- " << pg_.getRank() << " got the data " << stream.str() << std::endl << std::flush;

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
            reversedNameMap_[srcRank],
            requestId));
    } else if (type == RESPONSE_TYPE){
      Response response = Response::fromIValues(std::move(values));
      // TODO: handle errors
      futures_[requestId]->markCompleted(IValue(response.values()));
      futures_.erase(requestId);
    } else {
      throw std::runtime_error("Unrecognized message type.");
    }
  }
}

}
}
}
