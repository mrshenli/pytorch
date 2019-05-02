#include <torch/csrc/distributed/rpc/client.h>

namespace rpc {

void RpcWork::wait() {
  std::future<std::shared_ptr<Response>> future = promise_.get_future();
  response_ = future.get();
}

void RpcWork::notify(std::shared_ptr<Response> response) {
  promise_.set_value(response);
}

Stack RpcWork::get() {
  if (!response_) {
    wait();
  }
  return response_->values();
}

py::object RpcWork::get_py_obj() {
  return torch::jit::createPyObjectForStack(get());
}


Client::Client(std::shared_ptr<Transport> transport, int64_t rank)
    : rank_(rank), nextId_(100), transport_(transport) {
  transport->serveRpc(Response::load, [this](std::unique_ptr<Message> msg){
    this->processResponse(std::move(msg));
  });
}

std::shared_ptr<RpcWork> Client::sendRequest(
    std::shared_ptr<Operator> op, std::vector<IValue> args, int64_t dst) {
  std::shared_ptr<Request> request =
    std::make_shared<Request>(op, args, getId(), rank_, dst);
  transport_->send(request);
  std::shared_ptr<RpcWork> work = std::make_shared<RpcWork>(request);
  {
    std::lock_guard<std::mutex> lock{responseMutex_};
    pendingResponses_[request->id] = work;
  }
  return work;
}

void Client::processResponse(std::shared_ptr<Message> msg) {
  std::shared_ptr<Response> response =
      std::static_pointer_cast<Response>(msg);
  std::shared_ptr<RpcWork> work;
  {
    std::lock_guard<std::mutex> lock{responseMutex_};
    work = pendingResponses_[response->id];
    pendingResponses_.erase(response->id);
  }
  work->notify(response);
}

int64_t Client::getId() {
  std::lock_guard<std::mutex> lock{idMutex_};
  return nextId_++;
}


std::shared_ptr<RpcWork> invoke(Client& client,
                                std::string name,
                                int64_t dst,
                                py::args args,
                                py::kwargs kwargs) {
  Symbol symbol = Symbol::fromQualString(name);
  for (auto op: torch::jit::getAllOperatorsFor(symbol)) {
    try {
      Stack stack =
          torch::jit::createStackForSchema(op->schema(), args, kwargs);
      return client.sendRequest(op, stack, dst);
    } catch (std::runtime_error) {}
  }
  throw std::runtime_error("unrecognized function " + name + " with args ...");
}

}
