#pragma once

#include <future>

#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/csrc/distributed/rpc/Request.h>
#include <torch/csrc/distributed/rpc/Response.h>
#include <torch/csrc/distributed/rpc/Transport.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>

namespace rpc {

class RpcWork {

public:

 // TODO: use jit::future
  RpcWork(std::shared_ptr<Request> request) : request_(request) {}

  void wait() {
    std::future<std::shared_ptr<Response>> future = promise_.get_future();
    response_ = future.get();
  }

  void notify(std::shared_ptr<Response> response) {
    promise_.set_value(response);
  }

  Stack get() {
    if (!response_) {
      wait();
    }
    return response_->values();
  }

  py::object get_py_obj() {
    return torch::jit::createPyObjectForStack(get());
  }


private:
  const std::shared_ptr<Request> request_;
  std::promise<std::shared_ptr<Response>> promise_;
  std::shared_ptr<Response> response_;
};

class Client {
 public:

  Client(std::shared_ptr<Transport> transport, int64_t rank)
      : rank_(rank), nextId_(100), transport_(transport) {
    transport->serveRpc(Response::load, [this](std::unique_ptr<Message> msg){
      this->processResponse(std::move(msg));
    });
  }

  std::shared_ptr<RpcWork> sendRequest(
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

 private:

  int64_t getId() {
    std::lock_guard<std::mutex> lock{idMutex_};
    return nextId_++;
  }

  void processResponse(std::shared_ptr<Message> msg) {
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

  const int64_t rank_;
  int64_t nextId_;
  std::shared_ptr<Transport> transport_;
  std::unordered_map<int64_t, std::shared_ptr<RpcWork>> pendingResponses_;
  std::mutex idMutex_;
  std::mutex responseMutex_;
};


std::shared_ptr<RpcWork> invoke(Client& client,
               std::string name,
               int64_t dst_rank,
               py::args args,
               py::kwargs kwargs) {
  Symbol symbol = Symbol::fromQualString(name);
  for (auto op: torch::jit::getAllOperatorsFor(symbol)) {
    try {
      Stack stack = torch::jit::createStackForSchema(op->schema(), args, kwargs);
      return client.sendRequest(op, stack, dst_rank);
    } catch (std::runtime_error) {}
  }
  throw std::runtime_error("unrecognized function " + name + " with args ...");
}

}
