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
      : rank_(rank), nextId_(0), stop_(false), transport_(transport) {
    listenerThread_ = std::thread(&Client::listenerLoop, this);
  }

  std::shared_ptr<RpcWork> sendRequest(
      std::shared_ptr<Operator> op, std::vector<IValue> args, int64_t dstRank) {
    std::shared_ptr<Request> request =
      std::make_shared<Request>(op, args, getId(), rank_, dstRank);
    transport_->send(request);
    std::shared_ptr<RpcWork> work = std::make_shared<RpcWork>(request);
    {
      std::lock_guard<std::mutex> lock{responseMutex_};
      pendingResponses[request->id] = work;
    }
    return work;
  }

 private:

  int64_t getId() {
    std::lock_guard<std::mutex> lock{idMutex_};
    return nextId_++;
  }

  void listenerLoop() {
    while(!stop_) {
      std::shared_ptr<Response> response = transport_->receiveResponse();
      std::shared_ptr<RpcWork> work;
      {
        std::lock_guard<std::mutex> lock{responseMutex_};
        work = pendingResponses[response->id];
        pendingResponses.erase(response->id);
      }
      work->notify(response);
    }
  }

  const int64_t rank_;
  int64_t nextId_;
  bool stop_;
  std::shared_ptr<Transport> transport_;
  std::thread listenerThread_;
  std::unordered_map<int64_t, std::shared_ptr<RpcWork>> pendingResponses;
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
      //std::cout << "=== matching signature is " << op->schema() << std::endl;
      //std::cout << "=== op is " << c10::toString(op->schema()) << std::endl;
      //std::cout << "=== name is " << op->schema().name() << ", overload_name is " << op->schema().overload_name() << std::endl;
      return client.sendRequest(op, stack, dst_rank);
    } catch (std::runtime_error) {}
  }
  throw std::runtime_error("unrecognized function " + name + " with args ...");
}

}
