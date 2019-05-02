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

  void wait();
  void notify(std::shared_ptr<Response> response);
  // block and get IValues
  Stack get();
  // get IValues and convert it to Python obj
  py::object get_py_obj();

private:
  const std::shared_ptr<Request> request_;
  std::promise<std::shared_ptr<Response>> promise_;
  std::shared_ptr<Response> response_;
};

class Client {
 public:
  Client(std::shared_ptr<Transport> transport, int64_t rank);
  std::shared_ptr<RpcWork> sendRequest(std::shared_ptr<Operator> op,
                                       std::vector<IValue> args,
                                       int64_t dst);

 private:
  int64_t getId();
  void processResponse(std::shared_ptr<Message> msg);

  const int64_t rank_;
  int64_t nextId_;
  std::shared_ptr<Transport> transport_;
  std::unordered_map<int64_t, std::shared_ptr<RpcWork>> pendingResponses_;
  std::mutex idMutex_;
  std::mutex responseMutex_;
};

std::shared_ptr<RpcWork> invoke(Client& client,
                                std::string name,
                                int64_t dst,
                                py::args args,
                                py::kwargs kwargs);

}
