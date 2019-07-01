#pragma once

#include <torch/csrc/distributed/rpc/Request.h>
#include <torch/csrc/distributed/rpc/Response.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>

namespace torch {
namespace distributed {
namespace rpc {

using ResponseCallback =
    std::function<void(std::string, const int64_t, Response)>;

class RequestContext {
 public:
  RequestContext(ResponseCallback rcb,
                 const std::string fromWorker,
                 const int64_t requestId);

  void sendResponse(Response response);

 private:
  const ResponseCallback rcb_;
  const std::string fromWorker_;
  const uint64_t requestId_;
};

class RpcAgent {
 public:
  RpcAgent(std::string workerName);

  virtual ~RpcAgent() noexcept(false);

  virtual c10::intrusive_ptr<c10::ivalue::Future> sendRequest(
      std::string toWorker, Request request) = 0;

  virtual void shutdown() = 0;

 protected:
  friend class RequestContext;

  virtual void sendResponse(
      std::string toWorker, const int64_t requestId, Response response) = 0;

  const std::string workerName_;
};

void processRequest(Request request, RequestContext ctx);

}
}
}
