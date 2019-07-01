#pragma once

#include <torch/csrc/distributed/rpc/Request.h>
#include <torch/csrc/distributed/rpc/Response.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>

namespace torch {
namespace distributed {
namespace rpc {

using ResponseCallback =
    std::function<void(std::string, const int64_t, Response)>;

struct RequestContext {
  RequestContext(ResponseCallback rcb,
                 const std::string resolvedSrc,
                 const int64_t requestId)
      : rcb_(rcb),
        resolvedSrc_(std::move(resolvedSrc)),
        requestId_(requestId) {}

  void sendResponse(Response response) {
    rcb_(resolvedSrc_, requestId_, std::move(response));
  }

 private:
  const ResponseCallback rcb_;
  const std::string resolvedSrc_;
  const uint64_t requestId_;
};

struct NameStore {
  virtual ~NameStore() {};
  virtual const std::string resolve(std::string name) = 0;
};

class RpcAgent {
 public:
  RpcAgent(std::string workerName, NameStore& ns)
      : workerName_(std::move(workerName)),
        ns_(ns) {}

  virtual ~RpcAgent() {};

  virtual std::shared_ptr<c10::ivalue::Future> sendRequest(
      std::string dstName, Request request) = 0;

 protected:
  friend class RequestContext;

  virtual void sendResponse(
      std::string resolvedSrc,
      const int64_t requestId,
      Response response) = 0;

  const std::string workerName_;
  NameStore& ns_;
};

void processRequest(Request request, RequestContext ctx);

}
}
}
