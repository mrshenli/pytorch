#include <torch/csrc/distributed/rpc/RpcAgent.h>

namespace torch {
namespace distributed {
namespace rpc {


RequestContext::RequestContext(
    ResponseCallback rcb,
    const std::string fromWorker,
    const int64_t requestId)
    : rcb_(rcb),
      fromWorker_(std::move(fromWorker)),
      requestId_(requestId) {}

void RequestContext::sendResponse(Response response) {
  rcb_(fromWorker_, requestId_, std::move(response));
}

RpcAgent::RpcAgent(std::string workerName)
    : workerName_(std::move(workerName)) {}

RpcAgent::~RpcAgent() noexcept(false) {}

void processRequest(Request request, RequestContext ctx) {
  auto op = request.op();
  auto args = request.args();

  Stack stack;
  stack.insert(stack.end(), args.begin(), args.end());
  op->getOperation()(stack);

  ctx.sendResponse(Response(0, std::move(stack)));
}

}
}
}
