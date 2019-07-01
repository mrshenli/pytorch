#include <torch/csrc/distributed/rpc/RpcAgent.h>

namespace torch {
namespace distributed {
namespace rpc {

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
