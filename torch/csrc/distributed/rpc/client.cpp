#include <torch/csrc/distributed/rpc/client.h>

namespace torch {
namespace distributed {
namespace rpc {

void RpcFuture::notify(std::unique_ptr<Response> response) {
  if (response->code() == 0) {
    future_->markCompleted(IValue(response->values()));
  } else {
    throw std::runtime_error("Rpc Error");
  }
}

const Stack& RpcFuture::get() {
  if (!future_->completed()) {
    wait();
  }
  return future_->value().toGenericListRef();
}

Client::Client(std::shared_ptr<TransportFactory> transportFactory) {
  transport_ = transportFactory ->createTransport(
      processResponse,
      std::unique_ptr<ResponseDeserializerFactory>(
          new ResponseDeserializerFactory()),
      true);
}

std::shared_ptr<RpcFuture> Client::sendRequest(
    int64_t dst, std::shared_ptr<Operator> op, std::vector<IValue> args) {
  std::unique_ptr<Request> request =
      std::unique_ptr<Request>(new Request(op, args));
  std::unique_ptr<ClientSendContext> ctx = std::unique_ptr<ClientSendContext>();
  std::shared_ptr<RpcFuture> future = ctx->future();
  transport_->send(dst, std::move(request), std::move(ctx));
  return future;
}

std::unique_ptr<Message> Client::processResponse(
    int64_t src,
    std::unique_ptr<Message> msg,
    std::unique_ptr<SendContext> sendCtx) {

  std::unique_ptr<Response> response =
    static_unique_ptr_cast<Message, Response>(std::move(msg));
  std::unique_ptr<ClientSendContext> clientCtx =
    static_unique_ptr_cast<SendContext, ClientSendContext>(std::move(sendCtx));

  clientCtx->future()->notify(std::move(response));
  return nullptr;
}

}
}
}
