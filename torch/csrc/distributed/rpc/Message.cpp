#include <torch/csrc/distributed/rpc/Message.h>

namespace torch {
namespace distributed {
namespace rpc {

MessageSerializer::MessageSerializer(const std::vector<IValue>& values) : values_(values) {}


Message::Message(const std::vector<IValue> values) : values_(values) {}

}
}
}
