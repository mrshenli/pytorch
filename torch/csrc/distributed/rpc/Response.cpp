#include <torch/csrc/distributed/rpc/Response.h>

namespace torch {
namespace distributed {
namespace rpc {

// ResponseSerializer
int64_t ResponseSerializer::writeNext(std::ostream& os, uint64_t size) {
  auto starts_from = os.tellp();
  std::vector<at::Tensor> tensor_table;
  Pickler pickler(&tensor_table);

  pickler.start();
  for (auto value: values_) {
    pickler.addIValue(value);
  }
  pickler.addIValue(IValue(code_));
  pickler.finish();

  tensor_table.emplace_back(
    torch::from_blob((void *)pickler.stack().data(),
                     pickler.stack().size(),
                     {torch::kChar}));

  torch::save(tensor_table, os);
  return os.tellp() - starts_from;
}

// Response
int64_t Response::code() {
  return code_;
}

const std::vector<at::IValue> Response::values() {
  return values_;
}

std::unique_ptr<MessageSerializer> Response::serializer() {
  return std::unique_ptr<ResponseSerializer>(
      new ResponseSerializer(code_, values_));
}

// ResponseDeserializer
std::unique_ptr<Message> ResponseDeserializer::readNext(
    std::istream& is, int64_t size) {
  std::vector<at::Tensor> tensor_table;
  torch::load(tensor_table, is);

  auto meta_tensor = std::move(tensor_table.back());
  tensor_table.pop_back();

  Unpickler unpickler(meta_tensor.storage().data(),
                      meta_tensor.numel(),
                      &tensor_table);

  auto values = unpickler.parse_ivalue_list();

  auto code = values.back().toInt();
  values.pop_back();

  return std::unique_ptr<Response>(new Response(code, values));
}

} // namespace rpc
}
}
