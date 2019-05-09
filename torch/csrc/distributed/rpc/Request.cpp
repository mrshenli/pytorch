#include <torch/csrc/distributed/rpc/Request.h>

namespace torch {
namespace distributed {
namespace rpc {

std::shared_ptr<Operator> matchOperator(
    at::Symbol symbol, std::string str_schema) {
  for (auto op: torch::jit::getAllOperatorsFor(symbol)) {
    if (toString(op->schema()).compare(str_schema) == 0) {
      return op;
    }
  }
  throw std::runtime_error("Cannot find matching operator");
}

// RequestSerializer
int64_t RequestSerializer::writeNext(std::ostream& os, uint64_t size) {
  AT_CHECK(size == 0, "Streaming serialization not supported, but got "
      "serialize buffer size ", size);

  auto starts_from = os.tellp();
  std::vector<at::Tensor> tensor_table;
  Pickler pickler(&tensor_table);

  pickler.start();
  for (auto arg: values_) {
    pickler.addIValue(arg);
  }
  std::string str_schema = toString(op_->schema());
  pickler.addIValue(IValue(str_schema));
  pickler.finish();

  tensor_table.push_back(
    torch::from_blob((void *)pickler.stack().data(),
                     pickler.stack().size(),
                     {torch::kChar}));

  torch::save(tensor_table, os);
  return os.tellp() - starts_from;
}


// Request
std::shared_ptr<Operator> Request::op() {
  return op_;
}

std::vector<at::IValue> Request::args() {
  return values_;
}

std::unique_ptr<MessageSerializer> Request::serializer() {
  return std::unique_ptr<RequestSerializer>(
      new RequestSerializer(op_, values_));
}

// RequestDeserializer
std::unique_ptr<Message> RequestDeserializer::readNext(
    std::istream& is, int64_t size) {
  std::vector<at::Tensor> tensor_table;
  torch::load(tensor_table, is);
  auto meta_tensor = std::move(tensor_table.back());
  tensor_table.pop_back();

  Unpickler unpickler(meta_tensor.storage().data(),
                      meta_tensor.numel(),
                      &tensor_table);

  auto args = unpickler.parse_ivalue_list();
  auto str_schema = args.back().toStringRef();
  args.pop_back();

  auto str_symbol = str_schema.substr(0, str_schema.find("("));
  auto symbol = at::Symbol::fromQualString(str_symbol);
  auto op = matchOperator(symbol, str_schema);

  return std::unique_ptr<Request>(new Request(op, args));
}

} // namespace rpc
}
}
