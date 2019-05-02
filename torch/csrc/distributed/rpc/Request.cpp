#include <torch/csrc/distributed/rpc/Request.h>

namespace rpc {

Request::Request(std::shared_ptr<Operator> op,
        const std::vector<at::IValue> args,
        int64_t id,
        int64_t src,
        int64_t dst)
  : Message(id, src, dst), op_(op), args_(std::move(args)) {}

at::Symbol Request::symbol() {
  return at::Symbol::fromQualString(op_->schema().name());
}

std::shared_ptr<Operator> Request::op() {
  return op_;
}

std::vector<at::IValue> Request::args() {
  return args_;
}

void Request::save(std::ostream& stream) {
  std::vector<at::Tensor> tensor_table;
  Pickler pickler(&tensor_table);

  pickler.start();
  for (auto arg: args_) {
    pickler.addIValue(arg);
  }

  std::string str_schema = toString(op_->schema());
  pickler.addIValue(IValue(str_schema));
  pickler.addIValue(IValue(id));
  pickler.addIValue(IValue(src));
  pickler.addIValue(IValue(dst));
  pickler.finish();

  tensor_table.push_back(
    torch::from_blob((void *)pickler.stack().data(),
                     pickler.stack().size(),
                     {torch::kChar}));

  torch::save(tensor_table, stream);
}

std::unique_ptr<Request> Request::load(std::istream& stream) {
  std::vector<at::Tensor> tensor_table;
  torch::load(tensor_table, stream);

  auto meta_tensor = std::move(tensor_table.back());
  tensor_table.pop_back();

  Unpickler unpickler(meta_tensor.storage().data(),
                      meta_tensor.numel(),
                      &tensor_table);

  auto args = unpickler.parse_ivalue_list();

  auto dst = args.back().toInt();
  args.pop_back();

  auto src = args.back().toInt();
  args.pop_back();

  auto id = args.back().toInt();
  args.pop_back();

  auto str_schema = args.back().toStringRef();
  args.pop_back();

  auto str_symbol = str_schema.substr(0, str_schema.find("("));
  auto symbol = at::Symbol::fromQualString(str_symbol);

  auto op = matchOperator(symbol, str_schema);
  return std::unique_ptr<Request>(
    new Request(op, args, id, src, dst));
}

std::shared_ptr<Operator> Request::matchOperator(
    at::Symbol symbol, std::string str_schema) {
  for (auto op: torch::jit::getAllOperatorsFor(symbol)) {
    if (toString(op->schema()).compare(str_schema) == 0) {
      return op;
    }
  }
  throw std::runtime_error("Cannot find matching operator");
}

} // namespace rpc
