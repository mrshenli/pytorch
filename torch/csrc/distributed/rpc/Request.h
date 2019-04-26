#pragma once

#include <torch/csrc/distributed/rpc/Message.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/serialize.h>
#include <vector>

namespace rpc {

class Request : public Message {
 public:
  Request(std::shared_ptr<Operator> op,
          const std::vector<at::IValue> args,
          int64_t id,
          int64_t src_rank,
          int64_t dst_rank)
    : Message(id, src_rank, dst_rank), op_(op), args_(std::move(args)) {}

  at::Symbol symbol() {
    return at::Symbol::fromQualString(op_->schema().name());
  }

  std::vector<at::IValue> args() {
    return args_;
  }

  void save(std::ostream& stream) override {
    std::vector<at::Tensor> tensor_table;
    Pickler pickler(&tensor_table);

    pickler.start();
    for (auto arg: args_) {
      pickler.addIValue(arg);
    }
    pickler.addIValue(IValue(op_->schema().name()));
    pickler.addIValue(IValue(c10::toString(op_->schema())));
    pickler.addIValue(IValue(id));
    pickler.addIValue(IValue(src_rank));
    pickler.addIValue(IValue(dst_rank));
    pickler.finish();

    tensor_table.emplace_back(
      torch::from_blob((void *)pickler.stack().data(),
                       pickler.stack().size(),
                       {torch::kChar}));

    torch::save(tensor_table, stream);
  }

  static Request load(std::istream& stream) {
    std::vector<at::Tensor> tensor_table;
    torch::load(tensor_table, stream);

    auto meta_tensor = std::move(tensor_table.back());
    tensor_table.pop_back();

    Unpickler unpickler(meta_tensor.storage().data(),
                        meta_tensor.numel(),
                        &tensor_table);

    auto args = unpickler.parse_ivalue_list();
    std::string str_schema = args.back().toStringRef();
    args.pop_back();

    auto symbol = at::Symbol::fromQualString(args.back().toStringRef());
    args.pop_back();

    auto id = args.back().toInt();
    args.pop_back();

    auto src_rank = args.back().toInt();
    args.pop_back();

    auto dst_rank = args.back().toInt();
    args.pop_back();

    auto op = matchOperator(symbol, str_schema);
    return Request(op, args, id, src_rank, dst_rank);
  }
 private:

  static std::shared_ptr<Operator> matchOperator(
      at::Symbol symbol, std::string str_schema) {
    for (auto op: torch::jit::getAllOperatorsFor(symbol)) {
      if (c10::toString(op->schema()).compare(str_schema) == 0) {
        return op;
      }
    }
    throw std::runtime_error("Cannot find matching operator");
  }

  std::shared_ptr<Operator> op_;
  std::vector<at::IValue> args_;
};
}
