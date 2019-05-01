#pragma once

#include <torch/csrc/distributed/rpc/Message.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/serialize.h>
#include <vector>

namespace rpc {

class Response : public Message {
 public:

  Response(int64_t code,
           std::vector<at::IValue> values,
           int64_t id,
           int64_t src_rank,
           int64_t dst_rank)
    : Message(id, src_rank, dst_rank), code_(code), values_(std::move(values)) {}

  int code() {
    return code_;
  }

  const std::vector<at::IValue> values() {
    return values_;
  }

  void save(std::ostream& stream) override {
    std::vector<at::Tensor> tensor_table;
    Pickler pickler(&tensor_table);

    pickler.start();
    for (auto value: values_) {
      pickler.addIValue(value);
    }
    pickler.addIValue(IValue(code_));
    pickler.addIValue(IValue(id));
    pickler.addIValue(IValue(src));
    pickler.addIValue(IValue(dst));
    pickler.finish();

    tensor_table.emplace_back(
      torch::from_blob((void *)pickler.stack().data(),
                       pickler.stack().size(),
                       {torch::kChar}));

    torch::save(tensor_table, stream);
  }

  static std::unique_ptr<Response> load(std::istream& stream) {
    std::vector<at::Tensor> tensor_table;
    torch::load(tensor_table, stream);

    auto meta_tensor = std::move(tensor_table.back());
    tensor_table.pop_back();

    Unpickler unpickler(meta_tensor.storage().data(),
                        meta_tensor.numel(),
                        &tensor_table);

    auto values = unpickler.parse_ivalue_list();

    auto dst = values.back().toInt();
    values.pop_back();

    auto src = values.back().toInt();
    values.pop_back();

    auto id = values.back().toInt();
    values.pop_back();

    auto code = values.back().toInt();
    values.pop_back();

    return std::unique_ptr<Response>(
      new Response(code, values, id, src, dst));
  }

 private:

  const int64_t code_;
  const std::vector<at::IValue> values_;
};

}
