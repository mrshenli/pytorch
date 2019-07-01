#include <torch/csrc/distributed/rpc/Serializer.h>

namespace torch {
namespace distributed {
namespace rpc {

Serializer::Serializer(const std::vector<IValue>& values) : values_(values) {}

int64_t Serializer::writeNext(std::ostream& os, uint64_t size) {
  AT_CHECK(size == 0, "Streaming serialization not supported, but got "
      "serialize buffer size ", size);

  auto starts_from = os.tellp();
  std::vector<at::Tensor> tensor_table;
  Pickler pickler(&tensor_table);

  pickler.start();
  pickler.addIValue(values_);
  pickler.finish();

  tensor_table.push_back(
    torch::from_blob((void *)pickler.stack().data(),
                     pickler.stack().size(),
                     {torch::kChar}));

  torch::save(tensor_table, os);
  auto ret = os.tellp() - starts_from;
  return ret;
}

}
}
}
