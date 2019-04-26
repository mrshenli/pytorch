#pragma once

#include <torch/serialize.h>

namespace rpc {

class Message {
 public:
  Message(int64_t id, int64_t src_rank, int64_t dst_rank)
    : id(id), src_rank(src_rank), dst_rank(dst_rank) {}

  virtual void save(std::ostream& stream) = 0;

  const int64_t id;
  const int64_t src_rank;
  const int64_t dst_rank;
};

}
