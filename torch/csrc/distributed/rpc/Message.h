#pragma once

#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/serialize.h>

namespace rpc {

class Message {
 public:
  Message(int64_t id, int64_t src, int64_t dst)
    : id(id), src(src), dst(dst) {}

  virtual void save(std::ostream& stream) = 0;

  const int64_t id;
  const int64_t src;
  const int64_t dst;
};

}
