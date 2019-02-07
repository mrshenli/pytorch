#include "torch/csrc/autograd/function.h"

#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/ir.h"


#include <ATen/ATen.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <deque>

namespace torch { namespace autograd {

/// Monotonically incrementing (thread local!) counter to supply sequence
/// numbers.
thread_local uint64_t Function_next_sequence_nr_ = 0;

uint64_t Function::peek_at_next_sequence_nr() {
  return Function_next_sequence_nr_;
}

uint64_t& Function::get_next_sequence_nr() {
  return Function_next_sequence_nr_;
}

auto Function::name() const -> std::string {
  return at::demangle(typeid(*this).name());
}

AnomalyMetadata* Function::metadata() noexcept {
  if (!anomaly_metadata_) {
    anomaly_metadata_ = Engine::get_default_engine().make_anomaly_metadata();
  }
  return anomaly_metadata_.get();
}

}} // namespace torch::autograd
