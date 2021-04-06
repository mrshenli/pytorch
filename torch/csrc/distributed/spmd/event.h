#pragma once

#include <ATen/core/ivalue_inl.h>
#include <torch/csrc/distributed/spmd/event_schema.h>

#include <memory>


namespace torch {
namespace distributed {
namespace spmd {

// Event base class.
// NB: This class inherits torch::CustomClassHolder so that it can be a custom
// IValue type and can be wrapped with ivalue::Future.
class Event : public torch::CustomClassHolder {
 public:
  Event(EventSchema schema) : schema_(std::move(schema)) {}

  const EventSchema& schema() const {
    return schema_;
  }

 private:
  const EventSchema schema_;
};

} // spmd
} // distributed
} // torch
