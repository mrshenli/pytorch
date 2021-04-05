#pragma once

#include <ATen/core/ivalue_inl.h>
#include <torch/csrc/distributed/spmd/event_schema.h>

#include <memory>


namespace torch {
namespace distributed {
namespace spmd {

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
