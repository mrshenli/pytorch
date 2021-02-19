#pragma once

#include <ATen/core/ivalue_inl.h>
#include <c10/util/Exception.h>
#include <torch/csrc/distributed/ddp/event_schema.h>

#include <memory>

namespace torch {
namespace distributed {
namespace ddp {

struct EventContent {

};

class Event : public torch::CustomClassHolder {
 public:
  Event(EventSchema schema, std::unique_ptr<EventContent> content)
      : schema_(std::move(schema)), content_(std::move(content)) {}

  const EventSchema& schema() const {
    return schema_;
  }

  const std::unique_ptr<EventContent>& content() const {
    return content_;
  }

 private:
  const EventSchema schema_;
  // TODO: also expose static pybind functions to convert content types?? (e.g., XyzEvent.from(Event))
  // XyzEvent(py::object), AbcEvent(Gradient), LmnEvent(GradBucket), etc.
  // Trampoline for EventHandlers
  //
  const std::unique_ptr<EventContent> content_;
};

// TODO: add the following to ddp/init.cpp
// static const auto message = torch::class_<Event>("rpc", "_Event");

} // ddp
} // distributed
} // torch
