#pragma once

#include <torch/csrc/distributed/ddp/event_schema.h>

#include <vector>


namespace torch {
namespace distributed {
namespace ddp {

class EventHandler {
 public:
  virtual std::vector<EventSchema> ingressEvents() = 0;
  virtual std::vector<EventSchema> egressEvents() = 0;
};

class RootHandler : public EventHandler {
 public:
  std::vector<EventSchema> ingressEvents() override {
    return {};
  }

  std::vector<EventSchema> egressEvents() override {
    return {};
  }
};

} // namespace ddp
} // namespace distributed
} // namespace torch