#pragma once

namespace torch {
namespace distributed {
namespace ddp {

enum EventType {
  PREPARE_MODULE = 0,
  PREPARE_FORWARD = 1,
};

struct EventSchema {
  EventSchema(EventType type) : type_(type) {}

  const EventType type_;
};

} // namespace ddp
} // namespace distributed
} // namespace torch
