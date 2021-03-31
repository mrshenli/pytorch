#pragma once

namespace torch {
namespace distributed {
namespace spmd {

enum EventType {
  PREPARE_MODULE = 0,
  PRE_FORWARD = 1,
  POST_FORWARD = 2,
  GRAD_READY = 3,
};

struct EventSchema {
  EventSchema(EventType type) : type_(type) {}

  bool operator==(const EventSchema& rhs) const {
    return type_ == rhs.type_;
  }

  struct Hash {
    size_t operator()(const EventSchema& key) const {
      return key.type_;
    }
  };

  const EventType type_;
};

} // namespace spmd
} // namespace distributed
} // namespace torch
