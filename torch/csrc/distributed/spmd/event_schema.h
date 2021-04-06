#pragma once

namespace torch {
namespace distributed {
namespace spmd {

// TODO: add PRE_FORWARD, POST_FORWARD, PRE_BACKWARD, and POST_BACKWARD
enum EventType {
  PREPARE_MODULE = 0,
  PRE_FORWARD = 1,
  POST_FORWARD = 2,
  LOCAL_GRAD_READY = 3,
  BUCKET_READY = 4,
  COMM_DONE = 5,
  GLOBAL_GRAD_READY = 6,
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
