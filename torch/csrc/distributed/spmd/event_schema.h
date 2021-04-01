#pragma once

namespace torch {
namespace distributed {
namespace spmd {

enum EventType {
  PREPARE_MODULE = 0,
  PRE_FORWARD = 1,
  POST_FORWARD = 2,
  PRE_BACKWARD = 4,
  LOCAL_GRAD_READY = 5,
  //BUCKET_CONTENT_READY = 6,
  //BUCKET_TENSOR_READY = 7,
  BUCKET_READY = 6,
  COMM_DONE = 8,
  GLOBAL_GRAD_READY = 9,
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
