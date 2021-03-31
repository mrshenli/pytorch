#pragma once

#include <torch/csrc/distributed/spmd/event.h>

#include <vector>


namespace torch {
namespace distributed {
namespace spmd {

using c10::ivalue::Future;

class EventHandler {
 public:
  virtual std::vector<EventSchema> ingressEvents() = 0;
  virtual std::vector<EventSchema> egressEvents() = 0;
  // returned Future contents Event objects generated by this handler.
  virtual std::vector<std::shared_ptr<Future>> handleEvent(
      const c10::intrusive_ptr<Event>&) = 0;
};

class RootHandler : public EventHandler {
 public:
  std::vector<EventSchema> ingressEvents() override {
    return {};
  }

  std::vector<EventSchema> egressEvents() override {
    return {
        EventType::PREPARE_MODULE,
        EventType::PRE_FORWARD,
        EventType::POST_FORWARD};
  }

  std::vector<std::shared_ptr<Future>> handleEvent(
      const c10::intrusive_ptr<Event>& event) override {
    switch (event->schema().type_) {
      case EventType::PREPARE_MODULE: {
        auto pme =
            c10::dynamic_intrusive_pointer_cast<PrepareModuleEvent>(event);
        std::cout << "PREPARE_MODULE: " << pme->parameters().size() << std::endl << std::flush;
        return {};
      }
      default:
        return {};
    }
  }
};


class DefaultTrigger : public EventHandler {
 public:
  std::vector<EventSchema> ingressEvents() override {
    return {EventType::PREPARE_MODULE};
  }

  std::vector<EventSchema> egressEvents() override {
    return {EventType::GRAD_READY};
  }

  std::vector<std::shared_ptr<Future>> handleEvent(
      const c10::intrusive_ptr<Event>&) override {
    TORCH_INTERNAL_ASSERT(false);
  }
};

class DefaultBucketIndexer : public EventHandler {
 public:
  std::vector<EventSchema> ingressEvents() override {
    return {EventType::GRAD_READY, EventType::COMM_DONE};
  }

  std::vector<EventSchema> egressEvents() override {
    return {EventType::BUCKET_CONTENT_READY};
  }

  std::vector<std::shared_ptr<Future>> handleEvent(
      const c10::intrusive_ptr<Event>&) override {
    TORCH_INTERNAL_ASSERT(false);
  }
};

class DefaultBucketAllocator : public EventHandler {
 public:
  std::vector<EventSchema> ingressEvents() override {
    return {EventType::PREPARE_MODULE, EventType::BUCKET_CONTENT_READY};
  }

  std::vector<EventSchema> egressEvents() override {
    return {EventType::BUCKET_TENSOR_READY};
  }

  std::vector<std::shared_ptr<Future>> handleEvent(
      const c10::intrusive_ptr<Event>&) override {
    TORCH_INTERNAL_ASSERT(false);
  }
};

class AllReduceComm : public EventHandler {
 public:
  std::vector<EventSchema> ingressEvents() override {
    return {EventType::BUCKET_TENSOR_READY};
  }

  std::vector<EventSchema> egressEvents() override {
    return {EventType::COMM_DONE};
  }

  std::vector<std::shared_ptr<Future>> handleEvent(
      const c10::intrusive_ptr<Event>&) override {
    TORCH_INTERNAL_ASSERT(false);
  }
};


} // namespace spmd
} // namespace distributed
} // namespace torch
