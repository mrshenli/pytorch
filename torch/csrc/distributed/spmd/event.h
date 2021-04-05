#pragma once

#include <ATen/core/ivalue_inl.h>
#include <c10/util/Exception.h>
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

// TODO: add the following to ddp/init.cpp
// static const auto message = torch::class_<Event>("rpc", "_Event");


class PrepareModuleEvent : public Event {
 public:
  PrepareModuleEvent(std::vector<at::Tensor> params)
      : Event(EventSchema(EventType::PREPARE_MODULE)),
        params_(std::move(params)) {

  }

  const std::vector<at::Tensor>& parameters() const {
    return params_;
  }

 private:
  std::vector<at::Tensor> params_;
};

class LocalGradReadyEvent : public Event {
 public:
  LocalGradReadyEvent(size_t index, const at::Tensor& grad)
      : Event(EventSchema(EventType::LOCAL_GRAD_READY)),
        index_(index),
        grad_(grad) {}

  const at::Tensor& grad() {
    return grad_;
  }

  size_t index() const {
    return index_;
  }

 private:
  const size_t index_;
  const at::Tensor& grad_;
};

struct Bucket final {
  Bucket(
      size_t index,
      const at::Tensor tensor,
      std::vector<size_t> paramIndices)
      : index_(index),
        tensor_(std::move(tensor)),
        paramIndices_(paramIndices) {}

  const size_t index_;
  const at::Tensor tensor_;
  const std::vector<size_t> paramIndices_;
};

class BucketReadyEvent : public Event {
 public:
  BucketReadyEvent(std::shared_ptr<Bucket> bucket)
      : Event(EventSchema(EventType::BUCKET_READY)),
        bucket_(std::move(bucket)) {}

  const  std::shared_ptr<Bucket>& bucket() const {
    return bucket_;
  }

 private:
  const std::shared_ptr<Bucket> bucket_;
};

class CommDoneEvent : public Event {
 public:
  CommDoneEvent(std::shared_ptr<Bucket> bucket)
      : Event(EventSchema(EventType::COMM_DONE)),
        bucket_(std::move(bucket)) {}

  const  std::shared_ptr<Bucket>& bucket() const {
    return bucket_;
  }

 private:
  const std::shared_ptr<Bucket> bucket_;
};

class GlobalGradReadyEvent : public Event {
 public:
  GlobalGradReadyEvent(size_t index)
      : Event(EventSchema(EventType::LOCAL_GRAD_READY)),
        index_(index) {}

  size_t index() const {
    return index_;
  }

 private:
  const size_t index_;
};

} // spmd
} // distributed
} // torch
