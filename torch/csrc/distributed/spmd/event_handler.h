#pragma once

#include <torch/csrc/distributed/spmd/event.h>

#include <vector>

#include <ATen/core/ivalue.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/utils/lambda_post_hook.h>
#include <torch/csrc/utils/memory.h>



namespace torch {
namespace distributed {
namespace spmd {

using c10::ivalue::Future;
using c10::IValue;

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
      const c10::intrusive_ptr<Event>& /* unused */) override {
    TORCH_INTERNAL_ASSERT(false);
  }
};


class DefaultTrigger : public EventHandler {
 public:
  std::vector<EventSchema> ingressEvents() override {
    return {EventType::PREPARE_MODULE};
  }

  std::vector<EventSchema> egressEvents() override {
    return {EventType::LOCAL_GRAD_READY};
  }

  std::vector<std::shared_ptr<Future>> handleEvent(
      const c10::intrusive_ptr<Event>& event) override {
    switch (event->schema().type_) {
      case EventType::PREPARE_MODULE: {
        return handlePrepareModule(
            c10::static_intrusive_pointer_cast<PrepareModuleEvent>(event));
      }
      default:
        TORCH_INTERNAL_ASSERT(false, "unexcepted event type");
    }
  }

 private:
  std::vector<std::shared_ptr<Future>> handlePrepareModule(
      c10::intrusive_ptr<PrepareModuleEvent> event) {
    std::cout << "PREPARE_MODULE: " << event->parameters().size()
              << ", inserting hooks!" << std::endl << std::flush;

    params_ = event->parameters();
    std::vector<std::shared_ptr<Future>> futures;
    futures.reserve(params_.size());
    for (size_t index = 0; index < params_.size(); ++index) {
      auto& param = params_[index];
      futures.emplace_back(std::make_shared<Future>(at::AnyClassType::get()));

      auto gradAccumulator =
          torch::autograd::impl::grad_accumulator(param);
      // Hook to execute after the gradient accumulator has executed.
      gradAccumulator->add_post_hook(
          torch::make_unique<torch::autograd::utils::LambdaPostHook>(
              [this, index, localGradReadyFuture=futures.back()](
                  const torch::autograd::variable_list& outputs,
                  const torch::autograd::variable_list& /* unused */) {
                auto lgr = c10::make_intrusive<LocalGradReadyEvent>(
                    index, params_[index].mutable_grad());

                localGradReadyFuture->markCompleted(
                    IValue(c10::static_intrusive_pointer_cast<Event>(lgr)));
                return outputs;
              }));
      gradAccumulators_.push_back(std::move(gradAccumulator));
    }
    return futures;
  }

  // keep grad accumulators alive
  std::vector<std::shared_ptr<torch::autograd::Node>> gradAccumulators_;
  std::vector<at::Tensor> params_;
};

class DefaultBucketer : public EventHandler {
 public:
  // FIXME: we might need more advanced ingress/egress event specifications.
  // E.g., LOCAL_GRAD_READY -> BUCKET_READY; COMM_DONE -> GLOBAL_GRAD_READY,
  // otherwise, DefaultBucketer and AllReduceComm can form a cycle.
  std::vector<EventSchema> ingressEvents() override {
    // FIXME: consume PREPARE_MODULE to allocate buckets
    return {EventType::LOCAL_GRAD_READY, EventType::COMM_DONE};
  }

  std::vector<EventSchema> egressEvents() override {
    return {EventType::BUCKET_READY, EventType::GLOBAL_GRAD_READY};
  }

  std::vector<std::shared_ptr<Future>> handleEvent(
      const c10::intrusive_ptr<Event>& event) override {
    switch (event->schema().type_) {
      case EventType::LOCAL_GRAD_READY: {
        std::cout << "=== got LOCAL_GRAD_READY event " << std::endl << std::flush;
        return handleLocalGradReady(
            c10::static_intrusive_pointer_cast<LocalGradReadyEvent>(event));
      }
      default:
        TORCH_INTERNAL_ASSERT(false, "unexcepted event type");
    }
  }

 private:

  std::vector<std::shared_ptr<Future>> handleLocalGradReady(
      c10::intrusive_ptr<LocalGradReadyEvent> event) {
    auto future = std::make_shared<Future>(at::AnyClassType::get());
    auto br = c10::make_intrusive<BucketReadyEvent>(event->index(), event->grad());
    future->markCompleted(IValue(c10::static_intrusive_pointer_cast<Event>(br)));
    std::vector<std::shared_ptr<Future>> futures;
    futures.reserve(1);
    futures.emplace_back(std::move(future));
    std::cout << "=== bucket ready event for " << br->index() << std::endl << std::flush;
    return futures;
  }
};

class AllReduceComm : public EventHandler {
 public:
  std::vector<EventSchema> ingressEvents() override {
    return {EventType::BUCKET_READY};
  }

  std::vector<EventSchema> egressEvents() override {
    return {EventType::COMM_DONE};
  }

  std::vector<std::shared_ptr<Future>> handleEvent(
      const c10::intrusive_ptr<Event>& event) override {
    switch (event->schema().type_) {
      case EventType::BUCKET_READY: {
        std::cout << "=== got BUCKET_READY event " << std::endl << std::flush;
        return handleBucketReady(
            c10::static_intrusive_pointer_cast<BucketReadyEvent>(event));
      }
      default:
        TORCH_INTERNAL_ASSERT(false, "unexcepted event type");
    }
  }

 private:
  std::vector<std::shared_ptr<Future>> handleBucketReady(
      c10::intrusive_ptr<BucketReadyEvent> event) {
    return {};
  }
};


} // namespace spmd
} // namespace distributed
} // namespace torch