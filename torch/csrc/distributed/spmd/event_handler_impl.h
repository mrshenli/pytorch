#pragma once

#include <torch/csrc/distributed/spmd/event_handler.h>


namespace torch {
namespace distributed {
namespace spmd {

class TORCH_API RootHandler : public EventHandler {
 public:
  using EventHandler::EventHandler;
  std::vector<EventSchema> ingressEvents() override;
  std::vector<EventSchema> egressEvents() override;
  std::vector<std::shared_ptr<Future>> handleEvent(
      const c10::intrusive_ptr<Event>& /* unused */) override;
};


class TORCH_API DefaultTrigger : public EventHandler {
 public:
  using EventHandler::EventHandler;
  std::vector<EventSchema> ingressEvents() override;
  std::vector<EventSchema> egressEvents() override;
  std::vector<std::shared_ptr<Future>> handleEvent(
      const c10::intrusive_ptr<Event>& event) override;

 private:
  std::vector<std::shared_ptr<Future>> handlePrepareModule(
      c10::intrusive_ptr<PrepareModuleEvent> event);

  void autogradHook(
      size_t index,
      const std::shared_ptr<Future>& localGradReadyFuture);

  // keep grad accumulators alive
  std::vector<std::shared_ptr<torch::autograd::Node>> gradAccumulators_;
  std::vector<at::Tensor> params_;
};

class TORCH_API DefaultBucketer : public EventHandler {
 public:
  using EventHandler::EventHandler;
  // FIXME: we might need more advanced ingress/egress event specifications.
  // E.g., LOCAL_GRAD_READY -> BUCKET_READY; COMM_DONE -> GLOBAL_GRAD_READY,
  // otherwise, DefaultBucketer and AllReduceComm can form a cycle.
  std::vector<EventSchema> ingressEvents() override;
  std::vector<EventSchema> egressEvents() override;

  std::vector<std::shared_ptr<Future>> handleEvent(
      const c10::intrusive_ptr<Event>& event) override;

 private:

  std::vector<std::shared_ptr<Future>> handlePrepareModule(
      c10::intrusive_ptr<PrepareModuleEvent> event);

  std::vector<std::shared_ptr<Future>> handleLocalGradReady(
      c10::intrusive_ptr<LocalGradReadyEvent> event);

  std::vector<std::shared_ptr<Future>> handleCommDone(
      c10::intrusive_ptr<CommDoneEvent> event);

  std::vector<at::Tensor> params_;
};

class TORCH_API AllReduceComm : public EventHandler {
 public:
  using EventHandler::EventHandler;
  AllReduceComm(c10::intrusive_ptr<c10d::ProcessGroup> pg)
      : pg_(std::move(pg)) {}

  std::vector<EventSchema> ingressEvents() override;
  std::vector<EventSchema> egressEvents() override;
  std::vector<std::shared_ptr<Future>> handleEvent(
      const c10::intrusive_ptr<Event>& event) override;

 private:
  std::vector<std::shared_ptr<Future>> handleBucketReady(
      c10::intrusive_ptr<BucketReadyEvent> event);

  const c10::intrusive_ptr<c10d::ProcessGroup> pg_;
};


} // namespace spmd
} // namespace distributed
} // namespace torch
