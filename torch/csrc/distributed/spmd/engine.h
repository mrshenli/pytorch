#pragma once

#include <torch/csrc/distributed/spmd/event.h>
#include <torch/csrc/distributed/spmd/event_handler.h>

#include <unordered_map>
#include <unordered_set>


namespace torch {
namespace distributed {
namespace spmd {


// The event-based engine that maintains the event-handling graph and routes
// events to corresponding handlers properly.
//
// During construction, the Engine ctor takes a list of EventHandler instances.
// Each EventHandler specifies its ingress and egress events. Based on that
// information, the Engine builds a bipartite graph where the two sets of nodes
// are distinct Events and EventHandlers respectively. An edge pointing from an
// Event to an EventHandler means that the event is an ingress event for the
// handler, while an edge pointing from an EventHandler to an Event means that
// the event is an egress event from the handler. The figure below depicts an
// example snippet of a event-handling graph.
//
//      RootHandler    /-> DefaultTrigger  /-> DefaultBucketer   AllReduceComm
//           |  \     /       ^     \     /         ^     \           ^
//           |   \   /        |      \   /          |      \          |
//           |    \ /         |       \ /           |       \         |
//           |     X          |        X            |        \        |
//           V    / \         |       / \           |         \       |
//      PRE_FORWARD  \-> PREPARE_MODULE  \-> LOCAL_GRAD_READY  \-> BUCKET_READY
//
//
// The same Event instance can be consumed by multiple EventHandlers, e.g.,
// PREPARE_MODULE is consumed by both DefaultTrigger and DefaultBucketer.
// An EventHandler can consume and produce multiple types of Events, e.g.,
// RootHandler produces PREPARE_MODULE and PRE_FORWARD, and DefaultTrigger
// consumes PREPARE_MODULE and PRE_FORWARD.
//
// NB: In the current implementation, RootHandler produces type I events,
// namely, PREPARE_MODULE, PRE_FORWARD, and POST_FORWARD. All other Events and
// EventHandlers must be reacheable from type I Events. Otherwise, the graph
// is considered invalid, and Engine ctor will throw.
class TORCH_API Engine {
 public:

  explicit Engine(std::vector<std::shared_ptr<EventHandler>> handlers);
  void prepareModule(std::vector<at::Tensor> parameters);
  void preForward();

 private:

  struct Node {
    std::vector<std::shared_ptr<Node>> nextEdges_;
  };

  struct HandlerNode : Node {
    explicit HandlerNode(std::shared_ptr<EventHandler> handler)
        : handler_(std::move(handler)) {}
    const std::shared_ptr<EventHandler> handler_;
  };

  struct EventNode : Node {
    explicit EventNode(EventSchema schema)
        : schema_(std::move(schema)) {}
    const EventSchema schema_;
  };

  // NB: this function is thread-safe as it only reads eventNodes_. However, if
  // an EventHandler is not thread-safe, that EventHandler should use locks
  // accordingly.
  void processEvent(const c10::intrusive_ptr<Event>& event);

  std::unordered_map<EventSchema, std::shared_ptr<EventNode>, EventSchema::Hash> eventNodes_;
  std::vector<std::shared_ptr<EventHandler>> handlers_;
};


} // namespace spmd
} // namespace distributed
} // namespace torch
