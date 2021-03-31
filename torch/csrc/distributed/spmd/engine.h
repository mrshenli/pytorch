#pragma once

#include <torch/csrc/distributed/spmd/event.h>
#include <torch/csrc/distributed/spmd/event_handler.h>

#include <unordered_map>
#include <unordered_set>


namespace torch {
namespace distributed {
namespace spmd {

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


class Engine {
 public:
  Engine(std::vector<std::shared_ptr<EventHandler>> handlers) {
    handlers_ = std::move(handlers);
    handlers_.emplace_back(std::make_shared<RootHandler>());
    buildBiGraph();
  }

  void prepareModule(std::vector<at::Tensor> parameters) {
    processEvent(
        c10::make_intrusive<PrepareModuleEvent>(std::move(parameters)));
  }

 private:

  // NB: this function is thread-safe as it only reads eventNodes_. However, if
  // an EventHandler is not thread-safe, that EventHandler should use locks
  // accordingly.
  void processEvent(const c10::intrusive_ptr<Event>& event) {
    auto iter = eventNodes_.find(event->schema());
    //TORCH_CHECK(iter != eventNodes_.end());
    std::cout << "got event " << event->schema().type_ << ", registered events? " << eventNodes_.size()
              << ", found it? " << (iter != eventNodes_.end()) << std::endl << std::flush;
    if (iter != eventNodes_.end()) {
      std::cout << "==== yep, found it\n" << std::flush;
      for (auto& node : iter->second->nextEdges_) {
        auto handlerNode = std::static_pointer_cast<HandlerNode>(node);
        for (auto& futureEvent: handlerNode->handler_->handleEvent(event)) {
          std::weak_ptr<Future> wp = futureEvent;
          futureEvent->addCallback([this, wp](){
            auto fut = wp.lock();
            processEvent(fut->value().toCustomClass<Event>());
          });
        }
      }
    }

  }

  void buildBiGraph() {
    // temporary helper data structure
    std::unordered_map<EventSchema, std::vector<std::shared_ptr<HandlerNode>>, EventSchema::Hash> ingressMap;

    std::vector<std::shared_ptr<HandlerNode>> handlerNodes;
    for (auto& handler : handlers_) {
      auto handlerNode = std::make_shared<HandlerNode>(handler);
      handlerNodes.push_back(handlerNode);
      for (auto& eventSchema: handler->ingressEvents()) {
        // TODO: check no duplicated events
        ingressMap[eventSchema].push_back(handlerNode);
      }
    }

    // RootHandler generates Type I events
    std::shared_ptr<EventHandler> rootHandler = std::make_shared<RootHandler>();
    handlerNodes.push_back(std::make_shared<HandlerNode>(rootHandler));

    // build graph
    for (auto& handlerNode : handlerNodes) {
      for (auto& eventSchema : handlerNode->handler_->egressEvents()) {
        auto iter = eventNodes_.find(eventSchema);
        std::shared_ptr<EventNode> eventNode;
        if (iter == eventNodes_.end()) {
          eventNode = std::make_shared<EventNode>(eventSchema);
          std::cout << "=== registering event " << eventSchema.type_ << std::endl << std::flush;
          eventNodes_.emplace(eventSchema, eventNode);
        }

        handlerNode->nextEdges_.push_back(eventNode);

        for (auto& nextHandlerNode : ingressMap[eventSchema]) {
          eventNode->nextEdges_.push_back(nextHandlerNode);
        }
      }
    }

    // verify graph: all EventNodes and HandlerNodes must be reacheable from
    // Type I events.
    auto bfs = [this](
        const std::shared_ptr<Node>& from,
        std::unordered_set<Node*>& seen) {
          std::vector<Node*> queue;
          queue.push_back(from.get());
          while (!queue.empty()) {
            auto node = queue.back();
            queue.pop_back();
            for (const auto& nextNode : node->nextEdges_) {
              const bool inserted = seen.insert(nextNode.get()).second;
              if (inserted) {
                queue.push_back(nextNode.get());
              }
            }
          }
        };
    std::unordered_set<Node*> seen;
    for (auto& eventSchema: rootHandler->egressEvents()) {
      bfs(eventNodes_[eventSchema], seen);
    }

    std::cout << "seen = " << seen.size() << ", eventNodes_.size = " << eventNodes_.size()
              << ", handlerNodes = " << handlerNodes.size() << std::endl << std::flush;
    //TORCH_CHECK(seen.size() == eventNodes_.size() + handlerNodes.size());

  }

  std::unordered_map<EventSchema, std::shared_ptr<EventNode>, EventSchema::Hash> eventNodes_;
  std::vector<std::shared_ptr<EventHandler>> handlers_;
};


} // namespace spmd
} // namespace distributed
} // namespace torch
