#pragma once

#include <c10/util/Exception.h>
#include <torch/csrc/distributed/ddp/event_handler.h>

#include <memory>
#include <unordered_map>
#include <unordered_set>


namespace torch {
namespace distributed {
namespace ddp {

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
  Engine(std::vector<std::shared_ptr<EventHandler>> handlers)
      : handlers_(std::move(handlers)) {

  }



 private:
  void buildBiGraph() {
    // temporary helper data structure
    std::unordered_map<EventSchema, std::vector<std::shared_ptr<HandlerNode>>> ingressMap;

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
    auto rootHandler = std::make_shared<HandlerNode>(new RootHandler());
    handlerNodes.push_back(rootHandler);

    // build graph
    for (auto& handlerNode : handlerNodes) {
      for (auto& eventSchema : handlerNode->handler_->egressEvents()) {
        auto iter = eventNodes_.find(eventSchema);
        std::shared_ptr<EventNode> eventNode;
        if (iter == eventNodes_.end()) {
          eventNode = std::make_shared<EventNode>(eventSchema);
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

    TORCH_CHECK(seen.size() == eventNodes_.size() + handlerNodes.size());

  }

  std::unordered_map<EventSchema, std::shared_ptr<EventNode>> eventNodes_;
  const std::vector<std::shared_ptr<EventHandler>> handlers_;
};


} // namespace ddp
} // namespace distributed
} // namespace torch