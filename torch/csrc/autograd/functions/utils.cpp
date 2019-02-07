#include "torch/csrc/autograd/functions/utils.h"

#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

#include <sstream>
#include <vector>

namespace torch { namespace autograd {

variable_list wrap_outputs(const variable_list& inputs, tensor_list&& outputs,
                           function_constructor ctr) {
  variable_list result;
  result.reserve(outputs.size());
  if (!any_variable_requires_grad(inputs)) {
    for (auto& output : outputs) {
      if (output.defined()) {
        result.push_back(make_variable(output, /*requires_grad=*/false));
      } else {
        result.emplace_back();
      }
    }
  } else {
    auto grad_fn = ctr(collect_next_edges(inputs));
    for (auto& output : outputs) {
      if (output.defined()) {
        auto variable = autograd::make_variable(output, /*requires_grad=*/false);
        autograd::create_gradient_edge(variable, grad_fn);
        result.push_back(std::move(variable));
      } else {
        grad_fn->add_input_metadata(Function::undefined_input());
        result.emplace_back();
      }
    }
  }
  return result;
}

void check_input_variables(const char* name, const variable_list& inputs, int args, int required_args) {
  if (required_args == -1) {
    required_args = args;
  }
  if (inputs.size() != (size_t)args) {
    std::stringstream ss;
    ss << name << ": expected " << args << " arguments (got " << inputs.size();
    ss << ")";
    throw std::runtime_error(ss.str());
  }
  for (int i = 0; i < required_args; ++i) {
    if (!inputs[i].defined()) {
      std::stringstream ss;
      ss << name << ": expected Tensor at argument " << i << " (got None)";
      throw std::runtime_error(ss.str());
    }
  }
}


static void gatherFunctions(
    Function* func,
    std::vector<std::shared_ptr<Function>>& stack) {
  func->release_variables();




  for (auto& edge : func->next_edges()) {
    if (edge.function && edge.function->name().compare("BroadcastBackward") == 0) {
      std::cout << ">>>> use count is " << edge.function.use_count() << std::endl;
    }


    if (edge.function.use_count() == 1) {
      auto fn = dynamic_cast<PyFunction*>(edge.function.get());
      if ((fn && Py_REFCNT(fn->obj) == 1) || !fn) {
        std::cout << "---- gather " << edge.function -> name() << " by " << func->name() << ", " << edge.function.use_count() << ", " << edge.function << std::endl;
        stack.emplace_back(std::move(edge.function));
      }
    } else {
      edge.function.reset();
    }
  }
}

/*
  * Fix for #5534: prevent stack overflow on deletion of deep computation graph
  *
  * Sometimes one can end up with a very big computation graph of Functions
  * and Edges. Each std::shared_ptr<Function> contains a list of Edge, and
  * each Edge contains a std::shared_ptr<Function>. Deleting a
  * std::shared_ptr<Function> can trigger the recursive deletion of other
  * std::shared_ptr<Function>'s: this can stack overflow if the graph
  * is deep enough. Here is an example of such a graph:
  *
  * shared_ptr<Function> -> Edge -> shared_ptr<Function> -> Edge -> ... -> shared_ptr<Function>
  *
  * The solution here is to detect when we are decrementing away the last
  * reference to a Function, and when doing so to buffer up the Function's
  * that will be recursively decremented.  We can then decrement (and free)
  * the original Function without causing a recursive cascade, before
  * draining the buffer applying the same behavior.  This is, in effect,
  * converting recursion to a loop, using a heap buffer in place of the
  * recursive call stack.
  */
void deleteFunction(Function* function) {
  std::string name(function -> name());
  std::cout << "+++++ start " << name << "++++++\n";
  // To avoid stack overflow on large computational graphs,
  // we need to track reference decrementing and freeing
  // on the heap.
  function->release_variables();
  std::vector<std::shared_ptr<Function>> stack;
  gatherFunctions(function, stack);
  std::cout << "=== deleted function " << name << ", " << function << std::endl;

  delete function;

  while (!stack.empty()) {
    auto func = std::move(stack.back());
    stack.pop_back();
    gatherFunctions(func.get(), stack);
    // Reference count is decremented on the loop backedge.
  }

  std::cout << "+++++ end " << name << "++++++\n";

}

}} // namespace torch::autograd
