#include <torch/csrc/distributed/spmd/event.h>

#include <torch/types.h>

namespace torch {
namespace distributed {
namespace spmd {

namespace {

// NB: need to call torch::class_ to register Message in the map returned by
// c10::getCustomClassTypeMap(). Otherwise, Message cannot be wrapped within
// an IValue.
// NB: add this line here instead of in rpc/init.cpp because 1) we have C++
// only tests that won't run rpc/init.cpp; 2) Message is not meant to be
// visible from Python.

static const auto event = torch::class_<Event>("spmd", "_Event");
static const auto prepareModuleEvent =
    torch::class_<PrepareModuleEvent>("spmd", "_PrepareModuleEvent");
static const auto localgradReadyEvent =
    torch::class_<LocalGradReadyEvent>("spmd", "_LocalGradReadyEvent");
static const auto bucketReadyEvent =
    torch::class_<BucketReadyEvent>("spmd", "_BucketReadyEvent");
static const auto commDoneEvent =
    torch::class_<CommDoneEvent>("spmd", "_CommDoneEvent");

} // namespace

} // namespace spmd
} // namespace distributed
} // namespace torch
