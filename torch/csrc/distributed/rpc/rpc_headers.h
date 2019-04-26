#pragma once

#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/pickler.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>

namespace rpc {

using torch::jit::IValue;
using torch::jit::Operator;
using torch::jit::Pickler;
using torch::jit::Unpickler;
using torch::jit::Symbol;
using torch::jit::Stack;

}
