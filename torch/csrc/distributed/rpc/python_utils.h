#pragma once

#include <torch/csrc/utils/future.h>
#include <torch/csrc/utils/pybind.h>


namespace torch {
namespace distributed {
namespace rpc {

struct PyObj {
 public:
  PyObj() = default;
  explicit PyObj(py::object obj) : obj_(std::move(obj)) {}

  PyObj(const PyObj& other) = default;

  PyObj(PyObj&& other) noexcept = default;

  ~PyObj() {
    pybind11::gil_scoped_acquire ag;
    obj_ = py::none();
  }

  PyObj& operator=(const PyObj& rhs) {
    {
      pybind11::gil_scoped_acquire ag;
      obj_ = rhs.obj_;
    }
    return *this;
  }

  PyObj& operator=(PyObj&& rhs) {
    {
      pybind11::gil_scoped_acquire ag;
      obj_ = std::move(rhs.obj_);
    }
    return *this;
  }

  py::object obj_;
};

using FuturePyObj = torch::utils::Future<PyObj>;

} // namespace rpc
} // namespace distributed
} // namespace torch