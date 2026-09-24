#include <pybind11/pybind11.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
namespace py = pybind11;

static void check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}

// Python owns every allocation and graph. Calls are serialized by its engine lock.
class FastSession {
 public:
  FastSession(uintptr_t host, uintptr_t device, size_t bytes,
          int batch, int sequence, int options, int64_t pad,
          uintptr_t graph, uintptr_t stream, uintptr_t logits, uintptr_t actions,
          uintptr_t output, int action_width, bool captured_io)
      : h_(reinterpret_cast<uint8_t*>(host)), d_(reinterpret_cast<uint8_t*>(device)), bytes_(bytes),
        b_(batch), s_(sequence), k_(options), pad_(pad),
        graph_(reinterpret_cast<cudaGraphExec_t>(graph)), stream_(reinterpret_cast<cudaStream_t>(stream)),
        logits_(reinterpret_cast<float*>(logits)), actions_(reinterpret_cast<float*>(actions)),
        output_(reinterpret_cast<float*>(output)), a_(action_width), captured_io_(captured_io) {
    if (!h_ || b_ < 1 || s_ < 1 || k_ < 1) throw std::invalid_argument("Invalid packed buffer");
    ids_ = reinterpret_cast<int64_t*>(h_);
    mask_ = ids_ + b_ * s_;
    positions_ = mask_ + b_ * s_;
    types_ = positions_ + b_ * k_;
    marker_mask_ = reinterpret_cast<bool*>(types_ + b_);
    size_t needed = (2ULL*b_*s_ + b_*k_ + b_)*sizeof(int64_t) + b_*k_*sizeof(bool);
    if (needed > bytes_) throw std::invalid_argument("Packed buffer too small");
  }

  void fill(py::list items) {
    if (py::len(items) > static_cast<size_t>(b_)) throw std::invalid_argument("Batch exceeds buffer");
    // Clear previous request bytes, including padded dummy rows and marker slots.
    std::fill_n(ids_, b_*s_, pad_);
    std::memset(mask_, 0, bytes_ - (reinterpret_cast<uint8_t*>(mask_) - h_));
    for (int i=0; i<b_; ++i) { mask_[i*s_]=1; marker_mask_[i*k_]=true; }
    int row=0;
    for (py::handle item_handle : items) {
      py::dict item = py::reinterpret_borrow<py::dict>(item_handle);
      py::list ids = item["ids"].cast<py::list>();
      py::list positions = item["markers"].cast<py::list>();
      int n=py::len(ids), k=py::len(positions);
      if (n>s_ || k>k_) throw std::invalid_argument("Request exceeds packed shape");
      for (int j=0;j<n;++j) {
        ids_[row*s_+j]=PyLong_AsLongLong(PyList_GET_ITEM(ids.ptr(),j));
        if (PyErr_Occurred()) throw py::error_already_set();
      }
      std::fill_n(mask_+row*s_,n,1);
      for (int j=0;j<k;++j) {
        positions_[row*k_+j]=PyLong_AsLongLong(PyList_GET_ITEM(positions.ptr(),j));
        if (PyErr_Occurred()) throw py::error_already_set();
      }
      std::fill_n(marker_mask_+row*k_,k,true);
      types_[row]=item["qtype"].cast<int64_t>();
      ++row;
    }
  }

  void replay() {
    py::gil_scoped_release release;
    if (!captured_io_) check(cudaMemcpyAsync(d_,h_,bytes_,cudaMemcpyHostToDevice,stream_));
    check(cudaGraphLaunch(graph_,stream_));
    if (!captured_io_) {
      check(cudaMemcpyAsync(output_,logits_,b_*k_*sizeof(float),cudaMemcpyDeviceToHost,stream_));
      check(cudaMemcpyAsync(output_+b_*k_,actions_,b_*a_*sizeof(float),cudaMemcpyDeviceToHost,stream_));
    }
    check(cudaStreamSynchronize(stream_));
  }
  void run(py::list items) { fill(items); replay(); }
 private:
  uint8_t *h_, *d_; size_t bytes_; int b_,s_,k_; int64_t pad_;
  int64_t *ids_,*mask_,*positions_,*types_; bool *marker_mask_;
  cudaGraphExec_t graph_; cudaStream_t stream_;
  float *logits_,*actions_,*output_; int a_; bool captured_io_;
};

PYBIND11_MODULE(laya_fast_host, m) {
 py::class_<FastSession>(m,"Session")
  .def(py::init<uintptr_t,uintptr_t,size_t,int,int,int,int64_t,uintptr_t,uintptr_t,uintptr_t,uintptr_t,uintptr_t,int,bool>())
  .def("fill",&FastSession::fill).def("replay",&FastSession::replay).def("run",&FastSession::run);
}
