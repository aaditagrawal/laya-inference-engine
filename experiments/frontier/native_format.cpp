// Native orchestration of the installed NumPy loops; no replacement exp/log math.
// Protocol adapted from the Apache-2.0 Laya SDK; see repository NOTICE.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/dtype_api.h>
#include <numpy/ufuncobject.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
namespace py = pybind11;

// Documented by ufunc._get_strided_loop in the pinned NumPy 2.5.3 installation.
struct CallInfo {
  PyArrayMethod_StridedLoop *loop;
  PyArrayMethod_Context *context;
  NpyAuxData *auxdata;
  npy_bool requires_pyapi;
  npy_bool no_floatingpoint_errors;
};

class Loop {
  py::object capsule_;
  CallInfo *info_;
 public:
  Loop(py::object ufunc, int arity) {
    py::object dtype = py::module_::import("numpy").attr("dtype")("float32");
    py::tuple dtypes(arity + 1), strides(arity + 1);
    for (int i = 0; i <= arity; ++i) { dtypes[i] = dtype; strides[i] = py::none(); }
    capsule_ = ufunc.attr("_resolve_dtypes_and_context")(dtypes)[py::int_(1)];
    ufunc.attr("_get_strided_loop")(capsule_, py::arg("fixed_strides") = strides);
    info_ = reinterpret_cast<CallInfo*>(
      PyCapsule_GetPointer(capsule_.ptr(), "numpy_1.24_ufunc_call_info"));
    if (!info_) throw py::error_already_set();
    if (!info_->loop) throw std::runtime_error("Missing NumPy inner loop");
  }
  void unary(const float *x, float *y, npy_intp n) const {
    char *args[] = {reinterpret_cast<char*>(const_cast<float*>(x)), reinterpret_cast<char*>(y)};
    const npy_intp strides[] = {4, 4};
    if (info_->loop(info_->context, args, &n, strides, info_->auxdata) < 0)
      throw py::error_already_set();
  }
  void binary(const float *x, const float *y, float *z, npy_intp n, npy_intp ystride = 4) const {
    char *args[] = {reinterpret_cast<char*>(const_cast<float*>(x)),
                   reinterpret_cast<char*>(const_cast<float*>(y)), reinterpret_cast<char*>(z)};
    const npy_intp strides[] = {4, ystride, 4};
    if (info_->loop(info_->context, args, &n, strides, info_->auxdata) < 0)
      throw py::error_already_set();
  }
};

static py::object view(void *data, npy_intp n, int typenum) {
  PyObject *array = PyArray_SimpleNewFromData(1, &n, typenum, data);
  if (!array) throw py::error_already_set();
  return py::reinterpret_steal<py::object>(array);
}
static double sum(void *data, npy_intp n, int typenum) {
  py::object array = view(data, n, typenum);
  PyObject *result = PyArray_Sum(reinterpret_cast<PyArrayObject*>(array.ptr()), 0, typenum, nullptr);
  if (!result) throw py::error_already_set();
  py::object owner = py::reinterpret_steal<py::object>(result);
  double value = PyFloat_AsDouble(result);
  if (PyErr_Occurred()) throw py::error_already_set();
  return value;
}
static bool ordinary_array(PyObject *obj) {
  if (!PyArray_CheckExact(obj)) return false;
  auto *a = reinterpret_cast<PyArrayObject*>(obj);
  return PyArray_TYPE(a) == NPY_FLOAT32 && PyArray_ISCARRAY_RO(a) && PyArray_ISNOTSWAPPED(a)
         && PyArray_NDIM(a) == 2;
}
static bool safe_values(const float *x, npy_intp n) {
  for (npy_intp i = 0; i < n; ++i) {
    float a = std::abs(x[i]);
    if (!std::isfinite(a) || a > 10000.f || (a != 0.f && a < 1e-20f)) return false;
  }
  return true;
}
static void check_fpe(const char *name) {
  int status = PyUFunc_getfperr();
  if (status && PyUFunc_GiveFloatingpointErrors(name, status) < 0)
    throw py::error_already_set();
}

class Formatter {
  py::object np_, clamp_, round_, four_;
  Loop exp_, log_, divide_, subtract_, multiply_;
  std::vector<float> log_k_;
 public:
  Formatter()
      : np_(py::module_::import("numpy")),
        clamp_(py::module_::import("laya.common").attr("clamp_temperature")),
        round_(py::module_::import("builtins").attr("round")), four_(py::int_(4)),
        exp_(np_.attr("exp"), 1), log_(np_.attr("log"), 1),
        divide_(np_.attr("divide"), 2), subtract_(np_.attr("subtract"), 2),
        multiply_(np_.attr("multiply"), 2), log_k_(257) {
    if (py::str(np_.attr("__version__")).cast<std::string>() != "2.5.3")
      throw std::runtime_error("Native formatting requires NumPy 2.5.3");
    py::object log = py::module_::import("math").attr("log");
    for (int i = 2; i <= 256; ++i) log_k_[i] = static_cast<float>(log(i).cast<double>());
  }
  py::object rounded(double value) const { return round_(py::float_(value), four_); }

  py::object call(py::object prepared, py::object logits, py::object action_logits,
                  py::object temperature, py::object buckets) {
    if (!ordinary_array(logits.ptr()) || !ordinary_array(action_logits.ptr()) ||
        !PyList_CheckExact(temperature.ptr()) || !PyDict_CheckExact(buckets.ptr())) return py::none();
    py::object fields = prepared.attr("__dict__");
    for (const char *name : {"ids", "questions", "items", "input_tokens"})
      if (!PyDict_GetItemString(fields.ptr(), name)) return py::none();
    py::object ids = prepared.attr("ids"), questions = prepared.attr("questions"), items = prepared.attr("items");
    if (!PyList_CheckExact(ids.ptr()) || !PyList_CheckExact(questions.ptr()) || !PyList_CheckExact(items.ptr())) return py::none();
    npy_intp n = PyList_GET_SIZE(ids.ptr());
    if (n < 1 || n > 64 || PyList_GET_SIZE(questions.ptr()) != n || PyList_GET_SIZE(items.ptr()) != n) return py::none();
    auto *l = reinterpret_cast<PyArrayObject*>(logits.ptr());
    auto *a = reinterpret_cast<PyArrayObject*>(action_logits.ptr());
    npy_intp width = PyArray_DIM(l, 1), awidth = PyArray_DIM(a, 1);
    if (PyArray_DIM(l, 0) < n || PyArray_DIM(a, 0) < n || width < 1 || width > 256 || awidth < 1 || awidth > 256) return py::none();
    const float *lp = static_cast<const float*>(PyArray_DATA(l));
    const float *ap = static_cast<const float*>(PyArray_DATA(a));
    if (!safe_values(lp, n * width) || !safe_values(ap, n * awidth)) return py::none();
    std::vector<npy_intp> counts(n);
    std::vector<int> types(n);
    for (npy_intp row = 0; row < n; ++row) {
      PyObject *q = PyList_GET_ITEM(questions.ptr(), row), *item = PyList_GET_ITEM(items.ptr(), row);
      if (!PyDict_CheckExact(q) || !PyDict_CheckExact(item)) return py::none();
      PyObject *kind = PyDict_GetItemString(q, "t"), *crit = PyDict_GetItemString(q, "crit");
      PyObject *markers = PyDict_GetItemString(item, "markers");
      if (!kind || !crit || !markers || !PyUnicode_CheckExact(kind) || !PyUnicode_IS_ASCII(kind) || !PyList_CheckExact(markers)) return py::none();
      const char *text = PyUnicode_AsUTF8(kind);
      if (!text) throw py::error_already_set();
      int qt = std::strcmp(text, "choice") == 0 ? 0 : std::strcmp(text, "score") == 0 ? 1 : std::strcmp(text, "noul") == 0 ? 2 : -1;
      npy_intp k = PyList_GET_SIZE(markers);
      if (qt < 0 || k < 1 || k > width || (qt == 0 && (!PyDict_CheckExact(crit) || PyDict_Size(crit) != k)) ||
          (qt == 1 && (!PyList_CheckExact(crit) || PyList_GET_SIZE(crit) != k)) || (qt == 2 && k != 2)) return py::none();
      // Preflight every fallback before exp can emit warnings or call seterrcall.
      // Temperature is clamped to at least 0.5. A raw span <=39 keeps scaled
      // logits below 80 even after FP32 division rounding at the allowed magnitude.
      const float *values = lp + row * width;
      if (*std::max_element(values, values + k) - *std::min_element(values, values + k) > 39.f)
        return py::none();
      counts[row] = k; types[row] = qt;
    }
    std::vector<float> act(n * awidth);
    for (npy_intp row = 0; row < n; ++row) {
      const float *input = ap + row * awidth;
      float maximum = *std::max_element(input, input + awidth);
      subtract_.binary(input, &maximum, act.data() + row * awidth, awidth, 0);
    }
    PyUFunc_clearfperr();
    exp_.unary(act.data(), act.data(), n * awidth);
    check_fpe("exp");
    std::vector<float> denominators(n);
    for (npy_intp row = 0; row < n; ++row) {
      float *values = act.data() + row * awidth;
      denominators[row] = static_cast<float>(sum(values, awidth, NPY_FLOAT32));
    }
    PyUFunc_clearfperr();
    for (npy_intp row = 0; row < n; ++row) {
      float *values = act.data() + row * awidth;
      divide_.binary(values, &denominators[row], values, awidth, 0);
    }
    check_fpe("divide");
    py::dict answers, usage;
    usage["input_tokens"] = prepared.attr("input_tokens"); usage["output_tokens"] = 0;
    for (npy_intp row = 0; row < n; ++row) {
      py::dict q = py::reinterpret_borrow<py::dict>(PyList_GET_ITEM(questions.ptr(), row));
      int qt = types[row]; npy_intp k = counts[row];
      py::str kind(q["t"]);
      const char *size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
      py::str bucket(kind.cast<std::string>() + ":" + size);
      py::object default_temperature = temperature[py::int_(qt)];
      py::object chosen = buckets.attr("get")(bucket, default_temperature);
      float scale = static_cast<float>(clamp_(chosen).cast<double>());
      float p[256], z[256], clipped[256], logarithms[256], entropy_terms[256];
      divide_.binary(lp + row * width, &scale, z, k, 0);
      float maximum = *std::max_element(z, z + k);
      subtract_.binary(z, &maximum, z, k, 0);
      exp_.unary(z, p, k);
      float denominator = static_cast<float>(sum(p, k, NPY_FLOAT32));
      divide_.binary(p, &denominator, p, k, 0);
      float confidence = 1.f;
      if (k > 1) {
        for (npy_intp j = 0; j < k; ++j) clipped[j] = std::clamp(p[j], 1e-12f, 1.f);
        log_.unary(clipped, logarithms, k);
        multiply_.binary(p, logarithms, entropy_terms, k);
        float entropy = -static_cast<float>(sum(entropy_terms, k, NPY_FLOAT32));
        float normalized = entropy / log_k_[k];
        confidence = std::clamp(1.f - normalized, 0.f, 1.f);
      }
      py::dict action, answer;
      action["act_probability"] = rounded(act[row * awidth]);
      answer["type"] = kind;
      if (qt == 0) {
        py::dict crit(q["crit"]), probabilities;
        py::list keys = crit.attr("keys")();
        npy_intp winner = std::max_element(p, p + k) - p;
        answer["choice"] = keys[winner];
        for (npy_intp j = 0; j < k; ++j) probabilities[keys[j]] = rounded(p[j]);
        answer["probabilities"] = probabilities;
        answer["confidence"] = rounded(confidence);
      } else if (qt == 1) {
        py::list crit(q["crit"]); py::dict probabilities, legend;
        double weighted[256];
        for (npy_intp j = 0; j < k; ++j) {
          py::str key(std::to_string(j));
          weighted[j] = static_cast<double>(j) * static_cast<double>(p[j]);
          probabilities[key] = rounded(p[j]); legend[key] = crit[j];
        }
        answer["score"] = rounded(sum(weighted, k, NPY_FLOAT64));
        answer["legend"] = legend; answer["probabilities"] = probabilities;
        answer["confidence"] = rounded(confidence);
      } else {
        double probability = static_cast<double>(p[1]);
        answer["noul"] = rounded(probability);
        answer["confidence"] = rounded(std::max(probability, 1. - probability));
      }
      answer["action"] = action;
      answers[py::reinterpret_borrow<py::object>(PyList_GET_ITEM(ids.ptr(), row))] = answer;
    }
    py::dict result;
    result["model"] = "laya-rl-agent"; result["answers"] = answers; result["usage"] = usage;
    return result;
  }
  py::object inspect_math(py::object values, bool use_log, bool libm) {
    auto *input = reinterpret_cast<PyArrayObject*>(values.ptr());
    if (!PyArray_CheckExact(values.ptr()) || PyArray_TYPE(input) != NPY_FLOAT32 || !PyArray_ISCARRAY_RO(input))
      throw std::invalid_argument("Expected a contiguous float32 array");
    npy_intp n = PyArray_SIZE(input);
    auto result = py::reinterpret_steal<py::object>(PyArray_SimpleNew(1, &n, NPY_FLOAT32));
    float *out = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(result.ptr())));
    const float *x = static_cast<const float*>(PyArray_DATA(input));
    if (libm) for (npy_intp i = 0; i < n; ++i) out[i] = use_log ? std::log(x[i]) : std::exp(x[i]);
    else if (use_log) log_.unary(x, out, n);
    else exp_.unary(x, out, n);
    return result;
  }
};

PYBIND11_MODULE(laya_native_format, m) {
  if (_import_array() < 0) throw py::error_already_set();
  if (_import_umath() < 0) throw py::error_already_set();
  py::class_<Formatter>(m, "Formatter")
    .def(py::init<>()).def("__call__", &Formatter::call)
    .def("inspect_math", &Formatter::inspect_math);
}
