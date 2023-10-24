#ifndef TRT_HEPLER_
#define TRT_HEPLER_

#include <sys/time.h>
#include <vector>
#include <string>
#include <string.h>
#include <memory>
#include <cassert>
#include <iostream>

#include "NvInfer.h"

#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#  define CUDA_CHECK(status)                                                      \
    if (status != cudaSuccess) {                                                  \
      std::cout << "Cuda failure! Error=" << cudaGetErrorString(status) << std::endl; \
    }
#endif

template <typename T>
using cuda_shared_ptr = std::shared_ptr<T>;

struct InferDeleter {
  template <typename T>
  void operator()(T *obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

template <typename T>
std::shared_ptr<T> makeShared(T *obj) {
  if (!obj) {
    throw std::runtime_error("Failed to create object");
  }
  return std::shared_ptr<T>(obj, InferDeleter());
}

template <typename T>
struct CudaDeleter {
  void operator()(T* buf) {
    if (buf) cudaFree(buf);
  }
};

template <typename T>
void make_cuda_shared(cuda_shared_ptr<T>& ptr, void* cudaMem) {
  ptr.reset(static_cast<T*>(cudaMem), CudaDeleter<T>());
}

struct TrtDestroyer {
  template <typename T>
  void operator()(T *obj) const {
    if (obj) obj->destroy();
  }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer>;

template <typename T>
inline TrtUniquePtr<T> MakeUnique(T *t) {
  return TrtUniquePtr<T>{t};
}

template <typename T>
inline std::shared_ptr<T> MakeShared(T *t) {
  return std::shared_ptr<T>(t, TrtDestroyer());
}

struct sample{
    std::string qid;
    std::string label;
    std::vector<int> shape_info_0;
    std::vector<int> i0;
    std::vector<int> shape_info_1;
    std::vector<int> i1;
    std::vector<int> shape_info_2;
    std::vector<int> i2;
    std::vector<int> shape_info_3;
    std::vector<float> i3;
    std::vector<int> shape_info_4;
    std::vector<int> i4;
    std::vector<int> shape_info_5;
    std::vector<int> i5;
    std::vector<int> shape_info_6;
    std::vector<int> i6;
    std::vector<int> shape_info_7;
    std::vector<int> i7;
    std::vector<int> shape_info_8;
    std::vector<int> i8;
    std::vector<int> shape_info_9;
    std::vector<int> i9;
    std::vector<int> shape_info_10;
    std::vector<int> i10;
    std::vector<int> shape_info_11;
    std::vector<int> i11;
    std::vector<float> out_data;
    uint64_t timestamp;
};


// BEGIN_LIB_NAMESPACE {

// Undef levels to support LOG(LEVEL)


/**
 * \brief Trt TrtLogger 日志类，全局对象
 */
class TrtLogger : public nvinfer1::ILogger {
  using Severity = nvinfer1::ILogger::Severity;

 public:
  explicit TrtLogger(Severity level = Severity::kINFO);

  ~TrtLogger() = default;

  nvinfer1::ILogger& getTRTLogger();

  void log(Severity severity, const char* msg) noexcept override;

 private:
  Severity level_;
};

class TrtHepler {
 public:
  TrtHepler(std::string model_param, int dev_id);

  int Forward(sample& s);

  ~TrtHepler();

 private:
  int _dev_id;
  // NS_PROTO::ModelParam *_model_param_ptr;
  std::string _model_param;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t cuda_stream_;

  // The all dims of all inputs.
  std::vector<nvinfer1::Dims> inputs_dims_;
  std::vector<void*> device_bindings_;
};

// } // BEGIN_LIB_NAMESPACE

#endif // TRT_HEPLER_

