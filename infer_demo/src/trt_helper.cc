#include "trt_helper.h"

#include <string>
#include <fstream>
#include <sstream>

#include "NvInferPlugin.h"

using namespace std;

// BEGIN_LIB_NAMESPACE {

cuda_shared_ptr<void> CpuToDevice(const std::vector<int>& shape, int* data_ptr) {
  void *d_ptr;
  auto cpu_ptr = static_cast<void *>(data_ptr);
  int data_size = 1;
  for (int i = 0; i < shape.size(); i++) data_size *= shape[i];
  auto ret = cudaMalloc(&d_ptr, data_size * sizeof(int));
  //printf("int memory\n");
  if (ret) printf("memory error\n");
  ret = cudaMemcpy(d_ptr, cpu_ptr, data_size * sizeof(int), cudaMemcpyHostToDevice);
  if (ret) printf("memory error\n");
  cuda_shared_ptr<void> cuda_ptr;
  make_cuda_shared(cuda_ptr, d_ptr);
  return cuda_ptr;
}

cuda_shared_ptr<void> CpuToDevice(const std::vector<int>& shape, float* data_ptr) {
  void *d_ptr;
  auto cpu_ptr = static_cast<void *>(data_ptr);
  int data_size = 1;
  for (int i = 0; i < shape.size(); i++) data_size *= shape[i];
  auto ret = cudaMalloc(&d_ptr, data_size * sizeof(float));
  //printf("float memory\n");
  if (ret) printf("memory error\n");
  ret = cudaMemcpy(d_ptr, cpu_ptr, data_size * sizeof(float), cudaMemcpyHostToDevice);
  if (ret) printf("memory error\n");
  cuda_shared_ptr<void> cuda_ptr;
  make_cuda_shared(cuda_ptr, d_ptr);
  return cuda_ptr;
}

void DeviceToCpu(const std::vector<int>& shape, cuda_shared_ptr<void> cuda_ptr, float* data_ptr) {
  int data_size = 1;
  for (int i = 0; i < shape.size(); i++) data_size *= shape[i];
  if (data_size == 0) {
    std::cout << "data_size == 0" << std::endl;
    assert(0);
  }
  auto d_ptr = static_cast<void *>(data_ptr);
  auto ret = cudaMemcpy(d_ptr, cuda_ptr.get(), data_size * sizeof(float), cudaMemcpyDeviceToHost);
  printf("copy back\n");
  if (ret) printf("memory error\n");
}

TrtLogger::TrtLogger(nvinfer1::ILogger::Severity level) : level_(level) {}

nvinfer1::ILogger& TrtLogger::getTRTLogger() { return *this; }

// trt logger
void TrtLogger::log(Severity severity, const char* msg) noexcept {
  if (severity > level_) {
    return;
  }

  switch (severity) {
    case Severity::kINTERNAL_ERROR:
      std::cout << "[TRT] " << std::string(msg) << std::endl;
      break;
    case Severity::kERROR:
      std::cout << "[TRT] " << std::string(msg) << std::endl;
      break;
    case Severity::kWARNING:
      std::cout << "[TRT] " << std::string(msg) << std::endl;
      break;
    case Severity::kINFO:
      std::cout << "[TRT] " << std::string(msg) << std::endl;
      break;
    case Severity::kVERBOSE:
      std::cout << "[TRT] " << std::string(msg) << std::endl;
  }
}


TrtHepler::TrtHepler(std::string model_param, int dev_id)
    : _dev_id(dev_id), _model_param(model_param) {
  { // read model, deserializeCudaEngine and createExecutionContext
    std::ifstream t(_model_param);  // string pth
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string contents(buffer.str());

    CUDA_CHECK(cudaSetDevice(_dev_id));
    CUDA_CHECK(cudaStreamCreate(&cuda_stream_));

    TrtLogger trt_logger;
    initLibNvInferPlugins(&trt_logger.getTRTLogger(), "");
    auto runtime = MakeUnique(nvinfer1::createInferRuntime(trt_logger.getTRTLogger()));
    auto e = runtime->deserializeCudaEngine((void*)contents.c_str(),
                                            contents.size(), nullptr);
    engine_ = MakeShared(e);
    context_ = MakeShared(engine_->createExecutionContext());
    context_->setOptimizationProfile(0);
  }

}

int TrtHepler::Forward(sample& s) {
  cudaSetDevice(_dev_id);
  auto rc_ids_tensor = CpuToDevice(s.shape_info_0, s.i0.data());
  auto sent_ids_tensor = CpuToDevice(s.shape_info_1, s.i1.data());
  auto pos_ids_tensor = CpuToDevice(s.shape_info_2, s.i2.data());
  auto input_mask_tensor = CpuToDevice(s.shape_info_3, s.i3.data());
  auto tmp6_tensor = CpuToDevice(s.shape_info_4, s.i4.data());
  auto tmp7_tensor = CpuToDevice(s.shape_info_5, s.i5.data());
  auto tmp8_tensor = CpuToDevice(s.shape_info_6, s.i6.data());
  auto tmp9_tensor = CpuToDevice(s.shape_info_7, s.i7.data());
  auto tmp10_tensor = CpuToDevice(s.shape_info_8, s.i8.data());
  auto tmp11_tensor = CpuToDevice(s.shape_info_9, s.i9.data());
  auto tmp12_tensor = CpuToDevice(s.shape_info_10, s.i10.data());
  auto tmp13_tensor = CpuToDevice(s.shape_info_11, s.i11.data());

  void* out_ptr;
  auto ret_ = cudaMalloc(&out_ptr, s.shape_info_0[0] * sizeof(float));  // -1 * 1
  cuda_shared_ptr<void> cuda_out_ptr;
  make_cuda_shared(cuda_out_ptr, out_ptr);

  cudaEvent_t start, stop;
  float elapsed_time = 0.0;

  int binding_idx = 0;
  //std::vector<std::vector<int>> input_dims = {s.shape_info_0, s.shape_info_1, s.shape_info_2, s.shape_info_3,
                                              //s.shape_info_4, s.shape_info_5, s.shape_info_6, s.shape_info_7,
                                              //s.shape_info_8, s.shape_info_9, s.shape_info_10, s.shape_info_11};
  std::vector<std::vector<int>> input_dims = {s.shape_info_0, s.shape_info_1, s.shape_info_2, s.shape_info_3,
                                              s.shape_info_4, s.shape_info_5, s.shape_info_6, s.shape_info_7,
                                              s.shape_info_8, s.shape_info_9, s.shape_info_10, s.shape_info_11};
  // set device_bindings_ and setBindingDimensions
  for (size_t i = 0; i < input_dims.size(); i++) {
    std::vector<int> dims_vec = input_dims[i];
    nvinfer1::Dims trt_dims;
    trt_dims.nbDims = static_cast<int>(dims_vec.size());
    memcpy(trt_dims.d, dims_vec.data(), sizeof(int) * trt_dims.nbDims);
    context_->setBindingDimensions(binding_idx, trt_dims);
    binding_idx ++;
  }

  if (!context_->allInputDimensionsSpecified()) {
    //gLogFatal << "context_->allInputDimensionsSpecified() error";
    std::cout << ("context_->allInputDimensionsSpecified() error") << std::endl;
    assert(0);
  }

  // set the input dim

  void *device_bindings[13] = {rc_ids_tensor.get(), sent_ids_tensor.get(), pos_ids_tensor.get(),
                               input_mask_tensor.get(),
                               tmp6_tensor.get(), tmp7_tensor.get(),
                               tmp8_tensor.get(), tmp9_tensor.get(), tmp10_tensor.get(),
                               tmp11_tensor.get(), tmp12_tensor.get(), tmp13_tensor.get(),
                               cuda_out_ptr.get()};
  //printf("before enqueue\n");
  bool ret = context_->enqueueV2(device_bindings, cuda_stream_, nullptr);
  if (!ret) {
    std::cout << ("context_->enqueueV2 failed!") << std::endl;
    return -100;
  }

  cudaMemcpy(s.out_data.data(), cuda_out_ptr.get(), s.shape_info_0[0] * sizeof(float), cudaMemcpyDeviceToHost);
  cudaStreamSynchronize(cuda_stream_);
  struct timeval tv;
  gettimeofday(&tv, NULL);
  s.timestamp = tv.tv_sec * 1000000 + tv.tv_usec;

}

TrtHepler::~TrtHepler() {
  CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
}

// } // BEGIN_LIB_NAMESPACE

