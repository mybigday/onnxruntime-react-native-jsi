#include "global.h"

namespace onnxruntimereactnativejsi {

static bool ortInitialized = false;
static std::shared_ptr<Ort::Env> globalEnv = nullptr;
static std::shared_ptr<Object> globalTensorConstructor = nullptr;
static std::shared_ptr<Ort::RunOptions> globalDefaultRunOptions = nullptr;
static std::shared_ptr<facebook::react::CallInvoker> globalJsInvoker = nullptr;

void initOrtOnce(OrtLoggingLevel logLevel, std::shared_ptr<facebook::react::CallInvoker> jsInvoker, std::shared_ptr<Object> tensorConstructor) {
  if (ortInitialized) {
    return;
  }
  globalTensorConstructor = tensorConstructor;
  globalEnv = std::make_shared<Ort::Env>(logLevel, "onnxruntime-react-native-jsi");
  globalDefaultRunOptions = std::make_shared<Ort::RunOptions>();
  globalJsInvoker = jsInvoker;
  ortInitialized = true;
}

void cleanup() {
  ortInitialized = false;
  globalEnv.reset();
  globalTensorConstructor.reset();
  globalDefaultRunOptions.reset();
  globalJsInvoker.reset();
}

const std::shared_ptr<Ort::Env>& getOrtEnv() {
  if (!ortInitialized || !globalEnv) {
    throw std::runtime_error("ONNX Runtime not initialized. Call initOrtOnce first.");
  }
  return globalEnv;
}

const std::shared_ptr<Ort::RunOptions>& getDefaultRunOptions() {
  if (!ortInitialized || !globalDefaultRunOptions) {
    throw std::runtime_error("ONNX Runtime not initialized. Call initOrtOnce first.");
  }
  return globalDefaultRunOptions;
}

const std::shared_ptr<Object>& getTensorConstructor() {
  if (!ortInitialized || !globalTensorConstructor) {
    throw std::runtime_error("Tensor constructor not available. Call initOrtOnce first.");
  }
  return globalTensorConstructor;
}

const std::shared_ptr<facebook::react::CallInvoker>& getJsInvoker() {
  return globalJsInvoker;
}

} // namespace onnxruntimereactnativejsi