#pragma once

#include <onnxruntime_cxx_api.h>
#include <jsi/jsi.h>
#include <ReactCommon/CallInvokerHolder.h>
#include <memory>

using namespace facebook::jsi;

namespace onnxruntimereactnativejsi {

void initOrtOnce(
  OrtLoggingLevel logLevel,
  std::shared_ptr<facebook::react::CallInvoker> jsInvoker,
  std::shared_ptr<Object> tensorConstructor
);
void cleanup();
const std::shared_ptr<Ort::Env> &getOrtEnv();
const std::shared_ptr<Ort::RunOptions>& getDefaultRunOptions();
const std::shared_ptr<Object>& getTensorConstructor();
const std::shared_ptr<facebook::react::CallInvoker>& getJsInvoker();

} // namespace onnxruntimereactnativejsi
