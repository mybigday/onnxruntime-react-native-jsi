#pragma once

#include <jsi/jsi.h>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <string>

using namespace facebook::jsi;

namespace onnxruntimereactnativejsi {

class InferenceSessionHostObject : public HostObject, public std::enable_shared_from_this<InferenceSessionHostObject> {
public:
  InferenceSessionHostObject();
  ~InferenceSessionHostObject() override;

  std::vector<PropNameID> getPropertyNames(Runtime& rt) override;
  Value get(Runtime& runtime, const PropNameID& name) override;
  void set(Runtime& runtime, const PropNameID& name, const Value& value) override;
  
  void dispose();

  class LoadModelAsyncWorker;
  class RunAsyncWorker;

private:
  Ort::MemoryInfo memoryInfo_;
  std::unique_ptr<Ort::Session> session_;

  Value loadModelMethod(Runtime& runtime, const Value* arguments, size_t count);
  Value runMethod(Runtime& runtime, const Value* arguments, size_t count);
  Value disposeMethod(Runtime& runtime, const Value* arguments, size_t count);
  Value endProfilingMethod(Runtime& runtime, const Value* arguments, size_t count);
  Value getInputMetadata(Runtime& runtime);
  Value getOutputMetadata(Runtime& runtime);
};

} // namespace onnxruntimereactnativejsi
