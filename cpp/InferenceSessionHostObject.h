#pragma once

#include <jsi/jsi.h>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include "JsiHelper.hpp"
#include "Env.h"

using namespace facebook::jsi;

namespace onnxruntimereactnativejsi {

class InferenceSessionHostObject : public HostObject, public std::enable_shared_from_this<InferenceSessionHostObject> {
public:
  InferenceSessionHostObject(std::shared_ptr<Env> env);

  std::vector<PropNameID> getPropertyNames(Runtime& rt) override;
  Value get(Runtime& runtime, const PropNameID& name) override;
  void set(Runtime& runtime, const PropNameID& name, const Value& value) override;

  static inline facebook::jsi::Value constructor(
    std::shared_ptr<Env> env,
    facebook::jsi::Runtime& runtime,
    const facebook::jsi::Value& thisValue,
    const facebook::jsi::Value* arguments,
    size_t count
  ) {
    return facebook::jsi::Object::createFromHostObject(runtime, std::make_shared<InferenceSessionHostObject>(env));
  }

private:
  std::shared_ptr<Env> env_;
  std::unique_ptr<Ort::Session> session_;

  class LoadModelAsyncWorker;
  class RunAsyncWorker;

  DEFINE_METHOD(loadModel);
  DEFINE_METHOD(run);
  DEFINE_METHOD(dispose);
  DEFINE_METHOD(endProfiling);
  
  DEFINE_GETTER(inputMetadata);
  DEFINE_GETTER(outputMetadata);

  JsiMethodMap methods_;
  JsiGetterMap getters_;
};

} // namespace onnxruntimereactnativejsi
