#include "InferenceSessionHostObject.h"
#include "TensorUtils.h"
#include "JsiUtils.h"
#include "AsyncWorker.h"
#include "SessionUtils.h"

using namespace facebook::jsi;

namespace onnxruntimereactnativejsi {

InferenceSessionHostObject::InferenceSessionHostObject(
  std::shared_ptr<Env> env
):
  env_(env),
  methods_({
    METHOD_INFO(InferenceSessionHostObject, loadModel, 2),
    METHOD_INFO(InferenceSessionHostObject, run, 2),
    METHOD_INFO(InferenceSessionHostObject, dispose, 0),
    METHOD_INFO(InferenceSessionHostObject, endProfiling, 0),
  }),
  getters_({
    GETTER_INFO(InferenceSessionHostObject, inputMetadata),
    GETTER_INFO(InferenceSessionHostObject, outputMetadata),
  }) {}

std::vector<PropNameID> InferenceSessionHostObject::getPropertyNames(Runtime& rt) {
  std::vector<PropNameID> names;
  for (auto& [name, _] : methods_) {
    names.push_back(PropNameID::forUtf8(rt, name));
  }
  for (auto& [name, _] : getters_) {
    names.push_back(PropNameID::forUtf8(rt, name));
  }
  return names;
}

Value InferenceSessionHostObject::get(Runtime& runtime, const PropNameID& name) {
  auto propName = name.utf8(runtime);
  auto method = methods_.find(propName);
  if (method != methods_.end()) {
    return Function::createFromHostFunction(
      runtime, name, method->second.count,
      method->second.method
    );
  }

  auto getter = getters_.find(propName);
  if (getter != getters_.end()) {
    return getter->second(runtime);
  }

  return Value::undefined();
}

void InferenceSessionHostObject::set(Runtime& runtime, const PropNameID& name, const Value& value) {
  throw JSError(runtime, "InferenceSession properties are read-only");
}

class InferenceSessionHostObject::LoadModelAsyncWorker : public AsyncWorker {
  public:
    LoadModelAsyncWorker(
      Runtime& runtime,
      const Value* arguments, size_t count,
      std::shared_ptr<InferenceSessionHostObject> session
    ) : AsyncWorker(session->env_),
        session_(session) {
      if (count < 1) throw JSError(runtime, "loadModel requires at least 1 argument");
      if (arguments[0].isString()) {
        modelPath_ = arguments[0].asString(runtime).utf8(runtime);
        if (modelPath_.find("file://") == 0) {
          modelPath_ = modelPath_.substr(7);
        }
      } else if (arguments[0].isObject() && arguments[0].asObject(runtime).isArrayBuffer(runtime)) {
        auto arrayBuffer = arguments[0].asObject(runtime).getArrayBuffer(runtime);
        modelData_ = arrayBuffer.data(runtime);
        modelDataLength_ = arrayBuffer.size(runtime);
      } else {
        throw JSError(runtime, "Model path or buffer is required");
      }
      if (count > 1) {
        parseSessionOptions(runtime, arguments[1], sessionOptions_);
      }
    }

    void Execute() override {
      if (modelPath_.empty()) {
        session_->session_ = std::make_unique<Ort::Session>(session_->env_->getOrtEnv(), modelData_, modelDataLength_, sessionOptions_);
      } else {
        session_->session_ = std::make_unique<Ort::Session>(session_->env_->getOrtEnv(), modelPath_.c_str(), sessionOptions_);
      }
    }

  private:
    std::string modelPath_;
    void* modelData_;
    size_t modelDataLength_;
    std::shared_ptr<InferenceSessionHostObject> session_;
    Ort::SessionOptions sessionOptions_;
};

DEFINE_METHOD(InferenceSessionHostObject::loadModel) {
  auto worker = std::make_shared<LoadModelAsyncWorker>(runtime, arguments, count, shared_from_this());
  return worker->toPromise(runtime);
}

class InferenceSessionHostObject::RunAsyncWorker : public AsyncWorker {
  public:
    RunAsyncWorker(
      Runtime& runtime,
      const Value* arguments, size_t count,
      std::shared_ptr<InferenceSessionHostObject> session
    ) : AsyncWorker(session->env_),
        session_(session) {
      if (count < 1 || !arguments[0].isObject()) {
        throw JSError(runtime, "run requires feeds object as first argument");
      }
      if (count > 2 && !arguments[2].isUndefined()) {
        parseRunOptions(runtime, arguments[2], runOptions_);
      }
      Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
      { // feedObject
        auto feedObject = arguments[0].asObject(runtime);
        forEach(runtime, feedObject, [&](const std::string& key, const Value& value, size_t index) {
          inputNames.push_back(key);
          inputValues.push_back(
            TensorUtils::createOrtValueFromJSTensor(
              runtime,
              value.asObject(runtime),
              memoryInfo
            )
          );
        });
      }
      { // outputObject
        auto outputObject = arguments[1].asObject(runtime);
        auto size = outputObject.getPropertyNames(runtime).size(runtime);
        outputValues.resize(size);
        jsOutputValues.resize(size);
        forEach(runtime, outputObject, [&](const std::string& key, const Value& value, size_t index) {
          outputNames.push_back(key);
          if (value.isObject() && TensorUtils::isTensor(runtime, value.asObject(runtime))) {
            outputValues[index] = (
              TensorUtils::createOrtValueFromJSTensor(
                runtime,
                value.asObject(runtime),
                memoryInfo
              )
            );
            jsOutputValues[index] = std::make_unique<WeakObject>(runtime, value.asObject(runtime));
          }
        });
      }
    }

    void Execute() override {
      std::vector<const char*> inputNamesCStr(inputNames.size());
      std::vector<const char*> outputNamesCStr(outputNames.size());
      std::transform(inputNames.begin(), inputNames.end(), inputNamesCStr.begin(), [](const std::string& name) { return name.c_str(); });
      std::transform(outputNames.begin(), outputNames.end(), outputNamesCStr.begin(), [](const std::string& name) { return name.c_str(); });
      session_->session_->Run(runOptions_,
                              inputNames.empty() ? nullptr : inputNamesCStr.data(),
                              inputValues.empty() ? nullptr : inputValues.data(),
                              inputValues.size(),
                              outputNames.empty() ? nullptr : outputNamesCStr.data(),
                              outputNames.empty() ? nullptr : outputValues.data(),
                              outputNames.size());
    }

    Value OnSuccess(Runtime& runtime) override {
      auto resultObject = Object(runtime);
      auto tensorConstructor = session_->env_->getTensorConstructor(runtime).asObject(runtime);
      for (size_t i = 0; i < outputValues.size(); ++i) {
        if (jsOutputValues[i] && outputValues[i].IsTensor()) {
          resultObject.setProperty(runtime, outputNames[i].c_str(), jsOutputValues[i]->lock(runtime));
        } else {
          auto tensorObj = TensorUtils::createJSTensorFromOrtValue(runtime, outputValues[i], tensorConstructor);
          resultObject.setProperty(runtime, outputNames[i].c_str(), Value(runtime, tensorObj));
        }
      }
      return Value(runtime, resultObject);
    }

  private:
    std::shared_ptr<InferenceSessionHostObject> session_;
    Ort::RunOptions runOptions_;
    std::vector<std::string> inputNames;
    std::vector<Ort::Value> inputValues;
    std::vector<std::string> outputNames;
    std::vector<Ort::Value> outputValues;
    std::vector<std::unique_ptr<WeakObject>> jsOutputValues;
};

DEFINE_METHOD(InferenceSessionHostObject::run) {
  auto worker = std::make_shared<RunAsyncWorker>(runtime, arguments, count, shared_from_this());
  return worker->toPromise(runtime);
}

DEFINE_METHOD(InferenceSessionHostObject::dispose) {
  session_.reset();
  return Value::undefined();
}

DEFINE_METHOD(InferenceSessionHostObject::endProfiling) {
  try {
    Ort::AllocatorWithDefaultOptions allocator;
    auto filename = session_->EndProfilingAllocated(allocator);
    return String::createFromUtf8(runtime, std::string(filename.get()));
  } catch (const std::exception& e) {
    throw JSError(runtime, std::string(e.what()));
  }
}

DEFINE_GETTER(InferenceSessionHostObject::inputMetadata) {
  if (!session_) {
    return Array(runtime, 0);
  }
  try {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputs = session_->GetInputCount();
    auto array = Array(runtime, numInputs);
    
    for (size_t i = 0; i < numInputs; i++) {
      auto item = Object(runtime);
      auto inputName = session_->GetInputNameAllocated(i, allocator);
      item.setProperty(runtime, "name", String::createFromUtf8(runtime, std::string(inputName.get())));
      
      try {
        auto typeInfo = session_->GetInputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        
        // Get data type
        auto dataType = tensorInfo.GetElementType();
        item.setProperty(runtime, "type", static_cast<double>(dataType));
        
        // Get shape
        auto shape = tensorInfo.GetShape();
        auto shapeArray = Array(runtime, shape.size());
        for (size_t j = 0; j < shape.size(); j++) {
          shapeArray.setValueAtIndex(runtime, j, Value(static_cast<double>(shape[j])));
        }
        item.setProperty(runtime, "shape", shapeArray);
        
        item.setProperty(runtime, "isTensor", Value(true));

        // symbolicDimensions
        auto symbolicDimensions = tensorInfo.GetSymbolicDimensions();
        auto symbolicDimensionsArray = Array(runtime, symbolicDimensions.size());
        for (size_t j = 0; j < symbolicDimensions.size(); j++) {
          symbolicDimensionsArray.setValueAtIndex(runtime, j, String::createFromUtf8(runtime, symbolicDimensions[j]));
        }
        item.setProperty(runtime, "symbolicDimensions", symbolicDimensionsArray);
      } catch (const std::exception&) {
        // Fallback for unknown types
        item.setProperty(runtime, "type", String::createFromUtf8(runtime, "unknown"));
        item.setProperty(runtime, "shape", Array(runtime, 0));
        item.setProperty(runtime, "isTensor", Value(false));
      }
      
      array.setValueAtIndex(runtime, i, Value(runtime, item));
    }
    
    return Value(runtime, array);
  } catch (const Ort::Exception& e) {
    throw JSError(runtime, std::string(e.what()));
  }
}

DEFINE_GETTER(InferenceSessionHostObject::outputMetadata) {
  if (!session_) {
    return Array(runtime, 0);
  }
  try {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numOutputs = session_->GetOutputCount();
    auto array = Array(runtime, numOutputs);
    
    for (size_t i = 0; i < numOutputs; i++) {
      auto item = Object(runtime);
      auto outputName = session_->GetOutputNameAllocated(i, allocator);
      item.setProperty(runtime, "name", String::createFromUtf8(runtime, std::string(outputName.get())));
      
      try {
        auto typeInfo = session_->GetOutputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        
        // Get data type
        auto dataType = tensorInfo.GetElementType();
        item.setProperty(runtime, "type", static_cast<double>(dataType));
        
        // Get shape
        auto shape = tensorInfo.GetShape();
        auto shapeArray = Array(runtime, shape.size());
        for (size_t j = 0; j < shape.size(); j++) {
          shapeArray.setValueAtIndex(runtime, j, Value(static_cast<double>(shape[j])));
        }
        item.setProperty(runtime, "shape", shapeArray);
        
        item.setProperty(runtime, "isTensor", Value(true));

        // symbolicDimensions
        auto symbolicDimensions = tensorInfo.GetSymbolicDimensions();
        auto symbolicDimensionsArray = Array(runtime, symbolicDimensions.size());
        for (size_t j = 0; j < symbolicDimensions.size(); j++) {
          symbolicDimensionsArray.setValueAtIndex(runtime, j, String::createFromUtf8(runtime, symbolicDimensions[j]));
        }
        item.setProperty(runtime, "symbolicDimensions", symbolicDimensionsArray);
      } catch (const std::exception&) {
        // Fallback for unknown types
        item.setProperty(runtime, "type", String::createFromUtf8(runtime, "unknown"));
        item.setProperty(runtime, "shape", Array(runtime, 0));
        item.setProperty(runtime, "isTensor", Value(false));
      }
      
      array.setValueAtIndex(runtime, i, Value(runtime, item));
    }
    
    return Value(runtime, array);
  } catch (const Ort::Exception& e) {
    throw JSError(runtime, std::string(e.what()));
  }
}

} // namespace onnxruntimereactnativejsi
