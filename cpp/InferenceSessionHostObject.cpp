#include "InferenceSessionHostObject.h"
#include "TensorUtils.h"
#include "global.h"
#include "log.h"
#include "utils.h"
#include "AsyncWorker.hpp"
#include <cpu_provider_factory.h>
#ifdef USE_NNAPI
#include <nnapi_provider_factory.h>
#endif
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <vector>
#include <utility>

namespace onnxruntimereactnativejsi {

InferenceSessionHostObject::InferenceSessionHostObject() :
  memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)) {}

InferenceSessionHostObject::~InferenceSessionHostObject() {
  dispose();
}

std::vector<PropNameID> InferenceSessionHostObject::getPropertyNames(Runtime& rt) {
  return PropNameID::names(rt, 
    "loadModel", 
    "run", 
    "dispose", 
    "endProfiling",
    "inputMetadata",
    "outputMetadata"
  );
}

Value InferenceSessionHostObject::get(Runtime& runtime, const PropNameID& name) {
  auto propName = name.utf8(runtime);
  
  if (propName == "loadModel") {
    return Function::createFromHostFunction(
      runtime, name, 2,
      [this](Runtime& runtime, const Value& thisValue, const Value* arguments, size_t count) {
        return loadModelMethod(runtime, arguments, count);
      }
    );
  }
  
  if (propName == "run") {
    return Function::createFromHostFunction(
      runtime, name, 2,
      [this](Runtime& runtime, const Value& thisValue, const Value* arguments, size_t count) {
        return runMethod(runtime, arguments, count);
      }
    );
  }
  
  if (propName == "dispose") {
    return Function::createFromHostFunction(
      runtime, name, 0,
      [this](Runtime& runtime, const Value& thisValue, const Value* arguments, size_t count) {
        return disposeMethod(runtime, arguments, count);
      }
    );
  }
  
  if (propName == "endProfiling") {
    return Function::createFromHostFunction(
      runtime, name, 0,
      [this](Runtime& runtime, const Value& thisValue, const Value* arguments, size_t count) {
        return endProfilingMethod(runtime, arguments, count);
      }
    );
  }
  
  if (propName == "inputMetadata") {
    return getInputMetadata(runtime);
  }
  
  if (propName == "outputMetadata") {
    return getOutputMetadata(runtime);
  }
  
  return Value::undefined();
}

void InferenceSessionHostObject::set(Runtime& runtime, const PropNameID& name, const Value& value) {
  throw JSError(runtime, "InferenceSession properties are read-only");
}

// implement AddFreeDimensionOverrideByName for SessionOptions
class ExtendedSessionOptions : public Ort::SessionOptions {
  public:
    ExtendedSessionOptions() = default;

    void AppendExecutionProvider_CPU(int use_arena) {
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(this->p_, use_arena));
    }

    void AddFreeDimensionOverrideByName(const char* name, int64_t value) {
      Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(this->p_, name, value));
    }
#ifdef USE_NNAPI
    void AppendExecutionProvider_Nnapi(uint32_t nnapi_flags) {
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(this->p_, nnapi_flags));
    }
#endif
};

void parseSessionOptions(Runtime& runtime, const Value& optionsValue, Ort::SessionOptions& sessionOptions) {
  if (!optionsValue.isObject()) {
    return; // Skip if not an object
  }

  auto options = optionsValue.asObject(runtime);

  try {
    // intraOpNumThreads
    if (options.hasProperty(runtime, "intraOpNumThreads")) {
      auto prop = options.getProperty(runtime, "intraOpNumThreads");
      if (prop.isNumber()) {
        int numThreads = static_cast<int>(prop.asNumber());
        if (numThreads > 0) {
          sessionOptions.SetIntraOpNumThreads(numThreads);
        }
      }
    }
    
    // interOpNumThreads
    if (options.hasProperty(runtime, "interOpNumThreads")) {
      auto prop = options.getProperty(runtime, "interOpNumThreads");
      if (prop.isNumber()) {
        int numThreads = static_cast<int>(prop.asNumber());
        if (numThreads > 0) {
          sessionOptions.SetInterOpNumThreads(numThreads);
        }
      }
    }
    
    // freeDimensionOverrides
    if (options.hasProperty(runtime, "freeDimensionOverrides")) {
      auto prop = options.getProperty(runtime, "freeDimensionOverrides");
      if (prop.isObject()) {
        auto overrides = prop.asObject(runtime);
        for_each(runtime, overrides, [&](const std::string& key, const Value& value, size_t index) {
          reinterpret_cast<ExtendedSessionOptions&>(sessionOptions).AddFreeDimensionOverrideByName(
            key.c_str(),
            static_cast<int64_t>(value.asNumber())
          );
        });
      }
    }
    
    // graphOptimizationLevel
    if (options.hasProperty(runtime, "graphOptimizationLevel")) {
      auto prop = options.getProperty(runtime, "graphOptimizationLevel");
      if (prop.isString()) {
        std::string level = prop.asString(runtime).utf8(runtime);
        if (level == "disabled") {
          sessionOptions.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
        } else if (level == "basic") {
          sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
        } else if (level == "extended") {
          sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        } else if (level == "all") {
          sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        }
      }
    }
    
    // enableCpuMemArena
    if (options.hasProperty(runtime, "enableCpuMemArena")) {
      auto prop = options.getProperty(runtime, "enableCpuMemArena");
      if (prop.isBool()) {
        if (prop.asBool()) {
          sessionOptions.EnableCpuMemArena();
        } else {
          sessionOptions.DisableCpuMemArena();
        }
      }
    }
    
    // enableMemPattern
    if (options.hasProperty(runtime, "enableMemPattern")) {
      auto prop = options.getProperty(runtime, "enableMemPattern");
      if (prop.isBool()) {
        if (prop.asBool()) {
          sessionOptions.EnableMemPattern();
        } else {
          sessionOptions.DisableMemPattern();
        }
      }
    }
    
    // executionMode
    if (options.hasProperty(runtime, "executionMode")) {
      auto prop = options.getProperty(runtime, "executionMode");
      if (prop.isString()) {
        std::string mode = prop.asString(runtime).utf8(runtime);
        if (mode == "sequential") {
          sessionOptions.SetExecutionMode(ORT_SEQUENTIAL);
        } else if (mode == "parallel") {
          sessionOptions.SetExecutionMode(ORT_PARALLEL);
        }
      }
    }
    
    // optimizedModelFilePath
    if (options.hasProperty(runtime, "optimizedModelFilePath")) {
      auto prop = options.getProperty(runtime, "optimizedModelFilePath");
      if (prop.isString()) {
        std::string path = prop.asString(runtime).utf8(runtime);
        sessionOptions.SetOptimizedModelFilePath(path.c_str());
      }
    }
    
    // enableProfiling
    if (options.hasProperty(runtime, "enableProfiling")) {
      auto prop = options.getProperty(runtime, "enableProfiling");
      if (prop.isBool() && prop.asBool()) {
        sessionOptions.EnableProfiling("onnxruntime_profile_");
      }
    }
    
    // profileFilePrefix (if enableProfiling is true)
    if (options.hasProperty(runtime, "profileFilePrefix")) {
      auto enableProfilingProp = options.getProperty(runtime, "enableProfiling");
      if (enableProfilingProp.isBool() && enableProfilingProp.asBool()) {
        auto prop = options.getProperty(runtime, "profileFilePrefix");
        if (prop.isString()) {
          std::string prefix = prop.asString(runtime).utf8(runtime);
          sessionOptions.EnableProfiling(prefix.c_str());
        }
      }
    }
    
    // logId
    if (options.hasProperty(runtime, "logId")) {
      auto prop = options.getProperty(runtime, "logId");
      if (prop.isString()) {
        std::string logId = prop.asString(runtime).utf8(runtime);
        sessionOptions.SetLogId(logId.c_str());
      }
    }
    
    // logSeverityLevel
    if (options.hasProperty(runtime, "logSeverityLevel")) {
      auto prop = options.getProperty(runtime, "logSeverityLevel");
      if (prop.isNumber()) {
        int level = static_cast<int>(prop.asNumber());
        if (level >= 0 && level <= 4) {
          sessionOptions.SetLogSeverityLevel(level);
        }
      }
    }

    // externalData
    if (options.hasProperty(runtime, "externalData")) {
      auto prop = options.getProperty(runtime, "externalData").asObject(runtime);
      if (prop.isArray(runtime)) {
        auto externalDataArray = prop.asArray(runtime);
        std::vector<std::string> paths;
        std::vector<char*> buffs;
        std::vector<size_t> sizes;
        for_each(runtime, externalDataArray, [&](const Value& value, size_t index) {
          if (value.isObject()) {
            auto externalDataObject = value.asObject(runtime);
            if (externalDataObject.hasProperty(runtime, "path")) {
              auto pathValue = externalDataObject.getProperty(runtime, "path");
              if (pathValue.isString()) {
                paths.push_back(pathValue.asString(runtime).utf8(runtime));
              }
            }
            if (externalDataObject.hasProperty(runtime, "data")) {
              auto dataValue = externalDataObject.getProperty(runtime, "data").asObject(runtime);
              if (isTypedArray(runtime, dataValue)) {
                auto arrayBuffer = dataValue.getProperty(runtime, "buffer").asObject(runtime).getArrayBuffer(runtime);
                buffs.push_back(reinterpret_cast<char*>(arrayBuffer.data(runtime)));
                sizes.push_back(arrayBuffer.size(runtime));
              }
            }
          }
        });
        sessionOptions.AddExternalInitializersFromFilesInMemory(paths, buffs, sizes);
      }
    }
    
    // executionProviders
    if (options.hasProperty(runtime, "executionProviders")) {
      auto prop = options.getProperty(runtime, "executionProviders");
      if (prop.isObject() && prop.asObject(runtime).isArray(runtime)) {
        auto providers = prop.asObject(runtime).asArray(runtime);
        for_each(runtime, providers, [&](const Value& epValue, size_t index) {
          std::string epName;
          std::unique_ptr<Object> providerObj;
          if (epValue.isString()) {
            epName = epValue.asString(runtime).utf8(runtime);
          } else if (epValue.isObject()) {
            providerObj = std::make_unique<Object>(epValue.asObject(runtime));
            epName = providerObj->getProperty(runtime, "name").asString(runtime).utf8(runtime);
          }

          // Apply execution providers
          if (epName == "cpu") {
            int use_arena = 0;
            if (providerObj && providerObj->hasProperty(runtime, "useArena")) {
              auto useArena = providerObj->getProperty(runtime, "useArena");
              if (useArena.isBool() && useArena.asBool()) {
                use_arena = 1;
              }
            }
            reinterpret_cast<ExtendedSessionOptions&>(sessionOptions).AppendExecutionProvider_CPU(use_arena);
          } else if (epName == "xnnpack") {
            sessionOptions.AppendExecutionProvider("XNNPACK");
          }
#ifdef __APPLE__
          else if (epName == "coreml") {
            sessionOptions.AppendExecutionProvider_CoreML();
          }
#endif
#ifdef USE_NNAPI
          else if (epName == "nnapi") {
            uint32_t nnapi_flags = 0;
            if (providerObj && providerObj->hasProperty(runtime, "useFP16")) {
              auto useFP16 = providerObj->getProperty(runtime, "useFP16");
              if (useFP16.isBool() && useFP16.asBool()) {
                nnapi_flags |= NNAPI_FLAG_USE_FP16;
              }
            }
            if (providerObj && providerObj->hasProperty(runtime, "useNCHW")) {
              auto useNCHW = providerObj->getProperty(runtime, "useNCHW");
              if (useNCHW.isBool() && useNCHW.asBool()) {
                nnapi_flags |= NNAPI_FLAG_USE_NCHW;
              }
            }
            if (providerObj && providerObj->hasProperty(runtime, "cpuDisabled")) {
              auto cpuDisabled = providerObj->getProperty(runtime, "cpuDisabled");
              if (cpuDisabled.isBool() && cpuDisabled.asBool()) {
                nnapi_flags |= NNAPI_FLAG_CPU_DISABLED;
              }
            }
            if (providerObj && providerObj->hasProperty(runtime, "cpuOnly")) {
              auto cpuOnly = providerObj->getProperty(runtime, "cpuOnly");
              if (cpuOnly.isBool() && cpuOnly.asBool()) {
                nnapi_flags |= NNAPI_FLAG_CPU_ONLY;
              }
            }
            reinterpret_cast<ExtendedSessionOptions&>(sessionOptions).AppendExecutionProvider_Nnapi(nnapi_flags);
          }
#endif
#ifdef USE_QNN
          else if (epName == "qnn") {
            std::unordered_map<std::string, std::string> options;
            if (providerObj && providerObj->hasProperty(runtime, "backendType")) {
              options["backendType"] = providerObj->getProperty(runtime, "backendType").asString(runtime).utf8(runtime);
            }
            if (providerObj && providerObj->hasProperty(runtime, "backendPath")) {
              options["backendPath"] = providerObj->getProperty(runtime, "backendPath").asString(runtime).utf8(runtime);
            }
            if (providerObj && providerObj->hasProperty(runtime, "enableFp16Precision")) {
              auto enableFp16Precision = providerObj->getProperty(runtime, "enableFp16Precision");
              if (enableFp16Precision.isBool() && enableFp16Precision.asBool()) {
                options["enableFp16Precision"] = "1";
              } else {
                options["enableFp16Precision"] = "0";
              }
            }
            sessionOptions.AppendExecutionProvider("QNN", options);
          }
#endif
          else {
            throw JSError(runtime, "Unsupported execution provider: " + epName);
          }
        });
      }
    }
  } catch (const JSError& e) {
    throw e;
  } catch (const std::exception& e) {
    throw JSError(runtime, "Failed to parse session options: " + std::string(e.what()));
  }
}

class InferenceSessionHostObject::LoadModelAsyncWorker : public AsyncWorker {
  public:
    LoadModelAsyncWorker(
      Runtime& runtime,
      const Value* arguments, size_t count,
      std::shared_ptr<InferenceSessionHostObject> session
    ) : AsyncWorker(runtime),
        session_(session) {
      if (count < 1) {
        throw JSError(runtime, "loadModel requires at least 1 argument");
      }
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
        session_->session_ = std::make_unique<Ort::Session>(*getOrtEnv(), modelData_, modelDataLength_, sessionOptions_);
      } else {
        session_->session_ = std::make_unique<Ort::Session>(*getOrtEnv(), modelPath_.c_str(), sessionOptions_);
      }
    }

  private:
    std::string modelPath_;
    void* modelData_;
    size_t modelDataLength_;
    std::shared_ptr<InferenceSessionHostObject> session_;
    Ort::SessionOptions sessionOptions_;
};

Value InferenceSessionHostObject::loadModelMethod(Runtime& runtime, const Value* arguments, size_t count) {
  auto worker = std::make_shared<LoadModelAsyncWorker>(runtime, arguments, count, shared_from_this());
  return worker->toPromise(runtime);
}

void parseRunOptions(Runtime& runtime, const Value& optionsValue, Ort::RunOptions& runOptions) {
  if (!optionsValue.isObject()) {
    return; // Skip if not an object
  }
  
  auto options = optionsValue.asObject(runtime);
  
  try {
    // tag (run tag for logging/profiling)
    if (options.hasProperty(runtime, "tag")) {
      auto prop = options.getProperty(runtime, "tag");
      if (prop.isString()) {
        std::string tag = prop.asString(runtime).utf8(runtime);
        runOptions.SetRunTag(tag.c_str());
      }
    }
    
    // logSeverityLevel
    if (options.hasProperty(runtime, "logSeverityLevel")) {
      auto prop = options.getProperty(runtime, "logSeverityLevel");
      if (prop.isNumber()) {
        int level = static_cast<int>(prop.asNumber());
        if (level >= 0 && level <= 4) {
          runOptions.SetRunLogSeverityLevel(level);
        }
      }
    }
    
    // logVerbosityLevel
    if (options.hasProperty(runtime, "logVerbosityLevel")) {
      auto prop = options.getProperty(runtime, "logVerbosityLevel");
      if (prop.isNumber()) {
        int level = static_cast<int>(prop.asNumber());
        if (level >= 0) {
          runOptions.SetRunLogVerbosityLevel(level);
        }
      }
    }
    
    // terminate (early termination support)
    if (options.hasProperty(runtime, "terminate")) {
      auto prop = options.getProperty(runtime, "terminate");
      if (prop.isBool() && prop.asBool()) {
        // Enable termination support
        runOptions.SetTerminate();
      }
    }
    
  } catch (const std::exception& e) {
    throw JSError(runtime, "Failed to parse run options: " + std::string(e.what()));
  }
}

class InferenceSessionHostObject::RunAsyncWorker : public AsyncWorker {
  public:
    RunAsyncWorker(
      Runtime& runtime,
      const Value* arguments, size_t count,
      std::shared_ptr<InferenceSessionHostObject> session
    ) : AsyncWorker(runtime),
        session_(session) {
      if (count < 1 || !arguments[0].isObject()) {
        throw JSError(runtime, "run requires feeds object as first argument");
      }
      if (count > 2 && !arguments[2].isUndefined()) {
        parseRunOptions(runtime, arguments[2], runOptions_);
      }
      { // feedObject
        auto feedObject = arguments[0].asObject(runtime);
        for_each(runtime, feedObject, [&](const std::string& key, const Value& value, size_t index) {
          inputNames.push_back(key);
          inputValues.push_back(
            TensorUtils::createOrtValueFromJSTensor(
              runtime,
              value.asObject(runtime),
              session_->memoryInfo_
            )
          );
        });
      }
      { // outputObject
        auto outputObject = arguments[1].asObject(runtime);
        auto size = outputObject.getPropertyNames(runtime).size(runtime);
        outputValues.resize(size);
        jsOutputValues.resize(size);
        for_each(runtime, outputObject, [&](const std::string& key, const Value& value, size_t index) {
          outputNames.push_back(key);
          if (value.isObject() && TensorUtils::isTensor(runtime, value.asObject(runtime))) {
            outputValues[index] = (
              TensorUtils::createOrtValueFromJSTensor(
                runtime,
                value.asObject(runtime),
                session_->memoryInfo_
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
      auto tensorConstructor = getTensorConstructor();
      for (size_t i = 0; i < outputValues.size(); ++i) {
        if (jsOutputValues[i] && outputValues[i].IsTensor()) {
          resultObject.setProperty(runtime, outputNames[i].c_str(), jsOutputValues[i]->lock(runtime));
        } else {
          auto tensorObj = TensorUtils::createJSTensorFromOrtValue(runtime, outputValues[i], *tensorConstructor);
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

Value InferenceSessionHostObject::runMethod(Runtime& runtime, const Value* arguments, size_t count) {
  auto worker = std::make_shared<RunAsyncWorker>(runtime, arguments, count, shared_from_this());
  return worker->toPromise(runtime);
}

Value InferenceSessionHostObject::disposeMethod(Runtime& runtime, const Value* arguments, size_t count) {
  dispose();
  return Value::undefined();
}

Value InferenceSessionHostObject::endProfilingMethod(Runtime& runtime, const Value* arguments, size_t count) {
  try {
    // TODO: Implement profiling
    return Value::undefined();
  } catch (const std::exception& e) {
    throw JSError(runtime, "Failed to end profiling: " + std::string(e.what()));
  }
}

Value InferenceSessionHostObject::getInputMetadata(Runtime& runtime) {
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
    throw JSError(runtime, "Failed to get input metadata: " + std::string(e.what()));
  }
}

Value InferenceSessionHostObject::getOutputMetadata(Runtime& runtime) {
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
    throw JSError(runtime, "Failed to get output metadata: " + std::string(e.what()));
  }
}

void InferenceSessionHostObject::dispose() {
  session_.reset();
}

} // namespace onnxruntimereactnativejsi
