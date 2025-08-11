#include "InferenceSessionHostObject.h"
#include "TensorUtils.h"
#include "global.h"
#include "log.h"
#include "utils.h"
#include "AsyncWorker.hpp"
#include <stdexcept>

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
    
    // executionProviders
    if (options.hasProperty(runtime, "executionProviders")) {
      auto prop = options.getProperty(runtime, "executionProviders");
      if (prop.isObject() && prop.asObject(runtime).isArray(runtime)) {
        auto providersArray = prop.asObject(runtime).asArray(runtime);
        std::vector<std::string> providers;
        
        for (size_t i = 0; i < providersArray.size(runtime); ++i) {
          auto providerValue = providersArray.getValueAtIndex(runtime, i);
          if (providerValue.isString()) {
            providers.push_back(providerValue.asString(runtime).utf8(runtime));
          } else if (providerValue.isObject()) {
            // For provider objects, extract the name property
            auto providerObj = providerValue.asObject(runtime);
            if (providerObj.hasProperty(runtime, "name")) {
              auto nameValue = providerObj.getProperty(runtime, "name");
              if (nameValue.isString()) {
                providers.push_back(nameValue.asString(runtime).utf8(runtime));
              }
            }
          }
        }
        
        // Apply execution providers
        for (const auto& provider : providers) {
          if (provider == "cpu") {
            // sessionOptions.AppendExecutionProvider_CPU(OrtCPUProviderOptions{});
            // nothing to do
          }
          // Note: Other providers like CUDA, CoreML, NNAPI would need additional setup
          // For now, we support CPU which is always available
        }
      }
    }
    
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
      // feedObject
      auto feedObject = arguments[0].asObject(runtime);
      auto keys = getObjectKeys(runtime, feedObject);
      for (const auto& key : keys) {
        inputNames.push_back(strdup(key.c_str()));
      }
      for (const auto& key : inputNames) {
        inputValues.push_back(
          TensorUtils::createOrtValueFromJSTensor(
            runtime,
            feedObject.getProperty(runtime, key).asObject(runtime),
            session_->memoryInfo_
          )
        );
      }
      // outputObject
      auto outputObject = arguments[1].asObject(runtime);
      keys = getObjectKeys(runtime, outputObject);
      for (const auto& key : keys) {
        outputNames.push_back(strdup(key.c_str()));
      }
      outputValues.resize(outputNames.size());
    }

    void Execute() override {
      session_->session_->Run(runOptions_,
                              inputNames.empty() ? nullptr : inputNames.data(),
                              inputValues.empty() ? nullptr : inputValues.data(),
                              inputValues.size(),
                              outputNames.empty() ? nullptr : outputNames.data(),
                              outputNames.empty() ? nullptr : outputValues.data(),
                              outputNames.size());
    }

    Value OnSuccess(Runtime& runtime) override {
      auto resultObject = Object(runtime);
      auto tensorConstructor = getTensorConstructor();
      for (size_t i = 0; i < outputValues.size(); ++i) {
        auto tensorObj = TensorUtils::createJSTensorFromOrtValue(runtime, outputValues[i], *tensorConstructor);
        resultObject.setProperty(runtime, outputNames[i], Value(runtime, tensorObj));
      }
      return Value(runtime, resultObject);
    }

  private:
    std::shared_ptr<InferenceSessionHostObject> session_;
    Ort::RunOptions runOptions_;
    std::vector<const char*> inputNames;
    std::vector<Ort::Value> inputValues;
    std::vector<const char*> outputNames;
    std::vector<Ort::Value> outputValues;
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
