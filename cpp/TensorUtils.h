#pragma once

#include <jsi/jsi.h>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

using namespace facebook::jsi;

namespace onnxruntimereactnativejsi {

/**
 * Utility functions for ONNX Runtime tensor operations and conversions
 */
class TensorUtils {
public:
  // Data type conversion utilities
  static std::string dataTypeToString(ONNXTensorElementDataType dataType);
  static ONNXTensorElementDataType stringToDataType(const std::string& typeStr);
  static size_t getElementSize(ONNXTensorElementDataType dataType);
  
  // Tensor creation from onnxruntime-common Tensor objects
  static Ort::Value createOrtValueFromJSTensor(
    Runtime& runtime, 
    const Object& tensorObj, 
    const Ort::MemoryInfo& memoryInfo
  );
  
  // Tensor conversion to onnxruntime-common compatible JS objects
  static Object createJSTensorFromOrtValue(
    Runtime& runtime, 
    Ort::Value& ortValue,
    const Object& tensorConstructor
  );
  
  // Helper to parse tensor properties from JS object
  static bool isValidJSTensor(Runtime& runtime, const Object& obj);
};

} // namespace onnxruntimereactnativejsi
