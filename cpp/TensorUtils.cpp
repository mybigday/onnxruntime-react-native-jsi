#include "TensorUtils.h"
#include "log.h"
#include "utils.h"
#include <stdexcept>
#include <cstring>
#include <unordered_map>

namespace onnxruntimereactnativejsi {

static const std::unordered_map<ONNXTensorElementDataType, const char*> dataTypeToStringMap = {
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "float32"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, "uint8"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, "int8"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, "uint16"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, "int16"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, "int32"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, "int64"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, "string"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, "bool"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, "float16"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, "float64"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, "uint32"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, "uint64"}
};

static const std::unordered_map<ONNXTensorElementDataType, size_t> elementSizeMap = {
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, sizeof(float)},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, sizeof(uint8_t)},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, sizeof(int8_t)},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, sizeof(uint16_t)},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, sizeof(int16_t)},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, sizeof(int32_t)},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, sizeof(int64_t)},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, sizeof(bool)},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, 2},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, sizeof(double)},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, sizeof(uint32_t)},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, sizeof(uint64_t)},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, sizeof(char*)},
};

static const std::unordered_map<ONNXTensorElementDataType, const char*> dataTypeToTypedArrayMap = {
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "Float32Array"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, "Float64Array"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, "Int32Array"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, "BigInt64Array"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, "Uint32Array"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, "BigUint64Array"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, "Uint8Array"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, "Int8Array"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, "Uint16Array"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, "Int16Array"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, "Float16Array"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, "Array"},
  {ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, "Uint8Array"},
};

std::string TensorUtils::dataTypeToString(ONNXTensorElementDataType dataType) {
  auto it = dataTypeToStringMap.find(dataType);
  if (it != dataTypeToStringMap.end()) {
    return it->second;
  }
  throw std::invalid_argument("Unsupported or unknown tensor data type: " + std::to_string(static_cast<int>(dataType)));
}

ONNXTensorElementDataType TensorUtils::stringToDataType(const std::string& typeStr) {
  for (auto it = dataTypeToStringMap.begin(); it != dataTypeToStringMap.end(); ++it) {
    if (it->second == typeStr) {
      return it->first;
    }
  }
  throw std::invalid_argument("Unsupported or unknown tensor data type: " + typeStr);
}

size_t TensorUtils::getElementSize(ONNXTensorElementDataType dataType) {
  auto it = elementSizeMap.find(dataType);
  if (it != elementSizeMap.end()) {
    return it->second;
  }
  throw std::invalid_argument("Unsupported or unknown tensor data type: " + std::to_string(static_cast<int>(dataType)));
}

bool TensorUtils::isValidJSTensor(Runtime& runtime, const Object& obj) {
  return obj.hasProperty(runtime, "cpuData") &&
         obj.hasProperty(runtime, "dims") &&
         obj.hasProperty(runtime, "type");
}

Object getTypedArrayConstructor(Runtime& runtime, const ONNXTensorElementDataType type) {
  auto it = dataTypeToTypedArrayMap.find(type);
  if (it != dataTypeToTypedArrayMap.end()) {
    auto prop = runtime.global().getProperty(runtime, it->second);
    if (prop.isObject()) {
      return prop.asObject(runtime);
    } else {
      throw JSError(runtime, "TypedArray constructor not found: " + std::string(it->second));
    }
  }
  throw JSError(runtime, "Unsupported tensor data type for TypedArray creation: " + std::to_string(static_cast<int>(type)));
}

size_t getElementCount(const std::vector<int64_t>& shape) {
  size_t count = 1;
  for (auto dim : shape) {
    count *= dim;
  }
  return count;
}

Ort::Value TensorUtils::createOrtValueFromJSTensor(
  Runtime& runtime, 
  const Object& tensorObj, 
  const Ort::MemoryInfo& memoryInfo
) {
  if (!isValidJSTensor(runtime, tensorObj)) {
    throw JSError(runtime, "Invalid tensor object: missing cpuData, dims, or type properties");
  }
  
  auto dataProperty = tensorObj.getProperty(runtime, "cpuData");
  auto dimsProperty = tensorObj.getProperty(runtime, "dims");
  auto typeProperty = tensorObj.getProperty(runtime, "type");
  
  if (!dimsProperty.isObject() || !dimsProperty.asObject(runtime).isArray(runtime)) {
    throw JSError(runtime, "Tensor dims must be array");
  }
  
  if (!typeProperty.isString()) {
    throw JSError(runtime, "Tensor type must be string");
  }

  auto type = stringToDataType(typeProperty.asString(runtime).utf8(runtime));

  void* data = nullptr;
  auto dataObj = dataProperty.asObject(runtime);

  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    if (!dataObj.isArray(runtime)) {
      throw JSError(runtime, "Tensor data must be an array of strings");
    }
    auto array = dataObj.asArray(runtime);
    std::vector<char*> arrayData(array.size(runtime));
    for (size_t i = 0; i < arrayData.size(); ++i) {
      auto item = array.getValueAtIndex(runtime, i);
      auto str = item.asString(runtime).utf8(runtime);
      arrayData[i] = strdup(str.c_str());
    }
    data = new char*[arrayData.size()];
    memcpy(data, arrayData.data(), arrayData.size() * sizeof(char*));
  } else {
    if (!isTypedArray(runtime, dataObj)) {
      throw JSError(runtime, "Tensor data must be a TypedArray");
    }
    auto buffer = dataObj.getProperty(runtime, "buffer").asObject(runtime).getArrayBuffer(runtime);
    data = buffer.data(runtime);
  }

  std::vector<int64_t> shape;
  auto dimsArray = dimsProperty.asObject(runtime).asArray(runtime);
  for (size_t i = 0; i < dimsArray.size(runtime); ++i) {
    auto dim = dimsArray.getValueAtIndex(runtime, i);
    if (dim.isNumber()) {
      shape.push_back(static_cast<int64_t>(dim.asNumber()));
    }
  }

  return Ort::Value::CreateTensor(
    memoryInfo,
    data,
    getElementCount(shape) * getElementSize(type),
    shape.data(),
    shape.size(),
    type
  );
}

Object TensorUtils::createJSTensorFromOrtValue(Runtime& runtime, Ort::Value& ortValue, const Object& tensorConstructor) {
  // Get tensor info
  auto typeInfo = ortValue.GetTensorTypeAndShapeInfo();
  auto shape = typeInfo.GetShape();
  auto elementType = typeInfo.GetElementType();
  
  // Prepare constructor arguments: type, data, dims
  std::string typeStr = dataTypeToString(elementType);
  
  // Create dims array
  auto dimsArray = Array(runtime, shape.size());
  for (size_t j = 0; j < shape.size(); ++j) {
    dimsArray.setValueAtIndex(runtime, j, Value(static_cast<double>(shape[j])));
  }
  
  // Create TypedArray with tensor data (copy data for JSI transfer)
  void* rawData = ortValue.GetTensorMutableRawData();
  size_t elementCount = ortValue.GetTensorTypeAndShapeInfo().GetElementCount();
  size_t elementSize = getElementSize(elementType);
  size_t dataSize = elementCount * elementSize;
  
  // Create TypedArray based on data type
  auto typedArrayCtor = getTypedArrayConstructor(runtime, elementType);
  // Create TypedArray instance with the ArrayBuffer
  auto typedArrayInstance = typedArrayCtor.asFunction(runtime).callAsConstructor(runtime, static_cast<double>(elementCount));

  auto buffer = typedArrayInstance.asObject(runtime).getProperty(runtime, "buffer").asObject(runtime).getArrayBuffer(runtime);
  memcpy(buffer.data(runtime), rawData, dataSize);
  
  // Call: new Tensor(type, data, dims)
  // Create new Tensor instance using the constructor
  auto tensorInstance = tensorConstructor
    .asFunction(runtime)
    .callAsConstructor(runtime, typeStr, typedArrayInstance, dimsArray);
  
  return tensorInstance.asObject(runtime);
}

} // namespace onnxruntimereactnativejsi
