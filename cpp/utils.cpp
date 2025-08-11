#include "utils.h"

bool isTypedArray(Runtime &runtime, const Object &jsObj) {
  if (!jsObj.hasProperty(runtime, "buffer")) return false;
  if (!jsObj.getProperty(runtime, "buffer").asObject(runtime).isArrayBuffer(runtime)) return false;
  return true;
}

std::vector<std::string> getObjectKeys(Runtime& runtime, const Object& obj) {
  std::vector<std::string> keys;
  auto keysArray = runtime.global()
    .getPropertyAsObject(runtime, "Object")
    .getPropertyAsObject(runtime, "keys")
    .asFunction(runtime)
    .call(runtime, obj)
    .asObject(runtime)
    .asArray(runtime);
  size_t length = keysArray.size(runtime);
  keys.reserve(length);
  for (size_t i = 0; i < length; i++) {
    keys.push_back(keysArray.getValueAtIndex(runtime, i).asString(runtime).utf8(runtime));
  }
  return keys;
}
