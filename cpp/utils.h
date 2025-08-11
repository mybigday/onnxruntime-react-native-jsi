#pragma once

#include <jsi/jsi.h>
#include <vector>
#include <string>

using namespace facebook::jsi;

bool isTypedArray(Runtime &runtime, const Object &jsObj);
std::vector<std::string> getObjectKeys(Runtime& runtime, const Object& obj);
