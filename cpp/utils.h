#pragma once

#include <jsi/jsi.h>
#include <vector>
#include <string>
#include <iterator>
#include <utility>

using namespace facebook::jsi;

bool isTypedArray(Runtime &runtime, const Object &jsObj);

void for_each(Runtime &runtime, const Object &object, const std::function<void(const std::string&, const Value&, size_t)> &callback);

void for_each(Runtime &runtime, const Array &array, const std::function<void(const Value&, size_t)> &callback);
