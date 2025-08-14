#pragma once

#include <jsi/jsi.h>
#include <onnxruntime_cxx_api.h>

namespace onnxruntimereactnativejsi {

void parseSessionOptions(facebook::jsi::Runtime& runtime, const facebook::jsi::Value& optionsValue, Ort::SessionOptions& sessionOptions);

void parseRunOptions(facebook::jsi::Runtime& runtime, const facebook::jsi::Value& optionsValue, Ort::RunOptions& runOptions);

} // namespace onnxruntimereactnativejsi