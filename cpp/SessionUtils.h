#pragma once

#include <jsi/jsi.h>
#include <onnxruntime_cxx_api.h>

namespace onnxruntimereactnativejsi {

extern const std::vector<const char *> supportedBackends;

void parseSessionOptions(facebook::jsi::Runtime &runtime,
                         const facebook::jsi::Value &optionsValue,
                         Ort::SessionOptions &sessionOptions);

void parseRunOptions(facebook::jsi::Runtime &runtime,
                     const facebook::jsi::Value &optionsValue,
                     Ort::RunOptions &runOptions);

} // namespace onnxruntimereactnativejsi