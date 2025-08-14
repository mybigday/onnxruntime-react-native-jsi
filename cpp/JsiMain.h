#pragma once

#include <jsi/jsi.h>
#include <ReactCommon/CallInvoker.h>
#include "Env.h"

namespace onnxruntimereactnativejsi {

std::shared_ptr<Env> install(
  facebook::jsi::Runtime& runtime,
  std::shared_ptr<facebook::react::CallInvoker> jsInvoker = nullptr
);

} // namespace onnxruntimereactnativejsi
