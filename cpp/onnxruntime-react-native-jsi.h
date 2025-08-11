#ifndef ONNXRUNTIMEREACTNATIVEJSI_H
#define ONNXRUNTIMEREACTNATIVEJSI_H

#include <jsi/jsi.h>
#include <ReactCommon/CallInvokerHolder.h>

namespace onnxruntimereactnativejsi {

/**
 * Install the ONNX Runtime JSI bindings into the JavaScript runtime
 * This exposes the global OrtApi object with InferenceSession and other functionality
 */
void install(facebook::jsi::Runtime& runtime, std::shared_ptr<facebook::react::CallInvoker> jsInvoker = nullptr);

} // namespace onnxruntimereactnativejsi

#endif /* ONNXRUNTIMEREACTNATIVEJSI_H */