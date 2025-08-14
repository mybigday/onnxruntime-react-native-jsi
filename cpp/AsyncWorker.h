#pragma once

#include <jsi/jsi.h>
#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include "Env.h"

using namespace facebook::jsi;

namespace onnxruntimereactnativejsi {

class AsyncWorker: public HostObject, public std::enable_shared_from_this<AsyncWorker> {
public:
  AsyncWorker(std::shared_ptr<Env> env): env_(env), aborted_(false) {}

  ~AsyncWorker();

  facebook::jsi::Value toPromise(facebook::jsi::Runtime& runtime);

  virtual void OnAbort() {}
  virtual void Execute() = 0;
  virtual facebook::jsi::Value OnSuccess(facebook::jsi::Runtime& runtime) {
    return facebook::jsi::Value::undefined();
  }
  virtual facebook::jsi::Value OnError(facebook::jsi::Runtime& runtime, const std::string& message) {
    return facebook::jsi::Value(runtime, facebook::jsi::String::createFromUtf8(runtime, message));
  }

private:
  std::shared_ptr<Env> env_;
  std::shared_ptr<facebook::jsi::WeakObject> weakResolve_;
  std::shared_ptr<facebook::jsi::WeakObject> weakReject_;
  std::shared_ptr<facebook::jsi::WeakObject> weakPromise_;
  std::string error_;
  std::thread thread_;
  std::atomic<bool> aborted_;
};

} // namespace onnxruntimereactnativejsi
