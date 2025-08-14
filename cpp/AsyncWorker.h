#pragma once

#include <jsi/jsi.h>
#include <string>
#include <memory>
#include <thread>
#include "Env.h"

using namespace facebook::jsi;

namespace onnxruntimereactnativejsi {

class AsyncWorker: public HostObject, public std::enable_shared_from_this<AsyncWorker> {
public:
  AsyncWorker(std::shared_ptr<Env> env): env_(env) {}

  ~AsyncWorker();

  facebook::jsi::Value toPromise(facebook::jsi::Runtime& runtime);

  virtual void onAbort() {}
  virtual void Execute() = 0;
  virtual facebook::jsi::Value OnSuccess(facebook::jsi::Runtime& runtime) {
    return facebook::jsi::Value::undefined();
  }
  virtual facebook::jsi::Value OnError(facebook::jsi::Runtime& runtime, const std::string& message) {
    return facebook::jsi::Value(runtime, facebook::jsi::String::createFromUtf8(runtime, message));
  }

private:
  struct Resolver {
    Resolver(facebook::jsi::Runtime& runtime, const facebook::jsi::Value* arguments):
      resolve_(arguments[0].asObject(runtime).asFunction(runtime)),
      reject_(arguments[1].asObject(runtime).asFunction(runtime)) {}

    facebook::jsi::Function resolve_;
    facebook::jsi::Function reject_;
  };
  std::shared_ptr<Env> env_;
  std::shared_ptr<Resolver> resolver_;
  std::shared_ptr<facebook::jsi::WeakObject> weakPromise_;
  std::string error_;
  std::thread thread_;
};

} // namespace onnxruntimereactnativejsi
