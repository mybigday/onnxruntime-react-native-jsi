#include "AsyncWorker.h"
#include <signal.h>

using namespace facebook::jsi;

namespace onnxruntimereactnativejsi {

AsyncWorker::~AsyncWorker() {
  if (thread_.joinable()) {
    pthread_kill(thread_.native_handle(), SIGKILL);
    thread_.join();
  }
}

Value AsyncWorker::toPromise(Runtime& runtime) {
  auto jsPromise = runtime.global().getPropertyAsObject(runtime, "Promise");
  auto executor = Function::createFromHostFunction(
    runtime,
    PropNameID::forAscii(runtime, "executor"),
    2,
    [this](Runtime& runtime, const Value& thisValue, const Value* arguments, size_t count) -> Value {
      auto resolver = std::make_shared<Resolver>(runtime, arguments);
      this->resolver_ = resolver;
      thread_ = std::thread([this]() {
        auto& jsInvoker = env_->getJsInvoker();
        try {
          Execute();
          jsInvoker.invokeAsync([this](Runtime& runtime) {
            if (resolver_) {
              resolver_->resolve_.call(runtime, OnSuccess(runtime));
            }
            // release self
            auto promise = weakPromise_->lock(runtime);
            if (promise.isObject()) {
              promise.asObject(runtime).setProperty(runtime, "_p", Value::undefined());
            }
          });
        } catch (const std::exception& e) {
          error_ = std::string(e.what());
          jsInvoker.invokeAsync([this](Runtime& runtime) {
            if (resolver_) {
              resolver_->reject_.call(runtime, OnError(runtime, error_));
            }
            // release self
            auto promise = weakPromise_->lock(runtime);
            if (promise.isObject()) {
              promise.asObject(runtime).setProperty(runtime, "_p", Value::undefined());
            }
          });
        }
      });
      return Value::undefined();
    }
  );
  auto promise = jsPromise.asFunction(runtime).callAsConstructor(runtime, executor);
  // Hacking the promise to keep the AsyncWorker alive
  auto promiseObj = promise.asObject(runtime);
  weakPromise_ = std::make_shared<WeakObject>(runtime, promiseObj);
  promiseObj.setProperty(runtime, "_p", Object::createFromHostObject(runtime, shared_from_this()));
  return promise;
}

} // namespace onnxruntimereactnativejsi
