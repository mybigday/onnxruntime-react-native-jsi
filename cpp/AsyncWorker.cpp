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
      this->weakResolve_ = std::make_shared<WeakObject>(runtime, arguments[0].asObject(runtime));
      this->weakReject_ = std::make_shared<WeakObject>(runtime, arguments[1].asObject(runtime));
      thread_ = std::thread([this]() {
        auto jsInvoker = env_->getJsInvoker();
        if (!jsInvoker) return;
        try {
          Execute();
          jsInvoker->invokeAsync([this](Runtime& runtime) {
            auto resolve = weakResolve_->lock(runtime);
            if (resolve.isObject()) {
              resolve.asObject(runtime).asFunction(runtime).call(runtime, OnSuccess(runtime));
            }
            // release self
            auto promise = weakPromise_->lock(runtime);
            if (promise.isObject()) {
              promise.asObject(runtime).setProperty(runtime, "_p", Value::undefined());
            }
          });
        } catch (const std::exception& e) {
          error_ = std::string(e.what());
          jsInvoker->invokeAsync([this](Runtime& runtime) {
            auto reject = weakReject_->lock(runtime);
            if (reject.isObject()) {
              reject.asObject(runtime).asFunction(runtime).call(runtime, OnError(runtime, error_));
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
