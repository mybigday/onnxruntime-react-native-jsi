#pragma once

#include <jsi/jsi.h>
#include <string>
#include <memory>
#include <thread>
#include "global.h"
#include "log.h"

using namespace facebook::jsi;
using namespace facebook::react;

namespace onnxruntimereactnativejsi {

class AsyncWorker: public HostObject, public std::enable_shared_from_this<AsyncWorker> {
  public:
    inline AsyncWorker(Runtime& runtime): runtime_(runtime) {}

    inline ~AsyncWorker() {
      if (thread_.joinable()) {
        thread_.join();
      }
      LOGI("AsyncWorker destroyed");
    }

    inline Value toPromise(Runtime& runtime) {
      auto jsPromise = runtime.global().getPropertyAsObject(runtime, "Promise");
      auto executor = Function::createFromHostFunction(
        runtime,
        PropNameID::forAscii(runtime, "executor"),
        2,
        [this](Runtime& runtime, const Value& thisValue, const Value* arguments, size_t count) -> Value {
          auto resolver = std::make_shared<Resolver>(runtime, arguments);
          this->resolver_ = resolver;
          thread_ = std::thread([this]() {
            auto jsInvoker = getJsInvoker();
            try {
              Execute();
              jsInvoker->invokeAsync([this](Runtime& runtime) {
                if (!resolver_) return;
                resolver_->resolve_.call(runtime_, OnSuccess(runtime));
              });
            } catch (const std::exception& e) {
              error_ = std::string(e.what());
              jsInvoker->invokeAsync([this](Runtime& runtime) {
                if (!resolver_) return;
                resolver_->reject_.call(runtime_, OnError(runtime, error_));
              });
            }
          });
          return Value::undefined();
        }
      );
      auto promise = jsPromise.asFunction(runtime).callAsConstructor(runtime, executor);
      // Hacking the promise to keep the AsyncWorker alive
      promise
        .asObject(runtime)
        .setProperty(runtime, "_p", Object::createFromHostObject(runtime, shared_from_this()));
      return promise;
    }

    virtual void Execute() = 0;
    virtual Value OnSuccess(Runtime& runtime) { return Value::undefined(); }
    virtual Value OnError(Runtime& runtime, const std::string& message) {
      return Value(runtime, String::createFromUtf8(runtime, message));
    }

  private:
    struct Resolver {
      Resolver(Runtime& runtime, const Value* arguments):
        resolve_(arguments[0].asObject(runtime).asFunction(runtime)),
        reject_(arguments[1].asObject(runtime).asFunction(runtime)) {}

      Function resolve_;
      Function reject_;
    };
    Runtime& runtime_;
    std::shared_ptr<Resolver> resolver_;
    std::string error_;
    std::thread thread_;
};

} // namespace onnxruntimereactnativejsi
