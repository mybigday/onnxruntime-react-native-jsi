#include <jni.h>
#include <jsi/jsi.h>
#include <ReactCommon/CallInvokerHolder.h>
#include <ReactCommon/CallInvoker.h>
#include <ReactCommon/CallInvokerHolder.h>
#include <fbjni/fbjni.h>
#include <fbjni/detail/Registration.h>
#include "onnxruntime-react-native-jsi.h"
#include "global.h"

using namespace facebook;

class OnnxruntimeReactNativeJsiModule : public jni::JavaClass<OnnxruntimeReactNativeJsiModule> {
public:
  static constexpr auto kJavaDescriptor = "Lcom/onnxruntimereactnativejsi/OnnxruntimeReactNativeJsiModule;";

  static void registerNatives() {
    javaClassStatic()->registerNatives({
      makeNativeMethod("nativeInstall", OnnxruntimeReactNativeJsiModule::nativeInstall),
      makeNativeMethod("nativeCleanup", OnnxruntimeReactNativeJsiModule::nativeCleanup)
    });
  }

private:
  static void nativeInstall(
    jni::alias_ref<jni::JObject> thiz,
    jlong jsContextNativePointer,
    jni::alias_ref<react::CallInvokerHolder::javaobject> jsCallInvokerHolder
  ) {
    auto runtime = reinterpret_cast<jsi::Runtime*>(jsContextNativePointer);
    auto jsCallInvoker = jsCallInvokerHolder->cthis()->getCallInvoker();
    onnxruntimereactnativejsi::install(*runtime, jsCallInvoker);
  }

  static void nativeCleanup(jni::alias_ref<jni::JObject> thiz) {
    onnxruntimereactnativejsi::cleanup();
  }
};


JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
  return jni::initialize(vm, [] { OnnxruntimeReactNativeJsiModule::registerNatives(); });
}
