package com.onnxruntimereactnativejsi

import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.turbomodule.core.CallInvokerHolderImpl
import com.facebook.react.turbomodule.core.interfaces.CallInvokerHolder
import com.facebook.react.common.annotations.FrameworkAPI

class OnnxruntimeReactNativeJsiModule internal constructor(context: ReactApplicationContext) :
  OnnxruntimeReactNativeJsiSpec(context) {

  companion object {
    const val NAME = "OnnxruntimeReactNativeJsi"
    
    init {
      System.loadLibrary("onnxruntime-react-native-jsi")
    }

    @OptIn(FrameworkAPI::class)
    @JvmStatic
    external fun nativeInstall(jsiRuntimePointer: Long, jsCallInvoker: CallInvokerHolderImpl)
    
    @JvmStatic
    external fun nativeCleanup()
  }

  private var isInstalled = false

  override fun invalidate() {
    super.invalidate()
    nativeCleanup()
    isInstalled = false
  }

  override fun getName(): String {
    return NAME
  }

  @ReactMethod(isBlockingSynchronousMethod = true)
  override fun install(): Boolean {
    tryInstall()
    return isInstalled
  }

  @OptIn(FrameworkAPI::class)
  private fun tryInstall() {
    if (!isInstalled) {
      try {
        val jsContextHolder = reactApplicationContext.javaScriptContextHolder
        if (jsContextHolder != null) {
          val jsCallInvokerHolder = reactApplicationContext.catalystInstance.jsCallInvokerHolder as CallInvokerHolderImpl
          nativeInstall(jsContextHolder.get(), jsCallInvokerHolder)
          isInstalled = true
        }
      } catch (e: Exception) {
        e.printStackTrace()
      }
    }
  }
}
