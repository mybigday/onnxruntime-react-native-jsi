import { NativeModules, Platform } from 'react-native';

const LINKING_ERROR =
  `The package 'onnxruntime-react-native-jsi' doesn't seem to be linked. Make sure: \n\n` +
  Platform.select({ ios: "- You have run 'pod install'\n", default: '' }) +
  '- You rebuilt the app after installing the package\n' +
  '- You are not using Expo Go\n';

// @ts-expect-error
const isTurboModuleEnabled = global.__turboModuleProxy != null;

const OnnxruntimeReactNativeJsiModule = isTurboModuleEnabled
  ? require('./NativeOnnxruntimeReactNativeJsi').default
  : NativeModules.OnnxruntimeReactNativeJsi;

const OnnxruntimeReactNativeJsi = OnnxruntimeReactNativeJsiModule
  ? OnnxruntimeReactNativeJsiModule
  : new Proxy(
      {},
      {
        get() {
          throw new Error(LINKING_ERROR);
        },
      }
    );

export default OnnxruntimeReactNativeJsi;
