#ifdef __cplusplus
#import "onnxruntime-react-native-jsi.h"
#endif

#ifdef RCT_NEW_ARCH_ENABLED
#import "RNOnnxruntimeReactNativeJsiSpec.h"

@interface OnnxruntimeReactNativeJsi : NSObject <NativeOnnxruntimeReactNativeJsiSpec>
#else
#import <React/RCTBridgeModule.h>

@interface OnnxruntimeReactNativeJsi : NSObject <RCTBridgeModule>
#endif

@end
