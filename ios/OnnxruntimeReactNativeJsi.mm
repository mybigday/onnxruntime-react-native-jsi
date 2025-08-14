#import "OnnxruntimeReactNativeJsi.h"
#import <React/RCTBridge+Private.h>
#import <React/RCTUtils.h>
#import <jsi/jsi.h>

static std::shared_ptr<onnxruntimereactnativejsi::Env> env;

@implementation OnnxruntimeReactNativeJsi

@synthesize bridge = _bridge;
@synthesize methodQueue = _methodQueue;

RCT_EXPORT_MODULE()

+ (BOOL)requiresMainQueueSetup
{
    return YES;
}

- (instancetype)init
{
    self = [super init];
    if (self) {
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(handleJSReload)
                                                     name:RCTJavaScriptDidLoadNotification
                                                   object:nil];
    }
    return self;
}

- (void)dealloc
{
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}

- (void)handleJSReload
{
    env.reset();
}

- (void)setBridge:(RCTBridge *)bridge
{
    _bridge = bridge;
}

- (void)installImpl
{
    RCTCxxBridge *cxxBridge = (RCTCxxBridge *)self.bridge;
    if (!cxxBridge.runtime) {
        return;
    }
    
    auto jsiRuntime = (facebook::jsi::Runtime*) cxxBridge.runtime;
    if (jsiRuntime) {
        auto jsInvoker = cxxBridge.jsCallInvoker;
        env = onnxruntimereactnativejsi::install(*jsiRuntime, jsInvoker);
    }
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(install)
{
    [self installImpl];
    return @(YES);
}

// Don't compile this code when we build for the old architecture.
#ifdef RCT_NEW_ARCH_ENABLED
- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params
{
    return std::make_shared<facebook::react::NativeOnnxruntimeReactNativeJsiSpecJSI>(params);
}
#endif

@end
