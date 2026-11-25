#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>

@interface VorkConversion : NSObject {
@public
    CVMetalTextureCacheRef cache;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;
}
@end
@implementation VorkConversion
- (void)dealloc { if (cache) CFRelease(cache); }
@end

void *vork_conversion_open(void *raw, const char *shader) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)raw;
        VorkConversion *s = [VorkConversion new];
        if (CVMetalTextureCacheCreate(NULL, NULL, device, NULL, &s->cache)) return NULL;
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:shader]
                                                     options:nil error:&error];
        if (!library) { NSLog(@"footage shader: %@", error); return NULL; }
        s->pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"convert"] error:&error];
        s->queue = [device newCommandQueue];
        if (!s->pipeline || !s->queue) return NULL;
        return (__bridge_retained void *)s;
    }
}
void vork_conversion_close(void *ptr) { CFRelease(ptr); }

void *vork_conversion_frame(void *ptr, CVPixelBufferRef pixels, int matrix) {
    @autoreleasepool {
        VorkConversion *s = (__bridge VorkConversion *)ptr;
        NSUInteger w = CVPixelBufferGetWidth(pixels), h = CVPixelBufferGetHeight(pixels);
        CVMetalTextureRef planes[2] = {};
        id<MTLTexture> result = nil;
        for (unsigned i = 0; i < 2; i++) {
            if (CVMetalTextureCacheCreateTextureFromImage(NULL, s->cache, pixels, NULL,
                i ? MTLPixelFormatRG8Unorm : MTLPixelFormatR8Unorm, w >> i, h >> i, i, &planes[i]) ||
                !CVMetalTextureGetTexture(planes[i])) goto done;
        }
        {
            MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:w height:h mipmapped:NO];
            desc.storageMode = MTLStorageModePrivate;
            desc.usage = MTLTextureUsageShaderWrite;
            result = [s->queue.device newTextureWithDescriptor:desc];
            id<MTLCommandBuffer> buffer = [s->queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
            if (!result || !encoder) { result = nil; goto done; }
            [encoder setComputePipelineState:s->pipeline];
            for (unsigned i = 0; i < 2; i++) [encoder setTexture:CVMetalTextureGetTexture(planes[i]) atIndex:i];
            [encoder setTexture:result atIndex:2];
            [encoder setBytes:&matrix length:sizeof(matrix) atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake((w/2+15)/16,(h+15)/16,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
            [encoder endEncoding];
            [buffer commit];
            [buffer waitUntilCompleted];
            if (buffer.status != MTLCommandBufferStatusCompleted) result = nil;
        }
done:
        for (unsigned i = 0; i < 2; i++) if (planes[i]) CFRelease(planes[i]);
        return (__bridge_retained void *)result;
    }
}
