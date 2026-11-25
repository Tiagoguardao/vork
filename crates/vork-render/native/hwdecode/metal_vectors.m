// Test-only input and readback for the production CoreVideo/Metal converter.
// Hardware video decoders can reject I_PCM bitstreams before conversion.
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#include <string.h>

extern void *vork_conversion_open(void *device, const char *shader);
extern void *vork_conversion_frame(void *conversion, CVPixelBufferRef pixels, int matrix);

void *vork_vector_conversion_open(const char *shader) {
    @autoreleasepool {
        return vork_conversion_open((__bridge void *)MTLCreateSystemDefaultDevice(), shader);
    }
}

int vork_metal_vector(void *conversion, const unsigned char *nv12, int width, int height,
                      int matrix, unsigned char *rgba) {
    @autoreleasepool {
        NSDictionary *attrs = @{(id)kCVPixelBufferIOSurfacePropertiesKey: @{},
                                (id)kCVPixelBufferMetalCompatibilityKey: @YES};
        CVPixelBufferRef pixels = NULL;
        if (CVPixelBufferCreate(NULL, width, height, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                                (__bridge CFDictionaryRef)attrs, &pixels)) return 0;
        if (CVPixelBufferLockBaseAddress(pixels, 0)) {
            CFRelease(pixels);
            return 0;
        }
        for (int plane = 0; plane < 2; ++plane) {
            unsigned char *dst = CVPixelBufferGetBaseAddressOfPlane(pixels, plane);
            size_t pitch = CVPixelBufferGetBytesPerRowOfPlane(pixels, plane);
            for (int y = 0; y < (height >> plane); ++y) {
                memcpy(dst + y * pitch, nv12, width);
                nv12 += width;
            }
        }
        CVPixelBufferUnlockBaseAddress(pixels, 0);
        id<MTLTexture> result = (__bridge_transfer id<MTLTexture>)vork_conversion_frame(conversion, pixels, matrix);
        CFRelease(pixels);
        if (!result || result.mipmapLevelCount != 1) return 0;

        NSUInteger row = (width * 4 + 255) & ~255u;
        id<MTLBuffer> read = [result.device newBufferWithLength:row * height options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [result.device newCommandQueue];
        id<MTLCommandBuffer> buffer = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [buffer blitCommandEncoder];
        if (!blit || !read) return 0;
        [blit copyFromTexture:result sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
                   sourceSize:MTLSizeMake(width,height,1) toBuffer:read destinationOffset:0
       destinationBytesPerRow:row destinationBytesPerImage:row * height];
        [blit endEncoding];
        [buffer commit];
        [buffer waitUntilCompleted];
        if (buffer.status != MTLCommandBufferStatusCompleted) return 0;
        for (int y = 0; y < height; ++y)
            memcpy(rgba + y * width * 4, (const unsigned char *)read.contents + y * row, width * 4);
        return 1;
    }
}
