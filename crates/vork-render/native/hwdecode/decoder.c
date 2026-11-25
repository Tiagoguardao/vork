// Resolve the build-time libav ABI when opening a native footage session.
#include <CoreVideo/CoreVideo.h>
#define HW_FORMAT AV_PIX_FMT_VIDEOTOOLBOX
#define HW_DEVICE AV_HWDEVICE_TYPE_VIDEOTOOLBOX
#include "demux.h"
#define AVFUNCS(F) F(av_buffer_alloc) \
  F(av_buffer_unref) \
  F(av_codec_iterate) \
  F(av_codec_is_decoder) \
  F(avcodec_get_hw_config) \
  F(avcodec_alloc_context3) \
  F(avcodec_parameters_to_context) \
  F(avcodec_open2) \
  F(avcodec_send_packet) \
  F(avcodec_receive_frame) \
  F(avcodec_free_context) \
  F(avcodec_flush_buffers) \
  F(av_hwdevice_ctx_create) \
  F(av_frame_alloc) \
  F(av_frame_unref) \
  F(av_frame_free)

typedef struct {
  Demux media;
  AVFUNCS(AV_FIELD)
  AVCodecContext *codec;
  AVFrame *frame;
  int width, height;
} Decoder;

void vork_decode_close(Decoder *d) {
  if (!d)
    return;
  if (d->frame)
    d->av_frame_free(&d->frame);
  if (d->codec)
    d->avcodec_free_context(&d->codec);
  close_demux(&d->media);
  free(d);
}

static enum AVPixelFormat select_hardware(AVCodecContext *codec,
                                      const enum AVPixelFormat *formats) {
  (void)codec;
  for (; *formats != AV_PIX_FMT_NONE; formats++)
    if (*formats == HW_FORMAT)
      return *formats;
  return AV_PIX_FMT_NONE;
}

// The codec-ID default may be software-only (notably libdav1d for AV1).
// Select by the device/context contract used below; conversion checks the actual frame.
static const AVCodec *hardware_decoder(Decoder *d, enum AVCodecID id) {
  void *iter = NULL;
  const AVCodec *codec;
  while ((codec = d->av_codec_iterate(&iter))) {
    if (codec->id != id || !d->av_codec_is_decoder(codec))
      continue;
    const AVCodecHWConfig *config;
    for (int i = 0; (config = d->avcodec_get_hw_config(codec, i)); i++)
      if (config->device_type == HW_DEVICE && config->pix_fmt == HW_FORMAT &&
          (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX))
        return codec;
  }
  return NULL;
}

#ifdef VORK_DIST_AVFORMAT_MAJOR
_Static_assert(LIBAVFORMAT_VERSION_MAJOR == VORK_DIST_AVFORMAT_MAJOR, "unexpected release FFmpeg headers");
#endif

#include "mac_avformat.h"
Decoder *vork_decode_open(int input_fd, int width, int height) {
  Decoder *d = calloc(1, sizeof(*d));
  if (!d) {
    return NULL;
  }
  d->width = width;
  d->height = height;
  if (open_demux(&d->media, input_fd, open_mac_avformat()) < 0)
    goto fail;
#define FIND_AV(fn) if (!(d->fn = dlsym(d->media.format_lib, #fn))) goto fail;
  AVFUNCS(FIND_AV)
#undef FIND_AV
  AVStream *stream = d->media.input->streams[d->media.stream];
  const AVCodec *codec = hardware_decoder(d, stream->codecpar->codec_id);
  if (!codec || !(d->codec = d->avcodec_alloc_context3(codec)) ||
      d->avcodec_parameters_to_context(d->codec, stream->codecpar) < 0)
    goto fail;
  if (d->av_hwdevice_ctx_create(&d->codec->hw_device_ctx, HW_DEVICE, NULL, NULL, 0) < 0)
    goto fail;
  d->codec->get_format = select_hardware;
#if LIBAVUTIL_VERSION_MAJOR >= 60
  d->codec->flags |= AV_CODEC_FLAG_COPY_OPAQUE;
#endif
  if (d->avcodec_open2(d->codec, d->codec->codec, NULL) < 0)
    goto fail;
  d->frame = d->av_frame_alloc();
  if (!d->frame)
    goto fail;
  return d;
fail:
  vork_decode_close(d);
  return NULL;
}

int vork_decode_seek(Decoder *d, int64_t timestamp) {
  if (d->media.av_seek_frame(d->media.input, d->media.stream, timestamp, AVSEEK_FLAG_BACKWARD) <
      0)
    return -1;
  d->avcodec_flush_buffers(d->codec);
  d->media.av_packet_unref(d->media.packet);
  d->av_frame_unref(d->frame);
  d->media.draining = 0;
  return 0;
}

int vork_decode_next(Decoder *d, int64_t *position, int64_t *timestamp) {
  d->av_frame_unref(d->frame);
  for (;;) {
    int rc = d->avcodec_receive_frame(d->codec, d->frame);
    if (rc == 0) {
#if LIBAVUTIL_VERSION_MAJOR >= 60
      *position = -1;
      if (d->frame->opaque_ref && d->frame->opaque_ref->size == sizeof(*position))
        memcpy(position, d->frame->opaque_ref->data, sizeof(*position));
#else
      *position = d->frame->pkt_pos;
#endif
      *timestamp = d->frame->best_effort_timestamp;
      return 0;
    }
    if (rc != AVERROR(EAGAIN) || d->media.draining)
      return -1;
    rc = read_video_packet(&d->media);
    if (rc < 0 && rc != AVERROR_EOF)
      return -1;
    d->media.draining = rc == AVERROR_EOF;
#if LIBAVUTIL_VERSION_MAJOR >= 60
    if (!d->media.draining) {
      d->av_buffer_unref(&d->media.packet->opaque_ref);
      d->media.packet->opaque_ref = d->av_buffer_alloc(sizeof(d->media.packet->pos));
      if (!d->media.packet->opaque_ref)
        return -1;
      memcpy(d->media.packet->opaque_ref->data, &d->media.packet->pos, sizeof(d->media.packet->pos));
    }
#endif
    if (d->avcodec_send_packet(d->codec, d->media.draining ? NULL : d->media.packet) < 0)
      return -1;
  }
}

// Borrowed until the next decoder call. The Metal conversion completes before returning.
CVPixelBufferRef vork_decode_pixels(Decoder *d, int colorspace) {
  AVFrame *f = d->frame;
#define INVALID_FRAME NULL
  if (f->format != HW_FORMAT || !f->hw_frames_ctx ||
      ((AVHWFramesContext *)f->hw_frames_ctx->data)->sw_format !=
          AV_PIX_FMT_NV12 ||
      f->width != d->width || f->height != d->height ||
#if LIBAVUTIL_VERSION_MAJOR >= 59
      (f->flags & AV_FRAME_FLAG_INTERLACED) ||
#else
      f->interlaced_frame ||
#endif
      (f->color_range != AVCOL_RANGE_UNSPECIFIED &&
       f->color_range != AVCOL_RANGE_MPEG) ||
      (f->chroma_location != AVCHROMA_LOC_UNSPECIFIED &&
       f->chroma_location != AVCHROMA_LOC_LEFT) ||
      f->crop_top || f->crop_bottom || f->crop_left || f->crop_right)
    return INVALID_FRAME;
  for (int i = 0; i < f->nb_side_data; i++)
    if (f->side_data[i]->type == AV_FRAME_DATA_DISPLAYMATRIX)
      return INVALID_FRAME;
  if (!nv12_supported_matrix(colorspace))
    return INVALID_FRAME;
  CVPixelBufferRef pixels = (CVPixelBufferRef)f->data[3];
  if (!pixels || CVPixelBufferGetPixelFormatType(pixels) != kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ||
      !CVPixelBufferGetIOSurface(pixels) || CVPixelBufferGetPlaneCount(pixels) != 2 ||
      CVPixelBufferGetWidth(pixels) != d->width || CVPixelBufferGetHeight(pixels) != d->height)
    return NULL;
  // Metal reads both planes from (0, 0). The clean aperture is in encoded
  // pixels, independent of the presentation size for non-square pixels.
  CGRect visible = CVImageBufferGetCleanRect(pixels);
  if (visible.origin.x != 0 || visible.origin.y != 0 ||
      visible.size.width != d->width || visible.size.height != d->height)
    return NULL;
  return pixels;
}
