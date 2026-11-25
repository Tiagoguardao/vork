// libavformat supplies complete packets. The driver parser calls picture() in
// decode order and display() in presentation order on this reader's thread.
#include "demux.h"
#include <libavcodec/bsf.h>
#include "../../../vork-encode/native/nvcodec/dynlink_cuda.h"
#include "../../../vork-encode/native/nvcodec/dynlink_cuviddec.h"
#include "../../../vork-encode/native/nvcodec/dynlink_nvcuvid.h"

#define BSF_FUNCS(F) F(av_bsf_get_by_name) F(av_bsf_alloc) F(av_bsf_init) \
  F(av_bsf_send_packet) F(av_bsf_receive_packet) F(av_bsf_flush) F(av_bsf_free) \
  F(avcodec_parameters_copy) F(av_parser_init) F(av_parser_parse2) F(av_parser_close) \
  F(avcodec_alloc_context3) F(avcodec_free_context) F(avcodec_parameters_to_context)
#define CUDA_FUNCS(F) F(Init) F(DeviceGet) F(DeviceGetCount) F(DeviceGetUuid) \
  F(DevicePrimaryCtxGetState) F(DevicePrimaryCtxSetFlags) \
  F(DevicePrimaryCtxRetain) F(DevicePrimaryCtxRelease) F(CtxPushCurrent_v2) \
  F(CtxPopCurrent_v2) F(ImportExternalMemory) F(ExternalMemoryGetMappedBuffer) \
  F(DestroyExternalMemory) F(MemFree_v2) F(ModuleLoadData) F(ModuleGetFunction) \
  F(ModuleUnload) F(LaunchKernel) F(StreamSynchronize) F(StreamCreate) F(StreamDestroy_v2) \
  F(MemAlloc_v2) F(MemcpyDtoH_v2)
#define VIDEO_FUNCS(F) F(CreateVideoParser) F(ParseVideoData) F(DestroyVideoParser) \
  F(CreateDecoder) F(DestroyDecoder) F(DecodePicture) F(MapVideoFrame64) F(UnmapVideoFrame64)

typedef struct {
  int index;
  int64_t timestamp, position;
} DisplayedPicture;

typedef struct {
  int64_t position, pts;
  int discard;
} PacketIdentity;

typedef struct {
  Demux media;
  BSF_FUNCS(AV_FIELD)
#define CU_FIELD(fn) tcu##fn *fn;
  CUDA_FUNCS(CU_FIELD)
#define VIDEO_FIELD(fn) tcuvid##fn *fn;
  VIDEO_FUNCS(VIDEO_FIELD)
  tcuvidGetDecodeStatus *GetDecodeStatus;
  void *cuda_lib, *video_lib;
  CUdevice device;
  CUcontext context;
  CUstream stream;
  CUexternalMemory memory;
  CUdeviceptr pixels;
  CUmodule module;
  CUfunction kernel;
  int width, height, pitch, ten_bit;

  AVBSFContext *filter;
  cudaVideoCodec codec;
  CUvideoparser parser;
  CUvideodecoder decoder;
  unsigned surfaces, coded_width, coded_height;
  int failed, submitted_visible;
  // The demuxer splits container samples. Byte-stream codecs outside a
  // sample-framed container need an access-unit parser before ENDOFPICTURE.
  AVCodecParserContext *units;
  AVCodecContext *parameters;
  int packet_offset, packet_eof;
  PacketIdentity packet, *surface_packets;
  int parser_discard[AV_PARSER_PTS_NB], seen_discard;
  DisplayedPicture queue[32]; // At most one parser DPB at end-of-stream.
  unsigned queued, next;
  int current, mapped_index;
  unsigned long long mapped;
  unsigned stride;
  unsigned retries;
} Decoder;

static int unmap(Decoder *d) {
  if (!d->mapped)
    return 0;
  if (d->UnmapVideoFrame64(d->decoder, d->mapped))
    return -1;
  d->mapped = 0;
  return 0;
}

static int map(Decoder *d, int index) {
  if (d->mapped && d->mapped_index == index)
    return 0;
  if (unmap(d))
    return -1;
  CUVIDPROCPARAMS params = {.progressive_frame = 1, .output_stream = d->stream};
  unsigned long long address;
  unsigned stride;
  if (d->MapVideoFrame64(d->decoder, index, &address, &stride, &params))
    return -1;
  d->mapped = address;
  d->stride = stride;
  d->mapped_index = index;
  return 0;
}

static void reset_packet_identities(Decoder *d) {
  for (unsigned i = 0; i < d->surfaces; i++)
    d->surface_packets[i] = (PacketIdentity){.position = -1, .pts = AV_NOPTS_VALUE};
}

static int CUDAAPI sequence(void *opaque, CUVIDEOFORMAT *f) {
  Decoder *d = opaque;
  if (d->failed || f->codec != d->codec || f->chroma_format != cudaVideoChromaFormat_420 ||
      f->bit_depth_luma_minus8 != f->bit_depth_chroma_minus8 ||
      (f->bit_depth_luma_minus8 && (f->codec != cudaVideoCodec_HEVC || f->bit_depth_luma_minus8 != 2)) ||
      f->video_signal_description.video_full_range_flag ||
      f->display_area.left || f->display_area.top ||
      f->display_area.right != d->width || f->display_area.bottom != d->height)
    goto fail;
  if (d->decoder) {
    if (f->coded_width != d->coded_width || f->coded_height != d->coded_height ||
        f->min_num_decode_surfaces > d->surfaces || (f->bit_depth_luma_minus8 == 2) != d->ten_bit)
      goto fail;
    return d->surfaces;
  }
  d->ten_bit = f->bit_depth_luma_minus8 == 2;
  d->surfaces = f->min_num_decode_surfaces;
  d->coded_width = f->coded_width;
  d->coded_height = f->coded_height;
  CUVIDDECODECREATEINFO info = {
    .CodecType = d->codec, .ChromaFormat = f->chroma_format,
    .bitDepthMinus8 = f->bit_depth_luma_minus8,
    .OutputFormat = d->ten_bit ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12,
    .DeinterlaceMode = cudaVideoDeinterlaceMode_Weave,
    .ulWidth = f->coded_width, .ulHeight = f->coded_height,
    .ulTargetWidth = f->coded_width, .ulTargetHeight = f->coded_height,
    .ulNumOutputSurfaces = 1, .ulNumDecodeSurfaces = d->surfaces,
  };
  d->surface_packets = malloc(d->surfaces * sizeof(*d->surface_packets));
  if (!d->surface_packets)
    goto fail;
  reset_packet_identities(d);
  if (d->CreateDecoder(&d->decoder, &info))
    goto fail;
  return d->surfaces;
fail:
  d->failed = 1;
  return 0;
}

enum { RETRIES_EXHAUSTED = 2 };

static int CUDAAPI picture(void *opaque, CUVIDPICPARAMS *p) {
  Decoder *d = opaque;
  if (d->failed)
    return 0;
  if ((unsigned)p->CurrPicIdx >= d->surfaces || p->field_pic_flag || unmap(d))
    goto fail;
  // The parser can emit several displays in one call. Never overwrite an
  // unread surface if an unusual multi-picture packet recycles its index.
  for (unsigned i = d->next; i < d->queued; i++)
    if (d->queue[i].index == p->CurrPicIdx)
      goto fail;
  d->surface_packets[p->CurrPicIdx] = d->packet;
  // Hidden AV1/VP9 decodes may precede a show-existing command in this
  // same packet. Only a newly displayed picture supplies a new display ID.
  if ((d->codec != cudaVideoCodec_AV1 || p->CodecSpecific.av1.show_frame) &&
      (d->codec != cudaVideoCodec_VP9 || p->CodecSpecific.vp9.showFrame))
    d->submitted_visible = 1;
  for (unsigned attempt = 0; attempt <= 2; attempt++) {
    if (d->DecodePicture(d->decoder, p))
      goto fail;
    if (!d->GetDecodeStatus)
      return 1;
    // Mapping waits for completion. Retain this map for conversion, including
    // the common I/P case where this picture is also the next display.
    if (map(d, p->CurrPicIdx))
      goto fail;
    CUVIDGETDECODESTATUS status = {0};
    CUresult rc = d->GetDecodeStatus(d->decoder, p->CurrPicIdx, &status);
    if (!rc && status.decodeStatus == cuvidDecodeStatus_InProgress) {
      // Some mappings can return with output work pending on our stream.
      // Wait once on this reader and re-query before admitting a reference.
      if (d->StreamSynchronize(d->stream))
        goto fail;
      rc = d->GetDecodeStatus(d->decoder, p->CurrPicIdx, &status);
    }
    if ((int)rc == 801 && d->codec != cudaVideoCodec_H264) { // Status unsupported for this codec.
      d->GetDecodeStatus = NULL;
      return 1;
    }
    if (rc)
      goto fail;
    if (status.decodeStatus == cuvidDecodeStatus_Success)
      return 1;
    if (status.decodeStatus != cuvidDecodeStatus_Error &&
        status.decodeStatus != cuvidDecodeStatus_Error_Concealed)
      goto fail;
    if (attempt == 2) {
      d->failed = RETRIES_EXHAUSTED;
      return 0;
    }
    if (unmap(d))
      goto fail;
    d->retries++;
    // The parser owns p, its slices and offsets until this callback returns.
    // No subsequent picture has seen this reference.
  }
fail:
  d->failed = 1;
  return 0;
}

static int CUDAAPI display(void *opaque, CUVIDPARSERDISPINFO *info) {
  Decoder *d = opaque;
  if (d->failed)
    return 0;
  if (!info->progressive_frame || (unsigned)info->picture_index >= d->surfaces) {
    d->failed = 1;
    return 0;
  }
  int index = info->picture_index;
  PacketIdentity packet = d->surface_packets[index];
  // AV1 and VP9 can display an existing reference without submitting a new
  // decode. Its display packet owns the new position.
  if (!d->submitted_visible && !d->media.draining &&
      (d->codec == cudaVideoCodec_AV1 || d->codec == cudaVideoCodec_VP9) &&
      d->packet.pts != AV_NOPTS_VALUE && info->timestamp == d->packet.pts * (__int128)d->media.input->streams[d->media.stream]->time_base.num) {
    packet = d->packet;
  }
  // Edit-list preroll still participates in decode and reference checking.
  // Its reordered display alone is omitted, including during parser drain.
  if (packet.discard)
    return 1;
  if (d->queued == sizeof(d->queue)/sizeof(d->queue[0])) {
    d->failed = 1;
    return 0;
  }
  d->queue[d->queued++] = (DisplayedPicture){index, packet.pts, packet.position};
  return 1;
}

static int open_parser(Decoder *d) {
  AVStream *stream = d->media.input->streams[d->media.stream];
  CUVIDPARSERPARAMS params = {.ulClockRate = stream->time_base.den, .CodecType = d->codec, .ulMaxNumDecodeSurfaces = 1,
    .pUserData = d, .pfnSequenceCallback = sequence, .pfnDecodePicture = picture,
    .pfnDisplayPicture = display};
  if (d->CreateVideoParser(&d->parser, &params))
    return -1;
  AVCodecParameters *par = d->filter ? d->filter->par_out : d->media.input->streams[d->media.stream]->codecpar;
  CUVIDSOURCEDATAPACKET header = {.payload = par->extradata, .payload_size = par->extradata_size};
  if (d->codec == cudaVideoCodec_AV1 && header.payload_size && (header.payload[0] & 0x80)) {
    if (header.payload_size < 4)
      return -1;
    header.payload += 4;
    header.payload_size -= 4;
  }
  return header.payload_size && (d->ParseVideoData(d->parser, &header) || d->failed) ? -1 : 0;
}

static int setup_codec(Decoder *d) {
  AVStream *s = d->media.input->streams[d->media.stream];
  const char *filter = NULL;
  switch (s->codecpar->codec_id) {
    case AV_CODEC_ID_H264: d->codec = cudaVideoCodec_H264; filter = "h264_mp4toannexb"; break;
    case AV_CODEC_ID_HEVC: d->codec = cudaVideoCodec_HEVC; filter = "hevc_mp4toannexb"; break;
    case AV_CODEC_ID_AV1: d->codec = cudaVideoCodec_AV1; break;
    case AV_CODEC_ID_VP8: d->codec = cudaVideoCodec_VP8; break;
    case AV_CODEC_ID_VP9: d->codec = cudaVideoCodec_VP9; break;
    case AV_CODEC_ID_MJPEG: d->codec = cudaVideoCodec_JPEG; break;
    case AV_CODEC_ID_MPEG1VIDEO: d->codec = cudaVideoCodec_MPEG1; break;
    case AV_CODEC_ID_MPEG2VIDEO: d->codec = cudaVideoCodec_MPEG2; break;
    case AV_CODEC_ID_MPEG4: d->codec = cudaVideoCodec_MPEG4; filter = "mpeg4_unpack_bframes"; break;
    case AV_CODEC_ID_VC1: case AV_CODEC_ID_WMV3: d->codec = cudaVideoCodec_VC1; break;
    default: return -1;
  }
  if (d->codec == cudaVideoCodec_H264 || d->codec == cudaVideoCodec_HEVC || d->codec == cudaVideoCodec_JPEG) {
    d->GetDecodeStatus = dlsym(d->video_lib, "cuvidGetDecodeStatus");
    if (!d->GetDecodeStatus && d->codec == cudaVideoCodec_H264)
      return -1;
  }
  // An MPEG-4 Part 2 packet can contain a packed P/B pair even in AVI.
  // Split with FFmpeg's codec parser. Its pos/pts follow the beginning of
  // each parsed access unit across the one-packet lookahead and EOF.
  const char *format = d->media.input->iformat->name;
  int samples = !strcmp(format, "mov,mp4,m4a,3gp,3g2,mj2") || !strcmp(format, "matroska,webm");
  if (d->codec == cudaVideoCodec_MPEG4 || !samples) {
    d->units = d->av_parser_init(s->codecpar->codec_id);
    d->parameters = d->avcodec_alloc_context3(NULL);
    if (!d->units || !d->parameters ||
        d->avcodec_parameters_to_context(d->parameters, s->codecpar) < 0)
      return -1;
  }
  if (filter) {
    const AVBitStreamFilter *bsf = d->av_bsf_get_by_name(filter);
    if (!bsf || d->av_bsf_alloc(bsf, &d->filter) < 0 ||
        d->avcodec_parameters_copy(d->filter->par_in, s->codecpar) < 0)
      return -1;
    d->filter->time_base_in = s->time_base;
    if (d->av_bsf_init(d->filter) < 0 ||
        (d->parameters && d->avcodec_parameters_to_context(d->parameters, d->filter->par_out) < 0))
      return -1;
  }
  return open_parser(d);
}

static int close_parser(Decoder *d) {
  int error = unmap(d);
  if (d->parser && d->DestroyVideoParser(d->parser))
    error = -1;
  d->parser = NULL;
  return error;
}

void vork_decode_close(Decoder *d) {
  if (!d)
    return;
  if (d->context) {
    if (!d->CtxPushCurrent_v2(d->context)) {
      close_parser(d);
      if (d->decoder)
        d->DestroyDecoder(d->decoder);
      if (d->pixels)
        d->MemFree_v2(d->pixels);
      if (d->memory)
        d->DestroyExternalMemory(d->memory);
      if (d->stream)
        d->StreamDestroy_v2(d->stream);
      if (d->module)
        d->ModuleUnload(d->module);
      CUcontext popped;
      d->CtxPopCurrent_v2(&popped);
    }
    d->DevicePrimaryCtxRelease(d->device);
  }
  // Host metadata remains ours even if the CUDA context was lost.
  free(d->surface_packets);
  if (d->retries)
    fprintf(stderr, "vork: NVDEC resubmitted %u picture(s) after decode errors\n", d->retries);
  if (d->units)
    d->av_parser_close(d->units);
  if (d->parameters)
    d->avcodec_free_context(&d->parameters);
  if (d->filter)
    d->av_bsf_free(&d->filter);
  if (d->video_lib)
    dlclose(d->video_lib);
  if (d->cuda_lib)
    dlclose(d->cuda_lib);
  close_demux(&d->media);
  free(d);
}

Decoder *vork_decode_open(int input_fd, int fd, uint64_t bytes, const unsigned char *uuid,
                          int width, int height, int pitch, const char *ptx) {
  Decoder *d = calloc(1, sizeof(*d));
  if (!d)
    goto fail;
  if (open_demux(&d->media, input_fd, dlopen("libavformat.so." STR(LIBAVFORMAT_VERSION_MAJOR), RTLD_NOW | RTLD_LOCAL)))
    goto fail;
  d->width = width;
  d->height = height;
  d->pitch = pitch;
  d->cuda_lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  d->video_lib = dlopen("libnvcuvid.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!d->cuda_lib || !d->video_lib)
    goto fail;
#define FIND_BSF(fn) if (!(d->fn = dlsym(d->media.format_lib, #fn))) goto fail;
#define FIND_CU(fn) if (!(d->fn = dlsym(d->cuda_lib, "cu" #fn))) goto fail;
#define FIND_VIDEO(fn) if (!(d->fn = dlsym(d->video_lib, "cuvid" #fn))) goto fail;
  BSF_FUNCS(FIND_BSF)
  CUDA_FUNCS(FIND_CU)
  VIDEO_FUNCS(FIND_VIDEO)
  int count;
  if (d->Init(0) || d->DeviceGetCount(&count))
    goto fail;
  int ordinal;
  for (ordinal = 0; ordinal < count; ordinal++) {
    CUuuid id;
    if (!d->DeviceGet(&d->device, ordinal) && !d->DeviceGetUuid(&id, d->device) &&
        !memcmp(uuid, id.bytes, sizeof(id.bytes)))
      break;
  }
  unsigned flags;
  int active;
  if (ordinal == count || d->DevicePrimaryCtxGetState(d->device, &flags, &active))
    goto fail;
  // Blocking sync is a scheduling preference. Retain the usable context even
  // if the driver refuses the hint (for example, a racing reader activated it).
  if (!active && flags != CU_CTX_SCHED_BLOCKING_SYNC)
    (void)d->DevicePrimaryCtxSetFlags(d->device, CU_CTX_SCHED_BLOCKING_SYNC);
  if (d->DevicePrimaryCtxRetain(&d->context, d->device) ||
      d->CtxPushCurrent_v2(d->context))
    goto fail;
  int error = setup_codec(d);
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC desc = {.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
    .handle.fd = fd, .size = bytes, .flags = 1};
  int deep = fd < 0;
  if (!error && deep)
    error = d->MemAlloc_v2(&d->pixels, bytes);
  else if (!error && !(error = d->ImportExternalMemory(&d->memory, &desc))) {
    fd = -1;
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC buffer = {.size = (uint64_t)pitch * height};
    error = d->ExternalMemoryGetMappedBuffer(&d->pixels, d->memory, &buffer);
  }
  if (!error)
    error = d->StreamCreate(&d->stream, CU_STREAM_NON_BLOCKING) ||
            d->ModuleLoadData(&d->module, ptx) ||
            d->ModuleGetFunction(&d->kernel, d->module, deep ? "convert16" : "convert");
  CUcontext popped;
  d->CtxPopCurrent_v2(&popped);
  if (!error)
    return d;
fail:
  if (fd >= 0)
    close(fd);
  vork_decode_close(d);
  return NULL;
}

int vork_decode_seek(Decoder *d, int64_t timestamp) {
  Demux *m = &d->media;
  if (d->failed || m->av_seek_frame(m->input, m->stream, timestamp, AVSEEK_FLAG_BACKWARD) < 0 ||
      d->CtxPushCurrent_v2(d->context))
    return -1;
  int rc = close_parser(d);
  if (d->filter)
    d->av_bsf_flush(d->filter);
  m->av_packet_unref(m->packet);
  if (d->units) {
    d->av_parser_close(d->units);
    d->units = d->av_parser_init(m->input->streams[m->stream]->codecpar->codec_id);
    if (!d->units)
      rc = -1;
  }
  d->packet = (PacketIdentity){.position = -1, .pts = AV_NOPTS_VALUE};
  d->seen_discard = 0;
  reset_packet_identities(d);
  m->draining = d->packet_offset = d->packet_eof = d->next = d->queued = d->submitted_visible = 0;
  if (!rc)
    rc = open_parser(d);
  CUcontext popped;
  d->CtxPopCurrent_v2(&popped);
  return rc;
}

static int feed(Decoder *d) {
  Demux *m = &d->media;
  AVPacket *p = m->packet;
  if (m->draining)
    return -1;
  CUVIDSOURCEDATAPACKET data = {0};
  while (!data.payload_size) {
    if (d->packet_offset == p->size && !d->packet_eof) {
      int rc = read_video_packet(m);
      if (rc == AVERROR_EOF) {
        d->packet_eof = 1;
      } else if (rc < 0 || p->flags & AV_PKT_FLAG_CORRUPT) {
        return -1;
      } else if (d->filter && (d->av_bsf_send_packet(d->filter, p) < 0 ||
                                d->av_bsf_receive_packet(d->filter, p) < 0)) {
        return -1;
      }
      d->packet_offset = 0;
    }
    if (d->units) {
      uint8_t *unit;
      int size;
      int used = d->av_parser_parse2(d->units, d->parameters, &unit, &size,
        p->size ? p->data + d->packet_offset : NULL, p->size - d->packet_offset,
        p->pts, p->dts, p->pos);
      if (used < 0 || (!used && !size && !d->packet_eof))
        return -1;
      // The parser can emit the previous unit while consuming the next
      // packet. Keep flags on the same byte ranges it uses for PTS and pos.
      if (p->size) {
        d->parser_discard[d->units->cur_frame_start_index] = !!(p->flags & AV_PKT_FLAG_DISCARD);
        d->seen_discard |= !!(p->flags & AV_PKT_FLAG_DISCARD);
      }
      d->packet.discard = 0;
      if (size && d->seen_discard) {
        int i;
        for (i = 0; i < AV_PARSER_PTS_NB; i++) {
          if (d->units->cur_frame_offset[i] <= d->units->frame_offset &&
              d->units->frame_offset < d->units->cur_frame_end[i]) {
            d->packet.discard = d->parser_discard[i];
            break;
          }
        }
        if (i == AV_PARSER_PTS_NB)
          return -1;
      }
      d->packet_offset += used;
      data.payload = unit;
      data.payload_size = size;
      d->packet.pts = d->units->pts;
      d->packet.position = d->units->pos;
    } else {
      data.payload = p->data;
      data.payload_size = p->size;
      d->packet_offset = p->size;
      d->packet.pts = p->pts;
      d->packet.position = p->pos;
      d->packet.discard = !!(p->flags & AV_PKT_FLAG_DISCARD);
    }
    if (!data.payload_size && d->packet_eof) {
      m->draining = 1;
      data.flags = CUVID_PKT_ENDOFSTREAM;
      break;
    }
  }
  if (data.payload_size) {
    // A VP9 superframe or AV1 temporal unit can carry several coded frames.
    // These parsers process their own internal boundaries.
    if (d->codec != cudaVideoCodec_VP9 && d->codec != cudaVideoCodec_AV1)
      data.flags |= CUVID_PKT_ENDOFPICTURE;
    int64_t ticks;
    if (d->packet.pts != AV_NOPTS_VALUE && !__builtin_mul_overflow(d->packet.pts,
          m->input->streams[m->stream]->time_base.num, &ticks)) {
      data.flags |= CUVID_PKT_TIMESTAMP;
      data.timestamp = ticks;
    }
  }
  d->submitted_visible = 0;
  return d->ParseVideoData(d->parser, &data) || d->failed ? -1 : 0;
}

int vork_decode_next(Decoder *d, int64_t *position, int64_t *timestamp) {
  if (d->CtxPushCurrent_v2(d->context))
    return -1;
  int rc = 0;
  while (d->next == d->queued && !rc) {
    d->next = d->queued = 0;
    rc = feed(d);
  }
  if (!rc) {
    DisplayedPicture *pic = &d->queue[d->next++];
    d->current = pic->index;
    *position = pic->position;
    *timestamp = pic->timestamp;
  }
  CUcontext popped;
  d->CtxPopCurrent_v2(&popped);
  return d->failed == RETRIES_EXHAUSTED ? -2 : rc;
}

int vork_decode_convert(Decoder *d, int colorspace, void *host) {
  if (!nv12_supported_matrix(colorspace) || d->CtxPushCurrent_v2(d->context))
    return -1;
  int rc = map(d, d->current);
  if (!rc) {
    unsigned long long uv = d->mapped + (uint64_t)d->stride * d->coded_height;
    void *args[] = {&d->mapped, &uv, &d->stride, &d->stride, &d->ten_bit, &d->width, &d->height,
                    &d->pixels, &d->pitch, &colorspace};
    rc = d->LaunchKernel(d->kernel, (d->width / 2 + 15) / 16, (d->height + 15) / 16,
                         1, 16, 16, 1, 0, d->stream, args, NULL) || d->StreamSynchronize(d->stream);
    if (!rc && host)
      rc = d->MemcpyDtoH_v2(host, d->pixels, (size_t)d->pitch * d->height);
    if (unmap(d))
      rc = -1;
  }
  CUcontext popped;
  d->CtxPopCurrent_v2(&popped);
  return rc;
}
