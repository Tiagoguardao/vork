// Shared libavformat session for the native footage readers.
#define NV12_METADATA_ONLY
#include "nv12.h"
#include <dlfcn.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/display.h>
#include <libavutil/hwcontext.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#define STR1(x) #x
#define STR(x) STR1(x)
#define DEMUX_FUNCS(F) F(av_freep) \
  F(av_malloc) \
  F(avio_alloc_context) \
  F(avio_context_free) \
  F(avformat_alloc_context) \
  F(avformat_open_input) \
  F(avformat_find_stream_info) \
  F(avformat_close_input) \
  F(av_read_frame) \
  F(av_seek_frame) \
  F(av_packet_alloc) \
  F(av_packet_unref) \
  F(av_packet_free)

#define AV_FIELD(fn) __typeof__(&fn) fn;
typedef struct {
  void *format_lib;
  DEMUX_FUNCS(AV_FIELD)
  int input_fd;
  int64_t offset;
  AVIOContext *io;
  AVFormatContext *input;
  AVPacket *packet;
  int stream, draining;
} Demux;

// dup() holds the original inode. Each libav session advances its own offset.
static int read_media(void *opaque, uint8_t *buf, int size) {
  Demux *d = opaque;
  ssize_t n;
  do {
    n = pread(d->input_fd, buf, size, d->offset);
  } while (n < 0 && errno == EINTR);
  if (n <= 0)
    return n == 0 ? AVERROR_EOF : AVERROR(errno);
  d->offset += n;
  return (int)n;
}

static int64_t seek_media(void *opaque, int64_t delta, int whence) {
  Demux *d = opaque;
  int64_t base = 0, position;
  whence &= ~AVSEEK_FORCE;
  if (whence == AVSEEK_SIZE || whence == SEEK_END) {
    struct stat st;
    if (fstat(d->input_fd, &st) < 0)
      return AVERROR(errno);
    if (whence == AVSEEK_SIZE)
      return st.st_size;
    base = st.st_size;
  } else if (whence == SEEK_CUR) {
    base = d->offset;
  } else if (whence != SEEK_SET) {
    return AVERROR(EINVAL);
  }
  if (__builtin_add_overflow(base, delta, &position) || position < 0)
    return AVERROR(EINVAL);
  return d->offset = position;
}

static int open_demux(Demux *d, int input_fd, void *format_lib) {
  d->format_lib = format_lib;
  d->input_fd = fcntl(input_fd, F_DUPFD_CLOEXEC, 0);
  if (d->input_fd < 0 || !format_lib)
    return -1;
#define FIND_AV(fn) if (!(d->fn = dlsym(format_lib, #fn))) return -1;
  DEMUX_FUNCS(FIND_AV)
#undef FIND_AV
  const int size = 32768;
  unsigned char *buffer = d->av_malloc(size);
  if (!buffer)
    return -1;
  d->io = d->avio_alloc_context(buffer, size, 0, d, read_media, NULL, seek_media);
  if (!d->io) {
    d->av_freep(&buffer);
    return -1;
  }
  d->input = d->avformat_alloc_context();
  if (!d->input)
    return -1;
  d->input->pb = d->io;
  d->input->flags |= AVFMT_FLAG_CUSTOM_IO;
  if (d->avformat_open_input(&d->input, NULL, NULL, NULL) < 0 ||
      d->avformat_find_stream_info(d->input, NULL) < 0)
    return -1;
  for (unsigned i = 0;; i++) {
    if (i == d->input->nb_streams)
      return -1;
    AVStream *stream = d->input->streams[i];
    if (stream->codecpar->codec_type != AVMEDIA_TYPE_VIDEO)
      continue;
    d->stream = i;
    // FFmpeg CLI autorotates. Defer any display matrix to that reader.
#if LIBAVCODEC_VERSION_MAJOR >= 61
    for (int j = 0; j < stream->codecpar->nb_coded_side_data; j++)
      if (stream->codecpar->coded_side_data[j].type == AV_PKT_DATA_DISPLAYMATRIX)
        return -1;
#else
    for (int j = 0; j < stream->nb_side_data; j++)
      if (stream->side_data[j].type == AV_PKT_DATA_DISPLAYMATRIX)
        return -1;
#endif

    d->packet = d->av_packet_alloc();
    return d->packet ? 0 : -1;
  }
}

static void close_demux(Demux *d) {
  if (d->packet)
    d->av_packet_free(&d->packet);
  if (d->input)
    d->avformat_close_input(&d->input);
  if (d->io) {
    d->av_freep(&d->io->buffer);
    d->avio_context_free(&d->io);
  }
  if (d->input_fd >= 0)
    close(d->input_fd);
  if (d->format_lib)
    dlclose(d->format_lib);
}

static int read_video_packet(Demux *d) {
  int rc;
  do {
    d->av_packet_unref(d->packet);
    rc = d->av_read_frame(d->input, d->packet);
  } while (rc == 0 && d->packet->stream_index != d->stream);
  return rc;
}

int vork_decode_supported_size(int width, int height) {
  return nv12_supported_size(width, height);
}
