// Stable Homebrew opt links survive patch upgrades, including versioned kegs.
static void *open_mac_avformat(void) {
#define AV_DYLIB "/libavformat." STR(LIBAVFORMAT_VERSION_MAJOR) ".dylib"
#if defined(VORK_LOCAL_AV_LIBDIR) && !defined(VORK_DIST_AVFORMAT_MAJOR)
  void *local = dlopen(VORK_LOCAL_AV_LIBDIR AV_DYLIB, RTLD_NOW | RTLD_LOCAL);
  if (local) return local;
#endif
  const char *prefixes[] = {"/opt/homebrew", "/usr/local"};
  for (unsigned i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); ++i) {
    char versioned[96];
    // libavformat 60..63 corresponds to Homebrew ffmpeg@6..9.
    snprintf(versioned, sizeof(versioned), "%s/opt/ffmpeg@%d/lib" AV_DYLIB,
             prefixes[i], LIBAVFORMAT_VERSION_MAJOR - 54);
    char current[96], linked[96];
    snprintf(current, sizeof(current), "%s/opt/ffmpeg/lib" AV_DYLIB, prefixes[i]);
    snprintf(linked, sizeof(linked), "%s/lib" AV_DYLIB, prefixes[i]);
    const char *paths[] = {current, versioned, linked};
    for (unsigned j = 0; j < sizeof(paths) / sizeof(paths[0]); ++j) {
      void *lib = dlopen(paths[j], RTLD_NOW | RTLD_LOCAL);
      if (lib) return lib;
    }
  }
  return NULL;
}
