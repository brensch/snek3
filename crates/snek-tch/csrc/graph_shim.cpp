// Minimal C ABI over libtorch's at::cuda::CUDAGraph + CUDA stream helpers.
// tch (0.24) does not expose CUDA graph capture, so we bridge just the five
// calls we need: make a side stream, set it current, capture, replay, sync.
//
// Capture must run on a non-default stream, so callers create one here and set
// it current before doing any warmup / capture / replay work.

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdio>
#include <cstdlib>
#include <exception>

#define SNEK_GUARD(body)                                                       \
  try {                                                                        \
    body                                                                       \
  } catch (const std::exception &e) {                                          \
    std::fprintf(stderr, "snek_graph_shim: %s\n", e.what());                   \
    std::abort();                                                              \
  }

using at::cuda::CUDAGraph;
using c10::cuda::CUDAStream;

extern "C" {

void *snek_stream_new() {
  SNEK_GUARD(return new CUDAStream(c10::cuda::getStreamFromPool(false));)
}
void snek_stream_set_current(void *s) {
  SNEK_GUARD(c10::cuda::setCurrentCUDAStream(*static_cast<CUDAStream *>(s));)
}
void snek_stream_sync(void *s) {
  SNEK_GUARD(static_cast<CUDAStream *>(s)->synchronize();)
}
void snek_stream_free(void *s) { delete static_cast<CUDAStream *>(s); }

void *snek_graph_new() { SNEK_GUARD(return new CUDAGraph();) }
void snek_graph_capture_begin(void *g) {
  SNEK_GUARD(static_cast<CUDAGraph *>(g)->capture_begin();)
}
void snek_graph_capture_end(void *g) {
  // capture_end() instantiates the executable graph (keep_graph defaults false).
  SNEK_GUARD(static_cast<CUDAGraph *>(g)->capture_end();)
}
void snek_graph_replay(void *g) {
  SNEK_GUARD(static_cast<CUDAGraph *>(g)->replay();)
}
void snek_graph_free(void *g) { delete static_cast<CUDAGraph *>(g); }

} // extern "C"
