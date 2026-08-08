// Pure-C ABI wrapper around the single-header C++ Zimtohrli implementation.
//
// Everything in zimtohrli.h lives in an anonymous namespace inside
// `namespace zimtohrli`, so all symbols have internal linkage and this file
// must be the only translation unit that includes it.

#include <cstddef>
#include <cstring>

#include "zimt/mos.h"
#include "zimt/zimtohrli.h"

using namespace zimtohrli;

extern "C" {

void* zimt_cpp_new(float perceptual_sample_rate, float full_scale_sine_db) {
  Zimtohrli* z = new Zimtohrli{};
  z->perceptual_sample_rate = perceptual_sample_rate;
  z->full_scale_sine_db = full_scale_sine_db;
  return z;
}

void zimt_cpp_free(void* z) { delete static_cast<Zimtohrli*>(z); }

void* zimt_cpp_analyze(const void* z, const float* signal, size_t len) {
  return new Spectrogram(static_cast<const Zimtohrli*>(z)->Analyze(
      Span<const float>(signal, len)));
}

void* zimt_cpp_spec_clone(const void* s) {
  const Spectrogram* src = static_cast<const Spectrogram*>(s);
  Spectrogram* dst = new Spectrogram(src->num_steps, src->num_dims);
  std::memcpy(dst->values.get(), src->values.get(),
              src->num_steps * src->num_dims * sizeof(float));
  return dst;
}

void zimt_cpp_spec_free(void* s) { delete static_cast<Spectrogram*>(s); }

size_t zimt_cpp_spec_steps(const void* s) {
  return static_cast<const Spectrogram*>(s)->num_steps;
}

size_t zimt_cpp_spec_dims(const void* s) {
  return static_cast<const Spectrogram*>(s)->num_dims;
}

const float* zimt_cpp_spec_values(const void* s) {
  return static_cast<const Spectrogram*>(s)->values.get();
}

// Note: Distance/DistanceWithoutDtw rescale both spectrograms in place.

float zimt_cpp_distance(const void* z, void* a, void* b) {
  return static_cast<const Zimtohrli*>(z)->Distance(
      *static_cast<Spectrogram*>(a), *static_cast<Spectrogram*>(b));
}

float zimt_cpp_distance_without_dtw(const void* z, void* a, void* b) {
  return static_cast<const Zimtohrli*>(z)->DistanceWithoutDtw(
      *static_cast<Spectrogram*>(a), *static_cast<Spectrogram*>(b));
}

float zimt_cpp_mos_from_distance(float distance) {
  return MOSFromZimtohrli(distance);
}

}  // extern "C"
