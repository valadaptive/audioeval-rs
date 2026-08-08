//! FFI bindings to the original C++ Zimtohrli implementation
//! (`zimtohrli/cpp/zimt/zimtohrli.h`), compiled by the build script via `cc`
//! and reached through the pure-C ABI in `wrapper.cc`.
//!
//! Exists to benchmark and parity-check the Rust `zimtohrli` crate against
//! the reference implementation, so the API mirrors it (`analyze`,
//! `distance`, `distance_without_dtw`, `mos_from_distance`).
//!
//! Like the C++ original, [`CppZimtohrli::distance`] and
//! [`CppZimtohrli::distance_without_dtw`] rescale both spectrograms in place,
//! so spectrograms must be re-analyzed or cloned before reuse.

use std::ffi::c_void;

mod ffi {
    use std::ffi::c_void;

    unsafe extern "C" {
        pub(crate) fn zimt_cpp_new(
            perceptual_sample_rate: f32,
            full_scale_sine_db: f32,
        ) -> *mut c_void;
        pub(crate) fn zimt_cpp_free(z: *mut c_void);
        pub(crate) fn zimt_cpp_analyze(
            z: *const c_void,
            signal: *const f32,
            len: usize,
        ) -> *mut c_void;
        pub(crate) fn zimt_cpp_spec_clone(spec: *const c_void) -> *mut c_void;
        pub(crate) fn zimt_cpp_spec_free(spec: *mut c_void);
        pub(crate) fn zimt_cpp_spec_steps(spec: *const c_void) -> usize;
        pub(crate) fn zimt_cpp_spec_dims(spec: *const c_void) -> usize;
        pub(crate) fn zimt_cpp_spec_values(spec: *const c_void) -> *const f32;
        pub(crate) fn zimt_cpp_distance(z: *const c_void, a: *mut c_void, b: *mut c_void) -> f32;
        pub(crate) fn zimt_cpp_distance_without_dtw(
            z: *const c_void,
            a: *mut c_void,
            b: *mut c_void,
        ) -> f32;
        pub(crate) fn zimt_cpp_mos_from_distance(distance: f32) -> f32;
    }
}

/// The original C++ `zimtohrli::Zimtohrli` metric.
pub struct CppZimtohrli {
    ptr: *mut c_void,
}

impl CppZimtohrli {
    /// Creates a metric with the given perceptual sample rate (Hz) and
    /// reference dB SPL for a full-scale sine.
    pub fn new(perceptual_sample_rate: f32, full_scale_sine_db: f32) -> Self {
        // SAFETY: returns an owned pointer (or null on OOM, checked below).
        let ptr = unsafe { ffi::zimt_cpp_new(perceptual_sample_rate, full_scale_sine_db) };
        assert!(!ptr.is_null());
        Self { ptr }
    }

    /// Analyzes a 48kHz mono signal (samples in [-1, 1]) into a perceptual
    /// spectrogram.
    pub fn analyze(&self, signal: &[f32]) -> CppSpectrogram {
        // SAFETY: `signal` is valid for `signal.len()` floats; the returned
        // pointer is owned.
        let ptr = unsafe { ffi::zimt_cpp_analyze(self.ptr, signal.as_ptr(), signal.len()) };
        assert!(!ptr.is_null());
        CppSpectrogram { ptr }
    }

    /// Perceptual distance between two spectrograms, with exhaustive DTW time
    /// alignment. Rescales both spectrograms in place.
    pub fn distance(&self, spec_a: &mut CppSpectrogram, spec_b: &mut CppSpectrogram) -> f32 {
        // SAFETY: both spectrograms are valid, exclusively borrowed objects.
        unsafe { ffi::zimt_cpp_distance(self.ptr, spec_a.ptr, spec_b.ptr) }
    }

    /// Perceptual distance between two already time-aligned spectrograms of
    /// equal length. Rescales both spectrograms in place.
    pub fn distance_without_dtw(
        &self,
        spec_a: &mut CppSpectrogram,
        spec_b: &mut CppSpectrogram,
    ) -> f32 {
        // SAFETY: both spectrograms are valid, exclusively borrowed objects.
        unsafe { ffi::zimt_cpp_distance_without_dtw(self.ptr, spec_a.ptr, spec_b.ptr) }
    }

    /// Returns a _very approximate_ mean opinion score for a Zimtohrli
    /// distance.
    pub fn mos_from_distance(distance: f32) -> f32 {
        // SAFETY: pure function.
        unsafe { ffi::zimt_cpp_mos_from_distance(distance) }
    }
}

impl Default for CppZimtohrli {
    /// Matches the C++ (and Rust crate) defaults: `full_scale_sine_db = 78.3`,
    /// `perceptual_sample_rate = 48000 / floor(48000 / 85) ≈ 85.106`.
    fn default() -> Self {
        let samples_per_perceptual_block = (48_000.0f32 / 85.0) as usize;
        Self::new(48_000.0 / samples_per_perceptual_block as f32, 78.3)
    }
}

impl Drop for CppZimtohrli {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was created by `zimt_cpp_new` and is freed once.
        unsafe { ffi::zimt_cpp_free(self.ptr) }
    }
}

// SAFETY: instances are independently owned; the C++ code holds no global
// mutable state (`Analyze` is const, `Distance*` mutates only the
// spectrograms passed to it).
unsafe impl Send for CppZimtohrli {}
unsafe impl Sync for CppZimtohrli {}

/// A C++ `zimtohrli::Spectrogram`: row-major `[num_steps][num_dims]` floats.
pub struct CppSpectrogram {
    ptr: *mut c_void,
}

impl CppSpectrogram {
    pub fn num_steps(&self) -> usize {
        // SAFETY: `self.ptr` is a valid Spectrogram.
        unsafe { ffi::zimt_cpp_spec_steps(self.ptr) }
    }

    pub fn num_dims(&self) -> usize {
        // SAFETY: `self.ptr` is a valid Spectrogram.
        unsafe { ffi::zimt_cpp_spec_dims(self.ptr) }
    }

    /// Row-major view of the `num_steps * num_dims` spectrogram values.
    pub fn values(&self) -> &[f32] {
        // SAFETY: the C++ buffer holds exactly num_steps * num_dims floats
        // and outlives `&self`.
        unsafe {
            std::slice::from_raw_parts(
                ffi::zimt_cpp_spec_values(self.ptr),
                self.num_steps() * self.num_dims(),
            )
        }
    }
}

impl Clone for CppSpectrogram {
    fn clone(&self) -> Self {
        // SAFETY: returns an owned deep copy.
        let ptr = unsafe { ffi::zimt_cpp_spec_clone(self.ptr) };
        assert!(!ptr.is_null());
        Self { ptr }
    }
}

impl Drop for CppSpectrogram {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was created by analyze/clone and is freed once.
        unsafe { ffi::zimt_cpp_spec_free(self.ptr) }
    }
}

// SAFETY: a CppSpectrogram is an independently owned buffer with no interior
// mutability reachable through `&self`.
unsafe impl Send for CppSpectrogram {}
unsafe impl Sync for CppSpectrogram {}
