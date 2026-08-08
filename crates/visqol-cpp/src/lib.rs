//! FFI bindings to the original C++ ViSQOL implementation (the vendored
//! `visqol/` checkout), built as `libvisqol_capi.so` by bazel and reached
//! through the pure-C ABI in `visqol/src/visqol_capi.cc`.
//!
//! Exists to benchmark and parity-check the Rust `visqol` crate against the
//! reference implementation, so the API mirrors it: [`CppVisqol::audio`],
//! [`CppVisqol::speech_lattice`] and [`CppVisqol::speech_legacy`] construct
//! the default metric of each mode and [`CppVisqol::run`] compares two mono
//! signals.

use std::ffi::{CString, c_char, c_void};
use std::fmt::{self, Display};
use std::path::{Path, PathBuf};

#[repr(C)]
struct VisqolCppOptions {
    use_speech_mode: i32,
    use_unscaled_speech_mos_mapping: i32,
    use_lattice_model: i32,
    search_window_radius: i32,
    disable_global_alignment: i32,
    disable_realignment: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CppSimilarityResult {
    pub moslqo: f64,
    pub vnsim: f64,
}

mod ffi {
    use super::{CppSimilarityResult, VisqolCppOptions};
    use std::ffi::{c_char, c_void};

    unsafe extern "C" {
        pub(crate) fn visqol_cpp_create(
            model_path: *const c_char,
            options: VisqolCppOptions,
            error_out: *mut *mut c_char,
        ) -> *mut c_void;
        pub(crate) fn visqol_cpp_run(
            visqol: *mut c_void,
            ref_samples: *const f64,
            ref_len: usize,
            deg_samples: *const f64,
            deg_len: usize,
            sample_rate: usize,
            result_out: *mut CppSimilarityResult,
            error_out: *mut *mut c_char,
        ) -> i32;
        pub(crate) fn visqol_cpp_destroy(visqol: *mut c_void);
        pub(crate) fn visqol_cpp_free_string(s: *mut c_char);
    }
}

/// An error reported by the C++ ViSQOL implementation (via `absl::Status`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CppVisqolError(String);

impl Display for CppVisqolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CppVisqolError {}

/// Options for the C++ metric, mirroring `VisqolManager::Init`. The default
/// is audio mode: SVR quality model, no speech scoring, search window radius
/// 60 (the C++ default), global alignment and realignment enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CppVisqolOptions {
    pub use_speech_mode: bool,
    pub use_unscaled_speech_mos_mapping: bool,
    pub use_lattice_model: bool,
    pub search_window_radius: i32,
    pub disable_global_alignment: bool,
    pub disable_realignment: bool,
}

impl Default for CppVisqolOptions {
    fn default() -> Self {
        Self {
            use_speech_mode: false,
            use_unscaled_speech_mos_mapping: false,
            use_lattice_model: false,
            search_window_radius: 60,
            disable_global_alignment: false,
            disable_realignment: false,
        }
    }
}

/// The path to the default audio-mode SVR model in the vendored checkout.
pub fn default_svr_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../visqol/model/libsvm_nu_svr_model.txt")
}

/// The path to the default speech-mode lattice model in the vendored checkout
/// (the file the Rust crate's embedded lattice model was extracted from).
pub fn default_lattice_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../visqol/model/lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite",
    )
}

/// The original C++ `Visqol::VisqolManager` metric.
///
/// [`CppVisqol::run`] takes `&mut self` because the C++ object mutates
/// unguarded internal scratch state while running (in lattice mode
/// `TFLiteQualityMapper::PredictQuality` — a `const` method — writes input
/// tensors and calls `Invoke()` on the owned `tflite::Interpreter`).
pub struct CppVisqol {
    ptr: *mut c_void,
}

// SAFETY: no part of the C++ object has thread affinity (no thread-local
// state, a single-threaded interpreter, and `delete` is thread-agnostic), so
// sequential use from different threads is fine.
unsafe impl Send for CppVisqol {}
unsafe impl Sync for CppVisqol {}

impl CppVisqol {
    /// Creates a metric around the given similarity-to-quality mapper model
    /// file (e.g. [`default_svr_model_path`] for audio mode).
    pub fn new(model_path: &Path, options: CppVisqolOptions) -> Result<Self, CppVisqolError> {
        let model_path = CString::new(model_path.as_os_str().as_encoded_bytes())
            .map_err(|_| CppVisqolError("model path contains NUL".into()))?;
        let options = VisqolCppOptions {
            use_speech_mode: options.use_speech_mode as i32,
            use_unscaled_speech_mos_mapping: options.use_unscaled_speech_mos_mapping as i32,
            use_lattice_model: options.use_lattice_model as i32,
            search_window_radius: options.search_window_radius,
            disable_global_alignment: options.disable_global_alignment as i32,
            disable_realignment: options.disable_realignment as i32,
        };
        let mut error: *mut c_char = std::ptr::null_mut();
        // SAFETY: model_path is a valid NUL-terminated string; on failure a
        // null pointer is returned and `error` (if set) is taken ownership of
        // and freed below.
        let ptr = unsafe { ffi::visqol_cpp_create(model_path.as_ptr(), options, &mut error) };
        if ptr.is_null() {
            return Err(CppVisqolError(take_error(error)));
        }
        Ok(Self { ptr })
    }

    /// Creates a default audio-mode metric around the given SVR model file.
    pub fn audio(model_path: &Path) -> Result<Self, CppVisqolError> {
        Self::new(model_path, CppVisqolOptions::default())
    }

    /// Creates a default speech-mode metric (16kHz nominal) around the given
    /// deep lattice model file (e.g. [`default_lattice_model_path`]).
    pub fn speech_lattice(model_path: &Path) -> Result<Self, CppVisqolError> {
        Self::new(
            model_path,
            CppVisqolOptions {
                use_speech_mode: true,
                use_lattice_model: true,
                ..Default::default()
            },
        )
    }

    /// Creates a legacy speech-mode metric (16kHz nominal): VAD-gated patches
    /// and the exponential NSIM-to-MOS mapping (`--use_lattice_model=false`).
    ///
    /// The C++ implementation does not read a model file in this mode
    /// (`SpeechSimilarityToQualityMapper::Init` is a no-op), so unlike the
    /// other constructors this one takes no path.
    pub fn speech_legacy(use_unscaled_speech_mos_mapping: bool) -> Result<Self, CppVisqolError> {
        Self::new(
            Path::new(""),
            CppVisqolOptions {
                use_speech_mode: true,
                use_unscaled_speech_mos_mapping,
                ..Default::default()
            },
        )
    }

    /// Compares a reference and degraded mono signal (samples nominally in
    /// [-1, 1], both at `sample_rate` Hz; 48kHz required in audio mode).
    pub fn run(
        &mut self,
        reference: &[f64],
        degraded: &[f64],
        sample_rate: u32,
    ) -> Result<CppSimilarityResult, CppVisqolError> {
        let mut result = CppSimilarityResult {
            moslqo: 0.0,
            vnsim: 0.0,
        };
        let mut error: *mut c_char = std::ptr::null_mut();
        // SAFETY: both sample buffers are valid for their lengths; result_out
        // and error_out point to valid locals.
        let status = unsafe {
            ffi::visqol_cpp_run(
                self.ptr,
                reference.as_ptr(),
                reference.len(),
                degraded.as_ptr(),
                degraded.len(),
                sample_rate as usize,
                &mut result,
                &mut error,
            )
        };
        if status != 0 {
            return Err(CppVisqolError(take_error(error)));
        }
        Ok(result)
    }
}

impl Drop for CppVisqol {
    fn drop(&mut self) {
        // SAFETY: ptr was created by visqol_cpp_create and is destroyed once.
        unsafe { ffi::visqol_cpp_destroy(self.ptr) }
    }
}

/// Takes ownership of a heap-allocated C error string, if any.
fn take_error(error: *mut c_char) -> String {
    if error.is_null() {
        return "unknown error".into();
    }
    // SAFETY: error is a NUL-terminated string allocated by the library.
    let message = unsafe { std::ffi::CStr::from_ptr(error) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: error was allocated by the library's DupString (malloc) and is
    // freed by its own free function exactly once.
    unsafe { ffi::visqol_cpp_free_string(error) };
    message
}
