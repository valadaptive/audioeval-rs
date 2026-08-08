//! Links the original C++ ViSQOL implementation as a shared library.
//!
//! Unlike Zimtohrli (a single header), ViSQOL is a full bazel project with
//! heavy dependencies (abseil, protobuf, TensorFlow Lite, libsvm, pffft), so
//! we do not compile it ourselves. Instead, the vendored checkout in
//! `visqol/` carries a small pure-C ABI wrapper (`visqol/src/visqol_capi.cc`)
//! with a `//:visqol_capi` bazel target that links everything into one
//! self-contained `libvisqol_capi.so`; this script builds it (incrementally)
//! and arranges linking.
//!
//! Environment overrides:
//! - `VISQOL_DIR`: path to the visqol checkout (default: `../../visqol`
//!   relative to this crate).
//! - `VISQOL_CAPI_LIB_DIR`: use a prebuilt `libvisqol_capi.so` from this
//!   directory and skip bazel entirely (e.g. one built with a different CC
//!   or `--output_base`).
//! - `BAZEL`: bazel binary to invoke (default: `bazelisk`).
//! - `CC`/`CXX`: respected by bazel's C++ toolchain autoconfiguration;
//!   changing them triggers a full rebuild of all dependencies.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let visqol_dir = std::env::var_os("VISQOL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("../../visqol"))
        .canonicalize()
        .expect("visqol checkout not found (set VISQOL_DIR)");

    let lib_dir = if let Some(dir) = std::env::var_os("VISQOL_CAPI_LIB_DIR") {
        PathBuf::from(dir)
    } else {
        let bazel = std::env::var_os("BAZEL").unwrap_or_else(|| "bazelisk".into());
        let status = Command::new(&bazel)
            .args(["build", "//:visqol_capi"])
            .current_dir(&visqol_dir)
            .status()
            .unwrap_or_else(|e| {
                panic!("failed to run {bazel:?}: {e} (set VISQOL_CAPI_LIB_DIR to skip bazel)")
            });
        assert!(status.success(), "bazel build //:visqol_capi failed");

        let output = Command::new(&bazel)
            .args(["info", "bazel-bin"])
            .current_dir(&visqol_dir)
            .output()
            .expect("failed to run bazel info");
        assert!(output.status.success(), "bazel info bazel-bin failed");
        PathBuf::from(String::from_utf8(output.stdout).unwrap().trim())
    };

    let lib = lib_dir.join("libvisqol_capi.so");
    assert!(
        lib.is_file(),
        "{} not found; build it with `bazelisk build //:visqol_capi` in {}",
        lib.display(),
        visqol_dir.display()
    );
    // Canonicalize: bazel-bin may itself be a symlink, and an rpath should
    // not depend on the symlink surviving future builds.
    let lib_dir = lib_dir.canonicalize().unwrap();

    println!("cargo::warning=visqol-cpp: linking {}", lib.display());
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=visqol_capi");
    // Let test/bench binaries find the shared library at runtime.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    // rustc-link-arg does not propagate to dependent crates, so also export
    // the directory as metadata: dependent binaries/benches must re-emit an
    // rpath from DEP_VISQOL_CAPI_LIB_DIR in their own build script (see the
    // benchmarks crate for an example).
    println!("cargo::metadata=lib_dir={}", lib_dir.display());

    // Track the .so itself: a `bazel clean` (or output_base change) removes
    // it, and cargo treats a missing rerun-if-changed file as stale, which
    // reruns this script (and bazel) instead of leaving a dangling rpath.
    println!("cargo:rerun-if-changed={}", lib.display());
    println!("cargo:rerun-if-env-changed=VISQOL_DIR");
    println!("cargo:rerun-if-env-changed=VISQOL_CAPI_LIB_DIR");
    println!("cargo:rerun-if-env-changed=BAZEL");
    println!(
        "cargo:rerun-if-changed={}",
        visqol_dir.join("src/visqol_capi.cc").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        visqol_dir.join("BUILD").display()
    );
}
