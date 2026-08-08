//! Re-emits the rpath for `libvisqol_capi.so`: `cargo:rustc-link-arg` output
//! from the `visqol-cpp` build script only applies to that crate's own
//! targets, so this crate's bench binaries need their own rpath entry. The
//! library directory arrives as `DEP_VISQOL_CAPI_LIB_DIR` metadata (which is
//! why `visqol-cpp` is a normal, not dev-, dependency of this crate).

fn main() {
    if let Ok(lib_dir) = std::env::var("DEP_VISQOL_CAPI_LIB_DIR") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
    println!("cargo:rerun-if-env-changed=DEP_VISQOL_CAPI_LIB_DIR");
}
