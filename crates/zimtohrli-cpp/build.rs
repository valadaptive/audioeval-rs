//! Compiles the original single-header C++ Zimtohrli implementation
//! (`zimtohrli/cpp/zimt/zimtohrli.h`) together with the pure-C ABI wrapper in
//! `wrapper.cc`.
//!
//! The header only needs a C++17 standard library. Everything in it lives in
//! an anonymous namespace (internal linkage), so `wrapper.cc` is deliberately
//! the only translation unit that includes it.

use std::path::PathBuf;

fn has_clangxx() -> bool {
    std::process::Command::new("clang++")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let cpp_root = manifest_dir.join("../../zimtohrli/cpp");
    for header in ["zimt/zimtohrli.h", "zimt/mos.h"] {
        assert!(
            cpp_root.join(header).is_file(),
            "{} not found; is the zimtohrli checkout present?",
            cpp_root.join(header).display()
        );
    }

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .include(&cpp_root)
        .file(manifest_dir.join("wrapper.cc"))
        // Benchmarks must not inherit a debug opt level, whatever profile the
        // build script itself runs under.
        .opt_level(3);

    // The upstream filter loop is tuned for clang's auto-vectorizer ("clang
    // simdifies this"), so prefer clang++ when CXX is unset. Set CXX (and
    // CXXFLAGS, e.g. "-march=native") to override.
    if std::env::var_os("CXX").is_none() && has_clangxx() {
        build.compiler("clang++");
    }

    println!(
        "cargo::warning=zimtohrli-cpp: C++ compiler: {:?} (override with CXX)",
        build.get_compiler().path()
    );
    println!("cargo::rerun-if-env-changed=CXX");
    println!("cargo::rerun-if-env-changed=CXXFLAGS");
    println!("cargo::rerun-if-changed=wrapper.cc");
    println!(
        "cargo::rerun-if-changed={}",
        cpp_root.join("zimt/zimtohrli.h").display()
    );
    println!(
        "cargo::rerun-if-changed={}",
        cpp_root.join("zimt/mos.h").display()
    );

    build.compile("zimtohrli_cpp");
}
