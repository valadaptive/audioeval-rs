//! Runtime microarchitecture dispatch / function multiversioning: compile a closure once per feature
//! level and pick the best one the CPU supports.
//!
//! This is the same type of functionality provided by [the `multiversion` crate](https://crates.io/crates/multiversion), but more barebones/lightweight. In particular:
//! - It always dispatches based on a fixed set of x86_64 microarchitecture versions: x86-64-v2 (SSE4.2) and x86-64-v3 (AVX2). Other levels or architectures are omitted because:
//!   - AVX-512 support is spotty, and it's split up into a bunch of subsets, of which only some are supported between CPU vendors.
//!   - aarch64 always has NEON available.
//!   - WebAssembly doesn't support runtime feature detection at all; you must compile different versions of the entire library and do multiversioning on the *host*.
//!   - RISC-V is sort of a mess right now.
//! - It does not use any proc macros, and does not have any dependencies.
//!
//! The intent behind this crate is to get you most of the useful instructions added in or after x86-64-v2, while not pulling in a bunch of heavy dependencies.

/// Specialize a function into multiple microarchitecture-specific versions to
/// improve performance.
///
/// The passed function or closure, and any functions it calls that you want to
/// vectorize/specialize, *must* be marked `#[inline(always)]` in order for
/// specialization to occur.
#[inline(always)]
pub fn multiversion<R, F: FnOnce() -> R>(func: F) -> R {
    #[inline(never)]
    fn doit_baseline<R, F: FnOnce() -> R>(func: F) -> R {
        func()
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        #[target_feature(enable = "avx2,bmi1,bmi2,cmpxchg16b,f16c,fma,lzcnt,movbe,popcnt,xsave")]
        fn doit_avx2<R, F: FnOnce() -> R>(func: F) -> R {
            func()
        }
        #[target_feature(enable = "sse4.2,cmpxchg16b,popcnt")]
        fn doit_sse42<R, F: FnOnce() -> R>(func: F) -> R {
            func()
        }

        match x86_64_arch_level() {
            3 => return unsafe { doit_avx2(func) },
            2 => return unsafe { doit_sse42(func) },
            _ => {}
        }
    }

    doit_baseline(func)
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
fn x86_64_arch_level() -> usize {
    #[inline(never)]
    fn feature_detect() -> usize {
        let v2 = std::arch::is_x86_feature_detected!("sse4.2")
            && std::arch::is_x86_feature_detected!("cmpxchg16b")
            && std::arch::is_x86_feature_detected!("popcnt");
        let v3 = v2
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("bmi1")
            && std::arch::is_x86_feature_detected!("bmi2")
            && std::arch::is_x86_feature_detected!("f16c")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("lzcnt")
            && std::arch::is_x86_feature_detected!("movbe")
            && std::arch::is_x86_feature_detected!("xsave");

        if v3 {
            3
        } else if v2 {
            2
        } else {
            1
        }
    }

    use std::sync::atomic::{AtomicUsize, Ordering};

    static ARCH_LEVEL: AtomicUsize = AtomicUsize::new(0);

    match ARCH_LEVEL.load(Ordering::Relaxed) {
        0 => {
            let level = feature_detect();
            ARCH_LEVEL.store(level, Ordering::Relaxed);
            level
        }
        level => level,
    }
}
