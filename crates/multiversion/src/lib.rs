//! Runtime microarchitecture dispatch: compile a closure once per feature
//! level and pick the widest one the CPU supports.

/// Specialize a function into multiple microarchitecture-specific versions to
/// improve performance.
///
/// The passed function or closure, and any functions it calls that you want to
/// vectorize/specialize, *must* be marked `#[inline(always)]` in order for
/// specialization to occur.
pub fn multiversion<R, F: FnOnce() -> R>(func: F) -> R {
    #[inline(never)]
    fn doit_baseline<R, F: FnOnce() -> R>(func: F) -> R {
        func()
    }

    #[cfg(target_arch = "x86_64")]
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

#[cfg(target_arch = "x86_64")]
fn x86_64_arch_level() -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static ARCH_LEVEL: AtomicUsize = AtomicUsize::new(0);

    match ARCH_LEVEL.load(Ordering::Relaxed) {
        0 => {
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

            let level = if v3 {
                3
            } else if v2 {
                2
            } else {
                1
            };
            ARCH_LEVEL.store(level, Ordering::Relaxed);
            level
        }
        level => level,
    }
}
