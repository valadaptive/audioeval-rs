//! FFT helpers matching the C++ `FftManager`/`FastFourierTransform` behavior:
//! power-of-two sizes with a floor of 32, zero-padded inputs, and a `1/n`
//! scale on the inverse transform.
//!
//! The C++ implementation runs pffft in single precision; we use `realfft` in
//! double precision, which only affects results below the f32 noise floor.

use std::cell::RefCell;

use realfft::RealFftPlanner;

pub type Complex = num_complex::Complex<f64>;

pub const MIN_FFT_SIZE: usize = 32;

// The planner caches plans (twiddle tables) by size; keep one per thread so
// the many small per-patch FFTs during fine realignment don't re-plan.
thread_local! {
    static PLANNER: RefCell<RealFftPlanner<f64>> = RefCell::new(RealFftPlanner::new());
}

pub fn next_pow_two(n: usize) -> usize {
    n.next_power_of_two()
}

/// Forward real FFT of `signal` zero-padded to `fft_size` (power of two).
/// Returns the `fft_size / 2 + 1` non-redundant bins.
pub fn rfft(signal: &[f64], fft_size: usize) -> Vec<Complex> {
    debug_assert!(fft_size.is_power_of_two() && signal.len() <= fft_size);
    let r2c = PLANNER.with_borrow_mut(|p| p.plan_fft_forward(fft_size));
    let mut input = vec![0.0; fft_size];
    input[..signal.len()].copy_from_slice(signal);
    let mut spectrum = r2c.make_output_vec();
    r2c.process(&mut input, &mut spectrum)
        .expect("buffer sizes are correct by construction");
    spectrum
}

/// Inverse real FFT with the C++ `1/fft_size` scaling applied.
pub fn irfft(spectrum: &mut [Complex], fft_size: usize) -> Vec<f64> {
    debug_assert_eq!(spectrum.len(), fft_size / 2 + 1);
    let c2r = PLANNER.with_borrow_mut(|p| p.plan_fft_inverse(fft_size));
    let mut output = vec![0.0; fft_size];
    c2r.process(spectrum, &mut output)
        .expect("buffer sizes are correct by construction");
    let scale = 1.0 / fft_size as f64;
    for v in &mut output {
        *v *= scale;
    }
    output
}
