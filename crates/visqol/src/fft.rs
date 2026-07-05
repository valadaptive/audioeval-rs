//! FFT helpers matching the C++ `FftManager`/`FastFourierTransform` behavior:
//! power-of-two sizes with a floor of 32, zero-padded inputs, and a `1/n`
//! scale on the inverse transform.
//!
//! The C++ implementation runs pffft in single precision; we use `realfft` in
//! double precision, which only affects results below the f32 noise floor.

use std::cell::RefCell;

use realfft::RealFftPlanner;

pub type Complex = num_complex::Complex<f64>;

const MIN_FFT_SIZE: usize = 32;

// The planner caches plans (twiddle tables) by size; keep one per thread so
// the many small per-patch FFTs during fine realignment don't re-plan.
thread_local! {
    static PLANNER: RefCell<RealFftPlanner<f64>> = RefCell::new(RealFftPlanner::new());
}

/// Forward real FFT of `signal`, zero-padded to the next power of two with a
/// floor of 32 (the C++ `FftManager` constructor). Returns the
/// `fft_size / 2 + 1` non-redundant bins.
pub fn rfft(signal: &[f64]) -> Vec<Complex> {
    let fft_size = signal.len().next_power_of_two().max(MIN_FFT_SIZE);
    let r2c = PLANNER.with_borrow_mut(|p| p.plan_fft_forward(fft_size));
    let mut input = vec![0.0; fft_size];
    input[..signal.len()].copy_from_slice(signal);
    let mut spectrum = r2c.make_output_vec();
    r2c.process(&mut input, &mut spectrum)
        .expect("buffer sizes are correct by construction");
    spectrum
}

/// Inverse real FFT with the C++ `1/fft_size` scaling applied. The transform
/// size is implied by the spectrum length (`fft_size / 2 + 1` bins).
pub fn irfft(spectrum: &mut [Complex]) -> Vec<f64> {
    let fft_size = 2 * (spectrum.len() - 1);
    debug_assert!(fft_size.is_power_of_two() && fft_size >= MIN_FFT_SIZE);
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
