//! Analysis window parameters, a port of `analysis_window.cc`.

pub struct AnalysisWindow {
    pub size: usize,
    pub overlap: f64,
}

impl AnalysisWindow {
    pub const DURATION_SECONDS: f64 = 0.08;

    pub fn new(sample_rate: u32, overlap: f64) -> Self {
        AnalysisWindow {
            size: (sample_rate as f64 * Self::DURATION_SECONDS).round() as usize,
            overlap,
        }
    }

    pub fn hann_window(&self) -> Vec<f64> {
        let n = self.size;
        (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
            .collect()
    }
}
