//! Metrics for tracking training progress.

use candle::{Result, Tensor};

pub trait Metric {
    fn update(&mut self, preds: &Tensor, targets: &Tensor) -> Result<()>;
    fn compute(&self) -> f64;
    fn reset(&mut self);
}
