//! LightModule trait for trainable models.

use candle::{Result, Tensor, Var};
use std::collections::HashMap;

/// Output from a training or validation step.
pub struct StepOutput {
    pub loss: Tensor,
    pub metrics: HashMap<String, f64>,
}

impl StepOutput {
    pub fn new(loss: Tensor) -> Self {
        Self {
            loss,
            metrics: HashMap::new(),
        }
    }

    pub fn with_metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(name.into(), value);
        self
    }
}

/// Trait for models that can be trained with [`Trainer`](crate::Trainer).
pub trait LightModule {
    /// The batch type (e.g., `(Tensor, Tensor)` for (inputs, labels)).
    type Batch;

    fn training_step(&mut self, batch: Self::Batch, batch_idx: usize) -> Result<StepOutput>;

    fn validation_step(&self, batch: Self::Batch, batch_idx: usize) -> Result<StepOutput> {
        let _ = (batch, batch_idx);
        unimplemented!("validation_step not implemented")
    }

    fn parameters(&self) -> Vec<Var>;
}
