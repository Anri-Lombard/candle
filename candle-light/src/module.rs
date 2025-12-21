//! LightModule trait for trainable models.

use candle::{Result, Tensor, Var};
use std::collections::HashMap;

/// Output from a training or validation step.
pub struct StepOutput {
    /// The loss tensor for backpropagation.
    pub loss: Tensor,
    /// Optional metrics (e.g., accuracy, perplexity).
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
    /// Compute loss and metrics for a training batch.
    fn training_step(&mut self, batch: &Tensor, batch_idx: usize) -> Result<StepOutput>;

    /// Compute loss and metrics for a validation batch.
    fn validation_step(&self, batch: &Tensor, batch_idx: usize) -> Result<StepOutput> {
        let _ = (batch, batch_idx);
        unimplemented!("validation_step not implemented")
    }

    /// Return all trainable parameters.
    fn parameters(&self) -> Vec<Var>;
}
