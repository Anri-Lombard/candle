//! Callback system for training hooks.

mod checkpoint;
mod early_stopping;

pub use checkpoint::ModelCheckpoint;
pub use early_stopping::EarlyStopping;

use crate::{StepOutput, Trainer};
use candle::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Min,
    Max,
}

#[derive(Clone, Debug)]
pub struct EpochMetrics {
    pub train_loss: f32,
    pub val_loss: Option<f32>,
}

pub trait Callback: Send {
    fn on_fit_start(&mut self, _trainer: &Trainer) -> Result<()> {
        Ok(())
    }

    fn on_fit_end(&mut self, _trainer: &Trainer) -> Result<()> {
        Ok(())
    }

    fn on_epoch_start(&mut self, _trainer: &Trainer, _epoch: usize) -> Result<()> {
        Ok(())
    }

    fn on_epoch_end(
        &mut self,
        _trainer: &Trainer,
        _epoch: usize,
        _metrics: &EpochMetrics,
    ) -> Result<()> {
        Ok(())
    }

    fn on_train_batch_end(
        &mut self,
        _trainer: &Trainer,
        _batch_idx: usize,
        _output: &StepOutput,
    ) -> Result<()> {
        Ok(())
    }

    fn should_stop(&self) -> bool {
        false
    }
}
