//! Callback system for training hooks.

use crate::{StepOutput, Trainer};
use candle::Result;

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

    fn on_epoch_end(&mut self, _trainer: &Trainer, _epoch: usize) -> Result<()> {
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
