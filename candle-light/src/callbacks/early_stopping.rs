//! Early stopping callback.

use super::{Callback, EpochMetrics, Mode};
use crate::Trainer;
use candle::Result;

pub struct EarlyStopping {
    monitor: String,
    patience: usize,
    min_delta: f32,
    mode: Mode,
    best_metric: Option<f32>,
    epochs_without_improvement: usize,
    stopped: bool,
}

impl EarlyStopping {
    pub fn new() -> Self {
        Self {
            monitor: "val_loss".to_string(),
            patience: 3,
            min_delta: 0.0,
            mode: Mode::Min,
            best_metric: None,
            epochs_without_improvement: 0,
            stopped: false,
        }
    }

    pub fn monitor(mut self, metric: impl Into<String>) -> Self {
        self.monitor = metric.into();
        self
    }

    pub fn patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    pub fn min_delta(mut self, delta: f32) -> Self {
        self.min_delta = delta;
        self
    }

    pub fn mode(mut self, mode: Mode) -> Self {
        self.mode = mode;
        self
    }

    fn get_monitored_value(&self, metrics: &EpochMetrics) -> Option<f32> {
        match self.monitor.as_str() {
            "val_loss" => metrics.val_loss,
            "train_loss" => Some(metrics.train_loss),
            _ => None,
        }
    }

    fn is_improvement(&self, current: f32, best: f32) -> bool {
        match self.mode {
            Mode::Min => current < best - self.min_delta,
            Mode::Max => current > best + self.min_delta,
        }
    }
}

impl Default for EarlyStopping {
    fn default() -> Self {
        Self::new()
    }
}

impl Callback for EarlyStopping {
    fn on_epoch_end(
        &mut self,
        _trainer: &Trainer,
        _epoch: usize,
        metrics: &EpochMetrics,
    ) -> Result<()> {
        let Some(current) = self.get_monitored_value(metrics) else {
            return Ok(());
        };

        match self.best_metric {
            None => {
                self.best_metric = Some(current);
                self.epochs_without_improvement = 0;
            }
            Some(best) => {
                if self.is_improvement(current, best) {
                    self.best_metric = Some(current);
                    self.epochs_without_improvement = 0;
                } else {
                    self.epochs_without_improvement += 1;
                    if self.epochs_without_improvement >= self.patience {
                        self.stopped = true;
                    }
                }
            }
        }

        Ok(())
    }

    fn should_stop(&self) -> bool {
        self.stopped
    }
}
