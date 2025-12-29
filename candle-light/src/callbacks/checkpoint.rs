//! Model checkpointing callback.

use super::{Callback, EpochMetrics, Mode};
use crate::Trainer;
use candle::Result;
use candle_nn::VarMap;
use std::collections::BinaryHeap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

struct CheckpointEntry {
    metric: f32,
    path: PathBuf,
    mode: Mode,
}

impl PartialEq for CheckpointEntry {
    fn eq(&self, other: &Self) -> bool {
        self.metric == other.metric
    }
}

impl Eq for CheckpointEntry {}

impl PartialOrd for CheckpointEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CheckpointEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // For min mode: we want to remove largest values first (max-heap of values to remove)
        // For max mode: we want to remove smallest values first (min-heap, so reverse)
        let ordering = self.metric.partial_cmp(&other.metric).unwrap_or(std::cmp::Ordering::Equal);
        match self.mode {
            Mode::Min => ordering.reverse(), // Keep smallest, remove largest
            Mode::Max => ordering,           // Keep largest, remove smallest
        }
    }
}

pub struct ModelCheckpoint {
    varmap: Arc<Mutex<VarMap>>,
    dirpath: PathBuf,
    monitor: String,
    save_top_k: usize,
    mode: Mode,
    checkpoints: BinaryHeap<CheckpointEntry>,
    best_metric: Option<f32>,
}

impl ModelCheckpoint {
    pub fn new(varmap: Arc<Mutex<VarMap>>, dirpath: impl Into<PathBuf>) -> Self {
        Self {
            varmap,
            dirpath: dirpath.into(),
            monitor: "val_loss".to_string(),
            save_top_k: 1,
            mode: Mode::Min,
            checkpoints: BinaryHeap::new(),
            best_metric: None,
        }
    }

    pub fn monitor(mut self, metric: impl Into<String>) -> Self {
        self.monitor = metric.into();
        self
    }

    pub fn save_top_k(mut self, k: usize) -> Self {
        self.save_top_k = k;
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

    fn is_better(&self, current: f32, best: f32) -> bool {
        match self.mode {
            Mode::Min => current < best,
            Mode::Max => current > best,
        }
    }

    fn save_checkpoint(&mut self, epoch: usize, metric: f32) -> Result<()> {
        std::fs::create_dir_all(&self.dirpath).map_err(|e| candle::Error::wrap(e))?;

        let filename = format!("epoch_{:04}_{}_{:.4}.safetensors", epoch, self.monitor, metric);
        let path = self.dirpath.join(&filename);

        let varmap = self.varmap.lock().unwrap();
        varmap.save(&path)?;
        drop(varmap);

        self.checkpoints.push(CheckpointEntry {
            metric,
            path: path.clone(),
            mode: self.mode,
        });

        // Prune old checkpoints if we exceed save_top_k
        while self.checkpoints.len() > self.save_top_k {
            if let Some(entry) = self.checkpoints.pop() {
                let _ = std::fs::remove_file(&entry.path);
            }
        }

        Ok(())
    }
}

impl Callback for ModelCheckpoint {
    fn on_epoch_end(
        &mut self,
        _trainer: &Trainer,
        epoch: usize,
        metrics: &EpochMetrics,
    ) -> Result<()> {
        let Some(current) = self.get_monitored_value(metrics) else {
            return Ok(());
        };

        let should_save = match self.best_metric {
            None => true,
            Some(best) => self.is_better(current, best),
        };

        if should_save {
            self.best_metric = Some(current);
            self.save_checkpoint(epoch, current)?;
        }

        Ok(())
    }
}
