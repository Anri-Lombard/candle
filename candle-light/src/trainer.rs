//! Training loop implementation.

use crate::callbacks::EpochMetrics;
use crate::{Callback, LightModule, StepOutput, TrainerConfig};
use candle::Result;
use candle_nn::Optimizer;

pub struct Trainer {
    config: TrainerConfig,
    callbacks: Vec<Box<dyn Callback>>,
    current_epoch: usize,
    global_step: usize,
    should_stop: bool,
}

impl Trainer {
    pub fn new(config: TrainerConfig) -> Self {
        Self {
            config,
            callbacks: Vec::new(),
            current_epoch: 0,
            global_step: 0,
            should_stop: false,
        }
    }

    pub fn with_callback(mut self, callback: impl Callback + 'static) -> Self {
        self.callbacks.push(Box::new(callback));
        self
    }

    pub fn config(&self) -> &TrainerConfig {
        &self.config
    }

    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }

    pub fn global_step(&self) -> usize {
        self.global_step
    }

    pub fn fit<M, O, D>(
        &mut self,
        model: &mut M,
        optimizer: &mut O,
        train_data: D,
        val_data: Option<D>,
    ) -> Result<()>
    where
        M: LightModule,
        O: Optimizer,
        D: DataLoader<Batch = M::Batch>,
    {
        self.call_on_fit_start()?;

        for epoch in 0..self.config.max_epochs {
            self.current_epoch = epoch;
            self.call_on_epoch_start(epoch)?;

            let train_loss = self.train_epoch(model, optimizer, &train_data)?;

            let val_loss = match &val_data {
                Some(data) => Some(self.validate(model, data)?),
                None => None,
            };

            let metrics = EpochMetrics { train_loss, val_loss };
            self.call_on_epoch_end(epoch, &metrics)?;

            if self.config.log_every_n_steps > 0 {
                match val_loss {
                    Some(vl) => println!(
                        "Epoch {}: train_loss={:.4}, val_loss={:.4}",
                        epoch, train_loss, vl
                    ),
                    None => println!("Epoch {}: train_loss={:.4}", epoch, train_loss),
                }
            }

            if self.should_stop || self.check_callbacks_should_stop() {
                break;
            }
        }

        self.call_on_fit_end()?;
        Ok(())
    }

    fn train_epoch<M, O, D>(
        &mut self,
        model: &mut M,
        optimizer: &mut O,
        data: &D,
    ) -> Result<f32>
    where
        M: LightModule,
        O: Optimizer,
        D: DataLoader<Batch = M::Batch>,
    {
        let mut total_loss = 0f32;
        let mut batch_count = 0usize;

        for (batch_idx, batch) in data.iter().enumerate() {
            let output = model.training_step(batch, batch_idx)?;
            optimizer.backward_step(&output.loss)?;

            let loss_val = output.loss.to_scalar::<f32>()?;
            total_loss += loss_val;
            batch_count += 1;
            self.global_step += 1;

            self.call_on_train_batch_end(batch_idx, &output)?;
        }

        Ok(if batch_count > 0 {
            total_loss / batch_count as f32
        } else {
            0.0
        })
    }

    pub fn validate<M, D>(&self, model: &M, data: &D) -> Result<f32>
    where
        M: LightModule,
        D: DataLoader<Batch = M::Batch>,
    {
        let mut total_loss = 0f32;
        let mut batch_count = 0usize;

        for (batch_idx, batch) in data.iter().enumerate() {
            let output = model.validation_step(batch, batch_idx)?;
            let loss_val = output.loss.to_scalar::<f32>()?;
            total_loss += loss_val;
            batch_count += 1;
        }

        Ok(if batch_count > 0 {
            total_loss / batch_count as f32
        } else {
            0.0
        })
    }

    fn call_on_fit_start(&mut self) -> Result<()> {
        let mut callbacks = std::mem::take(&mut self.callbacks);
        for cb in &mut callbacks {
            cb.on_fit_start(self)?;
        }
        self.callbacks = callbacks;
        Ok(())
    }

    fn call_on_fit_end(&mut self) -> Result<()> {
        let mut callbacks = std::mem::take(&mut self.callbacks);
        for cb in &mut callbacks {
            cb.on_fit_end(self)?;
        }
        self.callbacks = callbacks;
        Ok(())
    }

    fn call_on_epoch_start(&mut self, epoch: usize) -> Result<()> {
        let mut callbacks = std::mem::take(&mut self.callbacks);
        for cb in &mut callbacks {
            cb.on_epoch_start(self, epoch)?;
        }
        self.callbacks = callbacks;
        Ok(())
    }

    fn call_on_epoch_end(&mut self, epoch: usize, metrics: &EpochMetrics) -> Result<()> {
        let mut callbacks = std::mem::take(&mut self.callbacks);
        for cb in &mut callbacks {
            cb.on_epoch_end(self, epoch, metrics)?;
        }
        self.callbacks = callbacks;
        Ok(())
    }

    fn call_on_train_batch_end(&mut self, batch_idx: usize, output: &StepOutput) -> Result<()> {
        let mut callbacks = std::mem::take(&mut self.callbacks);
        for cb in &mut callbacks {
            cb.on_train_batch_end(self, batch_idx, output)?;
        }
        self.callbacks = callbacks;
        Ok(())
    }

    fn check_callbacks_should_stop(&self) -> bool {
        self.callbacks.iter().any(|cb| cb.should_stop())
    }
}

/// Trait for data sources that can be iterated multiple times.
pub trait DataLoader {
    type Batch;
    type Iter<'a>: Iterator<Item = Self::Batch>
    where
        Self: 'a;

    fn iter(&self) -> Self::Iter<'_>;
}

impl<T: Clone> DataLoader for Vec<T> {
    type Batch = T;
    type Iter<'a> = std::iter::Cloned<std::slice::Iter<'a, T>> where T: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        <[T]>::iter(self).cloned()
    }
}

impl<T: Clone, const N: usize> DataLoader for [T; N] {
    type Batch = T;
    type Iter<'a> = std::iter::Cloned<std::slice::Iter<'a, T>> where T: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        <[T]>::iter(self).cloned()
    }
}
