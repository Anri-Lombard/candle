//! Training loop implementation.

use crate::{Callback, LightModule, TrainerConfig};
use candle::Result;

pub struct Trainer {
    config: TrainerConfig,
    callbacks: Vec<Box<dyn Callback>>,
}

impl Trainer {
    pub fn new(config: TrainerConfig) -> Self {
        Self {
            config,
            callbacks: Vec::new(),
        }
    }

    pub fn with_callback(mut self, callback: impl Callback + 'static) -> Self {
        self.callbacks.push(Box::new(callback));
        self
    }

    pub fn config(&self) -> &TrainerConfig {
        &self.config
    }

    pub fn fit<M, I>(&mut self, model: &mut M, train_data: I, _val_data: Option<I>) -> Result<()>
    where
        M: LightModule,
        I: IntoIterator<Item = Result<candle::Tensor>>,
    {
        let _ = (model, train_data);
        todo!("implement training loop")
    }
}
