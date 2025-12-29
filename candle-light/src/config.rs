//! Trainer configuration.

use candle::Device;

pub struct TrainerConfig {
    pub max_epochs: usize,
    pub device: Device,
    pub log_every_n_steps: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            max_epochs: 1000,
            device: Device::Cpu,
            log_every_n_steps: 50,
        }
    }
}

impl TrainerConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_epochs(mut self, epochs: usize) -> Self {
        self.max_epochs = epochs;
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn log_every_n_steps(mut self, n: usize) -> Self {
        self.log_every_n_steps = n;
        self
    }
}
