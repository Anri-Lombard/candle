//! High-level training framework for Candle, inspired by PyTorch Lightning.

pub mod callbacks;
pub mod config;
pub mod metrics;
pub mod module;
pub mod trainer;

pub use callbacks::Callback;
pub use config::TrainerConfig;
pub use module::{LightModule, StepOutput};
pub use trainer::Trainer;

pub use candle::{Module, ModuleT};
