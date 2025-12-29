#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{Device, Tensor, Var};
use candle_light::{Callback, EarlyStopping, EpochMetrics, LightModule, Mode, StepOutput, Trainer, TrainerConfig};
use candle_nn::{Linear, Module, Optimizer, SGD};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Clone)]
struct Batch {
    xs: Tensor,
    ys: Tensor,
}

struct LinearModel {
    linear: Linear,
    w: Var,
    b: Var,
}

impl LinearModel {
    fn new(device: &Device) -> Result<Self> {
        let w = Var::new(&[[0f32, 0.]], device)?;
        let b = Var::new(0f32, device)?;
        let linear = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));
        Ok(Self { linear, w, b })
    }
}

impl LightModule for LinearModel {
    type Batch = Batch;

    fn training_step(&mut self, batch: Batch, _batch_idx: usize) -> candle::Result<StepOutput> {
        let ys = self.linear.forward(&batch.xs)?;
        let loss = ys.sub(&batch.ys)?.sqr()?.sum_all()?;
        Ok(StepOutput::new(loss))
    }

    fn validation_step(&self, batch: Batch, _batch_idx: usize) -> candle::Result<StepOutput> {
        let ys = self.linear.forward(&batch.xs)?;
        let loss = ys.sub(&batch.ys)?.sqr()?.sum_all()?;
        Ok(StepOutput::new(loss))
    }

    fn parameters(&self) -> Vec<Var> {
        vec![self.w.clone(), self.b.clone()]
    }
}

#[test]
fn trainer_basic_training() -> Result<()> {
    let device = Device::Cpu;

    let w_gen = Tensor::new(&[[3f32, 1.]], &device)?;
    let b_gen = Tensor::new(-2f32, &device)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &device)?;
    let sample_ys = gen.forward(&sample_xs)?;

    let batches = vec![Batch {
        xs: sample_xs,
        ys: sample_ys,
    }];

    let mut model = LinearModel::new(&device)?;
    let mut optimizer = SGD::new(model.parameters(), 0.004)?;

    let config = TrainerConfig::new().max_epochs(1000).log_every_n_steps(0);
    let mut trainer = Trainer::new(config);
    trainer.fit(&mut model, &mut optimizer, batches, None)?;

    let w_vals = model.w.to_vec2::<f32>()?;
    let b_val = model.b.to_scalar::<f32>()?;

    assert!((w_vals[0][0] - 3.0).abs() < 0.01);
    assert!((w_vals[0][1] - 1.0).abs() < 0.01);
    assert!((b_val - (-2.0)).abs() < 0.03);

    Ok(())
}

#[test]
fn trainer_with_validation() -> Result<()> {
    let device = Device::Cpu;

    let w_gen = Tensor::new(&[[3f32, 1.]], &device)?;
    let b_gen = Tensor::new(-2f32, &device)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &device)?;
    let sample_ys = gen.forward(&sample_xs)?;

    let batches = vec![Batch {
        xs: sample_xs.clone(),
        ys: sample_ys.clone(),
    }];
    let val_batches = vec![Batch {
        xs: sample_xs,
        ys: sample_ys,
    }];

    let mut model = LinearModel::new(&device)?;
    let mut optimizer = SGD::new(model.parameters(), 0.004)?;

    let config = TrainerConfig::new().max_epochs(100).log_every_n_steps(0);
    let mut trainer = Trainer::new(config);
    trainer.fit(&mut model, &mut optimizer, batches, Some(val_batches))?;

    let val_loss = trainer.validate(&model, &vec![Batch {
        xs: Tensor::new(&[[2f32, 1.], [7., 4.]], &device)?,
        ys: Tensor::new(&[[5f32], [25.]], &device)?,
    }])?;

    assert!(val_loss.is_finite());
    Ok(())
}

struct CountingCallback {
    epoch_starts: Arc<AtomicUsize>,
    epoch_ends: Arc<AtomicUsize>,
    batch_ends: Arc<AtomicUsize>,
}

impl Callback for CountingCallback {
    fn on_epoch_start(&mut self, _trainer: &Trainer, _epoch: usize) -> candle::Result<()> {
        self.epoch_starts.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn on_epoch_end(
        &mut self,
        _trainer: &Trainer,
        _epoch: usize,
        _metrics: &EpochMetrics,
    ) -> candle::Result<()> {
        self.epoch_ends.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn on_train_batch_end(
        &mut self,
        _trainer: &Trainer,
        _batch_idx: usize,
        _output: &StepOutput,
    ) -> candle::Result<()> {
        self.batch_ends.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn trainer_callback_invocation() -> Result<()> {
    let device = Device::Cpu;

    let batches = vec![
        Batch {
            xs: Tensor::new(&[[1f32, 0.]], &device)?,
            ys: Tensor::new(&[[1f32]], &device)?,
        },
        Batch {
            xs: Tensor::new(&[[0f32, 1.]], &device)?,
            ys: Tensor::new(&[[1f32]], &device)?,
        },
    ];

    let epoch_starts = Arc::new(AtomicUsize::new(0));
    let epoch_ends = Arc::new(AtomicUsize::new(0));
    let batch_ends = Arc::new(AtomicUsize::new(0));

    let callback = CountingCallback {
        epoch_starts: epoch_starts.clone(),
        epoch_ends: epoch_ends.clone(),
        batch_ends: batch_ends.clone(),
    };

    let mut model = LinearModel::new(&device)?;
    let mut optimizer = SGD::new(model.parameters(), 0.01)?;

    let config = TrainerConfig::new().max_epochs(5).log_every_n_steps(0);
    let mut trainer = Trainer::new(config).with_callback(callback);
    trainer.fit(&mut model, &mut optimizer, batches, None)?;

    assert_eq!(epoch_starts.load(Ordering::SeqCst), 5);
    assert_eq!(epoch_ends.load(Ordering::SeqCst), 5);
    assert_eq!(batch_ends.load(Ordering::SeqCst), 10); // 2 batches * 5 epochs

    Ok(())
}

struct EarlyStopCallback {
    stop_at_epoch: usize,
    current_epoch: usize,
}

impl Callback for EarlyStopCallback {
    fn on_epoch_end(
        &mut self,
        _trainer: &Trainer,
        _epoch: usize,
        _metrics: &EpochMetrics,
    ) -> candle::Result<()> {
        self.current_epoch += 1;
        Ok(())
    }

    fn should_stop(&self) -> bool {
        self.current_epoch >= self.stop_at_epoch
    }
}

#[test]
fn trainer_early_stopping() -> Result<()> {
    let device = Device::Cpu;

    let batches = vec![Batch {
        xs: Tensor::new(&[[1f32, 0.]], &device)?,
        ys: Tensor::new(&[[1f32]], &device)?,
    }];

    let epoch_count = Arc::new(AtomicUsize::new(0));
    let counting = CountingCallback {
        epoch_starts: epoch_count.clone(),
        epoch_ends: Arc::new(AtomicUsize::new(0)),
        batch_ends: Arc::new(AtomicUsize::new(0)),
    };
    let stopper = EarlyStopCallback {
        stop_at_epoch: 3,
        current_epoch: 0,
    };

    let mut model = LinearModel::new(&device)?;
    let mut optimizer = SGD::new(model.parameters(), 0.01)?;

    let config = TrainerConfig::new().max_epochs(100).log_every_n_steps(0);
    let mut trainer = Trainer::new(config)
        .with_callback(counting)
        .with_callback(stopper);
    trainer.fit(&mut model, &mut optimizer, batches, None)?;

    assert_eq!(epoch_count.load(Ordering::SeqCst), 3);

    Ok(())
}

#[test]
fn builtin_early_stopping() -> Result<()> {
    let device = Device::Cpu;

    // Train on one relationship (y = 3x + z - 2)
    let w_gen = Tensor::new(&[[3f32, 1.]], &device)?;
    let b_gen = Tensor::new(-2f32, &device)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &device)?;
    let sample_ys = gen.forward(&sample_xs)?;

    // Validate on a DIFFERENT relationship (y = -x + 2z + 5)
    // This causes overfitting - as model fits train data, val loss increases
    let w_val = Tensor::new(&[[-1f32, 2.]], &device)?;
    let b_val = Tensor::new(5f32, &device)?;
    let val_gen = Linear::new(w_val, Some(b_val));
    let val_xs = Tensor::new(&[[1f32, 1.], [2., 3.], [0., 2.]], &device)?;
    let val_ys = val_gen.forward(&val_xs)?;

    let batches = vec![Batch {
        xs: sample_xs,
        ys: sample_ys,
    }];
    let val_batches = vec![Batch {
        xs: val_xs,
        ys: val_ys,
    }];

    let epoch_count = Arc::new(AtomicUsize::new(0));
    let counting = CountingCallback {
        epoch_starts: epoch_count.clone(),
        epoch_ends: Arc::new(AtomicUsize::new(0)),
        batch_ends: Arc::new(AtomicUsize::new(0)),
    };

    // EarlyStopping with patience=5 monitoring val_loss (min mode)
    let early_stopping = EarlyStopping::new()
        .monitor("val_loss")
        .patience(5)
        .mode(Mode::Min);

    let mut model = LinearModel::new(&device)?;
    let mut optimizer = SGD::new(model.parameters(), 0.01)?;

    let config = TrainerConfig::new().max_epochs(500).log_every_n_steps(0);
    let mut trainer = Trainer::new(config)
        .with_callback(counting)
        .with_callback(early_stopping);
    trainer.fit(&mut model, &mut optimizer, batches, Some(val_batches))?;

    // Should stop before 500 epochs due to early stopping on val_loss
    let epochs_run = epoch_count.load(Ordering::SeqCst);
    assert!(epochs_run < 500, "Early stopping should have triggered before 500 epochs, ran {}", epochs_run);
    assert!(epochs_run > 5, "Should run more than patience epochs before stopping");

    Ok(())
}
