// This should reach 98.52% accuracy.
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{DType, Device, Result, Tensor, Var, D};
use candle_light::{LightModule, StepOutput, Trainer, TrainerConfig};
use candle_nn::{loss, ops, Conv2d, Dropout, Linear, ModuleT, Optimizer, VarBuilder, VarMap};
use rand::prelude::*;
use rand::rng;

const LABELS: usize = 10;
const BATCH_SIZE: usize = 64;

#[derive(Clone)]
struct Batch {
    images: Tensor,
    labels: Tensor,
}

struct ConvNet {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    dropout: Dropout,
    varmap: VarMap,
    training: bool,
}

impl ConvNet {
    fn new(dev: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        let conv1 = candle_nn::conv2d(1, 32, 5, Default::default(), vs.pp("c1"))?;
        let conv2 = candle_nn::conv2d(32, 64, 5, Default::default(), vs.pp("c2"))?;
        let fc1 = candle_nn::linear(1024, 1024, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(1024, LABELS, vs.pp("fc2"))?;
        let dropout = Dropout::new(0.5);
        Ok(Self {
            conv1,
            conv2,
            fc1,
            fc2,
            dropout,
            varmap,
            training: true,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, _img_dim) = xs.dims2()?;
        let xs = xs
            .reshape((b_sz, 1, 28, 28))?
            .apply(&self.conv1)?
            .max_pool2d(2)?
            .apply(&self.conv2)?
            .max_pool2d(2)?
            .flatten_from(1)?
            .apply(&self.fc1)?
            .relu()?;
        self.dropout.forward_t(&xs, self.training)?.apply(&self.fc2)
    }
}

impl LightModule for ConvNet {
    type Batch = Batch;

    fn training_step(&mut self, batch: Batch, _batch_idx: usize) -> Result<StepOutput> {
        self.training = true;
        let logits = self.forward(&batch.images)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &batch.labels)?;
        Ok(StepOutput::new(loss))
    }

    fn validation_step(&self, batch: Batch, _batch_idx: usize) -> Result<StepOutput> {
        let logits = self.forward(&batch.images)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &batch.labels)?;

        let sum_ok = logits
            .argmax(D::Minus1)?
            .eq(&batch.labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let accuracy = sum_ok / batch.labels.dims1()? as f32;

        Ok(StepOutput::new(loss).with_metric("accuracy", accuracy as f64))
    }

    fn parameters(&self) -> Vec<Var> {
        self.varmap.all_vars()
    }
}

fn create_batches(images: &Tensor, labels: &Tensor, shuffle: bool) -> Result<Vec<Batch>> {
    let n_samples = images.dim(0)?;
    let n_batches = n_samples / BATCH_SIZE;

    let mut batch_idxs: Vec<usize> = (0..n_batches).collect();
    if shuffle {
        batch_idxs.shuffle(&mut rng());
    }

    let mut batches = Vec::with_capacity(n_batches);
    for batch_idx in batch_idxs {
        let start = batch_idx * BATCH_SIZE;
        batches.push(Batch {
            images: images.narrow(0, start, BATCH_SIZE)?,
            labels: labels.narrow(0, start, BATCH_SIZE)?,
        });
    }
    Ok(batches)
}

fn main() -> anyhow::Result<()> {
    let dev = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", dev);

    let m = candle_datasets::vision::mnist::load()?;
    println!("train-images: {:?}", m.train_images.shape());
    println!("test-images: {:?}", m.test_images.shape());

    let train_images = m.train_images.to_device(&dev)?;
    let train_labels = m.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    let test_images = m.test_images.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    let mut model = ConvNet::new(&dev)?;
    let params = candle_nn::ParamsAdamW {
        lr: 0.001,
        ..Default::default()
    };
    let mut optimizer = candle_nn::AdamW::new(model.parameters(), params)?;

    let train_batches = create_batches(&train_images, &train_labels, true)?;
    let val_batches = create_batches(&test_images, &test_labels, false)?;

    let config = TrainerConfig::new().max_epochs(10).log_every_n_steps(1);
    let mut trainer = Trainer::new(config);
    trainer.fit(&mut model, &mut optimizer, train_batches, Some(val_batches))?;

    let val_batches = create_batches(&test_images, &test_labels, false)?;
    let mut total_correct = 0f32;
    let mut total_samples = 0usize;
    for batch in &val_batches {
        let logits = model.forward(&batch.images)?;
        let preds = logits.argmax(D::Minus1)?;
        let correct = preds
            .eq(&batch.labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        total_correct += correct;
        total_samples += batch.labels.dims1()?;
    }
    println!(
        "Final test accuracy: {:.2}%",
        100.0 * total_correct / total_samples as f32
    );

    Ok(())
}
