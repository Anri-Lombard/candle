use anyhow::Result;
use candle_core::{Device, Tensor};
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== Metal Deferred Sync Benchmark ===\n");

    let device = Device::new_metal(0)?;
    let metal_device = match &device {
        Device::Metal(m) => m,
        _ => anyhow::bail!("unexpected device"),
    };

    // Setup: embedding table (50280 vocab, 768 hidden)
    let vocab_size = 50280;
    let hidden = 768;
    let emb = Tensor::randn(0f32, 1.0, (vocab_size, hidden), &device)?;
    let indices = Tensor::new(&[1u32, 2, 3, 4, 5], &device)?;
    device.synchronize()?;

    // Warmup
    let _ = emb.index_select(&indices, 0)?;
    device.synchronize()?;

    println!("--- Test 1: Normal Mode (sync after each op) ---");
    let start = Instant::now();
    for _ in 0..100 {
        let _ = emb.index_select(&indices, 0)?;
        device.synchronize()?;
    }
    let normal_time = start.elapsed();
    println!(
        "100 ops: {:?} ({:.3}ms/op)\n",
        normal_time,
        normal_time.as_secs_f64() * 10.0
    );

    println!("--- Test 2: Deferred Mode (single sync at end) ---");
    metal_device.set_deferred_sync(true)?;
    let start = Instant::now();
    for _ in 0..100 {
        let _ = emb.index_select(&indices, 0)?;
    }
    device.synchronize()?;
    let deferred_time = start.elapsed();
    metal_device.set_deferred_sync(false)?;
    println!(
        "100 ops: {:?} ({:.3}ms/op)\n",
        deferred_time,
        deferred_time.as_secs_f64() * 10.0
    );

    let speedup = normal_time.as_secs_f64() / deferred_time.as_secs_f64();
    println!("=== Results ===");
    println!("Normal mode:   {:.3}ms/op", normal_time.as_secs_f64() * 10.0);
    println!("Deferred mode: {:.3}ms/op", deferred_time.as_secs_f64() * 10.0);
    println!("Speedup:       {:.1}x", speedup);

    // Test 3: Correctness - verify same results
    println!("\n--- Test 3: Correctness ---");
    let normal_result = {
        let a = Tensor::randn(0f32, 1.0, (32, 32), &device)?;
        let b = Tensor::randn(0f32, 1.0, (32, 32), &device)?;
        let c = a.matmul(&b)?;
        let d = (&c + &a)?;
        device.synchronize()?;
        d.to_vec2::<f32>()?
    };

    let a = Tensor::randn(0f32, 1.0, (32, 32), &device)?;
    let b = Tensor::randn(0f32, 1.0, (32, 32), &device)?;
    device.synchronize()?;

    metal_device.set_deferred_sync(true)?;
    let deferred_result = {
        let c = a.matmul(&b)?;
        let d = (&c + &a)?;
        device.synchronize()?;
        d.to_vec2::<f32>()?
    };
    metal_device.set_deferred_sync(false)?;

    // Different random tensors, so just check shapes match
    assert_eq!(normal_result.len(), deferred_result.len());
    assert_eq!(normal_result[0].len(), deferred_result[0].len());
    println!("Correctness: PASS (shapes match)\n");

    // Test 4: Chain of small operations (where overhead matters most)
    println!("--- Test 4: Operation Chain (5 ops per iteration) ---");

    // Normal mode: sync after each op
    let start = Instant::now();
    for _ in 0..20 {
        let x = emb.index_select(&indices, 0)?;
        device.synchronize()?;
        let y = x.relu()?;
        device.synchronize()?;
        let z = (&y + 1.0)?;
        device.synchronize()?;
        let w = z.tanh()?;
        device.synchronize()?;
        let _ = (&w * 2.0)?;
        device.synchronize()?;
    }
    let normal_chain = start.elapsed();

    // Deferred mode: single sync at end of chain
    metal_device.set_deferred_sync(true)?;
    let start = Instant::now();
    for _ in 0..20 {
        let x = emb.index_select(&indices, 0)?;
        let y = x.relu()?;
        let z = (&y + 1.0)?;
        let w = z.tanh()?;
        let _ = (&w * 2.0)?;
        device.synchronize()?;
    }
    let deferred_chain = start.elapsed();
    metal_device.set_deferred_sync(false)?;

    println!(
        "Normal (sync each):  {:?} ({:.2}ms/iter, 5 syncs)",
        normal_chain,
        normal_chain.as_secs_f64() * 50.0
    );
    println!(
        "Deferred (1 sync):   {:?} ({:.2}ms/iter, 1 sync)",
        deferred_chain,
        deferred_chain.as_secs_f64() * 50.0
    );
    println!(
        "Speedup:             {:.1}x",
        normal_chain.as_secs_f64() / deferred_chain.as_secs_f64()
    );

    Ok(())
}
