/// NeuralSync Secure Channel — Custom stream cipher
/// Protocol: XOR with PRNG-derived keystream (deterministic per-message)
/// Format: Each message is encoded as hex(enc_key[4] ++ xor(payload, keystream))
/// Security: PRNG keystream with 32-byte shared secret, unique 4-byte key per message

/// Shared 32-byte key. Coordinator and Engine MUST use the same value.
pub const SHARED_KEY: &[u8; 32] = b"NeuralSyncV2_TrainingKey_2025!!X";

/// Generate keystream from key + encryption key using simple PRNG.
/// Fast enough for traffic obfuscation.
fn keystream(key: &[u8; 32], enc_key: u32, len: usize) -> Vec<u8> {
    let mut ks = Vec::with_capacity(len);
    let mut state = [0u32; 8];
    for (i, chunk) in key.chunks(4).enumerate() {
        state[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    state[7] ^= enc_key;

    let mut block = 0u32;
    let mut pos   = 0;
    while ks.len() < len {
        // Feistel-style mixing
        block = block
            .wrapping_add(state[pos % 8])
            .wrapping_mul(0x6c62272e)
            .rotate_right(13)
            ^ enc_key
            ^ (pos as u32 * 0x9e3779b9);
        pos += 1;
        let bytes = block.to_le_bytes();
        for b in bytes {
            if ks.len() < len { ks.push(b); }
        }
    }
    ks
}

/// Encrypt: [enc_key(4)] ++ [xor(data, keystream)] → hex string
pub fn encrypt(data: &[u8], enc_key: u32) -> String {
    let ks  = keystream(SHARED_KEY, enc_key, data.len());
    let mut enc = Vec::with_capacity(4 + data.len());

    // prepend 4-byte encryption key (little-endian)
    enc.extend_from_slice(&enc_key.to_le_bytes());
    // XOR encrypt
    for (d, k) in data.iter().zip(ks.iter()) {
        enc.push(d ^ k);
    }

    hex::encode(&enc)
}

/// Decrypt: hex string → original bytes
pub fn decrypt(hex_str: &str) -> Option<Vec<u8>> {
    if hex_str.len() < 8 || hex_str.len() % 2 != 0 { return None; }
    let bytes = hex::decode(hex_str).ok()?;
    if bytes.len() < 5 { return None; }

    let enc_key = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let ciphertext = &bytes[4..];

    let ks = keystream(SHARED_KEY, enc_key, ciphertext.len());
    let plain: Vec<u8> = ciphertext.iter().zip(ks.iter()).map(|(c, k)| c ^ k).collect();
    Some(plain)
}

/// Encrypt JSON string → hex string for WS transport
pub fn encrypt_json(json: &str, enc_key: u32) -> String {
    encrypt(json.as_bytes(), enc_key)
}

/// Decrypt hex string → JSON string
pub fn decrypt_json(hex_str: &str) -> Option<String> {
    let plain = decrypt(hex_str)?;
    String::from_utf8(plain).ok()
}

/// Generate encryption key from timestamp + random component
pub fn new_enc_key() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    // XOR with constant to ensure uniqueness within same millisecond
    t ^ 0xDEADBEEF
}

/// Quick check if a string looks like encrypted hex
pub fn looks_encrypted(s: &str) -> bool {
    s.len() >= 10
        && s.len() % 2 == 0
        && s.as_bytes()[0] != b'['
        && s.as_bytes()[0] != b'{'
        && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Convenience struct wrapper for coordinator compatibility.
/// Usage: `let cipher = Cipher::new(); cipher.encrypt(...)`
pub struct Cipher;

impl Cipher {
    pub fn new() -> Self { Cipher }

    pub fn encrypt_msg(&self, data: &[u8]) -> String {
        encrypt(data, new_enc_key())
    }

    pub fn decrypt_msg(&self, hex_str: &str) -> Option<Vec<u8>> {
        decrypt(hex_str)
    }

    pub fn encrypt_json_msg(&self, json: &str) -> String {
        encrypt_json(json, new_enc_key())
    }

    pub fn decrypt_json_msg(&self, hex_str: &str) -> Option<String> {
        decrypt_json(hex_str)
    }

    pub fn is_encrypted(&self, s: &str) -> bool {
        looks_encrypted(s)
    }
}
