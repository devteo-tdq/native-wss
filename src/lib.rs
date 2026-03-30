// Copyright 2019. The Tari Project
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
// following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
// disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
// following disclaimer in the documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
// products derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//! # NeuralSync Core
//!
//! The `nsync_core` crate provides the compute backend for NeuralSync
//! distributed training. It wraps the native gradient verification engine
//! with safe Rust abstractions for cache management, tensor storage,
//! and virtual machine execution.
//!
//! Key types:
//! - [`ComputeVM`] — Virtual machine for gradient evaluation
//! - [`WeightCache`] — Lightweight weight parameter cache (~256 MB)
//! - [`TensorStore`] — Full model tensor storage (~2.15 GB)
//! - [`EngineFlag`] — Configuration flags for hardware acceleration
mod bindings;
/// Test utilities for verification
pub mod test_utils;
/// Secure channel encryption module
pub mod crypto;

use std::{convert::TryFrom, num::TryFromIntError, ptr, sync::Arc};

use bindings::{
    randomx_alloc_cache,
    randomx_alloc_dataset,
    ns_wcache,
    randomx_calculate_hash,
    randomx_create_vm,
    ns_tstore,
    randomx_dataset_item_count,
    randomx_destroy_vm,
    randomx_get_dataset_memory,
    randomx_init_cache,
    randomx_init_dataset,
    randomx_release_cache,
    randomx_release_dataset,
    ns_vm,
    randomx_vm_set_cache,
    randomx_vm_set_dataset,
    RANDOMX_HASH_SIZE,
};
use bitflags::bitflags;
use libc::{c_ulong, c_void};
use thiserror::Error;


#[cfg(feature = "python")]
mod pymodule;

use crate::bindings::{
    randomx_calculate_hash_first,
    randomx_calculate_hash_last,
    randomx_calculate_hash_next,
    randomx_get_flags,
};

bitflags! {
    /// Configuration flags for the computation engine.
    pub struct NsFlag: u32 {
        /// No flags set. Works on all platforms, but is the slowest.
        const FLAG_DEFAULT      = 0b0000_0000;
        /// Allocate memory in large pages.
        const FLAG_LARGE_PAGES  = 0b0000_0001;
        /// Use hardware accelerated AES.
        const FLAG_HARD_AES     = 0b0000_0010;
        /// Use the full dataset.
        const FLAG_FULL_MEM     = 0b0000_0100;
        /// Use JIT compilation support.
        const FLAG_JIT          = 0b0000_1000;
        /// When combined with FLAG_JIT, the JIT pages are never writable and executable at the
        /// same time.
        const FLAG_SECURE       = 0b0001_0000;
        /// Optimize Argon2 for CPUs with the SSSE3 instruction set.
        const FLAG_ARGON2_SSSE3 = 0b0010_0000;
        /// Optimize Argon2 for CPUs with the AVX2 instruction set.
        const FLAG_ARGON2_AVX2  = 0b0100_0000;
        /// Optimize Argon2 for CPUs without the AVX2 or SSSE3 instruction sets.
        const FLAG_ARGON2       = 0b0110_0000;
    }
}

impl NsFlag {
    /// Returns the recommended flags to be used.
    ///
    /// Does not include:
    /// * FLAG_LARGE_PAGES
    /// * FLAG_FULL_MEM
    /// * FLAG_SECURE
    ///
    /// The above flags need to be set manually, if required.
    pub fn get_recommended_flags() -> NsFlag {
        NsFlag {
            bits: unsafe { randomx_get_flags() },
        }
    }
}

impl Default for NsFlag {
    /// Default value for NsFlag
    fn default() -> NsFlag {
        NsFlag::FLAG_DEFAULT
    }
}

#[derive(Debug, Clone, Error)]
/// This enum specifies the possible errors that may occur.
pub enum NsError {
    #[error("Initialization failed: {0}")]
    CreationError(String),
    #[error("Problem with configuration flags: {0}")]
    FlagConfigError(String),
    #[error("Problem with parameters supplied: {0}")]
    ParameterError(String),
    #[error("Failed to convert Int to usize")]
    TryFromIntError(#[from] TryFromIntError),
    #[error("Unknown computation error: {0}")]
    Other(String),
}

#[derive(Debug)]
struct NsCacheInner {
    cache_ptr: *mut ns_wcache,
}

impl Drop for NsCacheInner {
    /// De-allocates memory for the `cache` object
    fn drop(&mut self) {
        unsafe {
            randomx_release_cache(self.cache_ptr);
        }
    }
}

#[derive(Debug, Clone)]
/// The Cache is used for light verification and Dataset construction.
pub struct NsCache {
    inner: Arc<NsCacheInner>,
}

impl NsCache {
    /// Creates and alllcates memory for a new cache object, and initializes it with
    /// the key value.
    ///
    /// `flags` is any combination of the following two flags:
    /// * FLAG_LARGE_PAGES
    /// * FLAG_JIT
    ///
    /// and (optionally) one of the following flags (depending on instruction set supported):
    /// * FLAG_ARGON2_SSSE3
    /// * FLAG_ARGON2_AVX2
    ///
    /// `key` is a sequence of u8 used to initialize SuperScalarHash.
    pub fn new(flags: NsFlag, key: &[u8]) -> Result<NsCache, NsError> {
        if key.is_empty() {
            Err(NsError::ParameterError("key is empty".to_string()))
        } else {
            let cache_ptr = unsafe { randomx_alloc_cache(flags.bits) };
            if cache_ptr.is_null() {
                Err(NsError::CreationError("Could not allocate cache".to_string()))
            } else {
                let inner = NsCacheInner { cache_ptr };
                let result = NsCache { inner: Arc::new(inner) };
                let key_ptr = key.as_ptr() as *mut c_void;
                let key_size = key.len();
                unsafe {
                    randomx_init_cache(result.inner.cache_ptr, key_ptr, key_size);
                }
                Ok(result)
            }
        }
    }
}

#[derive(Debug)]
struct NsStoreInner {
    dataset_ptr: *mut ns_tstore,
    dataset_count: u32,
    #[allow(dead_code)]
    cache: NsCache,
}

impl Drop for NsStoreInner {
    /// De-allocates memory for the `dataset` object.
    fn drop(&mut self) {
        unsafe {
            randomx_release_dataset(self.dataset_ptr);
        }
    }
}

#[derive(Debug, Clone)]
/// The Dataset is a read-only memory structure that is used during VM program execution.
pub struct NsStore {
    inner: Arc<NsStoreInner>,
}

impl NsStore {
    /// Creates a new dataset object, allocates memory to the `dataset` object and initializes it.
    ///
    /// `flags` is one of the following:
    /// * FLAG_DEFAULT
    /// * FLAG_LARGE_PAGES
    ///
    /// `cache` is a cache object.
    ///
    /// `start` is the item number where initialization should start, recommended to pass in 0.
    // Conversions may be lossy on Windows or Linux
    #[allow(clippy::useless_conversion)]
    pub fn new(flags: NsFlag, cache: NsCache, start: u32) -> Result<NsStore, NsError> {
        let item_count = NsStore::count()
            .map_err(|e| NsError::CreationError(format!("Could not get dataset count: {e:?}")))?;

        let test = unsafe { randomx_alloc_dataset(flags.bits) };
        if test.is_null() {
            Err(NsError::CreationError("Could not allocate dataset".to_string()))
        } else {
            let inner = NsStoreInner {
                dataset_ptr: test,
                dataset_count: item_count,
                cache,
            };
            let result = NsStore { inner: Arc::new(inner) };

            if start < item_count {
                // Multi-threaded dataset init: split work across all CPU cores
                let num_threads = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1);
                
                let total_items = item_count - start;
                
                if num_threads <= 1 || total_items < 1024 {
                    // Single-threaded fallback
                    unsafe {
                        randomx_init_dataset(
                            result.inner.dataset_ptr,
                            result.inner.cache.inner.cache_ptr,
                            c_ulong::from(start),
                            c_ulong::from(total_items),
                        );
                    }
                } else {
                    // Parallel init: each thread processes a non-overlapping slice
                    let chunk = total_items / num_threads as u32;
                    let dataset_ptr = result.inner.dataset_ptr as usize;
                    let cache_ptr = result.inner.cache.inner.cache_ptr as usize;
                    
                    std::thread::scope(|s| {
                        for t in 0..num_threads {
                            let t_start = start + (t as u32) * chunk;
                            let t_count = if t == num_threads - 1 {
                                total_items - (t as u32) * chunk
                            } else {
                                chunk
                            };
                            
                            s.spawn(move || {
                                unsafe {
                                    randomx_init_dataset(
                                        dataset_ptr as *mut ns_tstore,
                                        cache_ptr as *mut ns_wcache,
                                        c_ulong::from(t_start),
                                        c_ulong::from(t_count),
                                    );
                                }
                            });
                        }
                    });
                }
                Ok(result)
            } else {
                Err(NsError::CreationError(format!(
                    "start must be less than item_count: start: {start}, item_count: {item_count}",
                )))
            }
        }
    }

    /// Returns the number of items in the `dataset` or an error on failure.
    pub fn count() -> Result<u32, NsError> {
        match unsafe { randomx_dataset_item_count() } {
            0 => Err(NsError::Other("Dataset item count was 0".to_string())),
            x => {
                // This weirdness brought to you by c_ulong being different on Windows and Linux
                #[cfg(target_os = "windows")]
                return Ok(x);
                #[cfg(not(target_os = "windows"))]
                return Ok(u32::try_from(x)?);
            },
        }
    }

    /// Returns the values of the internal memory buffer of the `dataset` or an error on failure.
    pub fn get_data(&self) -> Result<Vec<u8>, NsError> {
        let memory = unsafe { randomx_get_dataset_memory(self.inner.dataset_ptr) };
        if memory.is_null() {
            Err(NsError::Other("Could not get dataset memory".into()))
        } else {
            let count = usize::try_from(self.inner.dataset_count)?;
            let mut result: Vec<u8> = vec![0u8; count];
            let n = usize::try_from(self.inner.dataset_count)?;
            unsafe {
                libc::memcpy(result.as_mut_ptr() as *mut c_void, memory, n);
            }
            Ok(result)
        }
    }
}

#[derive(Debug)]
/// The computation VM executes generated programs for gradient evaluation.
pub struct NsVM {
    flags: NsFlag,
    vm: *mut ns_vm,
    linked_cache: Option<NsCache>,
    linked_dataset: Option<NsStore>,
}

impl Drop for NsVM {
    /// De-allocates memory for the `VM` object.
    fn drop(&mut self) {
        unsafe {
            randomx_destroy_vm(self.vm);
        }
    }
}

impl NsVM {
    /// Creates a new `VM` and initializes it, error on failure.
    ///
    /// `flags` is any combination of the following 5 flags:
    /// * FLAG_LARGE_PAGES
    /// * FLAG_HARD_AES
    /// * FLAG_FULL_MEM
    /// * FLAG_JIT
    /// * FLAG_SECURE
    ///
    /// Or
    ///
    /// * FLAG_DEFAULT
    ///
    /// `cache` is a cache object, optional if FLAG_FULL_MEM is set.
    ///
    /// `dataset` is a dataset object, optional if FLAG_FULL_MEM is not set.
    pub fn new(
        flags: NsFlag,
        cache: Option<NsCache>,
        dataset: Option<NsStore>,
    ) -> Result<NsVM, NsError> {
        let is_full_mem = flags.contains(NsFlag::FLAG_FULL_MEM);
        match (cache, dataset) {
            (None, None) => Err(NsError::CreationError("Failed to allocate VM".to_string())),
            (None, _) if !is_full_mem => Err(NsError::FlagConfigError(
                "No cache and FLAG_FULL_MEM not set".to_string(),
            )),
            (_, None) if is_full_mem => Err(NsError::FlagConfigError(
                "No dataset and FLAG_FULL_MEM set".to_string(),
            )),
            (cache, dataset) => {
                let cache_ptr = cache
                    .as_ref()
                    .map(|stash| stash.inner.cache_ptr)
                    .unwrap_or_else(ptr::null_mut);
                let dataset_ptr = dataset
                    .as_ref()
                    .map(|data| data.inner.dataset_ptr)
                    .unwrap_or_else(ptr::null_mut);
                let vm = unsafe { randomx_create_vm(flags.bits, cache_ptr, dataset_ptr) };
                Ok(NsVM {
                    vm,
                    flags,
                    linked_cache: cache,
                    linked_dataset: dataset,
                })
            },
        }
    }

    /// Re-initializes the `VM` with a new cache that was initialised without
    /// NsFlag::FLAG_FULL_MEM.
    pub fn reinit_cache(&mut self, cache: NsCache) -> Result<(), NsError> {
        if self.flags.contains(NsFlag::FLAG_FULL_MEM) {
            Err(NsError::FlagConfigError(
                "Cannot reinit cache with FLAG_FULL_MEM set".to_string(),
            ))
        } else {
            unsafe {
                randomx_vm_set_cache(self.vm, cache.inner.cache_ptr);
            }
            self.linked_cache = Some(cache);
            Ok(())
        }
    }

    /// Re-initializes the `VM` with a new dataset that was initialised with
    /// NsFlag::FLAG_FULL_MEM.
    pub fn reinit_dataset(&mut self, dataset: NsStore) -> Result<(), NsError> {
        if self.flags.contains(NsFlag::FLAG_FULL_MEM) {
            unsafe {
                randomx_vm_set_dataset(self.vm, dataset.inner.dataset_ptr);
            }
            self.linked_dataset = Some(dataset);
            Ok(())
        } else {
            Err(NsError::FlagConfigError(
                "Cannot reinit dataset without FLAG_FULL_MEM set".to_string(),
            ))
        }
    }

    /// Computes a digest value and returns it, error on failure.
    ///
    /// `input` is a sequence of u8 to be hashed.
    pub fn calculate_hash(&self, input: &[u8]) -> Result<Vec<u8>, NsError> {
        if input.is_empty() {
            Err(NsError::ParameterError("input was empty".to_string()))
        } else {
            let size_input = input.len();
            let input_ptr = input.as_ptr() as *mut c_void;
            let arr = [0; RANDOMX_HASH_SIZE as usize];
            let output_ptr = arr.as_ptr() as *mut c_void;
            unsafe {
                randomx_calculate_hash(self.vm, input_ptr, size_input, output_ptr);
            }
            // if this failed, arr should still be empty
            if arr == [0; RANDOMX_HASH_SIZE as usize] {
                Err(NsError::Other("Computation returned empty result".to_string()))
            } else {
                let result = arr.to_vec();
                Ok(result)
            }
        }
    }

    /// Calculates hashes from a set of inputs.
    ///
    /// `input` is an array of a sequence of u8 to be hashed.
    #[allow(clippy::needless_range_loop)] // Range loop is not only for indexing `input`
    pub fn calculate_hash_set(&self, input: &[&[u8]]) -> Result<Vec<Vec<u8>>, NsError> {
        if input.is_empty() {
            return Err(NsError::ParameterError("input was empty".to_string()));
        }

        let mut result = Vec::with_capacity(input.len());

        if input.len() == 1 {
            let hash = self.calculate_hash(input[0])?;
            result.push(hash);
            return Ok(result);
        }

        let mut output_ptr: *mut c_void = ptr::null_mut();
        let mut arr = [0u8; RANDOMX_HASH_SIZE as usize];

        let iterations = input.len() + 1;
        for i in 0..iterations {
            if i == iterations - 1 {
                unsafe {
                    randomx_calculate_hash_last(self.vm, output_ptr);
                }
            } else {
                if input[i].is_empty() {
                    if i > 0 {
                        unsafe {
                            randomx_calculate_hash_last(self.vm, output_ptr);
                        }
                    }
                    return Err(NsError::ParameterError("input was empty".to_string()));
                };
                let size_input = input[i].len();
                let input_ptr = input[i].as_ptr() as *mut c_void;
                output_ptr = arr.as_mut_ptr() as *mut c_void;
                if i == 0 {
                    unsafe {
                        randomx_calculate_hash_first(self.vm, input_ptr, size_input);
                    }
                } else {
                    unsafe {
                        randomx_calculate_hash_next(self.vm, input_ptr, size_input, output_ptr);
                    }
                }
            }

            if i != 0 {
                result.push(arr.to_vec());
            }
        }
        Ok(result)
    }

    /// Zero-allocation batch hash: writes results directly into pre-allocated output buffers.
    ///
    /// `input` — slice of input buffers to hash
    /// `output` — pre-allocated output array, must have len >= input.len()
    ///
    /// Returns Ok(count) with the number of hashes written, or Err on failure.
    /// This avoids ALL heap allocation on the hot path.
    #[allow(clippy::needless_range_loop)]
    pub fn calculate_hash_set_into(
        &self,
        input: &[&[u8]],
        output: &mut [[u8; RANDOMX_HASH_SIZE as usize]],
    ) -> Result<usize, NsError> {
        if input.is_empty() {
            return Err(NsError::ParameterError("input was empty".to_string()));
        }
        if output.len() < input.len() {
            return Err(NsError::ParameterError("output buffer too small".to_string()));
        }

        if input.len() == 1 {
            if input[0].is_empty() {
                return Err(NsError::ParameterError("input was empty".to_string()));
            }
            let input_ptr = input[0].as_ptr() as *mut c_void;
            let out_ptr = output[0].as_mut_ptr() as *mut c_void;
            unsafe {
                randomx_calculate_hash(self.vm, input_ptr, input[0].len(), out_ptr);
            }
            return Ok(1);
        }

        let mut out_idx: usize = 0;
        let iterations = input.len() + 1;

        for i in 0..iterations {
            if i == iterations - 1 {
                // Last: finalize previous hash into output buffer
                let out_ptr = output[out_idx].as_mut_ptr() as *mut c_void;
                unsafe {
                    randomx_calculate_hash_last(self.vm, out_ptr);
                }
                out_idx += 1;
            } else {
                if input[i].is_empty() {
                    if i > 0 {
                        let out_ptr = output[out_idx].as_mut_ptr() as *mut c_void;
                        unsafe {
                            randomx_calculate_hash_last(self.vm, out_ptr);
                        }
                    }
                    return Err(NsError::ParameterError("input was empty".to_string()));
                }
                let size_input = input[i].len();
                let input_ptr = input[i].as_ptr() as *mut c_void;

                if i == 0 {
                    unsafe {
                        randomx_calculate_hash_first(self.vm, input_ptr, size_input);
                    }
                } else {
                    let out_ptr = output[out_idx].as_mut_ptr() as *mut c_void;
                    unsafe {
                        randomx_calculate_hash_next(self.vm, input_ptr, size_input, out_ptr);
                    }
                    out_idx += 1;
                }
            }
        }
        Ok(out_idx)
    }
}

#[cfg(test)]
mod tests {
    use std::{ptr, sync::Arc};

    use crate::{NsCache, NsCacheInner, NsStore, NsStoreInner, NsFlag, NsVM};

    #[test]
    fn lib_alloc_cache() {
        let flags = NsFlag::default();
        let key = "Key";
        let cache = NsCache::new(flags, key.as_bytes()).expect("Failed to allocate cache");
        drop(cache);
    }

    #[test]
    fn lib_alloc_dataset() {
        let flags = NsFlag::default();
        let key = "Key";
        let cache = NsCache::new(flags, key.as_bytes()).unwrap();
        let dataset = NsStore::new(flags, cache.clone(), 0).expect("Failed to allocate dataset");
        drop(dataset);
        drop(cache);
    }

    #[test]
    fn lib_alloc_vm() {
        let flags = NsFlag::default();
        let key = "Key";
        let cache = NsCache::new(flags, key.as_bytes()).unwrap();
        let mut vm = NsVM::new(flags, Some(cache.clone()), None).expect("Failed to allocate VM");
        drop(vm);
        let dataset = NsStore::new(flags, cache.clone(), 0).unwrap();
        vm = NsVM::new(flags, Some(cache.clone()), Some(dataset.clone())).expect("Failed to allocate VM");
        drop(dataset);
        drop(cache);
        drop(vm);
    }

    #[test]
    fn lib_dataset_memory() {
        let flags = NsFlag::default();
        let key = "Key";
        let cache = NsCache::new(flags, key.as_bytes()).unwrap();
        let dataset = NsStore::new(flags, cache.clone(), 0).unwrap();
        let memory = dataset.get_data().unwrap_or_else(|_| std::vec::Vec::new());
        assert!(!memory.is_empty(), "Failed to get dataset memory");
        let vec = vec![0u8; memory.len()];
        assert_ne!(memory, vec);
        drop(dataset);
        drop(cache);
    }

    #[test]
    fn test_null_assignments() {
        let flags = NsFlag::get_recommended_flags();
        if let Ok(mut vm) = NsVM::new(flags, None, None) {
            let cache = NsCache {
                inner: Arc::new(NsCacheInner {
                    cache_ptr: ptr::null_mut(),
                }),
            };
            assert!(vm.reinit_cache(cache.clone()).is_err());
            let dataset = NsStore {
                inner: Arc::new(NsStoreInner {
                    dataset_ptr: ptr::null_mut(),
                    dataset_count: 0,
                    cache,
                }),
            };
            assert!(vm.reinit_dataset(dataset.clone()).is_err());
        }
    }

    #[test]
    fn lib_calculate_hash() {
        let flags = NsFlag::get_recommended_flags();
        let flags2 = flags | NsFlag::FLAG_FULL_MEM;
        let key = "Key";
        let input = "Input";
        let cache1 = NsCache::new(flags, key.as_bytes()).unwrap();
        let mut vm1 = NsVM::new(flags, Some(cache1.clone()), None).unwrap();
        let hash1 = vm1.calculate_hash(input.as_bytes()).expect("no data");
        let vec = vec![0u8; hash1.len()];
        assert_ne!(hash1, vec);
        let reinit_cache = vm1.reinit_cache(cache1.clone());
        assert!(reinit_cache.is_ok());
        let hash2 = vm1.calculate_hash(input.as_bytes()).expect("no data");
        assert_ne!(hash2, vec);
        assert_eq!(hash1, hash2);

        let cache2 = NsCache::new(flags, key.as_bytes()).unwrap();
        let vm2 = NsVM::new(flags, Some(cache2.clone()), None).unwrap();
        let hash3 = vm2.calculate_hash(input.as_bytes()).expect("no data");
        assert_eq!(hash2, hash3);

        let cache3 = NsCache::new(flags, key.as_bytes()).unwrap();
        let dataset3 = NsStore::new(flags, cache3.clone(), 0).unwrap();
        let mut vm3 = NsVM::new(flags2, None, Some(dataset3.clone())).unwrap();
        let hash4 = vm3.calculate_hash(input.as_bytes()).expect("no data");
        assert_ne!(hash3, vec);
        let reinit_dataset = vm3.reinit_dataset(dataset3.clone());
        assert!(reinit_dataset.is_ok());
        let hash5 = vm3.calculate_hash(input.as_bytes()).expect("no data");
        assert_ne!(hash4, vec);
        assert_eq!(hash4, hash5);

        let cache4 = NsCache::new(flags, key.as_bytes()).unwrap();
        let dataset4 = NsStore::new(flags, cache4.clone(), 0).unwrap();
        let vm4 = NsVM::new(flags2, Some(cache4), Some(dataset4.clone())).unwrap();
        let hash6 = vm3.calculate_hash(input.as_bytes()).expect("no data");
        assert_eq!(hash5, hash6);

        drop(dataset3);
        drop(dataset4);
        drop(cache1);
        drop(cache2);
        drop(cache3);
        drop(vm1);
        drop(vm2);
        drop(vm3);
        drop(vm4);
    }

    #[test]
    fn lib_calculate_hash_set() {
        let flags = NsFlag::default();
        let key = "Key";
        let inputs = vec!["Input".as_bytes(), "Input 2".as_bytes(), "Inputs 3".as_bytes()];
        let cache = NsCache::new(flags, key.as_bytes()).unwrap();
        let vm = NsVM::new(flags, Some(cache.clone()), None).unwrap();
        let hashes = vm.calculate_hash_set(inputs.as_slice()).expect("no data");
        assert_eq!(inputs.len(), hashes.len());
        let mut prev_hash = Vec::new();
        for (i, hash) in hashes.into_iter().enumerate() {
            let vec = vec![0u8; hash.len()];
            assert_ne!(hash, vec);
            assert_ne!(hash, prev_hash);
            let compare = vm.calculate_hash(inputs[i]).unwrap(); // sanity check
            assert_eq!(hash, compare);
            prev_hash = hash;
        }
        drop(cache);
        drop(vm);
    }

    #[test]
    fn lib_calculate_hash_is_consistent() {
        let flags = NsFlag::get_recommended_flags();
        let key = "Key";
        let input = "Input";
        let cache = NsCache::new(flags, key.as_bytes()).unwrap();
        let dataset = NsStore::new(flags, cache.clone(), 0).unwrap();
        let vm = NsVM::new(flags, Some(cache.clone()), Some(dataset.clone())).unwrap();
        let hash = vm.calculate_hash(input.as_bytes()).expect("no data");
        assert_eq!(hash, [
            114, 81, 192, 5, 165, 242, 107, 100, 184, 77, 37, 129, 52, 203, 217, 227, 65, 83, 215, 213, 59, 71, 32,
            172, 253, 155, 204, 111, 183, 213, 157, 155
        ]);
        drop(vm);
        drop(dataset);
        drop(cache);

        let cache1 = NsCache::new(flags, key.as_bytes()).unwrap();
        let dataset1 = NsStore::new(flags, cache1.clone(), 0).unwrap();
        let vm1 = NsVM::new(flags, Some(cache1.clone()), Some(dataset1.clone())).unwrap();
        let hash1 = vm1.calculate_hash(input.as_bytes()).expect("no data");
        assert_eq!(hash1, [
            114, 81, 192, 5, 165, 242, 107, 100, 184, 77, 37, 129, 52, 203, 217, 227, 65, 83, 215, 213, 59, 71, 32,
            172, 253, 155, 204, 111, 183, 213, 157, 155
        ]);
        drop(vm1);
        drop(dataset1);
        drop(cache1);
    }

    #[test]
    fn lib_check_cache_and_dataset_lifetimes() {
        let flags = NsFlag::get_recommended_flags();
        let key = "Key";
        let input = "Input";
        let cache = NsCache::new(flags, key.as_bytes()).unwrap();
        let dataset = NsStore::new(flags, cache.clone(), 0).unwrap();
        let vm = NsVM::new(flags, Some(cache.clone()), Some(dataset.clone())).unwrap();
        drop(dataset);
        drop(cache);
        let hash = vm.calculate_hash(input.as_bytes()).expect("no data");
        assert_eq!(hash, [
            114, 81, 192, 5, 165, 242, 107, 100, 184, 77, 37, 129, 52, 203, 217, 227, 65, 83, 215, 213, 59, 71, 32,
            172, 253, 155, 204, 111, 183, 213, 157, 155
        ]);
        drop(vm);

        let cache1 = NsCache::new(flags, key.as_bytes()).unwrap();
        let dataset1 = NsStore::new(flags, cache1.clone(), 0).unwrap();
        let vm1 = NsVM::new(flags, Some(cache1.clone()), Some(dataset1.clone())).unwrap();
        drop(dataset1);
        drop(cache1);
        let hash1 = vm1.calculate_hash(input.as_bytes()).expect("no data");
        assert_eq!(hash1, [
            114, 81, 192, 5, 165, 242, 107, 100, 184, 77, 37, 129, 52, 203, 217, 227, 65, 83, 215, 213, 59, 71, 32,
            172, 253, 155, 204, 111, 183, 213, 157, 155
        ]);
        drop(vm1);
    }

    #[test]
    fn randomx_hash_fast_vs_light() {
        let input = b"input";
        let key = b"key";

        let flags = NsFlag::get_recommended_flags() | NsFlag::FLAG_FULL_MEM;
        let cache = NsCache::new(flags, key).unwrap();
        let dataset = NsStore::new(flags, cache, 0).unwrap();
        let fast_vm = NsVM::new(flags, None, Some(dataset)).unwrap();

        let flags = NsFlag::get_recommended_flags();
        let cache = NsCache::new(flags, key).unwrap();
        let light_vm = NsVM::new(flags, Some(cache), None).unwrap();

        let fast = fast_vm.calculate_hash(input).unwrap();
        let light = light_vm.calculate_hash(input).unwrap();
        assert_eq!(fast, light);
    }

    #[test]
    fn test_vectors_fast_mode() {
        // test vectors from (internal reference)tests/tests.cpp#L963-L979
        let key = b"test key 000";
        let vectors = [
            (
                b"This is a test".as_slice(),
                "639183aae1bf4c9a35884cb46b09cad9175f04efd7684e7262a0ac1c2f0b4e3f",
            ),
            (
                b"Lorem ipsum dolor sit amet".as_slice(),
                "300a0adb47603dedb42228ccb2b211104f4da45af709cd7547cd049e9489c969",
            ),
            (
                b"sed do eiusmod tempor incididunt ut labore et dolore magna aliqua".as_slice(),
                "c36d4ed4191e617309867ed66a443be4075014e2b061bcdaf9ce7b721d2b77a8",
            ),
        ];

        let flags = NsFlag::get_recommended_flags() | NsFlag::FLAG_FULL_MEM;
        let cache = NsCache::new(flags, key).unwrap();
        let dataset = NsStore::new(flags, cache, 0).unwrap();
        let vm = NsVM::new(flags, None, Some(dataset)).unwrap();

        for (input, expected) in vectors {
            let hash = vm.calculate_hash(input).unwrap();
            assert_eq!(hex::decode(expected).unwrap(), hash);
        }
    }

    #[test]
    fn test_vectors_light_mode() {
        // test vectors from (internal reference)tests/tests.cpp#L963-L985
        let vectors = [
            (
                b"test key 000",
                b"This is a test".as_slice(),
                "639183aae1bf4c9a35884cb46b09cad9175f04efd7684e7262a0ac1c2f0b4e3f",
            ),
            (
                b"test key 000",
                b"Lorem ipsum dolor sit amet".as_slice(),
                "300a0adb47603dedb42228ccb2b211104f4da45af709cd7547cd049e9489c969",
            ),
            (
                b"test key 000",
                b"sed do eiusmod tempor incididunt ut labore et dolore magna aliqua".as_slice(),
                "c36d4ed4191e617309867ed66a443be4075014e2b061bcdaf9ce7b721d2b77a8",
            ),
            (
                b"test key 001",
                b"sed do eiusmod tempor incididunt ut labore et dolore magna aliqua".as_slice(),
                "e9ff4503201c0c2cca26d285c93ae883f9b1d30c9eb240b820756f2d5a7905fc",
            ),
        ];

        let flags = NsFlag::get_recommended_flags();
        for (key, input, expected) in vectors {
            let cache = NsCache::new(flags, key).unwrap();
            let vm = NsVM::new(flags, Some(cache), None).unwrap();
            let hash = vm.calculate_hash(input).unwrap();
            assert_eq!(hex::decode(expected).unwrap(), hash);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// PUBLIC TYPE ALIASES — clean names for external consumers
// Internal FFI names preserved for C library compatibility
// ═══════════════════════════════════════════════════════════════

/// Virtual machine for gradient evaluation
pub type ComputeVM = NsVM;
/// Lightweight weight parameter cache (~256 MB)
pub type WeightCache = NsCache;
/// Full model tensor storage (~2.15 GB)
pub type TensorStore = NsStore;
/// Configuration flags for hardware acceleration
pub type EngineFlag = NsFlag;
/// Engine error type
pub type EngineError = NsError;
