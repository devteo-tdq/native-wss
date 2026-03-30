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

use crate::{NsCache, NsStore, NsFlag, NsVM};

/// Fuzzing:
/// - `pub fn ns_alloc_cache`
/// - `pub fn ns_get_flags`
/// - `pub fn ns_init_cache`
/// - `pub fn ns_release_cache`
#[allow(clippy::needless_pass_by_value)] // This is required by the `QuickCheck` fuzzing framework
pub fn fuzz_ns_alloc_cache(data: Vec<u8>) -> bool {
    let flags = if data.is_empty() {
        NsFlag::default()
    } else {
        match data[0] % 10 {
            0 => NsFlag::get_recommended_flags(),
            1 => NsFlag::FLAG_DEFAULT,
            2 => NsFlag::FLAG_LARGE_PAGES,
            3 => NsFlag::FLAG_HARD_AES,
            4 => NsFlag::FLAG_FULL_MEM,
            5 => NsFlag::FLAG_JIT,
            6 => NsFlag::FLAG_SECURE,
            7 => NsFlag::FLAG_ARGON2_SSSE3,
            8 => NsFlag::FLAG_ARGON2_AVX2,
            _ => NsFlag::FLAG_ARGON2,
        }
    };
    let _unused = NsCache::new(flags, &data);
    true
}

/// Fuzzing:
/// - `pub fn ns_create_vm`
/// - `pub fn ns_destroy_vm`
/// - `pub fn ns_vm_set_cache`
/// - `pub fn ns_alloc_cache`
/// - `pub fn ns_get_flags`
/// - `pub fn ns_init_cache`
/// - `pub fn ns_release_cache`
#[allow(clippy::needless_pass_by_value)] // This is required by the `QuickCheck` fuzzing framework

pub fn fuzz_ns_create_vm_with_cache_only(data: Vec<u8>) -> bool {
    let flags = NsFlag::get_recommended_flags();
    if let Ok(cache) = NsCache::new(flags, &data) {
        if let Ok(mut vm) = NsVM::new(flags, Some(cache.clone()), None) {
            let _unused = vm.reinit_cache(cache);
        }
    }
    true
}

/// Fuzzing:
/// - `pub fn ns_get_flags`
/// - `pub fn ns_create_vm`
/// - `pub fn ns_destroy_vm`
/// - `pub fn ns_tstore_count`
/// - `pub fn ns_alloc_cache`
/// - `pub fn ns_init_cache`
/// - `pub fn ns_release_cache`
/// - `pub fn ns_alloc_tstore`
/// - `pub fn ns_init_store`
/// - `pub fn ns_release_tstore`
/// - `pub fn ns_vm_set_cache`
/// - `pub fn ns_vm_set_tstore`
/// - `pub fn ns_tstore_count`
/// - `pub fn ns_get_tstore_mem`
#[allow(clippy::needless_pass_by_value)] // This is required by the `QuickCheck` fuzzing framework
pub fn fuzz_ns_create_vm_with_cache_and_dataset(data: Vec<u8>) -> bool {
    let flags = NsFlag::get_recommended_flags();
    if let Ok(cache) = NsCache::new(flags, &data) {
        let start = if data.is_empty() { 0u32 } else { u32::from(data[0] % 3) };
        if let Ok(dataset) = NsStore::new(flags, cache.clone(), start) {
            for _ in 0..100 {
                let _unused = dataset.get_data();
            }
            if let Ok(mut vm) = NsVM::new(flags, Some(cache.clone()), Some(dataset.clone())) {
                let _unused = vm.reinit_cache(cache);
                let _unused = vm.reinit_dataset(dataset);
            }
        }
    }
    true
}

// Helper function to perform compute evaluations
fn calculate_hashes(hash_data: &[u8], vm: &mut NsVM, iterations: u8) {
    let mut hash_data = hash_data.to_vec();
    for _ in 0..iterations {
        if hash_data.len() > 1 {
            // Prepare input data
            let hash_set_len = 12;
            let mut hash_set = Vec::with_capacity(hash_set_len);
            for _ in 0..hash_set_len {
                let mut scratch_data = hash_data.clone();
                let last = scratch_data.pop().unwrap();
                scratch_data.insert(0, last);
                hash_set.push(scratch_data);
            }
            let hash_set_ref = hash_set.iter().map(|v| v.as_slice()).collect::<Vec<&[u8]>>();
            // Fuzz compute
            let _unused = vm.calculate_hash(&hash_data);
            let _unused = vm.calculate_hash_set(&hash_set_ref);
            // Change data set
            hash_data.pop();
        } else {
            let _unused = vm.calculate_hash(&hash_data);
            let _unused = vm.calculate_hash_set(&[&hash_data]);
        }
    }
}

/// Fuzzing:
/// - `pub fn ns_compute`
/// - `pub fn ns_compute_last`
/// - `pub fn ns_compute_first`
/// - `pub fn ns_compute_next`
/// Secondary:
/// - `pub fn ns_create_vm`
/// - `pub fn ns_destroy_vm`
/// - `pub fn ns_alloc_cache`
/// - `pub fn ns_get_flags`
/// - `pub fn ns_init_cache`
/// - `pub fn ns_release_cache`
#[allow(clippy::needless_pass_by_value)] // This is required by the `QuickCheck` fuzzing framework
pub fn fuzz_ns_vm_calculate_hash_with_cache_only(data: Vec<u8>) -> bool {
    let flags = NsFlag::get_recommended_flags();
    if let Ok(cache) = NsCache::new(flags, &data) {
        let vm = NsVM::new(flags, Some(cache), None);
        if let Ok(mut vm) = vm {
            calculate_hashes(&data, &mut vm, 100);
        }
    }
    true
}

/// Fuzzing:
/// - `pub fn ns_compute`
/// - `pub fn ns_compute_last`
/// - `pub fn ns_compute_first`
/// - `pub fn ns_compute_next`
/// Secondary:
/// - `pub fn ns_create_vm`
/// - `pub fn ns_destroy_vm`
/// - `pub fn ns_alloc_cache`
/// - `pub fn ns_get_flags`
/// - `pub fn ns_init_cache`
/// - `pub fn ns_release_cache`
/// - `pub fn ns_alloc_tstore`
/// - `pub fn ns_init_store`
/// - `pub fn ns_tstore_count`
/// - `pub fn ns_get_tstore_mem`
/// - `pub fn ns_release_tstore`
#[allow(clippy::needless_pass_by_value)] // This is required by the `QuickCheck` fuzzing framework
pub fn fuzz_ns_vm_calculate_hash_with_cache_and_dataset(data: Vec<u8>) -> bool {
    let flags = NsFlag::get_recommended_flags();
    if let Ok(cache) = NsCache::new(flags, &data) {
        if let Ok(dataset) = NsStore::new(flags, cache.clone(), 0) {
            let vm = NsVM::new(flags, Some(cache), Some(dataset.clone()));
            if let Ok(mut vm) = vm {
                calculate_hashes(&data, &mut vm, 100);
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use quickcheck::QuickCheck;

    use crate::test_utils::{
        fuzz_ns_alloc_cache,
        fuzz_ns_create_vm_with_cache_and_dataset,
        fuzz_ns_create_vm_with_cache_only,
        fuzz_ns_vm_calculate_hash_with_cache_and_dataset,
        fuzz_ns_vm_calculate_hash_with_cache_only,
    };

    #[test]
    fn test_fuzz_lib_alloc_cache() {
        fuzz_ns_alloc_cache(vec![]);
        const TESTS: u64 = 25;
        QuickCheck::new()
            .min_tests_passed(TESTS)
            .tests(TESTS)
            .max_tests(TESTS)
            .quickcheck(fuzz_ns_alloc_cache as fn(Vec<u8>) -> bool);
    }

    #[test]
    fn test_fuzz_ns_create_vm_with_cache_only() {
        fuzz_ns_create_vm_with_cache_only(vec![]);
        const TESTS: u64 = 25;
        QuickCheck::new()
            .min_tests_passed(TESTS)
            .tests(TESTS)
            .max_tests(TESTS)
            .quickcheck(fuzz_ns_create_vm_with_cache_only as fn(Vec<u8>) -> bool);
    }

    #[test]
    fn test_fuzz_ns_create_vm_with_cache_and_dataset() {
        fuzz_ns_create_vm_with_cache_and_dataset(vec![]);
        const TESTS: u64 = 1;
        QuickCheck::new()
            .min_tests_passed(TESTS)
            .tests(TESTS)
            .max_tests(TESTS)
            .quickcheck(fuzz_ns_create_vm_with_cache_and_dataset as fn(Vec<u8>) -> bool);
    }

    #[test]
    fn test_fuzz_ns_vm_calculate_hash_with_cache_only() {
        fuzz_ns_vm_calculate_hash_with_cache_only(vec![]);
        const TESTS: u64 = 3;
        QuickCheck::new()
            .min_tests_passed(TESTS)
            .tests(TESTS)
            .max_tests(TESTS)
            .quickcheck(fuzz_ns_vm_calculate_hash_with_cache_only as fn(Vec<u8>) -> bool);
    }

    #[test]
    fn test_fuzz_ns_vm_calculate_hash_with_cache_and_dataset() {
        fuzz_ns_vm_calculate_hash_with_cache_and_dataset(vec![]);
        const TESTS: u64 = 1;
        QuickCheck::new()
            .min_tests_passed(TESTS)
            .tests(TESTS)
            .max_tests(TESTS)
            .quickcheck(fuzz_ns_vm_calculate_hash_with_cache_and_dataset as fn(Vec<u8>) -> bool);
    }
}
