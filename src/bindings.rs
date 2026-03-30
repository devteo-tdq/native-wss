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

use libc::{c_uint, c_ulong, c_void};
pub const RANDOMX_HASH_SIZE: u32 = 32;

#[repr(C)]
pub struct ns_tstore {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct ns_wcache {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct ns_vm {
    _unused: [u8; 0],
}

extern "C" {
    #[link_name = "ns_alloc_wcache"]
    pub fn randomx_alloc_cache(flags: c_uint) -> *mut ns_wcache;
    #[link_name = "ns_init_wcache"]
    pub fn randomx_init_cache(cache: *mut ns_wcache, key: *const c_void, keySize: usize);
    #[link_name = "ns_release_wcache"]
    pub fn randomx_release_cache(cache: *mut ns_wcache);
    #[link_name = "ns_alloc_tstore"]
    pub fn randomx_alloc_dataset(flags: c_uint) -> *mut ns_tstore;
    #[link_name = "ns_tstore_count"]
    pub fn randomx_dataset_item_count() -> c_ulong;
    #[link_name = "ns_init_tstore"]
    pub fn randomx_init_dataset(
        dataset: *mut ns_tstore,
        cache: *mut ns_wcache,
        start_item: c_ulong,
        item_count: c_ulong,
    );
    #[link_name = "ns_get_tstore_mem"]
    pub fn randomx_get_dataset_memory(dataset: *mut ns_tstore) -> *mut c_void;
    #[link_name = "ns_release_tstore"]
    pub fn randomx_release_dataset(dataset: *mut ns_tstore);
    #[link_name = "ns_create_vm"]
    pub fn randomx_create_vm(
        flags: c_uint,
        cache: *mut ns_wcache,
        dataset: *mut ns_tstore,
    ) -> *mut ns_vm;
    #[link_name = "ns_vm_set_wcache"]
    pub fn randomx_vm_set_cache(machine: *mut ns_vm, cache: *mut ns_wcache);
    #[link_name = "ns_vm_set_tstore"]
    pub fn randomx_vm_set_dataset(machine: *mut ns_vm, dataset: *mut ns_tstore);
    #[link_name = "ns_destroy_vm"]
    pub fn randomx_destroy_vm(machine: *mut ns_vm);
    #[link_name = "ns_compute"]
    pub fn randomx_calculate_hash(
        machine: *mut ns_vm,
        input: *const c_void,
        input_size: usize,
        output: *mut c_void,
    );
    #[link_name = "ns_compute_first"]
    pub fn randomx_calculate_hash_first(machine: *mut ns_vm, input: *const c_void, input_size: usize);
    #[link_name = "ns_compute_next"]
    pub fn randomx_calculate_hash_next(
        machine: *mut ns_vm,
        input_next: *const c_void,
        input_size_next: usize,
        output: *mut c_void,
    );
    #[link_name = "ns_compute_last"]
    pub fn randomx_calculate_hash_last(machine: *mut ns_vm, output: *mut c_void);
    #[link_name = "ns_detect_caps"]
    pub fn randomx_get_flags() -> c_uint;
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use libc::{c_uint, c_void};

    use super::*;

    #[test]
    fn alloc_cache() {
        let key = b"Key";
        let flag: c_uint = 0;
        let cache = unsafe { randomx_alloc_cache(flag) };
        assert!(!cache.is_null(), "Failed to init cache");

        unsafe {
            randomx_init_cache(cache, key.as_ptr() as _, key.len());
        }
        unsafe {
            randomx_release_cache(cache);
        }
    }

    #[test]
    fn alloc_dataset() {
        let key = b"Key";
        let flag: c_uint = 0;
        let cache = unsafe { randomx_alloc_cache(flag) };

        unsafe {
            randomx_init_cache(cache, key.as_ptr() as _, key.len());
        }

        let dataset = unsafe { randomx_alloc_dataset(flag) };

        unsafe { randomx_init_dataset(dataset, cache, 0, 1) };

        assert_ne!(unsafe { randomx_dataset_item_count() }, 0);

        unsafe {
            randomx_release_dataset(dataset);
            randomx_release_cache(cache);
        }
    }

    #[test]
    fn alloc_vm() {
        let key = b"Key";
        let flag: c_uint = 0;

        let cache = unsafe { randomx_alloc_cache(flag) };

        unsafe {
            randomx_init_cache(cache, key.as_ptr() as _, key.len());
        }
        let mut vm = unsafe { randomx_create_vm(flag, cache, ptr::null_mut()) };
        if vm.is_null() {
            panic!("Failed to init vm with cache");
        }
        unsafe {
            randomx_vm_set_cache(vm, cache);
            randomx_destroy_vm(vm);
        }

        let dataset = unsafe { randomx_alloc_dataset(flag) };
        unsafe { randomx_init_dataset(dataset, cache, 0, 1) }

        vm = unsafe { randomx_create_vm(flag, cache, dataset) };
        if vm.is_null() {
            panic!("Failed to init vm with dataset");
        }
        unsafe {
            randomx_vm_set_dataset(vm, dataset);
        }

        unsafe {
            randomx_release_dataset(dataset);
            randomx_release_cache(cache);
            randomx_destroy_vm(vm);
        }
    }

    #[test]
    fn calculate_hash() {
        let key = b"test key 000";
        let input = b"This is a test";
        let expected = b"639183aae1bf4c9a35884cb46b09cad9175f04efd7684e7262a0ac1c2f0b4e3f";

        let flag: c_uint = 0;

        let arr = [0u8; RANDOMX_HASH_SIZE as usize];
        let output_ptr = arr.as_ptr() as *mut c_void;

        let cache = unsafe { randomx_alloc_cache(flag) };

        unsafe {
            randomx_init_cache(cache, key.as_ptr() as _, key.len());
        }

        let vm = unsafe { randomx_create_vm(flag, cache, ptr::null_mut()) };

        unsafe {
            randomx_calculate_hash(vm, input.as_ptr() as _, input.len(), output_ptr);
        }
        assert_eq!(hex::decode(expected).unwrap(), arr);

        unsafe {
            randomx_destroy_vm(vm);
            randomx_release_cache(cache);
        }
    }

    #[allow(clippy::cast_sign_loss)]
    #[test]
    fn calculate_hash_set() {
        let key = b"test key 000";
        let input = b"This is a test";
        let expected = "639183aae1bf4c9a35884cb46b09cad9175f04efd7684e7262a0ac1c2f0b4e3f";

        let input2 = b"Lorem ipsum dolor sit amet";
        let expected2 = "300a0adb47603dedb42228ccb2b211104f4da45af709cd7547cd049e9489c969";

        let input3 = b"sed do eiusmod tempor incididunt ut labore et dolore magna aliqua";
        let expected3 = "c36d4ed4191e617309867ed66a443be4075014e2b061bcdaf9ce7b721d2b77a8";

        let flag: c_uint = 0;

        let arr = [0u8; RANDOMX_HASH_SIZE as usize];
        let output_ptr = arr.as_ptr() as *mut c_void;

        let cache = unsafe { randomx_alloc_cache(flag) };

        unsafe {
            randomx_init_cache(cache, key.as_ptr() as _, key.len());
        }

        let vm = unsafe { randomx_create_vm(flag, cache, ptr::null_mut()) };

        unsafe {
            randomx_calculate_hash_first(vm, input.as_ptr() as _, input.len());
        }

        unsafe {
            randomx_calculate_hash_next(vm, input2.as_ptr() as _, input2.len(), output_ptr);
        }
        assert_eq!(hex::decode(expected).unwrap(), arr);

        unsafe {
            randomx_calculate_hash_next(vm, input3.as_ptr() as _, input3.len(), output_ptr);
        }
        assert_eq!(hex::decode(expected2).unwrap(), arr);

        unsafe {
            randomx_calculate_hash_last(vm, output_ptr);
        }
        assert_eq!(hex::decode(expected3).unwrap(), arr);

        unsafe {
            randomx_destroy_vm(vm);
            randomx_release_cache(cache);
        }
    }
}
