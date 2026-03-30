use std::env;
use std::path::Path;

fn main() {
    // 👉 Chỉ chạy build native nếu có thư mục native_core
    let native_dir = env::var("NATIVE_CORE_DIR").unwrap_or_else(|_| "native_core".to_string());

    if !Path::new(&native_dir).exists() {
        println!("cargo:warning=Skipping native_core build (not found)");
        return;
    }

    // Nếu tồn tại thì mới build (fallback an toàn)
    let portable = env::var("PORTABLE").unwrap_or_default() == "1";

    let mut cfg = cmake::Config::new(&native_dir);
    cfg.define("CMAKE_BUILD_TYPE", "Release");

    if portable {
        cfg.define("DARCH", "x86-64");
        cfg.cflag("-O3");
        cfg.cflag("-march=x86-64");
        cfg.cflag("-mtune=generic");
    } else {
        cfg.define("DARCH", "native");
        cfg.cflag("-O3");
        cfg.cflag("-march=native");
        cfg.cflag("-mtune=native");
    }

    cfg.cflag("-fvisibility=hidden");
    cfg.cxxflag("-fvisibility=hidden");
    cfg.cxxflag("-fvisibility-inlines-hidden");
    cfg.cxxflag("-fno-rtti");
    cfg.cflag("-ffunction-sections");
    cfg.cflag("-fdata-sections");

    if portable {
        cfg.cflag("-static-libstdc++");
        cfg.cflag("-static-libgcc");
    }

    let build_path = cfg.build();

    println!("cargo:rustc-link-search=native={}/lib64", build_path.display());
    println!("cargo:rustc-link-search=native={}/lib", build_path.display());
    println!("cargo:rustc-link-lib=static=nscore");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or("linux".to_string());
    let dylib_name = match target_os.as_str() {
        "freebsd" | "macos" | "ios" => "c++",
        "windows" => "msvcrt",
        _ => "stdc++",
    };
    println!("cargo:rustc-link-lib=dylib={}", dylib_name);

    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=advapi32");
    }
}