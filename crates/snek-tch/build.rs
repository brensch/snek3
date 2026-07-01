//! Compile the CUDA-graph C++ shim (`csrc/graph_shim.cpp`) against the same
//! libtorch this crate links via tch. Locates the venv's PyTorch by default;
//! override with `SNEK_TORCH_DIR`.

use std::path::PathBuf;

fn main() {
    let torch = std::env::var("SNEK_TORCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
            // crates/snek-tch -> repo root -> venv site-packages/torch
            manifest.join("../../.venv/lib/python3.12/site-packages/torch")
        });

    let inc = torch.join("include");
    let api_inc = inc.join("torch/csrc/api/include");
    // Host-compilable CUDA headers: the `nvidia-cuda-runtime` wheel ships them
    // flat (no `crt/` subdir), so cuda_runtime.h's `#include "crt/..."` fails.
    // Triton bundles a complete, version-matched (12.8) CUDA include tree.
    let cuda_inc = torch.join("../triton/backends/nvidia/include");

    if !inc.join("ATen/cuda/CUDAGraph.h").exists() {
        panic!(
            "libtorch headers not found at {} (set SNEK_TORCH_DIR)",
            inc.display()
        );
    }

    cc::Build::new()
        .cpp(true)
        .file("csrc/graph_shim.cpp")
        .include(&inc)
        .include(&api_inc)
        .include(&cuda_inc)
        .flag("-std=c++17")
        .flag_if_supported("-Wno-unused-parameter")
        .define("_GLIBCXX_USE_CXX11_ABI", "1")
        .compile("snek_graph_shim");

    let lib = torch.join("lib");
    println!("cargo:rustc-link-search=native={}", lib.display());
    for l in ["c10", "c10_cuda", "torch_cpu", "torch_cuda", "torch"] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }
    println!("cargo:rerun-if-changed=csrc/graph_shim.cpp");
    println!("cargo:rerun-if-changed=build.rs");
}
