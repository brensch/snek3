//! Compile the CUDA-graph C++ shim (`csrc/graph_shim.cpp`) against the same
//! libtorch this crate links via tch. Locates the venv's PyTorch by default;
//! override with `SNEK_TORCH_DIR`. Against a CPU-only libtorch (no
//! `libtorch_cuda.so`) the shim is skipped and `snek_tch::cudagraph` is
//! compiled out (cfg `snek_cuda`), so CPU serving builds work.

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
    if !inc.join("torch").exists() {
        panic!(
            "libtorch headers not found at {} (set SNEK_TORCH_DIR)",
            inc.display()
        );
    }

    let lib = torch.join("lib");
    println!("cargo:rustc-check-cfg=cfg(snek_cuda)");
    println!("cargo:rustc-link-search=native={}", lib.display());
    for l in ["c10", "torch_cpu", "torch"] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }

    if lib.join("libtorch_cuda.so").exists() {
        let api_inc = inc.join("torch/csrc/api/include");
        // Host-compilable CUDA headers: `SNEK_CUDA_INC` if set (e.g. a CUDA
        // toolkit's include dir in the docker build). Default to triton's
        // bundled, version-matched (12.8) include tree — the venv's
        // `nvidia-cuda-runtime` wheel ships its headers flat (no `crt/`
        // subdir), so cuda_runtime.h's `#include "crt/..."` fails there.
        let cuda_inc = std::env::var("SNEK_CUDA_INC")
            .map(PathBuf::from)
            .unwrap_or_else(|_| torch.join("../triton/backends/nvidia/include"));

        if !inc.join("ATen/cuda/CUDAGraph.h").exists() {
            panic!(
                "CUDA libtorch at {} is missing ATen/cuda/CUDAGraph.h",
                torch.display()
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

        for l in ["c10_cuda", "torch_cuda"] {
            println!("cargo:rustc-link-lib=dylib={l}");
        }
        println!("cargo:rustc-cfg=snek_cuda");
    } else {
        println!(
            "cargo:warning=CPU-only libtorch: CUDA-graph shim skipped (snek_tch::cudagraph unavailable)"
        );
    }

    println!("cargo:rerun-if-changed=csrc/graph_shim.cpp");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SNEK_TORCH_DIR");
    println!("cargo:rerun-if-env-changed=SNEK_CUDA_INC");
}
