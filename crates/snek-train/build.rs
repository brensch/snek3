fn main() -> Result<(), Box<dyn std::error::Error>> {
    prost_build::compile_protos(&["proto/snek.proto"], &["proto"])?;
    println!("cargo:rerun-if-changed=proto/snek.proto");
    Ok(())
}
