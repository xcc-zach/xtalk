use std::{
    collections::hash_map::DefaultHasher,
    env,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    process::Command,
};

fn run(mut command: Command, label: &str) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to run {label}: {error}"));
    assert!(status.success(), "{label} failed with {status}");
}

fn build_upstream(source: &Path, output: &Path, metal: bool, target: &str) -> PathBuf {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    metal.hash(&mut hasher);
    target.hash(&mut hasher);
    let build = output.join(format!("moss-transcribe-build-{:x}", hasher.finish()));
    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(source)
        .arg("-B")
        .arg(&build)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DMT_BUILD_CLI=OFF")
        .arg("-DMT_BUILD_TESTS=OFF")
        .arg("-DMT_SHARED=OFF")
        .arg("-DBUILD_SHARED_LIBS=OFF")
        .arg("-DGGML_NATIVE=OFF")
        .arg("-DGGML_LLAMAFILE=OFF")
        .arg(if metal {
            "-DMT_GGML_METAL=ON"
        } else {
            "-DMT_GGML_METAL=OFF"
        })
        .arg(if metal {
            "-DGGML_METAL=ON"
        } else {
            "-DGGML_METAL=OFF"
        });
    if target == "aarch64-apple-darwin" {
        configure
            .arg("-DCMAKE_OSX_ARCHITECTURES=arm64")
            .arg("-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0");
    } else if target == "x86_64-apple-darwin" {
        configure
            .arg("-DCMAKE_OSX_ARCHITECTURES=x86_64")
            .arg("-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0");
    }
    run(configure, "moss-transcribe.cpp CMake configure");

    let mut compile = Command::new("cmake");
    compile
        .arg("--build")
        .arg(&build)
        .arg("--target")
        .arg("moss-transcribe")
        .arg("--config")
        .arg("Release")
        .arg("--parallel");
    run(compile, "moss-transcribe.cpp build");
    build
}

fn link_upstream(build: &Path, metal: bool, target: &str) {
    let ggml = build.join("third_party/ggml/src");
    let blas = ggml.join("ggml-blas");
    let metal_dir = ggml.join("ggml-metal");
    for directory in [build, ggml.as_path(), blas.as_path(), metal_dir.as_path()] {
        println!("cargo:rustc-link-search=native={}", directory.display());
        println!(
            "cargo:rustc-link-search=native={}",
            directory.join("Release").display()
        );
    }
    println!("cargo:rustc-link-lib=static=moss-transcribe");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    let has_blas = static_library_exists(&blas, "ggml-blas", target);
    if has_blas {
        println!("cargo:rustc-link-lib=static=ggml-blas");
    }
    if metal {
        println!("cargo:rustc-link-lib=static=ggml-metal");
    }
    println!("cargo:rustc-link-lib=static=ggml-base");
    if target.contains("apple-darwin") {
        println!("cargo:rustc-link-lib=c++");
    } else if !target.contains("windows") {
        println!("cargo:rustc-link-lib=stdc++");
    }
    if target.contains("apple-darwin") {
        if has_blas {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
        if metal {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
        }
    }
}

fn static_library_exists(directory: &Path, name: &str, target: &str) -> bool {
    let filename = if target.contains("windows") {
        format!("{name}.lib")
    } else {
        format!("lib{name}.a")
    };
    directory.join(&filename).is_file() || directory.join("Release").join(filename).is_file()
}

fn main() {
    println!("cargo:rerun-if-changed=native/bridge.cpp");
    println!("cargo:rerun-if-changed=native/unavailable.cpp");
    println!("cargo:rerun-if-env-changed=MOSS_TRANSCRIBE_CPP_DIR");
    println!("cargo:rerun-if-env-changed=MOSS_TRANSCRIBE_METAL");

    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is required"));
    let target = env::var("TARGET").expect("TARGET is required");
    let Some(source) = env::var_os("MOSS_TRANSCRIBE_CPP_DIR").map(PathBuf::from) else {
        cc::Build::new()
            .cpp(true)
            .file("native/unavailable.cpp")
            .flag_if_supported("-std=c++17")
            .compile("xtalk_mtd_bridge");
        return;
    };
    assert!(
        source.join("CMakeLists.txt").is_file()
            && source.join("third_party/ggml/CMakeLists.txt").is_file(),
        "MOSS_TRANSCRIBE_CPP_DIR does not contain the assembled source tree"
    );
    println!(
        "cargo:rerun-if-changed={}",
        source.join(".xtalk-source-lock.json").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        source.join("CMakeLists.txt").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        source.join("third_party/ggml/CMakeLists.txt").display()
    );
    let metal = env::var("MOSS_TRANSCRIBE_METAL").as_deref() == Ok("1");
    let build = build_upstream(&source, &output, metal, &target);

    let mut bridge = cc::Build::new();
    bridge
        .cpp(true)
        .file("native/bridge.cpp")
        .include(source.join("include"))
        .include(source.join("src"))
        .include(source.join("third_party/ggml/include"))
        .flag_if_supported("-std=c++17");
    if target.contains("apple-darwin") {
        bridge.flag_if_supported("-mmacosx-version-min=11.0");
    }
    bridge.compile("xtalk_mtd_bridge");
    link_upstream(&build, metal, &target);
}
