#!/bin/bash

JNI_LIBS=../app/src/main/jniLibs

# Build the Rust library for different targets
cd rust
ORT_LIB_LOCATION=E:/onnxruntime/build/Windows/Release cargo build --target x86_64-linux-android --release --verbose
cd ..
#
## Remove existing JNI_LIBS directory and recreate it
#rm -rf $JNI_LIBS
#mkdir -p $JNI_LIBS/arm64-v8a
#mkdir -p $JNI_LIBS/x86_64
#
## Copy the generated .so files to the appropriate directories
#cp rust/target/x86_64-linux-android/release/librust.so $JNI_LIBS/x86_64/librust.so
#
## Wait for user input before continuing
read -n 1 -s -r -p "Press any key to continue..."
