#!/bin/bash

# Check for rm_build flag in script arguments
rm_build_flag=false
for arg in "$@"; do
    if [[ "$arg" == "--rm_build=true" || "$arg" == "-rm_build" ]]; then
        rm_build_flag=true
        break
    fi
done

cd /mm_resources/llama.cpp

# Update repository
git pull

# If the flag is set, remove the build folder
if $rm_build_flag; then
    echo "Removing existing build directory..."
    rm -rf build
else
    echo "Skipping build directory removal..."
fi

# Build process command
cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89" \
      -DGGML_CUDA_F16=ON -DGGML_CUDA_FA_ALL_QUANTS=ON && \
cmake --build build --config Release --parallel 50