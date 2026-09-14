#!/bin/bash
# whisper.cpp build for Spark — self-contained, logs to /tmp/wbuild.log
set -x
cd /opt/spark
rm -rf whisper.cpp
git clone --depth 1 https://github.com/ggml-org/whisper.cpp || exit 1
cd whisper.cpp
cmake -B build || exit 1
cmake --build build --config Release -j4 || exit 1
bash ./models/download-ggml-model.sh tiny.en || exit 1
bash ./models/download-ggml-model.sh base.en || exit 1
echo BUILD_DONE
