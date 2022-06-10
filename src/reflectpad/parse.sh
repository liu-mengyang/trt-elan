#!/usr/bin/env bash
python3 unit.py

trtexec \
        --onnx=unit.onnx \
        --explicitBatch \
        --minShapes=lr:1x3x64x64 \
        --optShapes=lr:1x3x80x80 \
        --maxShapes=lr:1x3x120x120 \
        --saveEngine=unit.plan \
        --workspace=40960 \
        --buildOnly \
        --noTF32 \
        --verbose \