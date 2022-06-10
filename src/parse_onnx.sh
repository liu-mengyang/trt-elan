#!/usr/bin/env bash

trtexec \
        --onnx=elan_x4_sed.onnx \
        --explicitBatch \
        --minShapes=lr:1x3x64x64 \
        --optShapes=lr:1x3x80x80 \
        --maxShapes=lr:1x3x120x120 \
        --saveEngine=elan_x4.plan \
        --workspace=40960 \
        --buildOnly \
        --noTF32 \
        --verbose \