#!/usr/bin/env bash

trtexec \
        --onnx=elan_x4_sed.onnx \
        --explicitBatch \
        --minShapes=lr:1x3x80x80 \
        --optShapes=lr:1x3x80x80 \
        --maxShapes=lr:1x3x80x80 \
        --saveEngine=elan_x4_tf32.plan \
        --workspace=40960 \
        --buildOnly \
        --verbose \