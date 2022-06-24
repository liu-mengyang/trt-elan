#!/usr/bin/env bash

trtexec \
        --onnx=elan_x4_sed.onnx \
        --explicitBatch \
        --minShapes=lr:1x3x304x208 \
        --optShapes=lr:1x3x304x208 \
        --maxShapes=lr:1x3x304x208 \
        --saveEngine=elan_x4_fp16.plan \
        --workspace=40960 \
        --buildOnly \
        --fp16 \
        --verbose \