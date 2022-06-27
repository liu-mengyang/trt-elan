#!/usr/bin/env bash

trtexec \
        --onnx=elan_x4_qat_3f1.onnx \
        --explicitBatch \
        --minShapes=lr:1x3x304x208 \
        --optShapes=lr:1x3x304x208 \
        --maxShapes=lr:1x3x304x208 \
        --saveEngine=elan_x4_qat_3f1.plan \
        --workspace=40960 \
        --buildOnly \
        --int8 \
        --verbose \