#!/usr/bin/env bash

trtexec \
        --onnx=elan_x4_sed_fixed.onnx \
        --saveEngine=elan_x4.plan \
        --workspace=40960 \
        --buildOnly \
        --noTF32 \
        --verbose \