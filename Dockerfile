FROM nvcr.io/nvidia/tensorrt:22.05-py3

RUN mkdir /workspace/trt-elan
COPY * /workspace/trt-elan/