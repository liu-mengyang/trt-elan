FROM registry.cn-hangzhou.aliyuncs.com/trt2022/trt-8.4-ga

RUN apt update && apt install libgl1-mesa-glx -y && \
    pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

RUN mkdir /workspace/trt-elan
ADD . /workspace/trt-elan

RUN pip3 install -r /workspace/trt-elan/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package && \
    python3 -m pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com
