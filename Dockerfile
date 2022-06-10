FROM nvcr.io/nvidia/tensorrt:22.05-py3

RUN mkdir /workspace/trt-elan
COPY src /workspace/trt-elan/src
COPY configs /workspace/trt-elan/configs
COPY weights /workspace/trt-elan/weights
COPY requirements.txt /workspace/trt-elan/

RUN apt update
RUN apt install libgl1-mesa-glx -y

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install -r /workspace/trt-elan/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
RUN python3 -m pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com