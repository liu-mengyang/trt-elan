nsys profile -t 'cuda,cudnn,cublas,nvtx,osrt,opengl' --gpu-metrics-device=0 -o example python3 example.py --config ../../configs/elan_x4_local.yml