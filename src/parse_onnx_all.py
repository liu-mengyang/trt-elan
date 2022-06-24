import os

from cuda import cudart
import tensorrt as trt


onnxFile = "./elan_x4_sed.onnx"
trtFile = "./elan_x4_partly_half.plan"

cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(logger, '')

builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.flags = config.flags | 1 << int(trt.BuilderFlag.FP16) | 1 << int(trt.BuilderFlag.DEBUG)
config.max_workspace_size = 3 << 30
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
parser = trt.OnnxParser(network, logger)
if not os.path.exists(onnxFile):
    print("Failed finding onnx file!")
    exit()
print("Succeeded finding onnx file!")
with open(onnxFile, 'rb') as model:
    if not parser.parse(model.read()):
        print("Failed parsing .onnx file!")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
    print("Succeeded parsing .onnx file!")

lr = network.get_input(0)

total = 0
for layer in network:
    if layer.type in [trt.LayerType.CONVOLUTION, """trt.LayerType.ELEMENTWISE""", trt.LayerType.MATRIX_MULTIPLY]:
        total += 1
        print(total, layer.type, layer.precision, layer.precision_is_set)
        layer.precision = trt.DataType.HALF
        print(total, layer.type, layer.precision, layer.precision_is_set)

profile.set_shape(lr.name, (1, 3, 304, 208), (1, 3, 304, 208), (1, 3, 304, 208))
config.add_optimization_profile(profile)

engineString = builder.build_serialized_network(network, config)
if engineString == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine!")
with open(trtFile, 'wb') as f:
    f.write(engineString)
