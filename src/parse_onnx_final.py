import os

from cuda import cudart
import tensorrt as trt


onnxFile = "./elan_x4_sed.onnx"
trtFile = "./elan_x4_final.plan"

cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.VERBOSE)
trt.init_libnvinfer_plugins(logger, '')

builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
config.max_workspace_size = 3 << 30
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
parser = trt.OnnxParser(network, logger)
with open(onnxFile, 'rb') as model:
    if not parser.parse(model.read()):
        print("Failed parsing .onnx file!")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
    print("Succeeded parsing .onnx file!")

except_layer = [
    'Conv_6885',
    'Conv_9344',
    'Conv_7698',
    'Conv_8528',
    'Conv_7712',
    'Conv_9333',
    'Conv_10976',
    'Conv_8978',
    'Conv_9335',
    'Conv_6530',
    'Conv_10152',
    'Conv_6533',
    'Conv_6882',
    'Conv_10608',
    'Conv_8982',
    'Conv_10146',
]

total = 0
for layer in network:
    if layer.precision != trt.DataType.FLOAT and layer.precision != trt.DataType.HALF:
        continue
    if layer.type not in [trt.LayerType.CONVOLUTION]:
        continue
    if layer.name[0:5] != "Conv_":
        continue
    if layer.name in except_layer or int(layer.name[5:]) > 10000:
        total += 1
        layer.precision = trt.DataType.FLOAT
        for i in range(layer.num_outputs):
            layer.get_output(i).dtype = trt.DataType.FLOAT
        print(total, layer.name, layer.type,layer.precision, layer.precision_is_set)

lr = network.get_input(0)

profile.set_shape(lr.name, (1, 3, 304, 208), (1, 3, 304, 208), (1, 3, 304, 208))
config.add_optimization_profile(profile)

engineString = builder.build_serialized_network(network, config)
if engineString == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine!")
with open(trtFile, 'wb') as f:
    f.write(engineString)
