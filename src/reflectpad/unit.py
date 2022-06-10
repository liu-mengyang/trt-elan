import torch
import torch.nn as nn
import torch.nn.functional as F



class ReflectPad(nn.Module):
    def __init__(self):
        super(ReflectPad, self).__init__()

    def forward(self, input):
        out = F.pad(input, (0, 1, 0, 2), "reflect")
        return out

input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).cuda()
print(input)
rp = ReflectPad().cuda()
out = rp(input)
print(out)

torch.onnx.export(rp,
                  input,
                  "unit.onnx",
                  input_names=["input"],
                  output_names=["output"],
                  verbose=True,
                  keep_initializers_as_inputs=True,
                  opset_version=13,
                  dynamic_axes={"input": {0: "batch_size"}})