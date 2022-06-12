import torch
import torch.nn as nn
import torch.nn.functional as F



class Roll(nn.Module):
    def __init__(self):
        super(Roll, self).__init__()

    def forward(self, input):
        out = torch.roll(input, shifts=(-2, -2), dims=(2,3))
        return out

input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).cuda()
print(input)
roll = Roll().cuda()
out = roll(input)
print(out)

torch.onnx.export(roll,
                  input,
                  "unit_roll.onnx",
                  input_names=["input"],
                  output_names=["output"],
                  verbose=True,
                  keep_initializers_as_inputs=True,
                  opset_version=13,
                  dynamic_axes={"input": {0: "batch_size"}})