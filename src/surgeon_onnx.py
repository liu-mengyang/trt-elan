import onnx_graphsurgeon as gs
import onnx

import numpy as np


graph = gs.import_onnx(onnx.load("elan_x4.onnx"))



# unsupport mod op
mod_13 = [node for node in graph.nodes if node.name=="Mod_13"][0]
mod_13_o = mod_13.o()
div_mod_13_out = gs.Variable("div_mod_13_out", dtype=np.int64)
div_mod_13 = gs.Node(name="div_mod_13", op="Div", inputs=mod_13.inputs, outputs=[div_mod_13_out])
graph.nodes.append(div_mod_13)
mul_mod_13_out = gs.Variable("mul_mod_13_out", dtype=np.int64)
mul_mod_13 = gs.Node(name="mul_mod_13", op="Mul", inputs=[div_mod_13_out, mod_13.inputs[1]], outputs=[mul_mod_13_out])
graph.nodes.append(mul_mod_13)
sub_mod_13_out = gs.Variable("sub_mod_13_out", dtype=np.int64)
sub_mod_13 = gs.Node(name="sub_mod_13", op="Sub", inputs=[mod_13.inputs[0], mul_mod_13.outputs[0]], outputs=[sub_mod_13_out])
graph.nodes.append(sub_mod_13)
mod_13_o.inputs[1] = sub_mod_13.outputs[0]
mod_13.outputs.clear()

mod_19 = [node for node in graph.nodes if node.name=="Mod_19"][0]
mod_19_o = mod_19.o()
div_mod_19_out = gs.Variable("div_mod_19_out", dtype=np.int64)
div_mod_19 = gs.Node(name="div_mod_19", op="Div", inputs=mod_19.inputs, outputs=[div_mod_19_out])
graph.nodes.append(div_mod_19)
mul_mod_19_out = gs.Variable("mul_mod_19_out", dtype=np.int64)
mul_mod_19 = gs.Node(name="mul_mod_19", op="Mul", inputs=[div_mod_19_out, mod_19.inputs[1]], outputs=[mul_mod_19_out])
graph.nodes.append(mul_mod_19)
sub_mod_19_out = gs.Variable("sub_mod_19_out", dtype=np.int64)
sub_mod_19 = gs.Node(name="sub_mod_19", op="Sub", inputs=[mod_19.inputs[0], mul_mod_19.outputs[0]], outputs=[sub_mod_19_out])
graph.nodes.append(sub_mod_19)
mod_19_o.inputs[1] = sub_mod_19.outputs[0]
mod_19.outputs.clear()

mod_17 = [node for node in graph.nodes if node.name=="Mod_17"][0]
mod_17_o0 = mod_17.o(0)
mod_17_o1 = mod_17.o(1)
div_mod_17_out = gs.Variable("div_mod_17_out", dtype=np.int64)
div_mod_17 = gs.Node(name="div_mod_17", op="Div", inputs=mod_17.inputs, outputs=[div_mod_17_out])
graph.nodes.append(div_mod_17)
mul_mod_17_out = gs.Variable("mul_mod_17_out", dtype=np.int64)
mul_mod_17 = gs.Node(name="mul_mod_17", op="Mul", inputs=[div_mod_17_out, mod_17.inputs[1]], outputs=[mul_mod_17_out])
graph.nodes.append(mul_mod_17)
sub_mod_17_out = gs.Variable("sub_mod_17_out", dtype=np.int64)
sub_mod_17 = gs.Node(name="sub_mod_17", op="Sub", inputs=[mod_17.inputs[0], mul_mod_17.outputs[0]], outputs=[sub_mod_17_out])
graph.nodes.append(sub_mod_17)
mod_17_o0.inputs[0] = sub_mod_17.outputs[0]
mod_17_o1.inputs[0] = sub_mod_17.outputs[0]
mod_17.outputs.clear()

mod_23 = [node for node in graph.nodes if node.name=="Mod_23"][0]
mod_23_o0 = mod_23.o(0)
mod_23_o1 = mod_23.o(1)
div_mod_23_out = gs.Variable("div_mod_23_out", dtype=np.int64)
div_mod_23 = gs.Node(name="div_mod_23", op="Div", inputs=mod_23.inputs, outputs=[div_mod_23_out])
graph.nodes.append(div_mod_23)
mul_mod_23_out = gs.Variable("mul_mod_23_out", dtype=np.int64)
mul_mod_23 = gs.Node(name="mul_mod_23", op="Mul", inputs=[div_mod_23_out, mod_23.inputs[1]], outputs=[mul_mod_23_out])
graph.nodes.append(mul_mod_23)
sub_mod_23_out = gs.Variable("sub_mod_23_out", dtype=np.int64)
sub_mod_23 = gs.Node(name="sub_mod_23", op="Sub", inputs=[mod_23.inputs[0], mul_mod_23.outputs[0]], outputs=[sub_mod_23_out])
graph.nodes.append(sub_mod_23)
mod_23_o0.inputs[0] = sub_mod_23.outputs[0]
mod_23_o1.inputs[0] = sub_mod_23.outputs[0]
mod_23.outputs.clear()

# unmatch pad op
# cast_51 = [node for node in graph.nodes if node.name=="Cast_51"][0]
# cast_51_new_out = gs.Variable("cast_51_new_out", dtype=np.INT32)
# cast_51_new = gs.Node(name="cast_51_new", op="Cast", inputs=cast_51.inputs, outputs=[cast_51_new_out], attrs={"to":getattr(onnx.TensorProto, 'FLOAT')})
# graph.nodes.append(cast_51_new)
# cast_51.o().inputs[1] = cast_51_new_out
# cast_51.outputs.clear()

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "elan_x4_sed.onnx")