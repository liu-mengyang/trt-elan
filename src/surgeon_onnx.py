import onnx_graphsurgeon as gs
import onnx


graph = gs.import_onnx(onnx.load("elan_x4.onnx"))



graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "elan_x4_sed.onnx")