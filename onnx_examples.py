import onnx
import numpy as np

###############################################
# Load .pb tensor and convert to numpy array
###############################################
tensor = onnx.load_tensor("/path/to/tensor.pb")
np_tensor = onnx.numpy_helper.to_array(tensor)
print(np_tensor.shape)

###############################################
# Convert opset
###############################################
model = onnx.load("/path/to/model.onnx")
print(model.opset_import[0].version)

opset = 13
new_model = onnx.version_converter.convert_version(model, opset)
onnx.checker.check_model(new_model)
print(new_model.opset_import[0].version)

inferred_model = onnx.shape_inference.infer_shapes(new_model)
onnx.checker.check_model(inferred_model)
onnx.save(inferred_model, "model_opset{}.onnx".format(opset))

###############################################
# Count operators
###############################################
model = onnx.load("/path/to/model.onnx")

op_type_dict = dict()

graph = model.graph
for node in graph.node:
    op_type = node.op_type
    if op_type in op_type_dict:
        op_type_dict[op_type] += 1
    else:
        op_type_dict[op_type] = 1

print(op_type_dict)

###############################################
# Add outputs from internal nodes
###############################################

# Edit NodeName, OutputName, dtype value and dims.
# Add more items following the same pattern to add more outputs.
additional_outputs = dict(
    NodeName=dict(name="OutputName", dtype=onnx.TensorProto.FLOAT, dims=[1, 2, 3]),
)

model_path = "/path/to/model.onnx"
model = onnx.load(model_path)
graph = model.graph
for k, v in outputs.items():
    name = v["name"]
    dtype = v["dtype"]
    dims = v["dims"]
    graph.output.append(onnx.helper.make_tensor_value_info(name, dtype, dims))
new_model = onnx.helper.make_model(graph, full_check=True)
onnx.checker.check_model(new_model)
new_model = onnx.shape_inference.infer_shapes(new_model)
onnx.save(new_model, model_path.split(".onnx")[0] + ".tmp.onnx")
