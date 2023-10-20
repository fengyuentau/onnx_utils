import onnx

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
