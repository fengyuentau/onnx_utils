# ONNX Utilities

Manipulating ONNX models as you wish. Install `onnx` and `numpy` to get started.

## Generate ONNX models

Call `gen.py` to generate ONNX models. Support ONNX model with a single operator for now.

### Customize your own ONNX model

Suppose your ONNX model is `custom.onnx`, then:
1. Learn the operator in https://github.com/onnx/onnx/blob/main/docs/Operators.md.
1. Make an new entry in `list_single_op` in `gen.py`:
    1. custom=dict(op_name="your_op", attributes="op_attrs_if_any", inputs="op_inputs", outputs="op_outputs")
    1. "your_op" should be the exact operator name in the link.
    1. "op_attrs_if_any" can be omitted.
    1. "op_inputs" has several dictionaries, each of which stands for an input. Each input should at least have `dtype` and `shape`. `initializer` is used for constant input, which can be omitted.
    1. "op_outputs" has the same definition as "op_inputs".
