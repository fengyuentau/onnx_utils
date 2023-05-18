import os
import sys
import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

# Available TensorProto data types: https://github.com/onnx/onnx/blob/e69872e7a55c6da5b22caa33739da4df17c7d01c/onnx/checker.cc#L201-L237
# (call with TensorProto.dtype): dtype can be one of the followings
# - FLOAT
# - COMPLEX64
# - DOUBLE
# - COMPLEX128
# - INT32
# - UINT8
# - INT8
# - UINT16
# - INT16
# - BOOL
# - FLOAT16
# - BFLOAT16
# - INT64
# - UINT32
# - UINT64
# - STRING

list_single_op = dict(
    qlinearsoftmax=dict(
        op_name="QLinearSoftmax",
        attributes=dict(
            domain="com.microsoft",
            axis=-1,
            opset=13,
        ),
        inputs=dict(
            input_quantized=dict(
            ),
            input1=dict(
                dtype=TensorProto.FLOAT,
                shape=[],
                initializer=np.array([0.20202656090259552], dtype=np.float32),
            ),
            input2=dict(
                dtype=TensorProto.INT8,
                shape=[],
                initializer=np.array([-8], dtype=np.int8),
            ),
            input3=dict(
                dtype=TensorProto.FLOAT,
                shape=[],
                initializer=np.array([0.00390625], dtype=np.float32),
            ),
            input4=dict(
                dtype=TensorProto.INT8,
                shape=[],
                initializer=np.array([-128], dtype=np.int8),
            ),
        ),
        outputs=dict(
            output_quantized=dict(
                dtype=TensorProto.INT8,
                shape=[100, 2],
            ),
        ),
    ),
    qlinearsoftmax_11=dict(
        op_name="QLinearSoftmax",
        attributes=dict(
            domain="com.microsoft",
            axis=-1,
            opset=11,
        ),
        inputs=dict(
            input_quantized=dict(
            ),
            input1=dict(
                dtype=TensorProto.FLOAT,
                shape=[],
                initializer=np.array([0.20202656090259552], dtype=np.float32),
            ),
            input2=dict(
                dtype=TensorProto.INT8,
                shape=[],
                initializer=np.array([-8], dtype=np.int8),
            ),
            input3=dict(
                dtype=TensorProto.FLOAT,
                shape=[],
                initializer=np.array([0.00390625], dtype=np.float32),
            ),
            input4=dict(
                dtype=TensorProto.INT8,
                shape=[],
                initializer=np.array([-128], dtype=np.int8),
            ),
        ),
        outputs=dict(
            output_quantized=dict(
                dtype=TensorProto.INT8,
                shape=[100, 2],
            ),
        ),
    ),

)

def gen_single_op(single_op, onnx_name, save_prefix="./models"):
    initializers = []

    inputs = []
    inputs.append(helper.make_tensor_value_info("input", TensorProto.FLOAT, [10, 2]))
    inputs.append(helper.make_tensor_value_info("input_scale", TensorProto.FLOAT, []))
    initializers.append(helper.make_tensor("input_scale", TensorProto.FLOAT, [], np.array([0.20202656090259552], dtype=np.float32)))
    inputs.append(helper.make_tensor_value_info("input_zero_point", TensorProto.INT8, []))
    initializers.append(helper.make_tensor("input_zero_point", TensorProto.INT8, [], np.array([-8], dtype=np.int8)))
    outputs = []
    outputs.append(helper.make_empty_tensor_value_info("input_quantized"))
    node_qlinear = helper.make_node("QuantizeLinear", ["input", "input_scale", "input_zero_point"], ["input_quantized"])

    # Create inputs (ValueInfoProto)
    input_names = []
    for name, prop in single_op.get("inputs").items():
        input_names.append(name)

        if "dtype" in prop.keys() and "shape" in prop.keys():
            dtype = prop.get("dtype")
            shape = prop.get("shape")
            inputs.append(
                helper.make_tensor_value_info(name, dtype, shape)
            )
            initializer = prop.get("initializer")
            if initializer is not None:
                initializers.append(
                    helper.make_tensor(name, dtype, shape, initializer)
                )
        else:
            inputs.append(
                helper.make_empty_tensor_value_info(name)
            )

    # Create outputs (ValueInfoProto)
    output_names = []
    for name, prop in single_op.get("outputs").items():
        output_names.append(name)

        dtype = prop.get("dtype")
        shape = prop.get("shape")
        outputs.append(
            helper.make_tensor_value_info(name, dtype, shape)
        )

    # Create a node (NodeProto)
    attributes = single_op.get("attributes", {})
    node_def = onnx.helper.make_node(
        single_op.get("op_name"),
        inputs=input_names,
        outputs=output_names,
        **attributes,
    )

    inputs.append(helper.make_tensor_value_info("output_quantized", TensorProto.FLOAT, [10, 2]))
    inputs.append(helper.make_tensor_value_info("output_scale", TensorProto.FLOAT, []))
    initializers.append(helper.make_tensor("output_scale", TensorProto.FLOAT, [], np.array([0.00390625], dtype=np.float32)))
    inputs.append(helper.make_tensor_value_info("output_zero_point", TensorProto.INT8, []))
    initializers.append(helper.make_tensor("output_zero_point", TensorProto.INT8, [], np.array([-128], dtype=np.int8)))
    outputs.append(helper.make_tensor_value_info("output", TensorProto.FLOAT, [10, 2]))
    node_deqlinear = helper.make_node("DequantizeLinear", ["output_quantized", "output_scale", "output_zero_point"], ["output"])

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_qlinear, node_def, node_deqlinear],        # nodes
        onnx_name,         # name
        [inputs[0]],            # inputs
        [outputs[-1]],           # outputs
        initializers       # initializer
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name="github.com/fengyuentau/onnx_utils")
    model_def.opset_import.extend([helper.make_opsetid("com.microsoft", 1)])
    #onnx.checker.check_model(model_def)
    onnx.save(model_def, "models/{}.onnx".format(onnx_name))

    return True

def main():
    to_gen_list = []
    for i in range(1, len(sys.argv)):
        to_gen_list.append(sys.argv[i])
    if not to_gen_list:
        to_gen_list = list(list_single_op.keys())
    print("Generating: {}".format(to_gen_list))

    save_prefix = "./models"
    gen_failed_list = []
    for to_gen in to_gen_list:
        if not gen_single_op(list_single_op[to_gen], to_gen, save_prefix):
            gen_failed_list.append(to_gen)
        else:
            print("\t - Done generating {}.onnx".format(os.path.join(save_prefix, to_gen)))
    if gen_failed_list:
        print("Failed to generate: {}".format(gen_failed_list))

if __name__ == '__main__':
    main()
