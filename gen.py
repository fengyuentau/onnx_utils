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
    gather_multiple_indices=dict(
        op_name="Gather",
        attributes=dict(
            axis=1,
        ),
        inputs=dict(
            data=dict( # name of input0
                dtype=TensorProto.FLOAT,
                shape=[1, 512, 640, 3],
            ),
            indices=dict( # name of input1
                dtype=TensorProto.INT32,
                shape=[128],
                initializer=np.arange(2, 512, 4, dtype=np.int32), # initializer has the same type and shape
            ),
        ),
        outputs=dict(
            y=dict(
                dtype=TensorProto.FLOAT,
                shape=[1, 128, 640, 3],
            ),
        ),
    ),
    tile=dict(
        op_name="Tile",
        inputs=dict(
            input=dict(
                dtype=TensorProto.FLOAT,
                shape=[2, 3, 4, 5],
            ),
            repeats=dict(
                dtype=TensorProto.INT64,
                shape=[4],
                initializer=np.array([7, 6, 4, 2], dtype=np.int64)
            ),
        ),
        outputs=dict(
            y=dict(
                dtype=TensorProto.FLOAT,
                shape=[14, 18, 16, 10]
            )
        )
    ),
    slice_neg_steps=dict(
        op_name="Slice",
        inputs=dict(
            x=dict(
                dtype=TensorProto.FLOAT,
                shape=[20, 10, 5],
            ),
            starts=dict(
                dtype=TensorProto.INT64,
                shape=[3],
                initializer=np.array([20, 10, 4], dtype=np.int64),
            ),
            ends=dict(
                dtype=TensorProto.INT64,
                shape=[3],
                initializer=np.array([0, 0, 1], dtype=np.int64),
            ),
            axes=dict(
                dtype=TensorProto.INT64,
                shape=[3],
                initializer=np.array([0, 1, 2], dtype=np.int64),
            ),
            steps=dict(
                dtype=TensorProto.INT64,
                shape=[3],
                initializer=np.array([-1, -3, -2], dtype=np.int64),
            ),
        ),
        outputs=dict(
            y=dict(
                dtype=TensorProto.FLOAT,
                shape=[19, 3, 2],
            ),
        ),
    ),
)

def gen_single_op(single_op, onnx_name, save_prefix="./models"):
    # Create inputs (ValueInfoProto)
    inputs = []
    input_names = []
    initializers = []
    for name, prop in single_op.get("inputs").items():
        input_names.append(name)

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

    # Create outputs (ValueInfoProto)
    outputs = []
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

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],        # nodes
        onnx_name,         # name
        inputs,            # inputs
        outputs,           # outputs
        initializers       # initializer
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name="github.com/fengyuentau/onnx_utils")
    onnx.checker.check_model(model_def)
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
