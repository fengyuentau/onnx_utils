import sys

import numpy as np
import onnx

import onnxscript as ost
from onnxscript import opset19 as op
msop = ost.values.Opset(domain="com.microsoft", version=1)

@ost.script()
def make_gather_shape1d_axis1(x: ost.FLOAT[1, 512, 640, 3]) -> ost.FLOAT[1, 128, 640, 3]:
    indices = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [128], np.arange(2, 512, 4, dtype=np.int64)))
    y = op.Gather(x, indices, axis=1)
    return y
@ost.script()
def make_tile_repeats1d(x: ost.FLOAT[2, 3, 4, 5]) -> ost.FLOAT[14, 18, 16, 10]:
    repeats = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [4], np.array([7, 6, 4, 2], dtype=np.int64)))
    y = op.Tile(x, repeats)
    return y
@ost.script()
def make_slice_neg_steps(x: ost.FLOAT[20, 10, 5]) -> ost.FLOAT[19, 3 ,2]:
    starts = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [3], np.array([20, 10, 4], dtype=np.int64)))
    ends = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [3], np.array([0, 0, 1], dtype=np.int64)))
    axes = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [3], np.array([0, 1, 2], dtype=np.int64)))
    steps = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [3], np.array([-1, -3, -2], dtype=np.int64)))
    y = op.Slice(x, starts, ends, axes, steps)
    return y
@ost.script()
def make_gather_shared_indices(x: ost.FLOAT[2, 1, 3, 4]) -> ost.FLOAT[3, 4]:
    indices = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [], np.array([0], dtype=np.int64)))
    y0 = op.Gather(x, indices, axis=0)
    y1 = op.Gather(y0, indices, axis=0)
    return y1
@ost.script()
def make_greater_input_dtype_int64(x: ost.INT64[27, 9]) -> ost.BOOL[27, 9]:
    y = op.Greater(x, op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [], np.array([61], dtype=np.int64))))
    return y

#######################################
### Quantized models from other domains
#######################################
@ost.script()
def make_qlinearsoftmax_opset13(x: ost.FLOAT[10, 2]) -> ost.FLOAT[10, 2]:
    # Create QuantizeLinear
    quant_scale = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [], np.array([0.20202656090259552], dtype=np.float32)))
    quant_zero_point = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT8, [], np.array([-8], dtype=np.int8)))
    quant_x = op.QuantizeLinear(x, quant_scale, quant_zero_point)

    # Create QLinearSoftmax
    dequant_scale = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [], np.array([0.00390625], dtype=np.float32)))
    dequant_zero_point = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT8, [], np.array([-128], dtype=np.int8)))
    quant_y = msop.QLinearSoftmax(quant_x, quant_scale, quant_zero_point, dequant_scale, dequant_zero_point)

    # Create Dequant
    y = op.DequantizeLinear(quant_y, dequant_scale, dequant_zero_point)
    return y

models = dict(
    gather_shape1d_axis1=make_gather_shape1d_axis1,
    qlinearsoftmax_opset13=make_qlinearsoftmax_opset13,
    tile_repeats1d=make_tile_repeats1d,
    slice_neg_steps=make_slice_neg_steps,
    gather_shared_indices=make_gather_shared_indices,
    greater_input_dtype_int64=make_greater_input_dtype_int64,
)

def make_and_save_model(k):
    model_proto = models[k].to_model_proto()
    try:
        onnx.checker.check_model(model_proto, full_check=True)
    except onnx.checker.ValidationError as e:
        print(f"\t Model {k} is invalid: {e}. Skipping ...")
        return False
    else:
        save_path = "./models/{}.onnx".format(k)
        print(f"\t Model {k} is valid! Saved to {save_path}")
        model_proto_ = onnx.shape_inference.infer_shapes(model_proto)
        onnx.save(model_proto_, save_path)
        return True

def main():
    l = []
    for i in range(1, len(sys.argv)):
        l.append(sys.argv[i])
    if not l:
        l = list(models.keys())
    print("Making models: {}".format(l))

    failed = []
    for m in l:
        if not make_and_save_model(m):
            failed.append(m)
    if failed:
        print("Failed to make: {}".format(failed))

if __name__ == "__main__":
    main()
