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
)

def make_and_save_model(k):
    model_proto = models[k].to_model_proto()
    try:
        onnx.checker.check_model(model_proto)
    except onnx.checker.ValidationError as e:
        print(f"\t The model is invalid: {e}. Skipping ...")
        return False
    else:
        save_path = "./models/{}.onnx".format(k)
        print("\t The model is valid! Saved to {}".format(save_path))
        onnx.save(model_proto, save_path)
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
