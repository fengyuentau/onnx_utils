# https://onnxruntime.ai/docs/api/python/api_summary.html
import onnxruntime as ort # build ort with RelWithDebugInfo to get more logs

so = ort.SessionOptions()
# Log severity level. Applies to session load, initialization, etc. 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.
so.log_severity_level = 0
net = ort.InferenceSession("/path/to/model.onnx", providers=["CPUExecutionProvider"], sess_options=so)
# some logs
net.run()
# some logs
