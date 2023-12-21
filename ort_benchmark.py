import numpy as np
import onnxruntime as ort
import time

np.random.seed(0)
A = np.random.rand(12, 197, 197).astype(np.float32)
B = np.random.rand(12, 197, 64).astype(np.float32)

net = ort.InferenceSession("./models/matmul.onnx", providers=["CPUExecutionProvider"])
net.run([], {"A": A, "B": B})

times = []
for _ in range(100):
    start = time.time()
    net.run([], {"A": A, "B": B})
    end = time.time()
    times.append((end - start) * 1000)

mean = sum(times) / len(times)
times = sorted(times)
median = (times[49] + times[50]) / 2
minimum = min(times)
print("mean={:.2f}, median={:.2f}, min={:.2f}".format(mean, median, minimum))
