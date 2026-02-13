import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cupy as cp
import cudf

print("cupy:", cp.__version__)
print("cudf:", cudf.__version__)

# 1) 最小 cupy
x = cp.zeros((1024, 1024), dtype=cp.float32)
cp.cuda.runtime.deviceSynchronize()
print("cupy alloc ok")

# 2) 最小 scatter_add
mat = cp.zeros((1000, 2000), dtype=cp.float32)
row = cp.arange(1000, dtype=cp.int32)
col = cp.arange(1000, dtype=cp.int32)
mat.scatter_add((row, col), 1)
cp.cuda.runtime.deviceSynchronize()
print("scatter_add ok")

# 3) 最小 cudf
df = cudf.DataFrame({"a": cudf.Series(cp.arange(10))})
print(df.head())
print("cudf ok")
