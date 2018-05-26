import numpy as np
import pandas as pd
np.random.RandomState(451).randn(4, 4).astype(np.float32)
np.random.RandomState(451).randn(4, 4).astype(np.float32).nbytes
np.random.RandomState(451).randn(4, 4).nbytes
df = pd.DataFrame(np.random.RandomState(451).randn(4, 4)).sample(frac=1, random_state=451).reset_index()
df

df.memory_usage().sum() # bytes

data = [1,2,3,4]
data = [1,2,3,4,5,6]

[x for x in data if x % 2 == 0]
[x for x in data if x % 4 == 0]


# from dask.distributed import Client
# client = Client()  # set up local cluster on your laptop
# client
#
# def square(x):
#         return x ** 2
#
# def neg(x):
#         return -x
#
# A = client.map(square, range(10))
# B = client.map(neg, A)
# total = client.submit(sum, B)
# total.result()
