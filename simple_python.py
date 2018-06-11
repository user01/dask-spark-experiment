import dask.array as da
from dask.distributed import Client
client = Client('toned-lion-dask-cluster-scaler-scheduler:8786')
client

x = da.random.normal(0, 1, size=(100000,100000), chunks=(1000, 1000))
fut = client.compute(x.mean())
fut

from __future__ import print_function

import thelib

import numpy as np
import pandas as pd

np.random.RandomState(451).randn(50000, 300).mean()
import time
time.time()

from dask.distributed import Client
from functools import partial

count_cols = 40
count_rows = 80500
common_data = pd.DataFrame(np.random.RandomState(451).randn(count_rows, count_cols)).sample(frac=1, random_state=451).reset_index()
byte_count = common_data.memory_usage().sum() # bytes
print("Generated common data of size {}".format(thelib.get_(byte_count)))

client = Client('192.168.1.151:8786')

# Makes this future available on all nodes with broadcase
future_common_data = client.scatter(common_data, broadcast=True)

the_work = partial(thelib.do_work)
futures = client.map(lambda idx: the_work(idx + 1, future_common_data), range(count_cols))
# futures = client.map(lambda idx: thelib.do_work(idx + 1, future_common_data), range(count_cols))

print(client.gather(futures))


from IPython.lib import passwd
password = passwd("batman123")
password

# Need:
# python code to unload library if loaded
# remove from system path if loaded
# create zip from module
# add zip to system path

# perhaps:
#  * unload libs if exist (unload and remove from path)
#  * zip up library
#  * load into memory
#  * scatter zipped library into workers
#  * store zip in known location
#  *

import itertools
list(itertools.chain.from_iterable([[1, 2, 3], [4, 5, 6]]))
list(itertools.chain.from_iterable([[1, 2, 3], [4, 5, 6], [[1], [8,2]]]))

import featuretools as ft
