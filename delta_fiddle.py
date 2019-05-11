# spark-class org.apache.spark.deploy.master.Master -h 127.0.0.1
# spark-class org.apache.spark.deploy.worker.Worker spark://127.0.0.1:7077 -c 1 -m 1GB -h 127.0.0.1

import random
import pandas as pd
import numpy as np
import datetime

from contextlib import contextmanager
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F



conf = SparkConf(
).setMaster(
    "spark://127.0.0.1:7077"
).setAppName(
    "mittens-local"
).set(
    "spark.executor.memory",
    "1024m",
).set(
    'spark.driver.extraClassPath',
    '/Users/erik/code/scala/foo-build/target/scala-2.12/hello_2.12-0.1.0-SNAPSHOT.jar',
)
# conf = SparkConf(
# ).setMaster(
#     "spark://127.0.0.1:7077"
# ).setAppName(
#     "mittens-local"
# ).set(
#     "spark.executor.memory",
#     "1024m",
# ).set(
#     'spark.jars.packages',
#     'io.delta:delta-core_2.11:0.1.0',
# )


spark_context = SparkContext(conf=conf)
spark = SparkSession(
    spark_context
)

# spark_context.stop()

NUM_SAMPLES = 100000000
def inside(p):
    x, y = random.random(), random.random()
    return x*x + y*y < 1

count = spark_context.parallelize(range(0, NUM_SAMPLES)).filter(inside).count()
pi = 4 * count / NUM_SAMPLES
f"Pi is roughly {pi}"

datetime.datetime.now().isoformat()

local_pth = "/tmp/delta-table-local"
data = spark.range(20, 30)
data.write.format("delta").mode("overwrite").save(local_pth)
data2 = spark.range(5_000, 10_000)
data2.write.format("delta").mode("overwrite").save(local_pth)

df = spark.read.format("delta").load(local_pth)
spark.read.format("delta").load(local_pth).count()

spark.read.format("delta").option("versionAsOf", 0).load(local_pth).count()
spark.read.format("delta").option("versionAsOf", 1).load(local_pth).count()
spark.read.format("delta").option("versionAsOf", 2).load(local_pth).count()
spark.read.format("delta").option("versionAsOf", 3).load(local_pth).count()

df1 = spark.read.format(
    "delta"
).option(
    "versionAsOf",
    0
).load(
    local_pth
)
df1.show()

df1 = spark.read.format("delta").option("timestampAsOf", timestamp_string).load("/delta/events")



iris_path = "/Users/erik/.keras/datasets/iris_training.csv"
df = spark.read.csv(iris_path)
df_header = spark.read.format("csv").option("header", "true").load(iris_path)

df_header.show()
df_header.dtypes
df_header['setosa'].astype(np.float32)
changedTypedf = df_header.withColumn("setosa", df_header["setosa"].cast("double"))
changedTypedf = df_header.withColumn("setosa_dbl", df_header["setosa"].cast("double") * 2)
changedTypedf.withColumn("Literal", F.lit(0)).dtypes
df = changedTypedf.withColumn(
    "Literal", F.lit(0)
)

df.withColumn("meta", df.setosa_dbl * 2).show()
df.withColumn("meta", df['setosa_dbl'] * 2).show()


changedTypedf['mittens'] = changedTypedf['setosa_dbl'] * 10
changedTypedf.sum('setosa_dbl')



df_header.take(2)
df_header.head(4)
df_header.head(4)

# Displays the content of the DataFrame to stdout
df.show()
df.assign
result_pdf = df.select("*").toPandas()
df.toPandas()
result_pdf.dtypes

df.write.parquet("output/proto.parquet")



data = spark_context.range(0, 5)
data.write.format("delta").save("/tmp/delta-table")
