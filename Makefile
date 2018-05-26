
spark-build:
	docker build -t spark-trial:v001 -f Dockerfile.spark .

spark-master:
	docker run --rm -it -p 8080:8080 -p 7077:7077 spark-trial:v001 bin/spark-class org.apache.spark.deploy.master.Master

spark-worker:
	docker run --rm -it spark-trial:v001 bin/spark-class org.apache.spark.deploy.worker.Worker spark://10.22.1.216:7077 -c 1 -m 1GB

spark-notebook:
	docker run --rm -it -p 8888:8888 spark-trial:v001 jupyter notebook --ip=0.0.0.0 --allow-root


dask-build:
	docker build -t dask2.7example:v0.4 -f Dockerfile .

dask-master:
	docker run --rm -it -p 8786:8786 -p 8787:8787 dask2.7example:v0.4 dask-scheduler

IP := $(shell ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1')
dask-worker:
	# echo $(IP)
	docker run --rm -it dask2.7example:v0.4 dask-worker $(IP):8786 --nthreads 2 --memory-limit 0.25

dask-notebook:
	docker run --rm -it -p 8888:8888 dask2.7example:v0.4 jupyter notebook --ip=0.0.0.0 --allow-root
