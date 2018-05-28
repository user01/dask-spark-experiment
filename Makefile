IP := $(shell ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -n1)

# http://192.168.0.59:8080/
spark-build:
	docker build -t spark-trial:v001 -f Dockerfile.spark .

spark-master:
	docker run --rm -it -p 8080:8080 -p 7077:7077 spark-trial:v001 bin/spark-class org.apache.spark.deploy.master.Master

spark-worker:
	docker run --rm -it spark-trial:v001 bin/spark-class org.apache.spark.deploy.worker.Worker spark://$(IP):7077 -c 1 -m 1GB

spark-notebook:
	docker run --rm -it -p 9999:8888 spark-trial:v001 jupyter notebook --ip=0.0.0.0 --allow-root


# http://localhost:8787/status
dask-build:
	docker build -t dask3.6example:v0.3 -f Dockerfile .

dask-master:
	docker run --rm -it -p 8786:8786 -p 8787:8787 dask3.6example:v0.3 dask-scheduler

dask-worker:
	# echo $(IP)
	docker run --rm -it dask3.6example:v0.3 dask-worker $(IP):8786 --nthreads 1 --memory-limit 0.2

dask-notebook:
	docker run --rm -it -p 8888:8888 -v $(PWD)/data/:/dask-tutorial/data/ dask3.6example:v0.3 jupyter lab --ip=0.0.0.0 --allow-root  --NotebookApp.token=''

dask-prep:
	docker run --rm -it -p 8888:8888 -v $(PWD)/data/:/dask-tutorial/data/ dask3.6example:v0.3 python prep.py
