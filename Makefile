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
	docker build -t user01e/dask:0.6 -f Dockerfile .

dask-master:
	docker run --rm -it -p 8786:8786 -p 8787:8787 user01e/dask:0.6 dask-scheduler

dask-worker:
	# echo $(IP)
	docker run --rm -it user01e/dask:0.6 dask-worker $(IP):8786 --nthreads 1 --memory-limit 0.2

dask-notebook:
	docker run --rm -it -p 8888:8888 -v $(PWD)/data/:/dask-tutorial/data/ user01e/dask:0.6 jupyter lab --ip=0.0.0.0 --allow-root  --NotebookApp.token=''

dask-prep:
	docker run --rm -it -p 8888:8888 -v $(PWD)/data/:/dask-tutorial/data/ user01e/dask:0.6 python prep.py


minikube:
	-minikube delete
	minikube start --memory 8192 --cpus 6
	# minikube start --memory 4096 --cpus 4  --extra-config kubelet.EnableCustomMetrics=true
	sleep 10
	helm init
	sleep 60
	# minikube addons enable heapster
	minikube addons enable metrics-server
	helm repo add coreos https://s3-eu-west-1.amazonaws.com/coreos-charts/stable/
	helm install coreos/prometheus-operator --name prometheus-operator --namespace monitoring
	helm install coreos/kube-prometheus --name kube-prometheus --set global.rbacEnable=true --namespace monitoring
	# kubectl port-forward -n monitoring prometheus-kube-prometheus-0 9090
	# kubectl port-forward $(kubectl get  pods --selector=app=kube-prometheus-grafana -n  monitoring --output=jsonpath="{.items..metadata.name}") -n monitoring  3000


# Get token for logging into main dashboard
# kubectl config view | grep -A10 "name: $(kubectl config current-context)" | awk '$1=="access-token:"{print $2}'

# give ourselves admin access
# gcloud info | grep Account
# kubectl create clusterrolebinding local-cluster-admin-binding \
# >  --clusterrole=cluster-admin \
# >  --user=erik.langenborg@gmail.com

# give the default the power
# kubectl create -f cluster_role.yaml
# kubectl create clusterrolebinding dask-role-something \
>  --clusterrole=dask-doer  \
>  --serviceaccount=default:default
