# Dask Helm Chart

## Chart Details

This chart will deploy the following:

-   1 x Jupyter notebook (optional) with port 443 exposed on an external LoadBalancer
-   All using Kubernetes Deployments

Cluster can use kubernetes cluster deployment.

## Installing the Chart

To install the chart with the release name `my-release`:

```bash
$ helm install --name my-release .
```

## Configuration

The following tables list the configurable parameters of the Dask chart and their default values.

### jupyter

| Parameter               | Description                      | Default                  |
|-------------------------|----------------------------------|--------------------------|
| `namespace`         | Deployed namespace             | `default`  |
| `image.repository`         | Container image name             | `user01e/dask`  |
| `image.tag`      | Container image tag              | `0.7`                 |
| `image.pullPolicy`      | Container image tag              | `IfNotPresent`                 |
| `jupyter.servicePort`   | k8s service port                 | `443`                     |

Specify each parameter using the `--set key=value[,key=value]` argument to `helm install`.

Alternatively, a YAML file that specifies the values for the parameters can be provided while installing the chart. For example,

```bash
$ helm install --name my-release -f values.yaml stable/dask
```

> **Tip**: You can use the default [values.yaml](values.yaml)

