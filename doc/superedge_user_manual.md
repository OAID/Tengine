# [SuperEdge](https://github.com/superedge/superedge "SuperEdge") & [Tengine](https://github.com/OAID/Tengine "Tengine")

------------
## Quickstart Guide
### Install edge Kubernetes master node

```shell
wget http://tengine2.openailab.com:9527/openailab/edgeadm-linux-amd64-v0.4.0.tgz
tar -zxvf edgeadm-linux-amd64-v0.4.0.tgz
cd edgeadm-linux-amd64-v0.4.0
./edgeadm init --kubernetes-version=1.18.2 --image-repository superedge.tencentcloudcr.com/superedge --service-cidr=10.96.0.0/12 --pod-network-cidr=10.224.0.0/16 --install-pkg-path ./kube-linux-*.tar.gz --apiserver-cert-extra-sans=<Master Public IP> --apiserver-advertise-address=<Master Intranet IP> --enable-edge=true
```
### Join edge node

```shell
wget http://tengine2.openailab.com:9527/openailab/edgeadm-linux-arm64-v0.4.0.tgz
tar -zxvf edgeadm-linux-arm64-v0.4.0.tgz
cd edgeadm-linux-arm64-v0.4.0
./edgeadm join <Master Public/Intranet IP Or Domain>:Port --token xxxx --discovery-token-ca-cert-hash sha256:xxxxxxxxxx --install-pkg-path kube-linux-arm64-v1.18.2.tar.gz --enable-edge=true
```
### Build Docker images on Khadas VIM3 Device

```shell
wget http://tengine2.openailab.com:9527/openailab/yolo.tar.gz
tar -zxvf yolo.tar.gz
cd superedge
docker build -t yolo:latest .
```
Dockerfile
```
FROM ubuntu:20.04
MAINTAINER openailab
RUN apt-get update
RUN apt-get install -y tzdata
RUN apt-get install -y libopencv-dev
RUN apt-get install -y libcanberra-gtk-module
RUN useradd -m openailab
COPY libtengine-lite.so /root/myapp/
COPY demo_yolo_camera /root/myapp/
COPY tm_330_330_330_1_3.tmcache /root/myapp/
ADD models /root/myapp/models/
COPY tm_88_88_88_1_1.tmcache /root/myapp/
COPY tm_classification_timvx /root/myapp/
COPY libOpenVX.so /lib/
COPY libGAL.so /lib/
COPY libVSC.so /lib/
COPY libArchModelSw.so /lib/
COPY libNNArchPerf.so /lib/
COPY libgomp.so.1 /lib/aarch64-linux-gnu/
COPY libm.so.6 /lib/aarch64-linux-gnu/
WORKDIR /root/myapp/
USER openailab
CMD ["./demo_yolo_camera"]
```
[Tengine lite source code Download](http://www.baidu.com "Tengine lite source code Download")

### RUN yolo docker container on  Khadas VIM3 Device

```shell
# Access to Xserver
# Execute script on device Terminal
xhost +
# Run
docker run -it --name yolo --privileged -v /dev:/dev -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=:0 -e GDK_SCALE -e GDK_DPI_SCALE yolo:latest
```
### Deploy yolo with SuperEdge

Edit yolo.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: yolo
  labels:
    name: yolo
spec:
  replicas: 1
  selector:
    matchLabels:
      name: yolo
  template:
    metadata:
      labels:
        name: yolo
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: kubernetes.io/hostname
                    operator: In
                    values:
                      - khadas
      containers:
        - name: yolo
          image: registry.cn-shenzhen.aliyuncs.com/edge_studio/yolo:v1.0
          env:
          - name: DISPLAY
            value: :0
          volumeMounts:
            - name: dev
              mountPath: /dev
            - name: unix
              mountPath: /tmp/.X11-unix
          securityContext:
            privileged: true
      volumes:
        - name: dev
          hostPath:
            path: /dev
        - name: unix
          hostPath:
            path: /tmp/.X11-unix
```
## Deploy yolo App

```shell
kubectl apply -f yolo.yaml
```
![](http://tengine2.openailab.com:9527/openailab/yolo_demo.jpg)