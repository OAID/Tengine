### Quickstart Guide
------------
### [SuperEdge](https://github.com/superedge/superedge "SuperEdge")
- 安装SuperEdge Master节点（x86_64机器）
```shell
wget https://superedge-1253687700.cos.ap-guangzhou.myqcloud.com/v0.4.0/amd64/edgeadm-linux-amd64-v0.4.0.tgz
tar -zxvf edgeadm-linux-amd64-v0.4.0.tgz
cd edgeadm-linux-amd64-v0.4.0
./edgeadm init --kubernetes-version=1.18.2 --image-repository superedge.tencentcloudcr.com/superedge --service-cidr=10.96.0.0/12 --pod-network-cidr=10.224.0.0/16 --install-pkg-path ./kube-linux-*.tar.gz --apiserver-cert-extra-sans=<Master Public IP> --apiserver-advertise-address=<Master Intranet IP> --enable-edge=true
#复制k8s配置文件到用户目录下
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
#去掉资源限制，解决khadas VIM3安装SuperEdge导致设备重启的问题
kubectl patch DaemonSet kube-proxy -n kube-system --type='json' -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/resources", "value":{}}]'
kubectl patch DaemonSet kube-flannel-ds -n kube-system --type='json' -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/resources", "value":{}}]'
kubectl patch DaemonSet tunnel-edge -n edge-system --type='json' -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/resources", "value":{}}]'
kubectl patch DaemonSet edge-health -n edge-system --type='json' -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/resources", "value":{}}]'
kubectl patch DaemonSet application-grid-wrapper-node -n edge-system --type='json' -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/resources", "value":{}}]'
```
- Khadas VIM3 设备加入集群
```shell
wget https://superedge-1253687700.cos.ap-guangzhou.myqcloud.com/v0.4.0/arm64/edgeadm-linux-arm64-v0.4.0.tgz
tar -zxvf edgeadm-linux-arm64-v0.4.0.tgz
cd edgeadm-linux-arm64-v0.4.0
#解决在Khadas上安装SuperEdge和CNI失败的问题
tar -zxvf kube-linux-arm64-v1.18.2.tar.gz
wget https://github.com/containernetworking/plugins/releases/download/v0.8.6/cni-plugins-linux-arm64-v0.8.6.tgz
mv cni-plugins-linux-arm64-v0.8.6.tgz edge-install/cni/cni-plugins-linux-arm64-v0.8.3.tgz
sed -i 's/\tload_kernel/# load_kernel/' edge-install/script/init-node.sh
tar -zcvf kube-linux-arm64-v1.18.2.1.tar.gz edge-install/
#加入集群
./edgeadm join <Master Public/Intranet IP Or Domain>:6443 --token xxxx --discovery-token-ca-cert-hash sha256:xxxxxxxxxx --install-pkg-path kube-linux-arm64-v1.18.2.1.tar.gz --enable-edge=true
```
- Khadas VIM3 设备开启Xserver授权
```shell
# Access to Xserver
# Execute script on device Terminal
xhost +
```

### [Tengine](https://github.com/OAID/Tengine "Tengine")
------------
- 在Khadas VIM3设备上构建Tengine物体识别应用docker镜像
```shell
wget http://tengine2.openailab.com:9527/openailab/yolo.tar.gz
tar -zxvf yolo.tar.gz
cd superedge
docker build -t yolo:latest .
```
Dockerfile文件如下所示
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
如果需要自己编译并生成demo_yolo_camera程序，具体操作参考[demo_videocapture user manual](https://github.com/OAID/Tengine/blob/tengine-lite/doc/demo_videocapture_user_manual.md "demo_videocapture user manual")

- 在SuperEdge Master节点上部署应用到Khadas VIM3设备上
编辑k8s编排文件yolo.yaml
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
执行编排文件
```shell
kubectl apply -f yolo.yaml
```
打开Khadas VIM设备的显示器，观察到如下效果

![](http://tengine2.openailab.com:9527/openailab/yolo_demo.jpg)
