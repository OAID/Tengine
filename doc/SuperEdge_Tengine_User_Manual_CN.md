# Tengine + SuperEdge 一条指令跨平台部署边缘AI应用
------------
## 案例说明

​		案例基于开源AI推理框架Tengine 实现容器调用边缘硬件NPU资源，完成高效物体检测的推理任务，并通过开源边缘容器方案 SuperEdge 轻松将应用调度到边缘计算节点，实现一条指令部署边缘计算跨平台AI应用案例。

​		[Tengine](https://github.com/OAID/Tengine "Tengine")由[OPEN AI LAB](http://www.openailab.com/)主导开发，该项目实现了深度学习神经网络模型在嵌入式设备上的**快速**、**高效**部署需求。为实现在众多**AIoT**应用中的跨平台部署，本项目使用**C语言**进行核心模块开发，针对嵌入式设备资源有限的特点进行了深度框架裁剪。同时采用了完全分离的前后端设计，有利于 CPU、GPU、NPU 等异构计算单元的快速移植和部署，降低评估、迁移成本。

​		[SuperEdge](https://github.com/superedge/superedge "SuperEdge") 是基于原生 Kubernetes 的**边缘容器**管理系统。该系统把云原生能力扩展到**边缘侧**，很好的实现了云端对边缘端的**管理**和控制，极大**简化**了应用从云端部署到边缘端的过程。SuperEdge 为应用实现**边缘原生化**提供了**强有力**的支持。

<img src="http://tengine2.openailab.com:9527/openailab/structure.png" alt="topo" style="zoom:100%;" />

## 硬件环境准备

| 物品         | 描述                                                         |
| ------------ | ------------------------------------------------------------ |
| Master服务器 | SuperEdge Master 服务器， 用于应用调度，可采用X86 or Arm 架构，本例中采用X86服务器 |
| Khadas VIM3  | 应用负载工作节点，内置 A311D SoC 的单板计算机，内置 5Tops NPU 加速器，各大商城有售 |
| USB 摄像头   | 连接Khadas VIM3，输入实时视频流                              |
| 液晶显示器   | 连接Khadas VIM3，控制台操作，实时输出示例运行结果            |
| HDMI连接线   | 由于Khadas VIM3 的 TYPE C 接口与 HDMI 接口过于紧凑，需要寻找小一点接口的 HMDI 连接线 |



## 操作步骤

### 1.安装SuperEdge环境

- 安装SuperEdge Master节点（x86_64）
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
# 由于demo使用了桌面GUI画图，开机登录界面导致应用无法正常启动，因此需设置设备开机桌面自动登录
#步骤：打开图形桌面右上角 settings-users-AutoLogin 配置，开机无需输入密码进入桌面，重新启动，无登录画面即可

#Disable fenix-zram-config service to disable the swap 
sudo systemctl disable fenix-zram-config
sudo systemctl status fenix-zram-config

# Download edgeadm arm64 version to install SuperEdge Node 
wget https://superedge-1253687700.cos.ap-guangzhou.myqcloud.com/v0.4.0/arm64/edgeadm-linux-arm64-v0.4.0.tgz
tar -zxvf edgeadm-linux-arm64-v0.4.0.tgz
cd edgeadm-linux-arm64-v0.4.0

#Upgrade cni-plugins from v0.8.3 to v0.8.6, 解决在Khadas上安装SuperEdge和CNI失败的问题,
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

### 2.（可选）构建Tengine demo容器镜像
------------
该步骤介绍如何构建Tengine Demo镜像，如采用Docker Hub镜像, 可跳过。

- 下载文件包到Khadas VIM3设备上，构建Tengine物体识别应用docker镜像
```shell
#Download docker build packeage [~91M] from OPEN AI LAB server
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

### 3.编写 yolo.yaml 编排文件 

- 在SuperEdge Master节点上编辑k8s编排文件yolo.yaml
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
          image: tengine3/yolo:v1.0
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
### 4. Tengine物体识别应用应用部署

执行编排文件

```shell
kubectl apply -f yolo.yaml
```
### 5.案例效果与验证

通过命令查看部署状态

```
peter@peter-VirtualBox:~$ kubectl get deployment yolo -o wide
NAME   READY   UP-TO-DATE   AVAILABLE   AGE   CONTAINERS   IMAGES               SELECTOR
yolo   1/1     1            1           21h   yolo         tengine3/yolo:v1.0   name=yolo

peter@peter-VirtualBox:~$ kubectl get pod yolo-76d95967bb-zxggk 
NAME                    READY   STATUS    RESTARTS   AGE
yolo-76d95967bb-zxggk   1/1     Running   3          79m

```

打开Khadas VIM设备的显示器，观察到如下效果

<img src="http://tengine2.openailab.com:9527/openailab/demo.jpg" style="zoom:67%;" />


