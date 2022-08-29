<p align="center">
<img src="https://raw.githubusercontent.com/OysterQAQ/Blog-Image/master/icon.png" width = "180" alt="pixivic_icon"/>
</p>


## Introduction

**deepix**全称为**dee**p learning based **pix**iv dataset。本repo会持续维护一些基于二次元插画相关的深度学习领域实践与探索。

目前模块包括：

* deepix-model：深度神经网络模型模块，目前包括基于resnet-inception-v2的多任务预测模型以及其去掉末层fc的特征抽取模型（当前拟合进度**35%**）
* deepix-webapp：对外提供web服务模块。目前包括开箱即用的二次元插画标签预测服务、以图搜图服务、特征抽取服务
* deepix-docker：基于容器化的部署模块，包括了部署所需要的配置文件与资源文件

## Architecture

<img src="https://raw.githubusercontent.com/OysterQAQ/Blog-Image/master/arch.png" alt="image-20220827172516288" style="border-radius:10px" />

## Technical overview

* Tensorflow 2.0作为模型训练引擎
* 基于resnet-inception-v2的深度神经网络架构
* 基于Spring Boot的web服务
* 基于TF-serving的模型部署与前向推理
* 基于Milvus实现的topk近似向量检索
* 基于docker-compose的容器化跨平台部署
* 基于Tendis的元数据存储

## Deploy

克隆本repo并在docker文件夹中使用docker-compose进行部署

```shell
#拉取项目
git clone https://github.com/OysterQAQ/deepix.git
cd deepix/deepix-docker
#使用docker-compose部署
docker-compose up -d
```

## Usage

基于restful api对外提供服务，以下是api文档（默认端口为8081，可在docker-compose.yaml中修改）：

### 获取插图特征向量

#### 基本信息

**Path：** /images/features

**Method：** POST

**接口描述：**


#### 请求参数

**Headers**

| 参数名称     | 参数值                | 是否必须 | 示例 | 备注 |
| ------------ | --------------------- | -------- | ---- | ---- |
| Content-Type | application/form-data | 是       |      |      |
| **Query**    |                       |          |      |      |

| 参数名称 | 是否必须 | 示例 | 备注     |
| -------- | -------- | ---- | -------- |
| file     | 是       |      | 插图文件 |

#### 返回数据

<table>
  <thead class="ant-table-thead">
    <tr>
      <th key=name>名称</th><th key=type>类型</th><th key=required>是否必须</th><th key=default>默认值</th><th key=desc>备注</th><th key=sub>其他信息</th>
    </tr>
  </thead><tbody className="ant-table-tbody"><tr key=0-0><td key=0><span style="padding-left: 0px"><span style="color: #8c8a8a"></span> message</span></td><td key=1><span>string</span></td><td key=2>非必须</td><td key=3></td><td key=4><span style="white-space: pre-wrap"></span></td><td key=5></td></tr><tr key=0-1><td key=0><span style="padding-left: 0px"><span style="color: #8c8a8a"></span> data</span></td><td key=1><span>string</span></td><td key=2>非必须</td><td key=3></td><td key=4><span style="white-space: pre-wrap">token</span></td><td key=5></td></tr>
               </tbody>
              </table>





## Thanks

本项目离不开以下开源项目

* [DeepDanbooru](https://github.com/KichangKim/DeepDanbooru)
* [Milvus](https://github.com/milvus-io/milvus)
* [TensorFlow](https://github.com/tensorflow/tensorflow)
* [Keras](https://github.com/keras-team/keras)
* [Spring Boot](https://github.com/spring-projects/spring-boot)
* [SeaweedFS](https://github.com/seaweedfs/seaweedfs)
* [TF-serving](https://github.com/tensorflow/serving)
* [Tendis](https://github.com/Tencent/Tendis)
* [Docker](https://github.com/docker/compose)

## Trend

![stars](https://starchart.cc/OysterQAQ/deepix.svg)

