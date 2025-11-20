# Head Pose Estimation 项目文档

## 项目概述
本项目是基于深度学习的人脸姿态估计（Head Pose Estimation）项目进行了功能添加，可以实现学生课堂专注度检测，主要以头部姿态在空间中的几何角度来进行度量。

## 快速开始

### 环境要求
- Python 3.6+
- PyTorch 1.7+
- OpenCV
- 其他依赖见 `requirements.txt`

### 安装
```bash
pip install -r requirements.txt
```
### 训练

#### 单GPU训练
```bash
python main.py --data data/300W_LP --network resnet18
```

#### 多GPU训练（DDP）
```bash
torchrun --nproc_per_node=num_gpus python main.py --data data/300W_LP --network resnet18
```

#### 支持的网络架构
* resnet18
* resnet34
* resnet50
* mobilenetv2
* mobilenetv3_small
* mobilenetv3_large

#### 训练参数说明
```bash
usage: main.py [-h] [--data DATA] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--network NETWORK] [--lr LR] [--num-workers NUM_WORKERS] [--checkpoint CHECKPOINT] [--lr-scheduler {StepLR,MultiStepLR}] [--step-size STEP_SIZE] [--gamma GAMMA] [--milestones MILESTONES [MILESTONES ...]] [--print-freq PRINT_FREQ] [--world-size WORLD_SIZE]
               [--local-rank LOCAL_RANK] [--save-path SAVE_PATH]

Head pose estimation training.

选项:
  -h, --help            显示帮助信息
  --data DATA           数据目录路径
  --epochs EPOCHS       最大训练轮数
  --batch-size BATCH_SIZE
                        批大小
  --network NETWORK     网络架构，当前支持: resnet18/34/50, mobilenetv2
  --lr LR               基础学习率
  --num-workers NUM_WORKERS
                        数据加载的工作进程数
  --checkpoint CHECKPOINT
                        继续训练的检查点路径
  --lr-scheduler {StepLR,MultiStepLR}
                        学习率调度器类型
  --step-size STEP_SIZE
                        StepLR的学习率衰减周期
  --gamma GAMMA         StepLR和ExponentialLR的学习率衰减乘数因子
  --milestones MILESTONES [MILESTONES ...]
                        MultiStepLR的学习率衰减里程碑（如果使用StepLR则忽略）
  --print-freq PRINT_FREQ
                        打印训练进度的频率（批次数）。默认: 100
  --world-size WORLD_SIZE
                        分布式进程数
  --local-rank LOCAL_RANK
                        分布式训练的本地排名
  --save-path SAVE_PATH
                        模型检查点保存路径。默认: `weights`
```
#### 训练示例
```bash
# 基础训练
python main.py --data data/300W_LP --network resnet18 --epochs 100 --batch-size 32

# 从检查点继续训练
python main.py --data data/300W_LP --network resnet18 --checkpoint weights/resnet18_epoch50.pt

# 使用学习率调度器
python main.py --data data/300W_LP --network resnet50 --lr-scheduler StepLR --step-size 30 --gamma 0.1

# 多GPU训练
torchrun --nproc_per_node=4 python main.py --data data/300W_LP --network resnet18 --batch-size 128
```

## 测试

### 测试命令
```bash
# ResNet18（快30-40%）
python detect.py --input assets/test.mp4 --weights weights/resnet18.pt --network resnet18 --attention --no-pose --scale-factor 0.8 --output results/output.mp4
```

## 注释
本项目为北京理工大学研究生课程数字媒体技术课程的课程设计，简单设计代码复现以及头部姿态检测内容。