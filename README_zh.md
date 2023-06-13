# KGTL
# 更新计划
- 加入过滤功能
- 集成更多模型

# 1 简介
知识图谱工具库包含四个模块——数据模块、工具模块、模型基类模块和模型模块。数据模块包含数据加载类和数据集，工具模块包含一些辅助函数，模型基类模块包含各个模型的基类，基类仅包含初始化和前向传播的过程，训练和评估都集成在模型模块的模型类中。

此外方法库还包含一个可执行main.py脚本，提供了模型训练、评估、保存checkpoint、根据实验数据绘图等功能。

# 2 快速开始

执行以下命令快速训练模型
```
# CyGNet
python main.py \
  --dataset ICEWS14s \
  --model cygnet \
  --epoch 50 \
  --amsgrad \
  --lr 0.001 \
  --early-stop 3
# RE-GCN
python main.py \
  --dataset ICEWS14s \
  --model regcn  \
  --epoch 50 \
  --lr 0.001 \
  --weight-decay 1e-5 \
  --early-stop 3
# CEN
python main.py \
  --dataset ICEWS14s \
  --model cen  \
  --epoch 50 \
  --lr 0.001 \
  --weight-decay 1e-5 \
  --early-stop 3
```
执行以下脚本加载checkpoint在测试集上进行评估
```sh
python main.py \
  --dataset ICEWS14s \
  --model cygnet \
  --test \
  --checkpoint cygnet_ICEWS14s_alpha5e-1_dim50_penalty-100
```
