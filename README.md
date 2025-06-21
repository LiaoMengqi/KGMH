# KGMH

# 最新更新

本方法库将长期进行维护与更新。
- 2023-10-25：修复过滤分数惩罚项为负可能导致分数变大的错误
- 2023-09-09：新增静态模型SACN
- 2023-09-07：集成最短路径搜索功能，新增DistMult模型
- 2023-09-02：支持自动选择显存最多的GPU，设置`--gpu -1` 来使用CPU，也可以通过指定GPU号来指定GPU`--gpu 0`
- 2023-08-31：修复CENET的小错误，新增静态模型RGCN
- 2023-08-26：可以通过`--monitor`选择早停监控的指标；采取过滤时，早停指标也改为过滤后的指标


# 1 简介

知识图谱工具库包含四个模块——数据模块(data)、工具模块(utils)、模型基类模块(base_models)和模型模块(models)。
数据模块包含数据加载类和数据集，工具模块包含一些辅助函数，模型基类模块包含各个模型的基类，基类仅包含初始化和前向传播的过程，训练和评估都集成在模型模块的模型类中。

静态模型（静态知识图谱上进行推理通常需要采取过滤）：

- [TransE (2013)](https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)
- [DistMult (2015)](http://arxiv.org/abs/1412.6575)
- [R-GCN (2018)](https://link.springer.com/chapter/10.1007/978-3-319-93417-4_38)
- [SACN (2019)](https://ojs.aaai.org/index.php/AAAI/article/view/4164)

时序外推：

- [CyGNet (2021)](https://ojs.aaai.org/index.php/AAAI/article/view/16604)
- [RE-GCN (2021)](https://dl.acm.org/doi/abs/10.1145/3404835.3462963)
- [CNE (2022)](https://arxiv.org/abs/2203.07782)
- [CENET (2023)](http://arxiv.org/abs/2211.10904)

使用者可以将集成的模型作为基线模型使用。也可以通过研究这些模型，在其基础上设计新模型。
方法库的模型训练与主干网络都尽量进行了解耦合，可以将模型基类模块作为新模型的一个模块去搭建网络。

此外方法库还包含一个可执行main.py脚本，提供了模型训练、评估、保存checkpoint、根据实验数据绘图等功能。

# 2 推荐配置

个人使用的配置，仅供参考，大版本保持一致应该都可以。

```
python 3.9.16
pytorch 2.0.1
```

# 3 快速开始

训练模型时至少需要指定模型和数据集两个参数，其他参数都有默认值，例如：

```shell
python main.py --dataset ICEWS14s --model cygnet
```

其他常用参数设置示例如下：

```shell
# --epoch 迭代轮数，默认30
python main.py --dataset ICEWS14s --model cygnet --epoch 100

# --batch-size batch的大小，默认1024
python main.py --dataset ICEWS14s --model cygnet --batch-size 256

# --lr 学习率，默认1e-3
python main.py --dataset ICEWS14s --model cygnet --lr 1e-5

# --gpu 选取gpu号（单gpu只能指定为0），默认自动选择显存占用最低的gpu
python main.py --dataset ICEWS14s --model cygnet --gpu 0

# --early-stop 早停轮数，默认不采取早停
python main.py --dataset ICEWS14s --model cygnet --early-stop 3

# --filter 采取过滤，默认不采取过滤
python main.py --dataset ICEWS14s --model cygnet --filter

# 随机种子，默认为0
python main.py --dataset ICEWS14s --model cygnet --seed 123

# 配置模型参数，默认采用模型默认参数，激活配置参数后需要手动输入模型的各个超参数
python main.py --dataset ICEWS14s --model cygnet --config
```

以下是各个模型推荐的配置，不一定是最佳，可以自己再对超参数进行调整

static

```shell
# TransE
python main.py --model transe --dataset FB15k  --early-stop 3 --filter

# R-GCN
pyhton main.py --mldel  rgcn --dataset FB15k --early-stop 3 --filter

# DistMult
pyhton main.py --model distmult --dataset FB15k --early-stop 3 --filter --lr 1e-4

# SACN
pyhton main.py --model sacn --dataset FB15k --early-stop 5 --filter --lr 1e-4
```

temporal

```shell
# CyGNet
python main.py --dataset ICEWS14s --model cygnet --amsgrad --early-stop 3

# RE-GCN
python main.py --dataset ICEWS14s --model regcn  --weight-decay 1e-5 --early-stop 3

# CEN
python main.py --dataset ICEWS14s --model cen  --weight-decay 1e-5 --early-stop 3

# CeNet
python main.py --dataset ICEWS14s --model cenet  --amsgrad --weight-decay 1e-5 --early-stop 3

```

加载保存的模型继续训练，至少需要指定模型和 checkpoint id，训练轮数和早停等参数可选

```shell
# 指定 checkpoint id, 以 checkpoint id 为 20230808085552 为例
python main.py --model cygnet --checkpoint 20230808085552
```

执行以下脚本加载checkpoint在测试集上进行评估

```shell
# 激活--test参数并指定 checkpoint id
python main.py  --model cygnet --test --checkpoint 20230808085552
```

# 更新历史
- 2025-06-21: 新增参数`----entity`，通过指定entity参数为`object`或`subject`，来评估模型预测头实体或尾实体的性能表现
- 2023-08-19：优化指标计算过程的性能，速度快了很多，在gup上训练时，大多数模型可以在1秒内完成评估
- 2023-08-19：设计所有模型的父类，统一接口
- 2023-08-16：打通静态知识图谱模型全流程，新增TransE模型
- 2023-08-14：支持选择浮点数精度，双精度`--fp fp64` 或者半精度`--fp fp16` ，默认单精度
- 2023-08-09：支持使用`--config`参数在运行main.py脚本时配置模型参数(之前需要修改默认模型参数配置代码)
- 2023-08-09：支持固定随机种子，例如执行main脚本时指定参数 `--seed 128`
- 2023-08-08：更改模型加载和保存逻辑，加载模型时不需要再指定模型超参数，可以从保存的文件中读取超参数，更加便捷易用。同时优化了模块的加载逻辑。
- 2023-08-07：添加CENET模型
- 2023-06-25：支持gpu选择,例如调用main.py脚本时指定0号gpu`--gpu 0`,修改数据按时间划分的函数,提升数据处理速度
- 2023-06-15：加入过滤功能,外推模型都支持过滤,调用main.py时添加`--filter`参数来进行过滤