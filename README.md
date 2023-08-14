# KGTL
# 更新日志
本方法库将长期进行维护与更新。
- 2023-08-14：支持选择浮点数精度，双精度`--fp fp64` 或者半精度`--fp fp16` ，默认单精度
- 2023-08-09：支持使用`--config`参数在运行main.py脚本时配置模型参数(之前需要修改默认模型参数配置代码)
- 2023-08-09：支持固定随机种子，例如执行main脚本时指定参数 `--seed 128`
- 2023-08-08：更改模型加载和保存逻辑，加载模型时不需要再指定模型超参数，可以从保存的文件中读取超参数，更加便捷易用。同时优化了模块的加载逻辑。
- 2023-08-07：添加CENET模型
- 2023-06-25：支持gpu选择,例如调用main.py脚本时指定0号gpu`--gpu 0`,修改数据按时间划分的函数,提升数据处理速度
- 2023-06-15：加入过滤功能,外推模型都支持过滤,调用main.py时添加`--filter`参数来进行过滤

# 1 简介
知识图谱工具库包含四个模块——数据模块(data)、工具模块(utils)、模型基类模块(base_models)和模型模块(models)。
数据模块包含数据加载类和数据集，工具模块包含一些辅助函数，模型基类模块包含各个模型的基类，基类仅包含初始化和前向传播的过程，训练和评估都集成在模型模块的模型类中。

目前已经集成的模型有：
- [CyGNet (2021)](https://ojs.aaai.org/index.php/AAAI/article/view/16604)
- [RE-GCN (2021)](https://dl.acm.org/doi/abs/10.1145/3404835.3462963)
- [CNE (2022)](https://arxiv.org/abs/2203.07782)
- [CENET (2023)](http://arxiv.org/abs/2211.10904)

使用者可以将集成的模型作为基线模型使用。也可以通过研究这些模型，在其基础上设计新模型。
方法库的模型训练与主干网络都尽量进行了解耦合，可以将模型基类模块作为新模型的一个模块去搭建网络。

此外方法库还包含一个可执行main.py脚本，提供了模型训练、评估、保存checkpoint、根据实验数据绘图等功能。

# 2 快速开始

执行以下命令快速训练模型
```
# CyGNet
python main.py --dataset ICEWS14s --model cygnet --epoch 50 --amsgrad --lr 0.001 --early-stop 3
# RE-GCN
python main.py --dataset ICEWS14s --model regcn --epoch 50 --lr 0.001 --weight-decay 1e-5 --early-stop 3
# CEN
python main.py --dataset ICEWS14s --model cen --epoch 50 --lr 0.001 --weight-decay 1e-5 --early-stop 3
#CeNet
python main.py --dataset ICEWS14s --model cenet --epoch 100 --amsgrad --lr 0.001 --gpu 0 --weight-decay 1e-5 --early-stop 3

```
加载保存的模型继续训练
```sh
# 指定 checkpoint id
python main.py --model cygnet --epoch 10 --checkpoint 20230808085552
```

执行以下脚本加载checkpoint在测试集上进行评估
```sh
# 激活--test参数并指定 checkpoint id
python main.py  --model cygnet --test --checkpoint 20230808085552
```
