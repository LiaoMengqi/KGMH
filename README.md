# KGTL

# 1 introduce

Knowledge Graph Tools Lib (KGTL) contains four modules, which are data , utils, base_models and models.

## 1.1 data

Firstly, data module is mainly used for data loading.

## 1.2 utils 
Utils module contains functions for data processing, file reading and writing, indicator calculation and data plotting.

## 1.3 base_models
This module contains some base models which only have initial function and forward function.

## 1.4 models 
This module contains complete models which integrate training, prediction, and loss calculations

# 2 Quick start

We can use the following command to quickly train an extrapolation model.

```sh
# CyGNet
python main.py --dataset ICEWS14s --model cygnet --epoch 3 --amsgrad --lr 0.001 --early-stop 3
# RE-GCN
python main.py --dataset ICEWS14s --model regcn  --epoch 50 --lr 0.001 --weight-decay 1e-5 --early-stop 3
# CEN
python main.py --dataset ICEWS14s --model cen  --epoch 50 --lr 0.001 --weight-decay 1e-5 --early-stop 3
```
We can use the following command to load a model and evaluate it on test dataset.
```sh
python main.py --dataset ICEWS14s --model cygnet --test --checkpoint cygnet_ICEWS14s_alpha5e-1_dim50_penalty-100
```
