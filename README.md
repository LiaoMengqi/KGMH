# KGTL
Knowledge Graph Tools Lib (KGTL) contains four modules, which are data , utils, base_models and models.
## data
Firstly, data module is mainly used for data loading.
 
## utils 
Utils module contains functions for data processing, file reading and writing, indicator calculation and data plotting.

## base_models
This module contains some base models which only have initial function and forward function.

## models 
This module contains complete models which integrate training, prediction, and loss calculations

## Quick start
cygnet
```angular2html
python main --dataset ICEWS14s --model cygnet --amsgrad --epoch 15 --lr 0.001
```
```angular2html

```