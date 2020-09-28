## 一个跑实验的代码模板

这是一个用来跑机器学习实验的代码模板，目前是第一版，欢迎各种改进建议。

该模板可以方便地进行写模型，跑实验，log记录结果，以及调参等方面。

创建模板的意义在于，不同工作之间有很多可以复用的代码，因此套用模板可以节省很多时间，可以更加专注于模型和实验分析。

## 感谢
本模板参考自 华为诺亚实验室开源出来的[一份代码](https://github.com/huawei-noah/trustworthyAI/tree/master/Causal_Structure_Learning/GAE_Causal_Structure_Learning)，觉得整体格式非常清晰值得学习，故整理了这套模板作为写模型跑实验的一个代码模板。

对此表示非常感谢。


## 整体框架/目录树
```
│  main.py
│  README.md
│
├─data_loader
│      synthetic_dataset.py
│      __init__.py
│
├─helpers
│      analyze_utils.py
│      config_utils.py
│      dir_utils.py
│      log_helper.py
│      tf_utils.py
│      torch_utils.py
│
├─models
│      gae.py
│      __init__.py
│
├─output
└─trainers
        al_trainer.py
        __init__.py
```

## 具体说明

(1) helpers

这个文件夹中主要放一些帮助记录结果和分析实验的代码

- analyze_utils.py
  
    定义计算模型performance、对结果进行可视化等进行结果分析的函数。

- config_utils.py

    设置代码中参数的默认值及其含义说明，包括数据集、模型、训练过程以及其他等需要改变的参数。

    同时定义函数将上述参数进行保存，这样跑实验的时候除了结果也把对应参数保存起来，方便复现。

- dir_utils.py

    定义跟路径相关的函数，比如创建一个输出路径。
    
- log_helper.py

    创建一个logger类，并设置这个logger的记录方式，方便对实验过程进行打log

- tf_utils.py

    如果模型中使用了TensorFlow，可以设置一些TF相关函数，比如固定TF内部的随机种子使得结果可复现和判断CUDA是否可用等等。

- torch_utils.py

    如果模型中使用了PyTorch，可以设置一些PyTorch相关函数，比如固定PyTorch的随机种子和设置使用几号CUDA等等。


(2) data_loader

这个文件夹用于存放实验过程中数据相关的代码。

- synthetic_dataset.py
  
    定义一个类，用于生成数据。方便调用

(3) models

用于存放模型相关的代码，一般定义模型的计算方式就足够了，模型优化求解的过程另外放。

- model.py

    定义模型的类，方便调用。

(4) trainers

用于存放模型优化求解的代码。

- trainer.py

    定义一个Trainer类，也就是一个优化器，用于优化我们定义的模型。

(5) output

用于存放输出文件的文件夹，包括实验结果和log，以执行时间命名一个文件夹，将对应输出存放到其中。


(6) main.py

整个实验的代码主入口，流程为：

> 获取实验参数 -> 创建输出文件夹以及设置logger -> 保存参数 ->
> 
> 设置随机种子和CUDA环境保证结果可复现 -> 获得数据集 -> 定义模型 ->
> 
> 定义优化器 -> 优化器训练模型，返回结果 ->
>  
> 对结果进行可视化及计算performance -> 保存实验结果。





