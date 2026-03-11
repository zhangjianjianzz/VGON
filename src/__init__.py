"""
Model 提供生成层级的函数
train 训练提供 loss 函数的输入接口和迭代算子

这样指定 Model optimizer DataLoader train_step+loss 即可以函数式训练
- 方便地嵌入循环
"""