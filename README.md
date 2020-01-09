# pytorch seq2seq


### 一、Requirement
```
anaconda3
pytorch
matplotlib
```

### 二、使用
1.准备数据集，data目录下train.txt格式

```
你/好                   #问题
你/好                   #答案
```

2.训练模型

```
anaconda3/bin/python main.py --train=1
```

3.模型预测

```
anaconda3/bin/python main.py --train=0 --msg=你好
```