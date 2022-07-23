# Pytorch常用操作

## 随机数

### torch.randn
从标准正态分布（均值为0，方差为1）中随机抽样的一组数据
例如
```python
torch.randn(2)
torch.randn((2,3))
```

## 矩阵操作

### 矩阵乘法

1. ```torch.mm```

2. ```@```