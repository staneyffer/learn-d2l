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

3. ```torch.matmul```
是多维的矩阵乘法，二维情况下与```torch.mm```一样，
参考: https://blog.csdn.net/qsmx666/article/details/105783610

4. ```torch.bmm```
批量矩阵乘法
带参数注意力汇聚，假定两个张量的形状分别是$(n, a, b)$和$(n, b, c)$，他们的批量矩阵乘法输出的形状为$(n, a, c)$
```python
X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
torch.bmm(X, Y).shape
# 输出
torch.Size([2, 1, 6])
```