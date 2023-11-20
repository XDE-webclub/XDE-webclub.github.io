
## SciPy的简介

SciPy是建立在在Numpy基础上的科学计算库，提供大量的科学计算支持。

在科学计算中我们往往会用到如下4个常用而又强大的功能：

* 数值优化：scipy.optimize, 最优问题的数值近似解问题能够得到求解
* 数值积分：scipy.integrate, 能够从数值角度求解积分问题，可以当作解析解的验证手段；同时也能够像Matlab一样求解微分方程
* 插值算法：scipy.interpolate, 采样精度过低时可以使用interpolate得到连续曲线
* 信号处理：scipy.signal, scipy.fftpack 滤波器和时间序列模型在数学上有异曲同工之妙，此处用这些库实现信号处理功能

SciPy的其他重要内容会在其他内容中反复提及，比如

* 线性代数：scipy.linalg直接映射了np.linalg的功能，如矩阵特征值分解、SVD分解、QR分解等。
* 统计功能：scipy.stats具有强大的统计分布生成、探索、检查功能，在统计分布/随机数生成部分中结合理论介绍。
* 稀疏矩阵：scipy.sparse在NumPy基础和NetworkX部分有所提及，在网络图模型中占有一席之地。
* 科学函数：scipy.special, 能够快速给出一些科学函数的值（常见的如贝塞尔函数，排列组合数等）

### 子模块

|子模块|描述|
|:----|:----|
|cluster|聚类算法|
|constants|物理数学常数|
|fftpack|快速傅里叶变换|
|integrate|积分和常微分方程求解|
|interpolate|插值|
|io|输入输出|
|linalg|线性代数|
|odr|正交距离回归|
|optimize|优化和求根|
|signal|信号处理|
|sparse|稀疏矩阵|
|spatial|空间数据结构和算法|
|special|特殊方程|
|stats|统计分布和函数|
|weave|调用C/C++|

使用scipy之前，基础模块需要导入：

```python
import numpy as np
```

```python
# 径向基函数
x = np.linspace(-3, 3, 100)
x
```

```python
# 高斯函数
plt.plot(x, np.exp(-1 * x ** 2))
plt.title("Gaussian")

plt.show()
```

```python
plt.savefig('Gaussian.png')
```

```python
import os
os.path.exists('Gaussian.png')
```

```python
# 高维 RBF 插值
# 三维数据点：
x, y = np.mgrid[-np.pi / 2:np.pi / 2:5j, -np.pi / 2:np.pi / 2:5j]
z = np.cos(np.sqrt(x ** 2 + y ** 2))
fig = plt.figure(figsize=(12, 6))
ax = fig.gca(projection="3d")
ax.scatter(x, y, z)
fig.savefig("mplot3d.jpg")
plt.show()
```

```python
os.remove('Gaussian.png')
os.remove('mplot3d.jpg')
```

## 统计模块：stats

Python 中常用的统计工具有 Numpy, Pandas, PyMC, StatsModels 等。
Scipy 中的子库 scipy.stats 中包含很多统计上的方法。

```python
# Numpy 自带简单的统计方法：
heights = np.array([1.46, 1.79, 2.01, 1.75, 1.56, 1.69, 1.88, 1.76, 1.88, 1.78])
print('mean,', heights.mean())
print('min,', heights.min())
print('max', heights.max())
print('stand deviation,', heights.std())
```

导入 Scipy 的统计模块：

```python
import scipy.stats.stats as st

print('mode, ', st.mode(heights))  # 众数及其出现次数
print('skewness, ', st.skew(heights))  # 偏度
print('kurtosis, ', st.kurtosis(heights))  # 峰度
```

### 概率分布

常见的连续概率分布有：
* 均匀分布
* 正态分布
* 学生t分布
* F分布
* Gamma分布
...
* 离散概率分布：
* 伯努利分布
* 几何分布
...

这些都可以在 scipy.stats 中找到。

它包含四类常用的函数：

* norm.cdf 返回对应的累计分布函数值
* norm.pdf 返回对应的概率密度函数值
* norm.rvs 产生指定参数的随机变量
* norm.fit 返回给定数据下，各参数的最大似然估计（MLE）值

 从正态分布产生500个随机点：

```python
# 正态分布
from scipy.stats import norm
x_norm = norm.rvs(size=500)
x_norm.shape
```

直方图：

```python
plt.ion() #开启interactive mode

h = plt.hist(x_norm)
print('counts, ', h[0])
print('bin centers', h[1])
figure = plt.figure(1)  # 创建图表1
plt.show()
```

归一化直方图（用出现频率代替次数），将划分区间变为 20（默认 10）：

```python
h = plt.hist(x_norm, bins=20)
plt.show()
```

在这组数据下，正态分布参数的最大似然估计值为：

```python
x_mean, x_std = norm.fit(x_norm)

print('mean, ', x_mean)
print('x_std, ', x_std)
```

将真实的概率密度函数与直方图进行比较：

```python
h = plt.hist(x_norm, bins=20)

x = np.linspace(-3, 3, 50)
p = plt.plot(x, norm.pdf(x), 'r-')
plt.show()
```

积分函数：

```python
from scipy.integrate import trapz

x1 = np.linspace(-2, 2, 108)
p = trapz(norm.pdf(x1), x1)
print('{:.2%} of the values lie between -2 and 2'.format(p))

plt.fill_between(x1, norm.pdf(x1), color='red')
plt.plot(x, norm.pdf(x), 'k-')
plt.show()
```

可以通过 loc 和 scale 来调整这些参数，一种方法是调用相关函数时进行输入：

```python
x = np.linspace(-3, 3, 50)
p = plt.plot(x, norm.pdf(x, loc=0, scale=1))
p = plt.plot(x, norm.pdf(x, loc=0.5, scale=2))
p = plt.plot(x, norm.pdf(x, loc=-0.5, scale=.5))
plt.show()
```

```python
# 不同参数的对数正态分布：
from scipy.stats import lognorm

x = np.linspace(0.01, 3, 100)

plt.plot(x, lognorm.pdf(x, 1), label='s=1')
plt.plot(x, lognorm.pdf(x, 2), label='s=2')
plt.plot(x, lognorm.pdf(x, .1), label='s=0.1')

plt.legend()
plt.show()
```

```python
# 离散分布
from scipy.stats import randint

# 离散均匀分布的概率质量函数（PMF）：
high = 10
low = -10

x = np.arange(low, high + 1, 0.5)
p = plt.stem(x, randint(low, high).pmf(x))  # 杆状图
plt.show()
```

### 假设检验

相关的函数：

1. 正态分布
2. 独立双样本 t 检验，配对样本 t 检验，单样本 t 检验
3. 学生 t 分布

导入函数：

```python
from scipy.stats import norm
from scipy.stats import ttest_ind

# 独立样本 t 检验
# 两组参数不同的正态分布：
n1 = norm(loc=0.3, scale=1.0)
n2 = norm(loc=0, scale=1.0)

# 从分布中产生两组随机样本：
n1_samples = n1.rvs(size=100)
n2_samples = n2.rvs(size=100)

# 将两组样本混合在一起：
samples = np.hstack((n1_samples, n2_samples))

# 最大似然参数估计：
loc, scale = norm.fit(samples)
n = norm(loc=loc, scale=scale)

# 比较：
x = np.linspace(-3, 3, 100)
```

```python
plt.hist([samples, n1_samples, n2_samples])
plt.plot(x, n.pdf(x), 'b-')
plt.plot(x, n1.pdf(x), 'g-')
plt.plot(x, n2.pdf(x), 'r-')
plt.show()
```

独立双样本 t 检验的目的在于判断两组样本之间是否有显著差异：

```python
t_val, p = ttest_ind(n1_samples, n2_samples)

print('t = {}'.format(t_val))
print('p-value = {}'.format(p))
# t = 0.868384594123
# p-value = 0.386235148899
```

p 值小，说明这两个样本有显著性差异。

```python

```

| [03_data_science/06_SciPy曲线拟合.ipynb](https://github.com/shibing624/python-tutorial/blob/master/03_data_science/06_SciPy曲线拟合.ipynb)  | Scipy曲线  |[Open In Colab](https://colab.research.google.com/github/shibing624/python-tutorial/blob/master/03_data_science/06_SciPy曲线拟合.ipynb) |

# SciPy曲线拟合

```python
# 导入基础包：
import matplotlib.pyplot as plt
import numpy as np
# 多项式拟合
from numpy import polyfit, poly1d

# 产生数据：
x = np.linspace(-5, 5, 100)
y = 4 * x + 1.5
noise_y = y + np.random.randn(y.shape[-1]) * 2.5

p = plt.plot(x, noise_y, 'rx')
p = plt.plot(x, y, 'b:')
plt.show()
```

进行线性拟合，polyfit 是多项式拟合函数，线性拟合即一阶多项式：

```python
coeff = polyfit(x, noise_y, 1)
coeff
```

一阶多项式 y=a1x+a0y=a1x+a0 拟合，返回两个系数 [a1,a0][a1,a0]。

画出拟合曲线：

```python
p = plt.plot(x, noise_y, 'rx')
p = plt.plot(x, coeff[0] * x + coeff[1], 'k-')
p = plt.plot(x, y, 'b--')
plt.show()
```

### 多项式拟合余弦函数

余弦函数：

```python
x = np.linspace(-np.pi, np.pi, 100)
y = np.cos(x)

# 用一阶到九阶多项式拟合，类似泰勒展开：
# 可以用 poly1d 生成一个以传入的 coeff 为参数的多项式函数：
y1 = poly1d(polyfit(x, y, 1))
y3 = poly1d(polyfit(x, y, 3))
y5 = poly1d(polyfit(x, y, 5))
y7 = poly1d(polyfit(x, y, 7))
y9 = poly1d(polyfit(x, y, 9))
x = np.linspace(-3 * np.pi, 3 * np.pi, 100)

p = plt.plot(x, np.cos(x), 'k')  # 黑色余弦
p = plt.plot(x, y1(x))
p = plt.plot(x, y3(x))
p = plt.plot(x, y5(x))
p = plt.plot(x, y7(x))
p = plt.plot(x, y9(x))

a = plt.axis([-3 * np.pi, 3 * np.pi, -1.25, 1.25])
plt.show()
```

黑色为原始的图形，可以看到，随着多项式拟合的阶数的增加，
曲线与拟合数据的吻合程度在逐渐增大。

### 最小二乘拟合

导入相关的模块：

```python
from scipy.stats import linregress

x = np.linspace(0, 5, 100)
y = 0.5 * x + np.random.randn(x.shape[-1]) * 0.35

plt.plot(x, y, 'x')
plt.show()
```

Scipy.linalg.lstsq 最小二乘解

可以使用 scipy.linalg.lstsq 求最小二乘解。

```python
X = np.hstack((x[:, np.newaxis], np.ones((x.shape[-1], 1))))
print(X[1:5])

# 求解：
from scipy.linalg import lstsq
C, resid, rank, s = lstsq(X, y)
print(C, resid, rank, s)
```

```python
# 画图：
p = plt.plot(x, y, 'rx')
p = plt.plot(x, C[0] * x + C[1], 'k--')
plt.show()

print("sum squared residual = {:.3f}".format(resid))
print("rank of the X matrix = {}".format(rank))
print("singular values of X = {}".format(s))
```

### Scipy.stats.linregress 线性回归

对于上面的问题，还可以使用线性回归进行求解：

```python
slope, intercept, r_value, p_value, stderr = linregress(x, y)
p = plt.plot(x, y, 'rx')
p = plt.plot(x, slope * x + intercept, 'k--')
plt.show()

print("R-value = {:.3f}".format(r_value))
print("p-value (probability there is no correlation) = {:.3e}".format(p_value))
print("Root mean squared error of the fit = {:.3f}".format(np.sqrt(stderr)))
```

可以看到，两者求解的结果是一致的，但是出发的角度是不同的。

### 高级的拟合

先定义这个非线性函数：y=ae^(−bsin(fx+ϕ))

```python
def function(x, a, b, f, phi):
    """a function of x with four parameters"""
    result = a * np.exp(-b * np.sin(f * x + phi))
    return result


# 画出原始曲线：
x = np.linspace(0, 2 * np.pi, 50)
actual_parameters = [3, 2, 1.25, np.pi / 4]
y = function(x, *actual_parameters)
p = plt.plot(x, y)
plt.show()
```

```python
# 加入噪声：
from scipy.stats import norm

y_noisy = y + 0.8 * norm.rvs(size=len(x))
p = plt.plot(x, y, 'k-')
p = plt.plot(x, y_noisy, 'rx')
plt.show()
```

高级的做法：

不需要定义误差函数，直接传入 function 作为参数：

```python
from scipy.optimize import curve_fit

p_est, err_est = curve_fit(function, x, y_noisy)
p_est
```

```python
p = plt.plot(x, y_noisy, "rx")
p = plt.plot(x, function(x, *p_est), "g--")
plt.show()
```

```python
# 这里 curve_fit 第一个返回的是函数的参数，第二个返回值为各个参数的协方差矩阵：
print(p_est)
print(err_est)

# 协方差矩阵的对角线为各个参数的方差：
print("normalized relative errors for each parameter")
print("   a\t    b\t    f\t    phi")
print(np.sqrt(err_est.diagonal()) / p_est)
```

```python

```
