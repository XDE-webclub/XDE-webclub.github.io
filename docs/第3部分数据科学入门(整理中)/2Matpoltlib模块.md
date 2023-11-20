
画图，引入matplotlib库的pyplot方法：

linspace 用来生成一组等间隔的数据：

```python
import numpy as np
from matplotlib import pyplot as plt

a = np.linspace(0, 2 * np.pi, 10)
a
# [ 0.          0.6981317   1.3962634   2.0943951   2.7925268   3.4906585
#   4.1887902   4.88692191  5.58505361  6.28318531]
```

```python
# 三角函数
b = np.sin(a)
b
```

```python
plt.plot(a, b)
plt.show() # 正弦图
```

```python
help(plt.plot)
```

从数组中选择元素：

假设我们想选取数组 b 中所有非负的部分，首先可以利用 b 产生一组布尔值：

```python
mask = b >= 0
mask
```

更密集的数据，更平滑的正弦曲线：

```python
x = np.linspace(0, 2 * np.pi, 50)
x
```

```python
plt.plot(np.sin(x))
plt.show()
```

```python
# 给定 x 和 y 值：
plt.plot(x, np.sin(x))
plt.show()
```

```python
# 画出多条数据线：
plt.plot(x, np.sin(x), x, np.sin(2 * x))
plt.show()
```

```python
# 使用字符串，给定线条参数：
plt.plot(x, np.sin(x), 'r-^')
plt.show()
```

```python
# 多线条：
plt.plot(x, np.sin(x), 'b-o', x, np.sin(2 * x), 'r-^')
plt.show()
```

### 散点图

```python
plt.plot(x, np.sin(x), 'bo')
plt.show()  # 二维散点图
```

用scatter画图，scatter 散点图

- scatter(x, y)
- scatter(x, y, size)
- scatter(x, y, size, color)

```python
plt.scatter(x, np.sin(x))
plt.show()  # 正弦函数
```

事实上，scatter函数与Matlab的用法相同，还可以指定它的大小，颜色等参数。

### 标签：label

可以在 plot 中加入 label ，使用 legend 加上图例：

```python
t = np.linspace(0, 2 * np.pi, 50)
x = np.sin(t)
plt.plot(t, x, 'bo', t, np.sin(2 * t), 'r-^', label='sin', color='red', )
plt.legend()
plt.xlabel('radians')
plt.ylabel('amplitude', fontsize='large')
plt.title('Sin(x)')
plt.grid()
plt.show()
```

```python
# 直方图
data = np.array([1234, 321, 400, 120, 11, 30, 2000])
plt.hist(data, 7)
plt.show()
```
