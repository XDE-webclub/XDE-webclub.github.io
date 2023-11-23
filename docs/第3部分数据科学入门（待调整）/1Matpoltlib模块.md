

## matplotlib模块




### 二维直线图


```python 
import numpy as np
from matplotlib import pyplot as plt
#  生成一个-3到3的等差数列，共100个数
a = np.linspace(-3, 3, 10)

# 三角函数
b = np.sin(a)
```


```python 
plt.plot(a, b)
# 等价于 plt.plot(b)
plt.show() # 正弦图
```


#### 绘制多条数据线


```python 
# 画出多条数据线：
plt.plot(a, np.sin(a))
plt.plot(a, np.sin(2 * a))
plt.show()
```


#### 线条修饰


```python 
# 使用字符串，给定线条参数：
# b:蓝色
# -- : 虚线
# o : 圆点
'''
完整参数可参考
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot
'''
plt.plot(a, np.sin(a), 'b--o')
plt.show()
```


### 散点图


```python 
plt.plot(a, np.sin(a), 'bo')
plt.show()  # 二维散点图
# 等价于
plt.scatter(a, np.sin(a),color='blue',marker='o')
plt.show() 
```


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




