---
sidebar_position: 0
title: K-邻近算法
---

## K-邻近算法

这个算法既可以解决分类问题，也可以用于回归问题，但工业上用于分类的情况更多。

KNN先记录所有已知数据，再利用一个距离函数，

找出已知数据中距离未知事件最近的K组数据，

最后按照这K组数据里最常见的类别预测该事件。

```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 创建一些示例数据
X = np.array([[1, 2], [2, 3], [2, 5], [3, 2], [3, 3], [4, 5]])  # 特征
y = np.array([0, 0, 1, 0, 1, 1])  # 目标标签

# 创建K-最近邻分类器
k = 3  # 选择K的值
model = KNeighborsClassifier(n_neighbors=k).fit(X, y)

# 预测新数据点
new_data_point = np.array([[3, 4]])  # 要预测的新数据点

# .predicts()方法返回一个数组，数组中包含了预测的类别
predicted_class = model.predict(new_data_point)

print("预测类别:", predicted_class)
```

### 简单实战

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 导入鸢尾花数据库
from sklearn.datasets import load_iris

# 加载数据集，数据集包含数据的特征、标签、类别等许多信息
iris = load_iris()
# 获取数据特征（即花的高度、宽度等）
iris_X = iris.data
# 获取数据标签（即花的品种，用0、1、2代替）
iris_y = iris.target
print(len(iris_X)) # 查看数据集的大小

# # 获取前2条数据,从0开始到2结束,不包括2。写法1
# print(iris_X[0:2])

# # 获取前2条数据,从0开始到2结束,不包括2。写法2，省略0
# print(iris_X[:2])

# # 获取前2条数据,从0开始到2结束,不包括2。写法3，省略0
# print(iris_X[:2,:])

# # 获取前2条数据,从0开始到2结束,不包括2。写法4，省略0,只取第一列
# print(iris_X[:2,0])

# # 查看花的类别
# print(iris_y)
# # 查看花的数据
# print(iris_X)
# # 合在一起查看
# print(list(zip(iris_X,iris_y)))

# 把数据打乱，并分成测试数据和训练数据，test_size是测试数据的比例，0.3表示为30%
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

'''
train_test_split详解

此方法会将数据和标签均分成两部分并打乱，一部分用于训练，一部分用于测试。

所以返回的数据有4个，我们用1、2、3、4给他们做上记号.

数据X [------70%---(1)-- | -30%(2)-]
标签y [------70%---(3)-- | -30%(4)-]

与上图对应，依次是:

训练的数据X(1),  测试的数据X(2),
     ↑↓              ↑↓
训练的标签y(3),  测试的标签y(4)

用(1)、(3)喂出一个模型

让模型预测(2)，获得预测结果

将预测结果与(4)进行比较来测试模型的准确率
'''

# 查看训练数据，已经被随机打乱了
# print(y_train)
# 实例化KNN分类器
knn = KNeighborsClassifier()
# .fit()方法用于训练模型，即让模型从数据中学习
knn.fit(X_train, y_train)
# .predicts()方法返回一个数组，数组中包含了预测的类别
print(knn.predict(X_test))
# 查看真实数据
print(y_test)
```

### 效果评估

```python
right = 0
error = 0
for i in zip(knn.predict(X_test),y_test):
    #print(i)
    if i[0] == i[1]:
        right +=1
    else:
        error +=1
print(right,error)
print('正确率：{}%'.format(right/(right+error)*100))
```

### 效果评估的改进

```python
print('正确率：{}%'.format(knn.score(X_test,y_test)*100))

# 正确率：100.0%
```
