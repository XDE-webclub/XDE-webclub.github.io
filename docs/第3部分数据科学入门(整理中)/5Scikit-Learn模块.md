
Scikit-learn（sklearn）、PyTorch和TensorFlow是三个在机器学习和深度学习领域广泛使用的库，各自有其优势和劣势。

如果你处理传统的机器学习问题，Scikit-learn是一个不错的选择。如果你主要进行深度学习研究或需要处理复杂的深度学习任务，PyTorch和TensorFlow是更好的选择，取决于你的偏好和需求。另外，TensorFlow在工业界有广泛的应用，因此在工业部署方面也有一定优势。

#### 数据的来源？

##### 小型标准数据集

scikit-learn 内置有一些小型标准数据集，不需要从某个外部网站下载任何文件。
这些数据集有助于快速说明在 scikit 中实现的各种算法的行为。
然而，它们数据规模往往太小，无法代表真实世界的机器学习任务。
但是作为学习使用刚刚好。

| 数据集名称 | 加载方法     | 模型类型               | 数据大小(样本数*特征数)       |         |
|-------|----------|--------------------|---------------------|---------|
| 0     | 波士顿房价数据集 | load_boston        | regression          | 506*13  |
| 1     | 鸢尾花数据集   | load_iris          | classification      | 150*4   |
| 2     | 手写数字数据集  | load_digits        | classification      | 1797*64 |
| 3     | 糖尿病数据集   | load_diabetes      | regression          | 442*10  |
| 4     | 葡萄酒数据集   | load_wine          | classification      | 178*13  |
| 5     | 乳腺癌数据集   | load_breast_cancer | classification      | 569*30  |
| 6     | 体能训练数据集  | load_linnerud      | 多重回归 | 20*3    |

```python
import sklearn.datasets

data = sklearn.datasets.load_wine()
data.data
```

##### 真实世界中的数据集

scikit-learn 提供加载较大数据集的工具，并在必要时下载这些数据集。

数据特征说明可参考
<https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets>

```python
from sklearn.datasets import fetch_california_housing
# 加载数据
housing = fetch_california_housing()
housing.data
```

##### 样本生成器

scikit-learn 包括各种随机样本的生成器，可以用来建立可控制的大小和复杂性人工数据集。

```python
from sklearn.datasets import make_blobs
# 创建KNN模型数据集
'''
X为样本特征，Y为样本簇类别，共1000个样本，

每个样本4个特征，共4个簇，簇中心在[-1,-1], [0,0], [1,1], [2,2],
簇方差分别为[0.4, 0.2, 0.2]
'''
x, y = make_blobs(n_samples=1000, 
                  n_features=2,
                  centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                  cluster_std=[0.4, 0.2, 0.2, 0.2],
                  )
from sklearn.datasets import make_regression
# 创建回归模型数据集
'''
X为样本特征，Y为样本簇类别，共1000个样本，

每个样本1个特征，
离散度为2
'''
x2,y2 = make_regression(n_samples=1000, n_features=1, n_targets=1, noise=2)
```

##### 自有数据集

我们手上可能刚好有一些数据集，可以通过pandas或者numpy读取

```python
# 通过pandas或者numpy读取
import pandas as pd
import numpy as np
data = pd.read_csv('./data/iris.csv')
# 通过numpy读取
data = np.loadtxt('./data/iris.csv', delimiter=",", skiprows=1)

```

## 分类

先学习下监督学习中的分类任务：

加载数据集：

```python
from sklearn import datasets
digits = datasets.load_digits()

digits.keys()
```

```python
digits.data
```

```python
print(digits.target)
digits.images[0]
```

显示图片：

```python
from sklearn import datasets
from matplotlib import pyplot as plt
digits = datasets.load_digits()
fig, ax = plt.subplots(
    nrows=4,
    ncols=4,
    sharex=True,
    sharey=True)
 
ax = ax.flatten()
for i in range(16):
    ax[i].imshow(digits.data[i].reshape((8, 8)), cmap='Greys', interpolation='nearest')
plt.show()
```

## 模型训练和预测

### SVM

```python
from sklearn import svm
clf = svm.SVC(gamma = 0.001, C=100)

clf.fit(digits.data[:-1], digits.target[:-1])
```

```python
clf.predict(digits.data[:2])
```

```python
from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)
```

```python
clf.predict(X[:150])
```

```python
y[:150]
```

使用pickle序列化模型：

```python
import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
predict_y = clf2.predict(X[:150])
predict_y
```

使用joblib保存：

```python
import joblib
joblib.dump(clf, 'clf.pkl')
```

```python
clf = joblib.load('clf.pkl')
clf
```

```python
import os
os.remove('clf.pkl')
```

### 随机映射 random projection

sklearn.random_projection 模块实现了一种简单和计算高效的方法，通过交易控制量的精度（作为附加方差），以缩短数据的维数，从而缩短处理时间和缩小模型大小。

该模块实现两种类型的非结构化随机矩阵：高斯随机矩阵 GaussianRandomProjection 和稀疏随机矩阵 SparseRandomProjection。

- 高斯随机矩阵：通过将原始输入空间投影在随机生成的矩阵上来降低维度。
- 稀疏随机矩阵：相比于高斯随机映射，稀疏随机映射会更能保证降维的质量，并带来内存的使用效率和运算效率。

```python
import numpy as np
from sklearn import random_projection
rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
X = np.array(X, dtype='float32')
X
```

```python
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)

print(X_new.dtype)
```

### LR

使用逻辑回归（Logistic Regression, LR）模型：

LR模型的详细介绍参考sklearn官方文档：[LR](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
)

```python
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
clf = LogisticRegression(solver='lbfgs', fit_intercept=False)
clf.fit(iris.data, iris.target)
```

```python
list(clf.predict(iris.data[:3]))
```

```python
y[:3]
```

如果模型的label为文本：

```python
clf.fit(iris.data, iris.target_names[iris.target])
```

```python
list(clf.predict(iris.data[:3]))
```

### set_params

设置模型参数

```python
import numpy as np
from sklearn.svm import SVC

rng = np.random.RandomState(0)
X = rng.rand(100, 10)
y = rng.binomial(1, 0.5, 100)
X_test = rng.rand(5, 10)

clf = SVC()
clf.set_params(kernel='linear').fit(X, y)
```

```python
clf.predict(X_test)
```

```python
clf.set_params(kernel='rbf').fit(X, y)
clf.predict(X_test)
```

## 多分类 vs. 多标签模型

多分类模型：

```python
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]
classify = OneVsRestClassifier(estimator=SVC(random_state=0))
model = classify.fit(X, y)
model
```

```python
model.predict(X)
```

```python
y = LabelBinarizer().fit_transform(y)
y
```

```python
classify.fit(X, y).predict(X) 
```

```python
classify.fit(X, y).score(X, y)
# 0.6 
```

多标签模型：

```python
from sklearn.preprocessing import MultiLabelBinarizer
y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
y
```

```python
y = MultiLabelBinarizer().fit_transform(y)
y
```

```python
classify = OneVsRestClassifier(estimator=SVC(random_state=0))
model = classify.fit(X, y)
model
```

```python
model.predict(X)
```

```python
model.score(X, y)
```

## 预测错误结果可视化

 `cross_val_predict`

```python
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

lr = LinearRegression()
boston = datasets.load_diabetes()
y = boston.target

boston.data[:3]
```

```python
predicted = cross_val_predict(lr, boston.data, y, cv=10)
predicted[:3]
```

```python
fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
```

| [03_data_science/10_Scikit-Learn聚类.ipynb](https://github.com/shibing624/python-tutorial/blob/master/03_data_science/10_Scikit-Learn聚类.ipynb)  | Scikit-Learn聚类  |[Open In Colab](https://colab.research.google.com/github/shibing624/python-tutorial/blob/master/03_data_science/10_Scikit-Learn聚类.ipynb) |

# 聚类：Cluster

测试数据演示聚类：

```python
from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=2, random_state=10).fit(X)
kmeans.labels_    # array([0, 0, 0, 1, 1, 1], dtype=int32)
```

```python
kmeans.predict([[0, 0], [5, 4]])   # array([0, 1], dtype=int32)
```

```python
kmeans.cluster_centers_   # array([[1., 2.], [4., 2.]])
```

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=1000, 
                  n_features=2,
                  centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                  cluster_std=[0.4, 0.2, 0.2, 0.2],
                  random_state=5)
X[999]
```

```python
y.shape
```

X为样本特征，Y为样本簇类别，共1000个样本，

每个样本4个特征，共4个簇，簇中心在[-1,-1], [0,0], [1,1], [2,2],
簇方差分别为[0.4, 0.2, 0.2]

```python
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()
```

## KMeans

使用KMeans聚类：

```python
from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
```

```python
from sklearn import metrics
metrics.calinski_harabasz_score(X, y_pred)
```

设置4个簇，n_clusters=4:

```python
y_pred = KMeans(n_clusters=4, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
```

```python
metrics.calinski_harabasz_score(X, y_pred)
```

## MiniBatchKMeans

```python
from sklearn.cluster import MiniBatchKMeans
y_pred = MiniBatchKMeans(n_clusters=2, batch_size=200, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
```

```python
metrics.calinski_harabasz_score(X, y_pred)
```

```python
y_pred = MiniBatchKMeans(n_clusters=3, batch_size=200, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
```

```python
metrics.calinski_harabasz_score(X, y_pred)
```

```python
y_pred = MiniBatchKMeans(n_clusters=4, batch_size=200, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
```

```python
metrics.calinski_harabasz_score(X, y_pred)
```

n_clusters的选择？

```python
plt.subplots_adjust(left=.02, right=.98, bottom=.096, top=.96, wspace=.1, hspace=.1)

plt.subplot(2, 2, 1)
y_pred = MiniBatchKMeans(n_clusters=2, batch_size=200, random_state=9).fit_predict(X)
score2 = metrics.calinski_harabasz_score(X, y_pred)  
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.text(.99, .01, ('k=%d, score: %.2f' % (2, score2)),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')


plt.subplot(2, 2, 2)
y_pred = MiniBatchKMeans(n_clusters=3, batch_size = 200, random_state=9).fit_predict(X)
score3 = metrics.calinski_harabasz_score(X, y_pred)  
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.text(.99, .01, ('k=%d, score: %.2f' % (3, score3)),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')

plt.subplot(2, 2, 3)
y_pred = MiniBatchKMeans(n_clusters=4, batch_size = 200, random_state=9).fit_predict(X)
score4 = metrics.calinski_harabasz_score(X, y_pred)  
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.text(.99, .01, ('k=%d, score: %.2f' % (4, score4)),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')

plt.subplot(2, 2, 4)
y_pred = MiniBatchKMeans(n_clusters=5, batch_size = 200, random_state=9).fit_predict(X)
score5 = metrics.calinski_harabasz_score(X, y_pred)  
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.text(.99, .01, ('k=%d, score: %.2f' % (5, score5)),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
plt.show()
```

简化的写法：

```python
plt.subplots_adjust(left=.02, right=.98, bottom=.096, top=.96, wspace=.1, hspace=.1)
for index, k in enumerate((2, 3, 4, 5)):
    plt.subplot(2, 2, index + 1)
    y_pred = MiniBatchKMeans(n_clusters=k, batch_size=200, random_state=9).fit_predict(X)
    score = metrics.calinski_harabasz_score(X, y_pred)  
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.text(.99, .01, ('k=%d, score: %.2f' % (k, score)),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
plt.show()
```

## 基于经验的k-means初始化方法

评估k均值初始化的方法，以使算法快速收敛，即到最近聚类中心的平方距离之和测量的。

比较K-Means和MiniBatchKMeans算法的聚类效果：

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.utils import shuffle
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans

random_state = np.random.RandomState(0)

# 每个策略的运行次数（使用随机生成的数据集），以便能够计算标准偏差的估计值
n_runs = 5

# k-means模型可以进行多次随机初始化，以便能够快速收敛
n_init_range = np.array([1, 5, 10, 15, 20])

# 数据生成参数
n_samples_per_center = 100
grid_size = 3
scale = 0.1
n_clusters = grid_size ** 2


def make_data(random_state, n_samples_per_center, grid_size, scale):
    random_state = check_random_state(random_state)
    centers = np.array([[i, j]
                        for i in range(grid_size)
                        for j in range(grid_size)])
    n_clusters_true, n_features = centers.shape

    noise = random_state.normal(
        scale=scale, size=(n_samples_per_center, centers.shape[1]))

    X = np.concatenate([c + noise for c in centers])
    y = np.concatenate([[i] * n_samples_per_center
                        for i in range(n_clusters_true)])
    return shuffle(X, y, random_state=random_state)
```

```python
# Part 1: 初始化方法的定量评价

plt.figure()
plots = []
legends = []

cases = [
    (KMeans, 'k-means++', {}),
    (KMeans, 'random', {}),
    (MiniBatchKMeans, 'k-means++', {'max_no_improvement': 3}),
    (MiniBatchKMeans, 'random', {'max_no_improvement': 3, 'init_size': 500}),
]

for factory, init, params in cases:
    print("Evaluation of %s with %s init" % (factory.__name__, init))
    inertia = np.empty((len(n_init_range), n_runs))

    for run_id in range(n_runs):
        X, y = make_data(run_id, n_samples_per_center, grid_size, scale)
        for i, n_init in enumerate(n_init_range):
            km = factory(n_clusters=n_clusters, init=init, random_state=run_id,
                         n_init=n_init, **params).fit(X)
            inertia[i, run_id] = km.inertia_
    p = plt.errorbar(n_init_range, inertia.mean(axis=1), inertia.std(axis=1))
    plots.append(p[0])
    legends.append("%s with %s init" % (factory.__name__, init))

plt.xlabel('n_init')
plt.ylabel('inertia')
plt.legend(plots, legends)
plt.title("Mean inertia for various k-means init across %d runs" % n_runs)
```

```python
# Part 2: 数据聚类结果的可视化显示

X, y = make_data(random_state, n_samples_per_center, grid_size, scale)
km = MiniBatchKMeans(n_clusters=n_clusters, init='random', n_init=1,
                     random_state=random_state).fit(X)

plt.figure()
for k in range(n_clusters):
    my_members = km.labels_ == k
    color = cm.nipy_spectral(float(k) / n_clusters, 1)
    plt.plot(X[my_members, 0], X[my_members, 1], 'o', marker='.', c=color)
    cluster_center = km.cluster_centers_[k]
    plt.plot(cluster_center[0], cluster_center[1], 'o',
             markerfacecolor=color, markeredgecolor='k', markersize=6)
    plt.title("Example cluster allocation with a single random init\n"
              "with MiniBatchKMeans")

plt.show()

```

第一个图显示了最佳初始化参数（``KMeans`` or ``MiniBatchKMeans``）和init方法（``init="random"`` or ``init="kmeans++"``）的选择。

第二个图显示了使用``init="random"`` and ``n_init=1``的``MiniBatchKMeans``一次运行结果。这种运行导致一个坏的收敛（局部最优）。

结论：初始化一致的情况下，K-Means和MiniBatchKMeans算法差别很小。

## 手写体数字的k均值聚类演示

在这个例子中，我们比较k-means的各种初始化策略的运行的效果。

评估聚类效果：

```python
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based",
              data=data)
print(82 * '_')

# #############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

```

## 基于K均值的颜色量化

对颐和园图像执行像素矢量量化（VQ），将显示图像所需的颜色数量从96615减少到64，同时保持整体外观质量。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

n_colors = 64

# Load the Summer Palace photo
china = load_sample_image("china.jpg")

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
china = np.array(china, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(china.shape)
assert d == 3
image_array = np.reshape(china, (w * h, d))

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))


codebook_random = shuffle(image_array, random_state=0)[:n_colors]
print("Predicting color indices on the full image (random)")
t0 = time()
labels_random = pairwise_distances_argmin(codebook_random,
                                          image_array,
                                          axis=0)
print("done in %0.3fs." % (time() - t0))


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

# Display all results, alongside original image
plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(china)

plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('Quantized image (64 colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

plt.figure(3)
plt.clf()
plt.axis('off')
plt.title('Quantized image (64 colors, Random)')
plt.imshow(recreate_image(codebook_random, labels_random, w, h))
```

可以看到聚类后，用更少的颜色数量达到了跟原图差不多的显示效果，强于随机选择。

```python

```
