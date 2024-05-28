
### 原理

卷积神经网络的核心是卷积核，卷积核在图像处理领域可以用来提取图像的纵向和横向特征。

卷积核的大小一般为奇数，如3x3，5x5，7x7等，卷积核通常与图像处理（over padding）后的图像进行卷积操作，卷积核在图像上滑动，每次滑动一个像素，对应位置的像素值与卷积核对应位置的值相乘，然后求和，最后将求和的结果作为卷积核中心像素的值，这样就得到了一个新的图像。

新的图像可以用更少的数据反应出图像的特征。这个过程就是特征提取。

#### 卷积核工作原理

![img](/imgs/卷积神经网络-卷积核工作原理.png)

![img](/imgs/卷积神经网络-卷积核工作原理动画.gif)


### 常见卷积核及用途


1. **水平边缘检测**：
   $$
   \begin{bmatrix}
   -1 & -1 & -1 \\
   0 & 0 & 0 \\
   1 & 1 & 1
   \end{bmatrix}
   $$
   用途：检测水平边缘。

2. **垂直边缘检测**：
   $$
   \begin{bmatrix}
   -1 & 0 & 1 \\
   -1 & 0 & 1 \\
   -1 & 0 & 1
   \end{bmatrix}
   $$
   用途：检测垂直边缘。

3. **Sobel算子（水平）**：
   $$
   \begin{bmatrix}
   -1 & 0 & 1 \\
   -2 & 0 & 2 \\
   -1 & 0 & 1
   \end{bmatrix}
   $$
   用途：检测水平边缘和梯度。

4. **Sobel算子（垂直）**：
   $$
   \begin{bmatrix}
   1 & 2 & 1 \\
   0 & 0 & 0 \\
   -1 & -2 & -1
   \end{bmatrix}
   $$
   用途：检测垂直边缘和梯度。

5. **拉普拉斯算子**：
   $$
   \begin{bmatrix}
   0 & 1 & 0 \\
   1 & -4 & 1 \\
   0 & 1 & 0
   \end{bmatrix}
   $$
   用途：检测图像的二阶导数，强调边缘。

6. **锐化**：
   $$
   \begin{bmatrix}
   0 & -1 & 0 \\
   -1 & 5 & -1 \\
   0 & -1 & 0
   \end{bmatrix}
   $$
   用途：提高图像的清晰度。

7. **高斯模糊（3x3）**：
   $$
   \frac{1}{16}
   \begin{bmatrix}
   1 & 2 & 1 \\
   2 & 4 & 2 \\
   1 & 2 & 1
   \end{bmatrix}
   $$
   用途：平滑图像，减少噪声。

8. **高斯模糊（5x5）**：
   $$
   \frac{1}{256}
   \begin{bmatrix}
   1 & 4 & 6 & 4 & 1 \\
   4 & 16 & 24 & 16 & 4 \\
   6 & 24 & 36 & 24 & 6 \\
   4 & 16 & 24 & 16 & 4 \\
   1 & 4 & 6 & 4 & 1
   \end{bmatrix}
   $$
   用途：更强的平滑效果。

9. **边缘增强**：
   $$
   \begin{bmatrix}
   -1 & -1 & -1 \\
   -1 & 9 & -1 \\
   -1 & -1 & -1
   \end{bmatrix}
   $$
   用途：增强边缘，使图像轮廓更加明显。

10. **均值滤波**：
    $$
    \frac{1}{9}
    \begin{bmatrix}
    1 & 1 & 1 \\
    1 & 1 & 1 \\
    1 & 1 & 1
    \end{bmatrix}
    $$
    用途：均匀地平滑图像。

### 效果查看

```python

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
# 替换为你系统中支持中文的字体路径(windows)
font_path = r'C:\Windows\Fonts\simhei.ttf'  
# mac（如果有的话）
# font_path = '/System/Library/Fonts/STHeiti Light.ttc' 
font_prop = FontProperties(fname=font_path)
# 读取灰度图像
image = cv2.imread('people.bmp', cv2.IMREAD_GRAYSCALE)

# 定义卷积核
kernels = {
    '水平边缘': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
    '垂直边缘': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    'Sobel水平': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    'Sobel垂直': np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
    '拉普拉斯': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    '锐化': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    '高斯模糊3x3': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
    '高斯模糊5x5': np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256,
    '边缘增强': np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
    '均值滤波': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
}

# 应用卷积核
results = {}
for name, kernel in kernels.items():
    filtered_image = cv2.filter2D(image, -1, kernel)
    results[name] = filtered_image

# 显示结果
plt.figure(figsize=(15, 8))
for i, (name, result) in enumerate(results.items()):
    plt.subplot(3, 4, i + 1)
    plt.imshow(result, cmap='gray')
    plt.title(name, fontproperties=font_prop)
    plt.axis('off')

plt.tight_layout()
plt.show()

```

你可以使用人像、车牌等不同物体，来查看不同卷积核的卷积效果。

### 卷积神经网络对手写数字识别

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

#### 导入库和数据预处理

<Tabs>
  <TabItem value="tf" label="TensorFlow" default>

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

# 加载数据
digits = load_digits()
X = digits.images
y = digits.target

# 数据预处理
X = X[..., np.newaxis]  # 增加通道维度 (n_samples, 8, 8, 1)
X = X.astype(np.float32) / 16.0  # 归一化

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

  </TabItem>
  <TabItem value="torch" label="Pytorch">

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

# 加载数据
digits = load_digits()
X = digits.images
y = digits.target

# 数据预处理
X = X[:, np.newaxis, :, :]  # 增加通道维度 (n_samples, 1, 8, 8)
X = X.astype(np.float32) / 16.0  # 归一化

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).long())
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).long())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

  </TabItem>
</Tabs>

#### 定义模型

<Tabs>
  <TabItem value="tf" label="TensorFlow" default>

```python
model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(8, 8, 1), padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

  </TabItem>
  <TabItem value="torch" label="Pytorch">

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()
```

  </TabItem>
</Tabs>

#### 训练和评估模型

<Tabs>
  <TabItem value="tf" label="TensorFlow" default>

```python
# 训练
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 评估
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')
```

  </TabItem>
  <TabItem value="torch" label="Pytorch">

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

# 评估
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
```

  </TabItem>
</Tabs>