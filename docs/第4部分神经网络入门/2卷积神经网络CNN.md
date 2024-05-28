
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