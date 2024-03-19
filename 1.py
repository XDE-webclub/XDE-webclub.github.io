import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ImageThresholding:
    def __init__(self, image_path):
        self.image = np.array(Image.open(image_path).convert('L'))

    def threshold_fixed(self, threshold=127):
        return (self.image > threshold).astype(int)

    def threshold_mean(self):
        threshold = self.image.mean()
        return (self.image > threshold).astype(int)

    def threshold_adaptive(self, window_size=3):
        binary_image = np.zeros_like(self.image)
        h, w = self.image.shape
        for i in range(h):
            for j in range(w):
                i_start = max(0, i - window_size // 2)
                i_end = min(h, i + window_size // 2 + 1)
                j_start = max(0, j - window_size // 2)
                j_end = min(w, j + window_size // 2 + 1)
                window = self.image[i_start:i_end, j_start:j_end]
                threshold = window.mean()
                binary_image[i, j] = self.image[i, j] > threshold
        return binary_image.astype(int)

    def threshold_otsu(self):
        histogram = np.bincount(self.image.ravel(), minlength=256)
        histogram = histogram / np.sum(histogram)
        mu_T = np.sum([i * histogram[i] for i in range(256)])
        best_sigma_B = 0
        best_threshold = 0
        for t in range(256):
            p_0 = np.sum(histogram[:t])
            p_1 = np.sum(histogram[t:])
            mu_0 = np.sum([i * histogram[i] for i in range(t)]) / p_0 if p_0 > 0 else 0
            mu_1 = np.sum([i * histogram[i] for i in range(t, 256)]) / p_1 if p_1 > 0 else 0
            sigma_B = p_0 * (mu_0 - mu_T) ** 2 + p_1 * (mu_1 - mu_T) ** 2
            if sigma_B > best_sigma_B:
                best_sigma_B = sigma_B
                best_threshold = t
        return (self.image > best_threshold).astype(int)

    def show_image(self, image, title):
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    thresholding = ImageThresholding('img.bmp')
    thresholding.show_image(thresholding.threshold_fixed(), 'Fixed Thresholding')
    thresholding.show_image(thresholding.threshold_mean(), 'Mean Thresholding')
    thresholding.show_image(thresholding.threshold_adaptive(), 'Adaptive Thresholding')
    thresholding.show_image(thresholding.threshold_otsu(), 'Otsu Thresholding')
