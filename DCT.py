import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import time
from scipy.fftpack import dct, idct

# 设置字体 保证打印图片标题时不会为空方框
font = font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')

# 噪声测试，添加噪声
def add_noise_to_dct(dct_image, noise_factor=70, high_freq_fraction=0.1):
    """向DCT加密图像添加噪声，主要添加到高频部分"""
    rows, cols = dct_image.shape
    high_freq_rows = int(rows * high_freq_fraction)
    high_freq_cols = int(cols * high_freq_fraction)

    noise = np.zeros_like(dct_image)
    noise[rows - high_freq_rows:, cols - high_freq_cols:] = noise_factor * np.random.randn(high_freq_rows, high_freq_cols)
    return dct_image + noise

# 读取图像并转为灰度图像
img = cv2.imread('data/test.JPG')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.imread('data/test.JPG', cv2.IMREAD_GRAYSCALE)

# DCT加密解密过程
image = img_gray
start_time = time.time()
J = dct(dct(image.T, norm='ortho').T, norm='ortho')
s = image.shape
r = np.random.rand(s[0], s[1]) * 20
image_en = J * r
end_time = time.time()
print(f"加密所需时间: {end_time - start_time} seconds")

start_time = time.time()
K = image_en / r
image_de = idct(idct(K.T, norm='ortho').T, norm='ortho') / 255.0
end_time = time.time()
print(f"解密所需时间: {end_time - start_time} seconds")

# 添加噪声并解密
noise_image = add_noise_to_dct(image_en)
K_noise = noise_image / r
noise_truth = idct(idct(K_noise.T, norm='ortho').T, norm='ortho') / 255.0

# 显示原图、加密图和解密图
images = [img_rgb, img_gray, image_en, image_de, noise_image, noise_truth]
titles = ['原始图像', '灰度图像', '加密图像', '解密图像', '噪声加密图像', '噪声解密图像']

plt.figure(figsize=(15, 5))

for i, (image, title) in enumerate(zip(images, titles)):
    plt.subplot(1, 6, i + 1)
    plt.imshow(image, cmap='gray' if i > 0 else None)  # 灰度图像从第二个开始
    plt.title(title, fontproperties=font)

plt.show()

# 解密函数
def decode(r, enimage):
    image_truth = enimage / r
    return idct(idct(image_truth.T, norm='ortho').T, norm='ortho') / 255.0

# 添加不同程度的噪声
noise_factors = [10, 50, 100, 500, 1000]
noise_images = [add_noise_to_dct(image_en, factor) for factor in noise_factors]
noise_images_truth = [decode(r, noise_image) for noise_image in noise_images]

# 显示不同噪声强度下的解密图像
plt.figure(figsize=(15, 5))
for i, (noise_image_truth, factor) in enumerate(zip(noise_images_truth, noise_factors)):
    plt.subplot(1, 5, i + 1)
    plt.imshow(noise_image_truth, cmap='gray')
    plt.title(f'noise_factor={factor}', fontproperties=font)

plt.show()
