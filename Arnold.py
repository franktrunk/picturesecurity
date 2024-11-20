import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import time

# 设置字体 保证打印图片标题时不会为空方框
font = font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')  # 选择一个可用的字体路径


# 向图像添加噪声
def add_noise(image, noise_factor=50):
    return np.clip(image + noise_factor * np.random.randn(*image.shape), 0, 255).astype(np.uint8)


# Arnold变换加密与解密
def arnold_transform(image, shuffle_times, a, b, decode=False):
    h, w = image.shape[:2]
    N = h
    result_image = np.zeros_like(image)

    for _ in range(shuffle_times):
        for x in range(h):
            for y in range(w):
                new_x = (x * (a * b + 1) - b * y) % N if decode else (x + b * y) % N
                new_y = (-a * x + y) % N if decode else (a * x + (a * b + 1) * y) % N
                result_image[new_x, new_y, :] = image[x, y, :]
    return result_image


# 读取并预处理图像
img = cv2.imread('data/test.JPG')
img = cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2RGB)  # 调整为256x256

# 图像加密
start_time = time.time()
encrypted_img = arnold_transform(img, 1, 1, 1)
end_time = time.time()
print(f"加密所需时间: {end_time - start_time}秒")

# 图像解密
start_time = time.time()
decrypted_img = arnold_transform(encrypted_img, 1, 1, 1, decode=True)
end_time = time.time()
print(f"解密所需时间: {end_time - start_time}秒")

# 添加噪声并解密
noise_img = add_noise(encrypted_img)
decrypted_noise_img = arnold_transform(noise_img, 1, 1, 1, decode=True)

# 显示结果
# 定义图像和对应的标题
images = [img, encrypted_img, decrypted_img, noise_img, decrypted_noise_img]
titles = ['原始图像', '加密后图像', '解密后图像', '噪声加密图像', '噪声解密图像']

# 创建图形窗口
plt.figure(figsize=(15, 5))

# 使用循环绘制每个子图
for i in range(5):
    plt.subplot(1, 5, i + 1)  # 1行5列，第i+1个位置
    plt.imshow(images[i])      # 显示图像
    plt.title(titles[i], fontproperties=font)  # 设置标题

# 显示所有图像
plt.show()

# 测试不同噪声强度下的解密效果
noise_factors = [10, 50, 100, 500, 1000]
plt.figure(figsize=(15, 5))
for i, noise_factor in enumerate(noise_factors):
    noise_img = add_noise(encrypted_img, noise_factor)
    decrypted_noise_img = arnold_transform(noise_img, 1, 1, 1, decode=True)
    plt.subplot(1, 5, i + 1)
    plt.imshow(decrypted_noise_img)
    plt.title(f'噪声强度={noise_factor}', fontproperties=font)
plt.show()
