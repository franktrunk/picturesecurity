import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import time

# 设置字体，保证打印图片标题时不会为空方框
font = font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')  # 选择一个可用的字体路径，Windows 用户可以使用 'MSYH.TTC' 或其他中文字体

# 噪声测试，添加高斯噪声
def add_noise(image_test, noise_factor=100):
    """向图像添加噪声"""
    noisy_image = image_test + noise_factor * np.random.randn(*image_test.shape)
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

# 加密函数
def encrypt(image):
    """对图像进行加密，使用行列置换"""
    M, N = image.shape
    # 获取Lorenz系统的轨迹
    s = np.zeros(M + N)
    # Lorenz系统初始状态
    x0, y0, z0, w0 = 1.1, 2.2, 3.3, 4.4
    a, b, c = 10, 8 / 3, 28
    h = 0.002  # 步长
    t = 800  # 初始时间，用于忽略系统的初始过渡过程
    r=-20
    for i in range(1, M + N + t):
        K11 = a * (y0 - x0) + w0
        K12 = a * (y0 - (x0 + K11 * h / 2)) + w0
        K13 = a * (y0 - (x0 + K12 * h / 2)) + w0
        K14 = a * (y0 - (x0 + K13 * h)) + w0
        x1 = x0 + (K11 + K12 + K13 + K14) * h / 6

        K21 = c * x1 - y0 - x1 * z0
        K22 = c * x1 - (y0 + K21 * h / 2) - x1 * z0
        K23 = c * x1 - (y0 + K22 * h / 2) - x1 * z0
        K24 = c * x1 - (y0 + K23 * h) - x1 * z0
        y1 = y0 + (K21 + K22 + K23 + K24) * h / 6

        K31 = x1 * y1 - b * z0
        K32 = x1 * y1 - b * (z0 + K31 * h / 2)
        K33 = x1 * y1 - b * (z0 + K32 * h / 2)
        K34 = x1 * y1 - b * (z0 + K33 * h)
        z1 = z0 + (K31 + K32 + K33 + K34) * h / 6

        K41 = -y1 * z1 + r * w0
        K42 = -y1 * z1 + r * (w0 + K41 * h / 2)
        K43 = -y1 * z1 + r * (w0 + K42 * h / 2)
        K44 = -y1 * z1 + r * (w0 + K43 * h)
        w1 = w0 + (K41 + K42 + K43 + K44) * h / 6

        # 更新状态变量
        x0, y0, z0, w0 = x1, y1, z1, w1

        # 从第t+1步开始，记录Lorenz系统轨迹
        if i > t:
            s[i - t - 1] = x1  # 存储x1值（Lorenz轨迹的一部分）

    X = (np.floor((s[:M] + 100) * 10 ** 10) % M).astype(int) + 1
    Y = (np.floor((s[M:M + N] + 100) * 10 ** 10) % N).astype(int) + 1

    # 行置换
    A = image.copy()
    for i in range(M):
        t = A[i, :].copy()
        A[i, :] = A[X[i] - 1, :]
        A[X[i] - 1, :] = t

    # 列置换
    B = A.copy()
    for j in range(N):
        t = B[:, j].copy()
        B[:, j] = B[:, Y[j] - 1]
        B[:, Y[j] - 1] = t

    return B, X, Y

# 解密函数
def decode(N, M, X, Y, encrypted_image):
    for j in range(N - 1, -1, -1):  # 从后往前恢复列
        t = encrypted_image[:, j].copy()
        encrypted_image[:, j] = encrypted_image[:, Y[j] - 1]
        encrypted_image[:, Y[j] - 1] = t
    decrypted_image = encrypted_image.copy()
    for i in range(M - 1, -1, -1):  # 从后往前恢复行
        t = decrypted_image[i, :].copy()
        decrypted_image[i, :] = decrypted_image[X[i] - 1, :]
        decrypted_image[X[i] - 1, :] = t
    return decrypted_image

# 加载彩色图像
img = cv2.imread('data/test.jpg')  # 确保图像路径正确
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 对RGB图像的每个通道进行加密
def encrypt_rgb(image):
    # 分离RGB通道
    R, G, B = cv2.split(image)

    # 对每个通道执行加密
    R_encrypted, X_r, Y_r = encrypt(R)
    G_encrypted, X_g, Y_g = encrypt(G)
    B_encrypted, X_b, Y_b = encrypt(B)

    # 合并加密后的通道
    encrypted_image = cv2.merge([R_encrypted, G_encrypted, B_encrypted])
    return encrypted_image, X_r, Y_r, X_g, Y_g, X_b, Y_b

# 对RGB图像的每个通道进行解密
def decrypt_rgb(encrypted_image, X_r, Y_r, X_g, Y_g, X_b, Y_b):
    # 分离RGB通道
    R, G, B = cv2.split(encrypted_image)

    # 对每个通道执行解密
    R_decrypted = decode(R.shape[1], R.shape[0], X_r, Y_r, R)
    G_decrypted = decode(G.shape[1], G.shape[0], X_g, Y_g, G)
    B_decrypted = decode(B.shape[1], B.shape[0], X_b, Y_b, B)

    # 合并解密后的通道
    decrypted_image = cv2.merge([R_decrypted, G_decrypted, B_decrypted])
    return decrypted_image

# 对加噪声图像进行处理
def add_noise_rgb(image, noise_factor=100):
    # 分离RGB通道
    R, G, B = cv2.split(image)

    # 对每个通道添加噪声
    R_noisy = add_noise(R, noise_factor)
    G_noisy = add_noise(G, noise_factor)
    B_noisy = add_noise(B, noise_factor)

    # 合并噪声图像
    noisy_image = cv2.merge([R_noisy, G_noisy, B_noisy])
    return noisy_image

# 主函数
def main():
    # 加密图像
    encrypted_image, X_r, Y_r, X_g, Y_g, X_b, Y_b = encrypt_rgb(img)

    # 添加噪声
    noisy_image = add_noise_rgb(encrypted_image, noise_factor=100)

    # 解密
    decrypted_image = decrypt_rgb(noisy_image, X_r, Y_r, X_g, Y_g, X_b, Y_b)

    # 显示结果
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title("Original Image", fontproperties=font)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(encrypted_image)
    plt.title("Encrypted Image", fontproperties=font)
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(noisy_image)
    plt.title("Noisy Image", fontproperties=font)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(decrypted_image)
    plt.title("Decrypted Image", fontproperties=font)
    plt.axis('off')

    plt.show()

# 运行主函数
if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Processing time: {time.time() - start_time:.2f} seconds")
