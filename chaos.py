import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import time
from analysis import analyze_image_correlations, plot_correlation_heatmap, plot_image_histogram, plot_entropy
from permutation import  show_images
# 设置字体，保证打印图片标题时不会为空方框
font = font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')  # 替换为合适的字体路径
plt.rcParams['font.family'] = font.get_name()


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
    x0, y0, z0, w0 = 1.1, 2.2, 3.3, 4.4
    a, b, c = 10, 8 / 3, 28
    h = 0.002
    t = 800
    r = -20
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

        x0, y0, z0, w0 = x1, y1, z1, w1

        if i > t:
            s[i - t - 1] = x1

    X = (np.floor((s[:M] + 100) * 10 ** 10) % M).astype(int) + 1
    Y = (np.floor((s[M:M + N] + 100) * 10 ** 10) % N).astype(int) + 1

    A = image.copy()
    for i in range(M):
        t = A[i, :].copy()
        A[i, :] = A[X[i] - 1, :]
        A[X[i] - 1, :] = t

    B = A.copy()
    for j in range(N):
        t = B[:, j].copy()
        B[:, j] = B[:, Y[j] - 1]
        B[:, Y[j] - 1] = t

    return B, X, Y


# 解密函数
def decode(N, M, X, Y, encrypted_image):
    for j in range(N - 1, -1, -1):
        t = encrypted_image[:, j].copy()
        encrypted_image[:, j] = encrypted_image[:, Y[j] - 1]
        encrypted_image[:, Y[j] - 1] = t
    decrypted_image = encrypted_image.copy()
    for i in range(M - 1, -1, -1):
        t = decrypted_image[i, :].copy()
        decrypted_image[i, :] = decrypted_image[X[i] - 1, :]
        decrypted_image[X[i] - 1, :] = t
    return decrypted_image


# 加载彩色图像
img = cv2.imread('data/test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encrypt_rgb(image):
    R, G, B = cv2.split(image)
    R_encrypted, X_r, Y_r = encrypt(R)
    G_encrypted, X_g, Y_g = encrypt(G)
    B_encrypted, X_b, Y_b = encrypt(B)
    encrypted_image = cv2.merge([R_encrypted, G_encrypted, B_encrypted])
    return encrypted_image, X_r, Y_r, X_g, Y_g, X_b, Y_b


def decrypt_rgb(encrypted_image, X_r, Y_r, X_g, Y_g, X_b, Y_b):
    R, G, B = cv2.split(encrypted_image)
    R_decrypted = decode(R.shape[1], R.shape[0], X_r, Y_r, R)
    G_decrypted = decode(G.shape[1], G.shape[0], X_g, Y_g, G)
    B_decrypted = decode(B.shape[1], B.shape[0], X_b, Y_b, B)
    decrypted_image = cv2.merge([R_decrypted, G_decrypted, B_decrypted])
    return decrypted_image


def process_image(img_path, noise_factors):
    """处理图像：加密、添加噪声、解密并保存图像"""

    # 读取图像
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 加密
    start_time = time.perf_counter()
    encrypted_image, X_r, Y_r, X_g, Y_g, X_b, Y_b = encrypt_rgb(img_rgb)
    end_time = time.perf_counter()
    print(f"加密所需时间: {end_time - start_time:.6f}秒")
    cv2.imwrite("data/permutation/encrypt_and_decrypt/encrypted_image.jpg", encrypted_image)


    # 解密
    start_time = time.perf_counter()
    decrypted_image = decrypt_rgb(encrypted_image, X_r, Y_r, X_g, Y_g, X_b, Y_b)
    end_time = time.perf_counter()
    print(f"解密所需时间: {end_time - start_time:.6f}秒")  # 保留 6 位小数
    cv2.imwrite("data/permutation/encrypt_and_decrypt/decrypted_image.jpg", decrypted_image)
    # 添加噪声并解密
    noisy_images = []
    noisy_truths = []
    for nf in noise_factors:
        noisy_image = add_noise(encrypted_image, nf)
        noisy_images.append(noisy_image)
        noisy_truth = decrypt_rgb(noisy_image, X_r, Y_r, X_g, Y_g, X_b, Y_b)
        noisy_truths.append(noisy_truth)

    # 保存噪声加密图像和解密后的图像
    for i, (noisy, truth) in enumerate(zip(noisy_images, noisy_truths), start=1):
        noisy_filename = f"data/permutation/noise/noisy_image_{i}.jpg"
        truth_filename = f"data/permutation/noise/noisy_truth_{i}.jpg"
        cv2.imwrite(noisy_filename, noisy)
        cv2.imwrite(truth_filename, cv2.cvtColor(truth, cv2.COLOR_RGB2BGR))

    show_images(
        [img_rgb, encrypted_image, decrypted_image],
        ["原始图像", "加密图像", "解密图像"],
        font,
        save_dir="data/permutation/comparison",  # 保存图像的文件夹路径
        file_name="decrypt_and_encrypt_comparison.jpg",
        main_title="原始图像，加密图像，解密图像对比图"
    )
    titles = [f"noise_factor={nf}" for nf in noise_factors]
    show_images(noisy_truths, titles, font, save_dir="data/permutation/comparison",
                file_name="noise_comparison.jpg", main_title="不同程度噪音下解密图像对比图")  # 保存噪声图像



    # 计算并展示图像的相关性和直方图等
    correlations_before = analyze_image_correlations(img_rgb, "加密前图像")
    correlations_after = analyze_image_correlations(encrypted_image, "加密后图像")
    plot_correlation_heatmap(correlations_before, correlations_after, "图像相关性分析")
    plot_image_histogram(img_rgb, encrypted_image)
    plot_entropy(img_rgb, encrypted_image, decrypted_image)


# 调用主函数处理图像
if __name__ == "__main__":
    process_image('data/test.jpg', noise_factors=[10, 50, 100, 500, 1000])  # 可以自定义噪声因子

