
from skimage.color import rgb2gray
import time
from matplotlib import font_manager
from analysis import *

# 设置字体 保证打印图片标题时不会为空方框
font = font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family'] = font.get_name()
# 加密函数
def encrypt_image(image):
    """对图像进行DCT变换并加密"""
    img_dct = np.fft.fft2(image)
    size =image.shape

    r = np.random.choice(size[0], size[0], replace=False)
    c = np.random.choice(size[1], size[1], replace=False)
    img_dct_s = img_dct[r, :]
    img_dct_ss = img_dct_s[:, c]
    return   img_dct_ss,r, c

# 解密函数
def decrypt_image(img_dct_ss, r, c):
    """对加密图像进行解密"""
    f = []
    for i in range(len(c)):
        f.append(np.where(c == i)[0][0])
    img_dct_d = img_dct_ss[:, f]
    g = []
    for j in range(len(r)):
        g.append(np.where(r == j)[0][0])
    img_dct_dd = img_dct_d[g, :]
    img_dct_de = np.fft.ifft2(img_dct_dd) / 255

    return img_dct_de

# 噪声添加函数
def add_noise_to_dct(dct_image, noise_factor=1000, high_freq_fraction=0.1):
    """向DCT加密图像添加噪声，主要添加到高频部分"""
    rows, cols = dct_image.shape
    high_freq_rows = int(rows * high_freq_fraction)
    high_freq_cols = int(cols * high_freq_fraction)
    noise = np.zeros_like(dct_image)
    noise[rows - high_freq_rows:, cols - high_freq_cols:] = noise_factor * np.random.randn(high_freq_rows, high_freq_cols)
    return dct_image + noise

def preprocess_image(image):
    """
    对输入图像进行预处理，确保其适用于 OpenCV 函数。
    转换为实数，归一化到 [0, 255]，并转为 uint8 类型。
    """
    # 确保图像为实数
    image = np.abs(image)

    # 将图像归一化到 [0, 255] 范围
    if np.max(image) > 0:  # 避免除以零
        image = (image / np.max(image) * 255).astype(np.uint8)
    else:
        image = np.zeros_like(image, dtype=np.uint8)

    return image
# 处理图像
def process_image(img_path, noise_factors):
    """处理图像：加密、添加噪声、解密并保存图像"""

    # 读取图像
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = rgb2gray(img_rgb)

    # 图像加密
    start_time = time.time()
    encrypted, r, c = encrypt_image(image)
    end_time = time.time()
    print(f"加密所需时间: {end_time - start_time} seconds")
    encrypted_abs = np.abs(encrypted)  # 取绝对值
    # 图像解密
    start_time = time.time()
    decrypted = decrypt_image(encrypted, r, c)
    end_time = time.time()
    print(f"解密所需时间: {end_time - start_time} seconds")

    # 噪声测试
    noisy_images = [add_noise_to_dct(encrypted, factor) for factor in noise_factors]
    noisy_decrypted_images = [decrypt_image(noisy_image, r, c) for noisy_image in noisy_images]

    # 图像分析
    correlations_before = analyze_image_correlations(img_rgb, "加密前图像")
    correlations_after = analyze_image_correlations(encrypted_abs, "加密后图像")
    plot_correlation_heatmap(correlations_before, correlations_after, "图像相关性分析")
    plot_image_histogram(img_rgb, encrypted_abs)
    plot_entropy(preprocess_image(image), preprocess_image(encrypted), preprocess_image(decrypted))

    # 显示图像
    plt.figure(figsize=(15, 8))

    # 原始图像
    plt.subplot(2, 5, 1)
    plt.imshow(img_rgb)
    plt.title("原始图像", fontproperties=font)

    # 灰度图像
    plt.subplot(2, 5, 2)
    plt.imshow(image, cmap='gray')
    plt.title("灰度图像", fontproperties=font)

    # 加密图像
    plt.subplot(2, 5, 3)
    plt.imshow(encrypted_abs, cmap='gray')  # 显示绝对值
    plt.title("加密图像", fontproperties=font)

    # 解密图像
    plt.subplot(2, 5, 4)
    plt.imshow(np.abs(decrypted), cmap='gray')
    plt.title("解密图像", fontproperties=font)

    # 噪声解密结果
    for i, (noisy_decrypted, factor) in enumerate(zip(noisy_decrypted_images, noise_factors)):
        plt.subplot(2, 5, 6 + i)
        plt.imshow(np.abs(noisy_decrypted), cmap='gray')
        plt.title(f"噪声解密: {factor}", fontproperties=font)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_image('data/test.jpg', noise_factors=[10, 50, 100, 500, 1000])
    process_image('data/test.jpg', noise_factors=[10, 50, 250, 1250, 6250])