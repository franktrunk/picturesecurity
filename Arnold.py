import time
from matplotlib import font_manager
from analysis import *  # 导入图像分析部分
font = font_manager.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # 使用 Windows 系统中的 SimHei 字体
plt.rcParams['font.family'] = font.get_name()

# 向图像添加噪声
def add_noise(image, noise_factor):
    """添加噪声到图像"""
    noise = np.random.normal(0, noise_factor, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


# Arnold加密变换
def arnold_encrypt(image, shuffle_times, a, b):
    """使用 Arnold 变换对图像进行加密。"""
    h, w = image.shape[:2]
    N = h
    result_image = np.zeros_like(image)

    for _ in range(shuffle_times):
        for x in range(h):
            for y in range(w):
                new_x = (x + b * y) % N
                new_y = (a * x + (a * b + 1) * y) % N
                result_image[new_x, new_y, :] = image[x, y, :]
    return result_image


# Arnold解密变换
def arnold_decrypt(image, shuffle_times, a, b):
    """使用 Arnold 变换对图像进行解密。"""
    h, w = image.shape[:2]
    N = h
    result_image = np.zeros_like(image)

    for _ in range(shuffle_times):
        for x in range(h):
            for y in range(w):
                new_x = (x * (a * b + 1) - b * y) % N
                new_y = (-a * x + y) % N
                result_image[new_x, new_y, :] = image[x, y, :]
    return result_image


def show_images(images, titles, font, save_dir="data/permutation", file_name="11.jpg", main_title=None):
    """显示多幅图像并保存最后一幅，可以添加大标题"""
    plt.figure(figsize=(15, 5))

    # 添加大标题（如果传入了大标题参数）
    if main_title:
        plt.suptitle(main_title, fontsize=16, fontweight='bold', fontproperties=font)

    for i, (image, title) in enumerate(zip(images, titles), 1):
        plt.subplot(1, len(images), i)
        plt.imshow(image)
        plt.title(title, fontproperties=font)
        plt.axis("off")

    # 只保存最后一幅图像
    save_path = f"{save_dir}/{file_name}"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)  # 保存最后一张图像，去除边距

    plt.show()

# 图像预处理
def preprocess_image(image_path, size=(256, 256)):
    """读取图像并调整大小。"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_BGR2RGB)
    return img


# 处理图像的主函数
def process_image(img_path, noise_factors):
    """处理图像：加密、添加噪声、解密并保存图像"""
    # 读取图像
    img = preprocess_image(img_path)

    # 加密
    shuffle_times, a, b = 1, 1, 1
    start_time = time.perf_counter()
    encrypted = arnold_encrypt(img, shuffle_times, a, b)
    end_time = time.perf_counter()
    print(f"加密所需时间: {end_time - start_time:.6f}秒")

    # 解密
    start_time = time.perf_counter()
    decrypted = arnold_decrypt(encrypted, shuffle_times, a, b)
    end_time = time.perf_counter()
    print(f"解密所需时间: {end_time - start_time:.6f}秒")

    # 添加噪声并解密
    noisy_images = []
    noisy_truths = []
    for nf in noise_factors:
        noisy_image = add_noise(encrypted, nf)
        noisy_images.append(noisy_image)
        noisy_truth = arnold_decrypt(noisy_image, shuffle_times, a, b)
        noisy_truths.append(noisy_truth)

    # 显示并保存图像
    show_images(
        [img, encrypted, decrypted],
        ["原始图像", "加密图像", "解密图像"],
        font,
        save_dir="data/permutation/comparison",  # 保存图像的文件夹路径
        file_name="decrypt_and_encrypt_comparison.jpg",
        main_title="原始图像，加密图像，解密图像对比图"
    )

    # 显示噪声加密图像及解密结果
    titles = [f"noise_factor={nf}" for nf in noise_factors]
    show_images(noisy_truths, titles, font, save_dir="data/permutation/comparison",
                file_name="noise_comparison.jpg", main_title="不同噪声因子下解密图像")

    # 图像分析
    correlations_before = analyze_image_correlations(img, "加密前图像")
    correlations_after = analyze_image_correlations(encrypted, "加密后图像")
    plot_correlation_heatmap(correlations_before, correlations_after, "图像相关性分析")

    plot_image_histogram(img, encrypted)
    plot_entropy(img, encrypted, decrypted)


# 调用主函数处理图像
if __name__ == "__main__":
    process_image('data/test.jpg', noise_factors=[10, 50, 100, 500, 1000])
