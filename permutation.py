
from matplotlib import font_manager
from  analysis import *
import  time
# 设置字体路径
font = font_manager.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # 使用 Windows 系统中的 SimHei 字体
plt.rcParams['font.family'] = font.get_name()
def encrypt_image(image):
    """对图像进行行列置乱加密"""
    s = image.shape
    r = np.random.permutation(s[0])  # 随机行置乱
    c = np.random.permutation(s[1])  # 随机列置乱
    encrypted = image[r, :, :][:, c, :]
    return encrypted, r, c


def decrypt_image(encrypted, r, c):
    """对图像进行行列置乱解密"""
    r_inv = np.argsort(r)  # 行逆映射
    c_inv = np.argsort(c)  # 列逆映射
    decrypted = encrypted[r_inv, :, :][:, c_inv, :]
    return decrypted


def decode(r_inv, c_inv, enimage):
    """解密函数：适用于噪声图像"""
    decrypted = enimage[r_inv, :, :][:, c_inv, :]
    return cv2.cvtColor(decrypted, cv2.COLOR_BGR2RGB)


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


def process_image(img_path, noise_factors):
    """处理图像：加密、添加噪声、解密并保存图像"""

    # 读取图像
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 加密
    start_time = time.perf_counter()
    encrypted, r, c = encrypt_image(img)
    end_time = time.perf_counter()
    print(f"加密所需时间: {end_time - start_time:.6f}秒")
    encrypted_rgb = cv2.cvtColor(encrypted, cv2.COLOR_BGR2RGB)

    # 保存加密图像
    cv2.imwrite("data/permutation/encrypt_and_decrypt/encrypted_image.jpg", encrypted)

    # 解密
    start_time = time.perf_counter()
    decrypted = decrypt_image(encrypted, r, c)
    end_time = time.perf_counter()
    decrypted_rgb = cv2.cvtColor(decrypted, cv2.COLOR_BGR2RGB)
    print(f"解密所需时间: {end_time - start_time:.6f}秒")  # 保留 6 位小数
    # 保存解密图像
    cv2.imwrite("data/permutation/encrypt_and_decrypt/decrypted_image.jpg", decrypted)

    # 添加噪声并解密
    noisy_images = []
    noisy_truths = []
    for nf in noise_factors:
        noisy_image = add_noise(encrypted, nf)
        noisy_images.append(noisy_image)
        noisy_truth = decode(np.argsort(r), np.argsort(c), noisy_image)
        noisy_truths.append(noisy_truth)

    # 保存噪声加密图像和解密后的图像
    for i, (noisy, truth) in enumerate(zip(noisy_images, noisy_truths), start=1):
        noisy_filename = f"data/permutation/noise/noisy_image_{i}.jpg"
        truth_filename = f"data/permutation/noise/noisy_truth_{i}.jpg"
        cv2.imwrite(noisy_filename, noisy)
        cv2.imwrite(truth_filename, cv2.cvtColor(truth, cv2.COLOR_RGB2BGR))

    # 显示并保存原始、加密和解密图像
    show_images(
        [img_rgb, encrypted_rgb, decrypted_rgb],
        ["原始图像", "加密图像", "解密图像"],
        font,
        save_dir="data/permutation/comparison",  # 保存图像的文件夹路径
        file_name="decrypt_and_encrypt_comparison.jpg",
        main_title="原始图像，加密图像，解密图像对比图"
    )

    # 显示并保存噪声加密与解密图像
    titles = [f"noise_factor={nf}" for nf in noise_factors]
    show_images(noisy_truths, titles, font, save_dir="data/permutation/comparison",
                file_name="noise_comparison.jpg",main_title="不同程度噪音下解密图像对比图")  # 保存噪声图像


    correlations_before = analyze_image_correlations(img, "加密前图像")
    correlations_after = analyze_image_correlations(encrypted, "加密后图像")
    plot_correlation_heatmap(correlations_before, correlations_after, "图像相关性分析")
    plot_image_histogram(img, encrypted)
    plot_entropy(img, encrypted,decrypted)

# 调用主函数处理图像
if __name__ == "__main__":
    process_image('data/test.jpg', noise_factors=[10, 50, 100, 500, 1000])
