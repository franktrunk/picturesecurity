import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置字体路径
font = font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')

def add_noise(image, noise_factor=100):
    """向图像添加高斯噪声"""
    noise_matrix = noise_factor * np.random.randn(*image.shape)  # 生成噪声矩阵
    noise_matrix_normalized = (noise_matrix - np.min(noise_matrix)) / (np.max(noise_matrix) - np.min(noise_matrix))

    # 显示噪声矩阵
    plt.imshow(noise_matrix_normalized, cmap='gray')  # 使用灰度色图显示归一化后的噪声矩阵
    plt.title("Noise Matrix")
    plt.colorbar()  # 添加颜色条

    # 保存噪声矩阵图像
    noise_image_path = f"data/permutation/noise/noise_matrix.jpg"
    plt.savefig(noise_image_path, bbox_inches='tight', pad_inches=0.1)  # 保存噪声矩阵
    plt.close()  # 关闭当前图像，避免显示多个图像
    noisy_image = image + noise_matrix
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

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


def show_images(images, titles, font, save_dir="data/permutation",file_name="11.jpg"):
    """显示多幅图像并保存最后一幅"""
    plt.figure(figsize=(15, 5))

    for i, (image, title) in enumerate(zip(images, titles), 1):
        plt.subplot(1, len(images), i)
        plt.imshow(image)
        plt.title(title, fontproperties=font)
        plt.axis("off")

    # 只保存最后一幅图像
    save_path = f"{save_dir}/{file_name}"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)  # 保存最后一张图像，去除边距

    plt.show()

if __name__ == "__main__":
    # 读取图像
    img = cv2.imread('data/test.JPG')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 加密
    encrypted, r, c = encrypt_image(img)
    encrypted_rgb = cv2.cvtColor(encrypted, cv2.COLOR_BGR2RGB)

    # 保存加密图像
    cv2.imwrite("data/permutation/encrypt_and_decrypt/encrypted_image.jpg", encrypted)

    # 解密
    decrypted = decrypt_image(encrypted, r, c)
    decrypted_rgb = cv2.cvtColor(decrypted, cv2.COLOR_BGR2RGB)

    # 保存解密图像
    cv2.imwrite("data/permutation/encrypt_and_decrypt/decrypted_image.jpg", decrypted)

    # 添加噪声并解密
    noise_factors = [10, 50, 100, 500, 1000]
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
        save_dir="data/permutation/comparison" ,  # 保存图像的文件夹路径
        file_name="decrypt_and_encrypt_comparison.jpg"
    )

    # 显示并保存噪声加密与解密图像
    titles = [f"noise_factor={nf}" for nf in noise_factors]
    show_images(noisy_truths, titles, font, save_dir="data/permutation/comparison",file_name ="noise_comparison.jpg")  # 保存噪声图像
