import  time
from Crypto.Cipher import Blowfish
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from analysis import *  # 引入图像分析函数
from matplotlib import font_manager

font = font_manager.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # 使用 Windows 系统中的 SimHei 字体
plt.rcParams['font.family'] = font.get_name()


def add_gaussian_noise(image, mean=0, sigma=25):
    """
    向图像添加高斯噪声。参数：
    image: 输入的图像。
    mean: 高斯噪声的均值，默认为0。
    sigma: 高斯噪声的标准差，默认为25。

    返回：带噪声的图像。
    """
    row, column, depth = image.shape
    gauss = np.random.normal(mean, sigma, (row, column, depth))  # 生成高斯噪声
    noisy_image = np.array(image, dtype=float) + gauss  # 将噪声添加到图像
    noisy_image = np.clip(noisy_image, 0, 255)  # 限制像素值在[0, 255]范围内
    return noisy_image.astype(np.uint8)


class BlowfishImageEncryptor:
    def __init__(self, key=None):
        """
        初始化加密器。参数：key: 加密密钥 (bytes)，若为 None，则随机生成密钥。
        """
        self.key = key if key else get_random_bytes(16)  # Blowfish 支持 4-56 字节的密钥
        self.block_size = Blowfish.block_size

    def encrypt_image(self, image):
        """
        加密图像。参数：image: 输入图像。返回：encrypted_image: 加密后的图像，original_shape: 原始图像形状。
        """
        row_orig, column_orig, depth_orig = image.shape

        # 图像转字节流
        image_bytes = image.tobytes()

        # 填充数据
        padded_bytes = pad(image_bytes, self.block_size)

        # 创建加密器
        cipher = Blowfish.new(self.key, Blowfish.MODE_ECB)
        ciphertext = cipher.encrypt(padded_bytes)

        # 重建加密图像
        bytes_per_row = column_orig * depth_orig
        total_size = ((len(ciphertext) + bytes_per_row - 1) // bytes_per_row) * bytes_per_row
        void = total_size - len(ciphertext)
        encrypted_bytes = ciphertext + bytes([0] * void)
        encrypted_image = np.frombuffer(encrypted_bytes, dtype=image.dtype).reshape(-1, column_orig, depth_orig)

        return encrypted_image, (row_orig, column_orig, depth_orig)

    def decrypt_image(self, encrypted_image, original_shape):
        """
        解密图像。参数：encrypted_image: 加密图像，original_shape: 原始图像形状。返回：decrypted_image: 解密后的图像。
        """
        encrypted_bytes = encrypted_image.tobytes()

        # 去除多余填充
        ciphertext = encrypted_bytes.rstrip(b'\x00')

        # 创建解密器
        cipher = Blowfish.new(self.key, Blowfish.MODE_ECB)
        padded_bytes = cipher.decrypt(ciphertext)
        image_bytes = unpad(padded_bytes, self.block_size)

        # 重建解密图像
        row_orig, column_orig, depth_orig = original_shape
        decrypted_image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(row_orig, column_orig, depth_orig)
        return decrypted_image

    def display_image(self, image, title):
        """
        显示图像。参数：image: 图像数组，title: 图像标题。
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()

    def process_images(self, input_image_path, encrypted_image_path, decrypted_image_path):
        """
        处理加密和解密流程，并进行图像分析。参数：
        input_image_path: 原始图像路径，encrypted_image_path: 加密图像保存路径，decrypted_image_path: 解密图像保存路径。
        """
        img = cv2.imread(input_image_path)
        if img is None:
            raise ValueError("无法加载图像，请检查路径！")

        # 原图加密解密
        start_time = time.perf_counter()
        encrypted_image, original_shape = self.encrypt_image(img)
        end_time = time.perf_counter()
        print(f"加密所需时间: {end_time - start_time:.6f}秒")
        # 保存加密图像
        cv2.imwrite(encrypted_image_path, encrypted_image)
        print(f"加密图像已保存到 {encrypted_image_path}")

        # 解密图像
        print("开始解密图像...")
        start_time = time.perf_counter()
        decrypted_image = self.decrypt_image(encrypted_image, original_shape)
        end_time = time.perf_counter()
        print(f"加密所需时间: {end_time - start_time:.6f}秒")
        # 图像列表
        images = [img, encrypted_image, decrypted_image]
        titles = ["Original Image", "Encrypted Image", "Decrypted Image"]

        # 使用 matplotlib 展示三张图像
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # 使用循环展示每张图像
        for i, (image, title) in enumerate(zip(images, titles)):
            axs[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axs[i].set_title(title)
            axs[i].axis('off')

        # 显示图像
        plt.show()
        # 保存解密图像
        cv2.imwrite(decrypted_image_path, decrypted_image)
        print(f"解密图像已保存到 {decrypted_image_path}")

        # 图像分析
        print("开始进行图像分析...")
        correlations_before = analyze_image_correlations(img, "加密前图像")
        correlations_after = analyze_image_correlations(encrypted_image, "加密后图像")
        plot_correlation_heatmap(correlations_before, correlations_after, "图像相关性分析")
        plot_image_histogram(img, encrypted_image)
        plot_entropy(img, encrypted_image, decrypted_image)

        # 噪声攻击：不同噪声强度
        noise_levels = [10, 50, 100, 500, 1000]  # 不同的噪声强度
        noisy_images = []
        decrypted_noisy_images = []

        for sigma in noise_levels:
            print(f"对图像添加噪声 (sigma={sigma}) 并进行加密解密...")
            noisy_image = add_gaussian_noise(img, sigma=sigma)

            # 加密噪声图像
            encrypted_noisy_image, original_shape = self.encrypt_image(noisy_image)

            # 解密噪声图像
            decrypted_noisy_image = self.decrypt_image(encrypted_noisy_image, original_shape)

            # 将噪声图像和解密结果保存
            noisy_images.append(noisy_image)
            decrypted_noisy_images.append(decrypted_noisy_image)

        # 显示噪声攻击下的不同噪声图像
        fig, axs = plt.subplots(1, len(noise_levels), figsize=(15, 5))

        # 显示不同噪声强度下的噪声图像
        for i, sigma in enumerate(noise_levels):
            axs[i].imshow(cv2.cvtColor(noisy_images[i], cv2.COLOR_BGR2RGB))
            axs[i].set_title(f'Noisy (Noise: {sigma})')
            axs[i].axis('off')

        plt.show()
        print("所有图像已解密并展示。")


if __name__ == "__main__":
    # 文件路径
    input_image_path = "data/test.jpg"
    encrypted_image_path = "data/topsecretEnc.jpg"
    decrypted_image_path = "data/decryptedImage.jpg"

    # 初始化加密器并处理图像
    encryptor = BlowfishImageEncryptor()
    encryptor.process_images(input_image_path, encrypted_image_path, decrypted_image_path)
