import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Crypto.Cipher import Blowfish
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

# 加密和解密函数
def encrypt_image(image_path, key, block_size):
    # 读取图像
    image_orig = cv2.imread(image_path)
    if image_orig is None:
        raise ValueError("无法加载图像，请检查路径！")
    row_orig, column_orig, depth_orig = image_orig.shape
    print(f"原始图像形状: {image_orig.shape}")

    # 图像转字节流
    image_bytes = image_orig.tobytes()
    print(f"原始字节长度: {len(image_bytes)}")

    # 填充数据
    padded_bytes = pad(image_bytes, block_size)
    print(f"填充后字节长度: {len(padded_bytes)} (应为 {block_size} 的倍数)")

    # 创建加密器
    cipher = Blowfish.new(key, Blowfish.MODE_ECB)
    ciphertext = cipher.encrypt(padded_bytes)
    print(f"密文长度: {len(ciphertext)}")

    # 重建加密图像（扩展行数以适应加密数据）
    bytes_per_row = column_orig * depth_orig
    total_size = ((len(ciphertext) + bytes_per_row - 1) // bytes_per_row) * bytes_per_row
    void = total_size - len(ciphertext)
    encrypted_bytes = ciphertext + bytes([0] * void)
    encrypted_image = np.frombuffer(encrypted_bytes, dtype=image_orig.dtype).reshape(-1, column_orig, depth_orig)

    return encrypted_image, (row_orig, column_orig, depth_orig), key

def decrypt_image(encrypted_image, original_shape, key, block_size):
    # 提取加密字节
    encrypted_bytes = encrypted_image.tobytes()

    # 去除多余填充
    ciphertext = encrypted_bytes.rstrip(b'\x00')

    # 创建解密器
    cipher = Blowfish.new(key, Blowfish.MODE_ECB)
    padded_bytes = cipher.decrypt(ciphertext)
    image_bytes = unpad(padded_bytes, block_size)

    # 重建解密图像
    row_orig, column_orig, depth_orig = original_shape
    decrypted_image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(row_orig, column_orig, depth_orig)
    return decrypted_image

# 主程序
if __name__ == "__main__":
    # 输入和输出文件
    input_image_path = "data/test.jpg"
    encrypted_image_path = "data/topsecretEnc.jpg"
    decrypted_image_path = "data/decryptedImage.jpg"

    # 设置密钥
    key = get_random_bytes(16)  # Blowfish 支持 4-56 字节的密钥
    block_size = Blowfish.block_size

    # 加密图像
    encrypted_image, original_shape, key = encrypt_image(input_image_path, key, block_size)
    print(f"加密图像形状: {encrypted_image.shape}")

    # 显示加密后的图像
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(encrypted_image, cv2.COLOR_BGR2RGB))
    plt.title("Encrypted Image")
    plt.axis('off')
    plt.show()

    # 保存加密图像
    cv2.imwrite(encrypted_image_path, encrypted_image)

    # 解密图像
    decrypted_image = decrypt_image(encrypted_image, original_shape, key, block_size)
    print(f"解密图像形状: {decrypted_image.shape}")

    # 显示解密后的图像
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(decrypted_image, cv2.COLOR_BGR2RGB))
    plt.title("Decrypted Image")
    plt.axis('off')
    plt.show()

    # 保存解密图像
    cv2.imwrite(decrypted_image_path, decrypted_image)
    cv2.destroyAllWindows()
