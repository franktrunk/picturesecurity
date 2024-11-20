import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

# 设置AES加密模式
#mode = AES.MODE_CBC
mode = AES.MODE_ECB
if mode not in [AES.MODE_CBC, AES.MODE_ECB]:
    print("Only CBC and ECB mode supported...")
    sys.exit()

# 设置密钥长度和IV长度
keySize = 32
ivSize = AES.block_size if mode == AES.MODE_CBC else 0

# 加载图像
imageOrig = cv2.imread("data/test.jpg")
if imageOrig is None:
    print("Error: Image not loaded. Check the file path.")
    sys.exit()

rowOrig, columnOrig, depthOrig = imageOrig.shape
print(f"Original image shape: {imageOrig.shape}")

# 检查宽度是否满足加密最小宽度要求
minWidth = (AES.block_size + AES.block_size) // depthOrig + 1
if columnOrig < minWidth:
    print(f"The minimum width of the image must be {minWidth} pixels.")
    sys.exit()

# 显示原始图像
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(imageOrig, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
plt.show()

# 图像转字节流
imageOrigBytes = imageOrig.tobytes()
print(f"Original bytes length: {len(imageOrigBytes)}")

# 数据填充
imageOrigBytesPadded = pad(imageOrigBytes, AES.block_size)
print(f"Padded bytes length: {len(imageOrigBytesPadded)} (should be multiple of {AES.block_size})")

# 加密
key = get_random_bytes(keySize)
iv = get_random_bytes(ivSize)
cipher = AES.new(key, mode, iv) if mode == AES.MODE_CBC else AES.new(key, mode)
ciphertext = cipher.encrypt(imageOrigBytesPadded)
print(f"Ciphertext length: {len(ciphertext)}")

# 填充加密图像数据
bytes_per_row = columnOrig * depthOrig
total_size = (rowOrig + 1) * bytes_per_row
void = total_size - ivSize - len(ciphertext)
if void < 0:
    print("Error: Void size is negative, check dimensions.")
    sys.exit()

ivCiphertextVoid = iv + ciphertext + bytes([0] * void)
if len(ivCiphertextVoid) != total_size:
    print(f"Error: Data size mismatch. Expected {total_size}, got {len(ivCiphertextVoid)}.")
    sys.exit()

# 重建加密图像
imageEncrypted = np.frombuffer(ivCiphertextVoid, dtype=imageOrig.dtype).reshape(rowOrig + 1, columnOrig, depthOrig)

# 显示加密图像
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(imageEncrypted, cv2.COLOR_BGR2RGB))
plt.title("Encrypted Image")
plt.axis('off')
plt.show()

# 保存加密图像
cv2.imwrite("topsecretEnc.jpg", imageEncrypted)

# 解密部分
rowEncrypted, columnOrig, depthOrig = imageEncrypted.shape
rowOrig = rowEncrypted - 1
encryptedBytes = imageEncrypted.tobytes()

# 提取IV和密文
iv = encryptedBytes[:ivSize]
encrypted_data_size = len(encryptedBytes) - ivSize - void
encrypted = encryptedBytes[ivSize: ivSize + encrypted_data_size]

# 检查密文长度是否正确
if len(encrypted) % AES.block_size != 0:
    print(f"Error: Encrypted data length {len(encrypted)} is not a multiple of {AES.block_size}.")
    sys.exit()

# 解密
cipher = AES.new(key, mode, iv) if mode == AES.MODE_CBC else AES.new(key, mode)
decryptedImageBytesPadded = cipher.decrypt(encrypted)
decryptedImageBytes = unpad(decryptedImageBytesPadded, AES.block_size)
print(f"Decrypted bytes length: {len(decryptedImageBytes)}")

# 重建解密图像
decryptedImage = np.frombuffer(decryptedImageBytes, dtype=imageOrig.dtype).reshape(rowOrig, columnOrig, depthOrig)

# 显示解密图像
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(decryptedImage, cv2.COLOR_BGR2RGB))
plt.title("Decrypted Image")
plt.axis('off')
plt.show()

# 保存解密图像
cv2.imwrite("decryptedImage.jpg", decryptedImage)
cv2.destroyAllWindows()
