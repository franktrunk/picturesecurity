import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import cv2
import  math


'''加入噪声'''
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



'''图像相关性分析'''


def calculate_pixel_correlation(image, direction="horizontal"):
  """
  计算图像相邻像素的相关性系数。

  :param image: 输入图像（灰度图像），二维 numpy 数组。
  :param direction: 方向，'horizontal', 'vertical', 'positive_diagonal', 'negative_diagonal'。
  :return: 相邻像素的相关性系数。
  """


  if direction == "horizontal":
    # 水平方向：获取每行的相邻像素对
    x = image[:, :-1].flatten()  # 去掉最后一列
    y = image[:, 1:].flatten()  # 去掉第一列

  elif direction == "vertical":
    # 垂直方向：获取每列的相邻像素对
    x = image[:-1, :].flatten()  # 去掉最后一行
    y = image[1:, :].flatten()  # 去掉第一行

  elif direction == "positive_diagonal":
    # 正对角方向：获取从左上到右下的相邻像素对
    x = image[:-1, :-1].flatten()  # 去掉最后一行和最后一列
    y = image[1:, 1:].flatten()  # 去掉第一行和第一列

  elif direction == "negative_diagonal":
    # 反对角方向：获取从右上到左下的相邻像素对
    x = image[:-1, 1:].flatten()  # 去掉最后一行和第一列
    y = image[1:, :-1].flatten()  # 去掉第一行和最后一列

  else:
    raise ValueError(
      "Invalid direction. Choose from 'horizontal', 'vertical', 'positive_diagonal', or 'negative_diagonal'.")

  # 计算皮尔逊相关系数
  correlation, _ = pearsonr(x, y)
  return correlation


def analyze_image_correlations(image, title="图像"):
  """
  分析图像在不同方向的相关性。

  :param image: 输入图像（灰度图像），二维 numpy 数组。
  :param title: 图像的标题（用于标识）。

  :return: 包含相关性的字典。
  """
  if image is None or image.size == 0:
    raise ValueError("输入的图像无效或为空。")

  correlations = {
    "horizontal": calculate_pixel_correlation(image, "horizontal"),
    "vertical": calculate_pixel_correlation(image, "vertical"),
    "positive_diagonal": calculate_pixel_correlation(image, "positive_diagonal"),
    "negative_diagonal": calculate_pixel_correlation(image, "negative_diagonal"),
  }

  print(f"{title}的相关性: {correlations}")
  return correlations


def plot_correlation_heatmap(correlations_before, correlations_after, title="相关性分析"):
    """
    绘制两个图像相关性对比热图，并在热图中标明相关性数值。

    :param correlations_before: 加密前图像的相关性字典。
    :param correlations_after: 加密后图像的相关性字典。
    :param title: 热图的标题。
    """
    directions = ["horizontal", "vertical", "positive_diagonal", "negative_diagonal"]

    values_before = [correlations_before[direction] for direction in directions]
    values_after = [correlations_after[direction] for direction in directions]

    # 创建图形和子图
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True, gridspec_kw={'width_ratios': [1, 1]})
    fig.suptitle(title, fontsize=16)

    # 设置颜色范围，确保两个热图的颜色范围一致
    vmin = min(min(values_before), min(values_after))
    vmax = max(max(values_before), max(values_after))

    # 绘制加密前图像相关性热图
    im1 = ax[0].imshow([values_before], cmap="coolwarm", aspect="auto", vmin=vmin, vmax=vmax)
    ax[0].set_xticks(np.arange(len(directions)))
    ax[0].set_xticklabels(directions, rotation=45, ha='right')
    ax[0].set_yticks([])
    ax[0].set_title("加密前图像", fontsize=14)
    for i, value in enumerate(values_before):
        ax[0].text(i, 0, f"{value:.2f}", ha='center', va='center', color='black')

    # 绘制加密后图像相关性热图
    im2 = ax[1].imshow([values_after], cmap="coolwarm", aspect="auto", vmin=vmin, vmax=vmax)
    ax[1].set_xticks(np.arange(len(directions)))
    ax[1].set_xticklabels(directions, rotation=45, ha='right')
    ax[1].set_yticks([])
    ax[1].set_title("加密后图像", fontsize=14)
    for i, value in enumerate(values_after):
        ax[1].text(i, 0, f"{value:.2f}", ha='center', va='center', color='black')

    # 添加共享颜色条
    cbar = fig.colorbar(im1, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label("相关性值", fontsize=12)

    # 显示图形
    plt.show()




def plot_image_histogram(image, encrypted_image=None):
    """
    生成并显示图像的直方图对比（原图和加密后的图像）。

    参数:
        image (np.array): 原图像数据。
        encrypted_image (np.array 或 None): 加密后的图像数据。如果没有传入，只有原图会显示。
    """
    plt.figure(figsize=(12, 6))  # 创建一个更大的图形窗口
    numbins=50
    # 将彩色图像转换为灰度图像
    if len(image.shape) == 3:  # 如果是彩色图像，转换为灰度图像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 判断并绘制原图直方图
    plt.subplot(1, 2, 1)  # 在1x2的网格中的第1个位置
    plt.hist(image.ravel(), bins=numbins, range=(0, 256), color='blue', alpha=0.7, linewidth=2)
    plt.title("Original Image Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.grid()

    # 如果有加密图像，先转换为灰度图像，再绘制直方图
    if encrypted_image is not None:
        if len(encrypted_image.shape) == 3:  # 如果是彩色图像，转换为灰度图像
            encrypted_image = cv2.cvtColor(encrypted_image, cv2.COLOR_BGR2GRAY)

        plt.subplot(1, 2, 2)  # 在1x2的网格中的第2个位置
        plt.hist(encrypted_image.ravel(), bins=numbins, range=(0, 256), color='blue', alpha=0.7, linewidth=2)
        plt.title("Encrypted Image Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.grid()

    plt.tight_layout()
    plt.show()


def calculate_entropy(image):
    """
    计算图像的熵值。

    参数:
        image (ndarray): 图像数据，应该是已经通过cv2打开的图像。

    返回:
        float: 图像的熵值。
    """
    # 确保图像是灰度图像
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算图像的直方图
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # 归一化直方图（将频率转换为概率）
    hist /= hist.sum()

    # 计算熵值
    entropy = 0
    for p in hist:
        if p > 0:
            entropy -= p * math.log2(p)

    return float(entropy)  # 确保返回的是标量

def plot_entropy(original, encrypted, decrypted):
    """
    绘制加密前、加密后和解密后三张图像的对比图，并显示熵值。

    参数:
        original (ndarray): 加密前的图像。
        encrypted (ndarray): 加密后的图像。
        decrypted (ndarray): 解密后的图像。
    """
    # 确保图像是灰度图（如果是彩色图像）
    if len(original.shape) == 3:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    if len(encrypted.shape) == 3:
        encrypted = cv2.cvtColor(encrypted, cv2.COLOR_BGR2GRAY)
    if len(decrypted.shape) == 3:
        decrypted = cv2.cvtColor(decrypted, cv2.COLOR_BGR2GRAY)

    # 计算熵值
    entropy_original = calculate_entropy(original)
    entropy_encrypted = calculate_entropy(encrypted)
    entropy_decrypted = calculate_entropy(decrypted)

    # 准备图像和标题
    images = [original, encrypted, decrypted]
    entropies = [entropy_original, entropy_encrypted, entropy_decrypted]
    titles = ['加密前图像', '加密后图像', '解密后图像']

    # 绘制图像
    plt.figure(figsize=(15, 5))

    for i, (img, entropy, title) in enumerate(zip(images, entropies, titles)):
        plt.subplot(1, 3, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'{title}\n熵值: {entropy:.2f}', pad=40)  # 增加标题的间距
        plt.axis('off')

    # 调整布局，避免标题过于靠上
    plt.subplots_adjust(top=0.8)  # 这个调整可微调，减小图像上方的空白

    plt.tight_layout()
    plt.show()