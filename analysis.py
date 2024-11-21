import cv2
import numpy as np
import matplotlib.pyplot as plt
import  random
import  math


'''密钥敏感性分析'''
'''
计算像素数变化率
'''
def NPCR(img1,img2):
  #opencv颜色通道顺序为BGR
  img1=cv2.imread(img1)
  img2=cv2.imread(img2)
  w,h,_=img1.shape

  #图像通道拆分
  B1,G1,R1=cv2.split(img1)
  B2,G2,R2=cv2.split(img2)
  #返回数组的排序后的唯一元素和每个元素重复的次数
  ar,num=np.unique((R1!=R2),return_counts=True)
  R_npcr=(num[0] if ar[0]==True else num[1])/(w*h)
  ar,num=np.unique((G1!=G2),return_counts=True)
  G_npcr=(num[0] if ar[0]==True else num[1])/(w*h)
  ar,num=np.unique((B1!=B2),return_counts=True)
  B_npcr=(num[0] if ar[0]==True else num[1])/(w*h)

  return R_npcr,G_npcr,B_npcr

'''
两张图像之间的平均变化强度
'''

def UACI(img1,img2):
  img1=cv2.imread(img1)
  img2=cv2.imread(img2)
  w,h,_=img1.shape
  #图像通道拆分
  B1,G1,R1=cv2.split(img1)
  B2,G2,R2=cv2.split(img2)
  #元素为uint8类型取值范围：0到255
  # print(R1.dtype)

  #强制转换元素类型，为了运算
  R1=R1.astype(np.int16)
  R2=R2.astype(np.int16)
  G1=G1.astype(np.int16)
  G2=G2.astype(np.int16)
  B1=B1.astype(np.int16)
  B2=B2.astype(np.int16)

  sumR=np.sum(abs(R1-R2))
  sumG=np.sum(abs(G1-G2))
  sumB=np.sum(abs(B1-B2))
  R_uaci=sumR/255/(w*h)
  G_uaci=sumG/255/(w*h)
  B_uaci=sumB/255/(w*h)

  return R_uaci,G_uaci,B_uaci


'''def main():
  #img='./lena.png'
  img1='./lena_encrypt1.png'
  img2='./lena_encrypt2.png'

  R_npcr,G_npcr,B_npcr=NPCR(img1,img2)
  print('*********PSNR*********')
  #百分数表示，保留小数点后4位
  print('Red  :{:.4%}'.format(R_npcr))
  print('Green:{:.4%}'.format(G_npcr))
  print('Blue :{:.4%}'.format(B_npcr))


  R_uaci,G_uaci,B_uaci=UACI(img1,img2)
  print('*********UACI*********')
  #百分数表示，保留小数点后4位
  print('Red  :{:.4%}'.format(R_uaci))
  print('Green:{:.4%}'.format(G_uaci))
  print('Blue :{:.4%}'.format(B_uaci))


if __name__== '__main__':
  main()
'''




'''
绘制灰度直方图
'''


def hist(img):
  img = cv2.imread(img)
  B, G, R = cv2.split(img)
  # 转成一维
  R = R.flatten(order='C')
  G = G.flatten(order='C')
  B = B.flatten(order='C')

  # 结果展示
  plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
  plt.subplot(232)
  # plt.imshow(img[:,:,(2,1,0)])
  plt.hist(img.flatten(order='C'), bins=range(257), color='gray')
  plt.title('原图像')
  # 子图2，通道R
  plt.subplot(234)
  # imshow()对图像进行处理，画出图像，show()进行图像显示
  plt.hist(R, bins=range(257), color='red')
  plt.title('通道R')
  # plt.show()
  # 不显示坐标轴
  # plt.axis('off')

  # 子图3，通道G
  plt.subplot(235)
  plt.hist(G, bins=range(257), color='green')
  plt.title('通道G')
  # plt.show()
  # plt.axis('off')

  # 子图4，通道B
  plt.subplot(236)
  plt.hist(B, bins=range(257), color='blue')
  plt.title('通道B')
  # plt.axis('off')
  # #设置子图默认的间距
  plt.tight_layout()
  plt.show()

'''
def main():
  img = './lena.png'
  # 图像lean的灰度直方图
  hist(img)


if __name__ == '__main__':
  main()
'''



'''鲁棒性，噪音攻击'''


def gauss_noise(image, mean=0, var=0.001):
  '''
      添加高斯噪声
      mean : 均值
      var : 方差
  '''
  image = np.array(image / 255, dtype=float)
  noise = np.random.normal(mean, var ** 0.5, image.shape)
  out = image + noise
  if out.min() < 0:
    low_clip = -1.
  else:
    low_clip = 0.
  out = np.clip(out, low_clip, 1.0)
  out = np.uint8(out * 255)
  return out


# 默认10%的椒盐噪声
def salt_and_pepper_noise(noise_img, proportion=0.1):
  height, width, _ = noise_img.shape
  num = int(height * width * proportion)  # 多少个像素点添加椒盐噪声
  for i in range(num):
    w = random.randint(0, width - 1)
    h = random.randint(0, height - 1)
    if random.randint(0, 1) == 0:
      noise_img[h, w] = 0
    else:
      noise_img[h, w] = 255
  return noise_img

'''
def main():
  img = './lena.png'
  img1 = './lena_encrypt1.png'
  img2 = './lena_encrypt2.png'
  im = cv2.imread(img1)
  gauss_img = gauss_noise(im, mean=0, var=0.0005)
  salt_img = salt_and_pepper_noise(im, proportion=0.05)
  cv2.imwrite('./gauss_img.png', gauss_img)
  cv2.imwrite('./salt_img.png', salt_img)


if __name__ == '__main__':
  main()
'''

'''
计算图像的信息熵
'''



def entropy(img):
  img = cv2.imread(img)
  w, h, _ = img.shape
  B, G, R = cv2.split(img)
  gray, num1 = np.unique(R, return_counts=True)
  gray, num2 = np.unique(G, return_counts=True)
  gray, num3 = np.unique(B, return_counts=True)
  R_entropy = 0
  G_entropy = 0
  B_entropy = 0

  for i in range(len(gray)):
    p1 = num1[i] / (w * h)
    p2 = num2[i] / (w * h)
    p3 = num3[i] / (w * h)
    R_entropy -= p1 * (math.log(p1, 2))
    G_entropy -= p2 * (math.log(p2, 2))
    B_entropy -= p3 * (math.log(p3, 2))
  return R_entropy, G_entropy, B_entropy


def main():
  img = './lena.png'
  img1 = './lena_encrypt1.png'
  img2 = './lena_encrypt2.png'
  # 图像lena的熵
  R_entropy, G_entropy, B_entropy = entropy(img)
  print('***********信息熵*********')
  print('通道R:{:.4}'.format(R_entropy))
  print('通道G:{:.4}'.format(G_entropy))
  print('通道B:{:.4}'.format(B_entropy))


if __name__ == '__main__':
  main()
