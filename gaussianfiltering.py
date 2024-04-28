from scipy.ndimage import convolve
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Gaussian 필터링 함수 구현
def gaussian_filtering(input_image):
    # Gaussian kernel (3x3 approximation)
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 16
    # 이미지에 커널 적용
    filtered_image = convolve(input_image, kernel, mode='constant', cval=0.0)
    return filtered_image

# 이미지 로드 및 흑백 변환
orig_img = Image.open('orig_img.PNG').convert('L')
orig_img_array = np.array(orig_img)

# Gaussian 필터링 적용
filtered_img_array = gaussian_filtering(orig_img_array)

# 결과 이미지 표시
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(orig_img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_img_array, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')

plt.show()

# 필터링된 이미지 저장
filtered_img = Image.fromarray(filtered_img_array)
filtered_img_path = './filtered_img.PNG'
filtered_img.save(filtered_img_path)

filtered_img_path