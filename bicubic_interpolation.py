import numpy as np
import cv2

def bicubic_kernel(x):
    a = -0.5
    abs_x = np.abs(x)
    if abs_x <= 1:
        return (a + 2) * np.power(abs_x, 3) - (a + 3) * np.power(abs_x, 2) + 1
    elif 1 < abs_x < 2:
        return a * np.power(abs_x, 3) - 5*a * np.power(abs_x, 2) + 8*a * abs_x - 4*a
    else:
        return 0

def bicubic_interpolate(img, scale_factor):
    H, W = img.shape
    H_new, W_new = int(H * scale_factor), int(W * scale_factor)
    img_new = np.zeros((H_new, W_new))

    for i in range(H_new):
        for j in range(W_new):
            x = i / scale_factor
            y = j / scale_factor

            x_floor = np.floor(x).astype(int)
            y_floor = np.floor(y).astype(int)

            sum_val = 0
            for m in range(-1, 3):
                for n in range(-1, 3):
                    if 0 <= x_floor + m < H and 0 <= y_floor + n < W:
                        p = img[x_floor + m, y_floor + n]
                        sum_val += p * bicubic_kernel(x - (x_floor + m)) * bicubic_kernel(y - (y_floor + n))

            img_new[i, j] = np.clip(sum_val, 0, 255)

    return img_new.astype(np.uint8)

# 이미지 불러오기 및 색상 채널 분리
img = cv2.imread('orig_img.PNG', cv2.IMREAD_GRAYSCALE)  # 예시로 그레이스케일 이미지를 사용
scale_factor = 2.0  # 이미지를 2배로 확대

# 바이큐빅 보간 수행
img_interpolated = bicubic_interpolate(img, scale_factor)
img_interpolated = cv2.resize(img_interpolated, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC);

# 결과 출력
cv2.imshow('Original Image', img)
cv2.imshow('Bicubic Interpolation', img_interpolated)
cv2.imwrite('bicubic_img.PNG', img_interpolated)
cv2.waitKey(0)
cv2.destroyAllWindows()
