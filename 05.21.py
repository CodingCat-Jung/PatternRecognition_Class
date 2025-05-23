#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2

# 최소값 & 최대값 필터링 함수
def minmax_filter(image, ksize, mode):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.uint8)  # 출력 영상 초기화
    center = ksize // 2                    # 마스크 중심 위치

    for i in range(center, rows - center):
        for j in range(center, cols - center):
            # 마스크 영역 설정
            y1, y2 = i - center, i + center + 1
            x1, x2 = j - center, j + center + 1
            mask = image[y1:y2, x1:x2]     # 마스크 영역
            dst[i, j] = cv2.minMaxLoc(mask)[mode]  # mode=0:최소값, mode=1:최댓값

    return dst

# 영상 입력
image = cv2.imread("C:/Users/SAMSUNG/Desktop/Pattern_Recognition_Class/images6/min_max.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")

# 최소값 필터링 (mode=0), 최대값 필터링 (mode=1)
minfilter_img = minmax_filter(image, 3, 0)
maxfilter_img = minmax_filter(image, 3, 1)

# 결과 출력
cv2.imshow("image", image)
cv2.imshow("minfilter_img", minfilter_img)
cv2.imshow("maxfilter_img", maxfilter_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import numpy as np
import cv2

# 평균값 필터링 함수 정의
def average_filter(image, ksize):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.uint8)  # 출력 영상 초기화
    center = ksize // 2                     # 마스크 중심 위치

    for i in range(rows):
        for j in range(cols):
            # 마스크 범위 계산
            y1, y2 = i - center, i + center + 1
            x1, x2 = j - center, j + center + 1

            # 경계 조건 확인
            if y1 < 0 or y2 > rows or x1 < 0 or x2 > cols:
                dst[i, j] = image[i, j]  # 경계 밖은 원래 값 유지 (cv2.BORDER_CONSTANT 방식)
            else:
                mask = image[y1:y2, x1:x2]     # 마스크 지정
                dst[i, j] = cv2.mean(mask)[0]  # 평균값 할당

    return dst

# 영상 입력
image = cv2.imread("C:/Users/SAMSUNG/Desktop/Pattern_Recognition_Class/images6/filter_avg.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")

# 사용자 정의 평균값 필터 적용
avg_img = average_filter(image, 5)

# OpenCV 함수 평균값 비교
blur_img = cv2.blur(image, (5, 5), anchor=(-1, -1), borderType=cv2.BORDER_REFLECT)  # 평균 필터
box_img = cv2.boxFilter(image, ddepth=-1, ksize=(5, 5))                             # 박스 필터

# 결과 출력
cv2.imshow("image", image)
cv2.imshow("avg_img", avg_img)
cv2.imshow("blur_img", blur_img)
cv2.imshow("box_img", box_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import numpy as np
import cv2

# 사용자 정의 미디언 필터 함수
def median_filter(image, ksize):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.uint8)  # 출력 영상 초기화
    center = ksize // 2  # 마스크 중심 위치

    for i in range(center, rows - center):
        for j in range(center, cols - center):
            y1, y2 = i - center, i + center + 1
            x1, x2 = j - center, j + center + 1
            mask = image[y1:y2, x1:x2].flatten()  # 마스크를 1차원 배열로 변환
            sort_mask = cv2.sort(mask, cv2.SORT_EVERY_COLUMN)  # 오름차순 정렬
            dst[i, j] = sort_mask[sort_mask.size // 2].item()  # 중앙값 할당

    return dst

# 소금-후추 노이즈 생성 함수
def salt_pepper_noise(img, n):
    h, w = img.shape[:2]
    x = np.random.randint(0, w, n)
    y = np.random.randint(0, h, n)
    noise = img.copy()
    for (x_, y_) in zip(x, y):
        noise[y_, x_] = 0 if np.random.rand() < 0.5 else 255
    return noise

# 이미지 입력
image = cv2.imread("C:/Users/SAMSUNG/Desktop/Pattern_Recognition_Class/images6/median.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")

# 소금-후추 노이즈 추가
noise = salt_pepper_noise(image, 500)

# 사용자 정의 미디언 필터
med_img1 = median_filter(noise, 5)

# OpenCV 제공 미디언 필터
med_img2 = cv2.medianBlur(noise, 5)

# 결과 출력
cv2.imshow("image", image)
cv2.imshow("noise", noise)
cv2.imshow("median - User", med_img1)
cv2.imshow("Median - OpenCV", med_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import numpy as np
import cv2

# 가우시안 마스크 생성 함수
def getGaussianMask(ksize, sigmaX, sigmaY):
    # sigma 값이 음수이면 ksize 기반으로 기본값 계산
    sigma = 0.3 * ((np.array(ksize) - 1.0) * 0.5 - 1.0) + 0.8

    if sigmaX <= 0:
        sigmaX = sigma[0]
    if sigmaY <= 0:
        sigmaY = sigma[1]

    # 커널 절반 크기 (중심 기준)
    u = np.array(ksize) // 2

    # x, y 방향 범위 설정
    x = np.arange(-u[0], u[0] + 1, 1)
    y = np.arange(-u[1], u[1] + 1, 1)

    # 2차원 평방 필드로 변환 (그리드 생성)
    x, y = np.meshgrid(x, y)

    # 가우시안 정규 계수 및 지수항 계산
    ratio = 1 / (sigmaX * sigmaY * 2 * np.pi)
    v1 = x**2 / (2 * sigmaX**2)
    v2 = y**2 / (2 * sigmaY**2)
    mask = ratio * np.exp(-(v1 + v2))  # 가우시안 수식

    return mask / np.sum(mask)  # 마스크 정규화 (전체 합 1로 유지)

# 영상 읽기
image = cv2.imread("C:/Users/SAMSUNG/Desktop/Pattern_Recognition_Class/images6/smoothing.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")

# 커널 크기 지정 (가로 × 세로)
ksize = (17, 5)

# 사용자 정의 2D 가우시안 마스크
gaussian_2d = getGaussianMask(ksize, 0, 0)

# OpenCV 제공 1D 가우시안 커널
gaussian_1dX = cv2.getGaussianKernel(ksize[0], 0, cv2.CV_32F)  # 가로
gaussian_1dY = cv2.getGaussianKernel(ksize[1], 0, cv2.CV_32F)  # 세로

# 사용자 정의 마스크 적용 (2D 컨볼루션)
gauss_img1 = cv2.filter2D(image, -1, gaussian_2d)

# OpenCV 제공 2D 가우시안 블러
gauss_img2 = cv2.GaussianBlur(image, ksize, 0)

# 분리형 필터 (OpenCV sepFilter2D)
gauss_img3 = cv2.sepFilter2D(image, -1, gaussian_1dX, gaussian_1dY)

# 결과 출력
titles = ['image', 'gauss_img1', 'gauss_img2', 'gauss_img3']
for t in titles:
    cv2.imshow(t, eval(t))

cv2.waitKey(0)
cv2.destroyAllWindows()

