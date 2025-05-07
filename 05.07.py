#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import cv2

# 사용자 정의 히스토그램 계산 함수
def calc_histo(image, hsize, ranges=[0, 256]):
    hist = np.zeros((hsize, 1), np.float32)  # 히스토그램 누적 행렬
    gap = ranges[1] / hsize                   # 구간 간격

    for i in (image / gap).flat:  # 이미지의 각 픽셀 값을 bin 크기로 나누어 해당 bin에 카운트
        hist[int(i)] += 1

    return hist

# 영상 읽기
image = cv2.imread("C:/Users/SAMSUNG/Desktop/Pattern_Recognition_Class/pixel.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상 파일 읽기 오류 발생")

# 히스토그램 파라미터
hsize, ranges = [32], [0, 256]  # 히스토그램 간격수, 값 범위
gap = ranges[1] / hsize[0]      # 구간 간격

# 사용자 정의 히스토그램 계산
hist1 = calc_histo(image, hsize[0], ranges)

# OpenCV 함수로 히스토그램 계산
hist2 = cv2.calcHist([image], [0], None, hsize, ranges)

# NumPy를 이용한 히스토그램 계산
ranges_gap = np.arange(0, ranges[1] + 1, gap)
hist3, bins = np.histogram(image, ranges_gap)

# 히스토그램 결과 출력
print("User 함수: \n", hist1.flatten())
print("OpenCV 함수: \n", hist2.flatten())
print("numpy 함수: \n", hist3.astype(float))

# 영상 출력
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[7]:


import numpy as np
import cv2

def draw_histo(hist, shape=(200, 256)):  # 기본 영상 크기: 200x256
    hist_img = np.full(shape, 255, np.uint8)  # 흰색 배경 영상 생성

    cv2.normalize(hist, hist, 0, shape[0], cv2.NORM_MINMAX)  # 정규화: 높이를 0~200 사이로 설정
    gap = hist_img.shape[1] / hist.shape[0]  # bin당 너비 계산

    for i, h in enumerate(hist):
        x = int(round(i * gap))
        w = int(round(gap))
        cv2.rectangle(hist_img, (x, 0), (x + w, int(h[0])), 0, cv2.FILLED)

    return cv2.flip(hist_img, 0)  # 영상 상하 반전하여 반환 (Y축 기준)


# In[8]:


import cv2

# 영상 읽기
image = cv2.imread("C:/Users/SAMSUNG/Desktop/Pattern_Recognition_Class/pixel.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류 발생")

# 히스토그램 계산 (OpenCV 함수 사용)
hist = cv2.calcHist([image], [0], None, [32], [0, 256])

# 히스토그램 시각화 이미지 생성 (사용자 정의 함수 호출)
hist_img = draw_histo(hist)  # draw_histo() 함수는 이전에 정의된 함수 사용

# 영상 및 히스토그램 출력
cv2.imshow("image", image)
cv2.imshow("hist_img", hist_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[10]:


import numpy as np
import cv2

# 값이 있는 첫 번째/마지막 계급 검색 함수
def search_value_idx(hist, bias=0):
    for i in range(hist.shape[0]):
        idx = np.abs(bias - i)
        if hist[idx] > 0:
            return idx
    return -1

# 영상 불러오기
image = cv2.imread("C:/Users/SAMSUNG/Desktop/Pattern_Recognition_Class/hist_stretch.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")

# 히스토그램 설정
bsize, ranges = [64], [0, 256]  # 계급 수 및 범위 설정
hist = cv2.calcHist([image], [0], None, bsize, ranges)

# 히스토그램 기반 최소/최대 화소값 계산
bin_width = ranges[1] / bsize[0]
low = search_value_idx(hist, 0) * bin_width
high = search_value_idx(hist, bsize[0] - 1) * bin_width

# 룩업테이블(LUT) 생성: 선형 스케일링 공식 적용
idx = np.arange(0, 256)
idx = (idx - low) / (high - low) * 255
idx[0:int(low)] = 0
idx[int(high) + 1:] = 255

# LUT 적용하여 히스토그램 스트레칭 결과 생성
dst = cv2.LUT(image, idx.astype('uint8'))

# 히스토그램 재계산 및 시각화
hist_dst = cv2.calcHist([dst], [0], None, bsize, ranges)
hist_img = draw_histo(hist, (200, 360))
hist_dst_img = draw_histo(hist_dst, (200, 360))

# 출력
print("high_value =", high)
print("low_value =", low)

cv2.imshow("image", image)
cv2.imshow("dst", dst)
cv2.imshow("hist_img", hist_img)
cv2.imshow("hist_dst_img", hist_dst_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[12]:


import numpy as np
import cv2

def draw_histo(hist, shape=(200, 256)):  # 기본 영상 크기: 200x256
    hist_img = np.full(shape, 255, np.uint8)  # 흰색 배경 영상 생성

    cv2.normalize(hist, hist, 0, shape[0], cv2.NORM_MINMAX)  # 정규화: 높이를 이미지 세로 범위로 맞춤
    gap = hist_img.shape[1] / hist.shape[0]  # bin당 너비 계산

    for i, h in enumerate(hist):
        x = int(round(i * gap))
        w = int(round(gap))
        cv2.rectangle(hist_img, (x, 0), (x + w, int(h[0])), 0, cv2.FILLED)

    return cv2.flip(hist_img, 0)  # 영상 상하 반전하여 반환 (Y축 기준)

# 영상 읽기
image = cv2.imread("C:/Users/SAMSUNG/Desktop/Pattern_Recognition_Class/equalize.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")

# 히스토그램 계산
bins, ranges = [256], [0, 256]
hist = cv2.calcHist([image], [0], None, bins, ranges)

# 누적 히스토그램 계산
accum_hist = np.zeros(hist.shape[:2], np.float32)
accum_hist[0] = hist[0]
for i in range(1, hist.shape[0]):
    accum_hist[i] = accum_hist[i - 1] + hist[i]

# 누적합 정규화: [0, 255] 범위로
accum_hist = (accum_hist / sum(hist)) * 255

# 정규화 누적합을 기반으로 픽셀 매핑 (직접 매핑)
dst1 = [[accum_hist[val] for val in row] for row in image]
dst1 = np.array(dst1, np.uint8)

# OpenCV 내장 함수 사용
dst2 = cv2.equalizeHist(image)

# 히스토그램 재계산
hist1 = cv2.calcHist([dst1], [0], None, bins, ranges)
hist2 = cv2.calcHist([dst2], [0], None, bins, ranges)

# 시각화
hist_img = draw_histo(hist)
hist_img1 = draw_histo(hist1)
hist_img2 = draw_histo(hist2)

# 출력
cv2.imshow("image", image)
cv2.imshow("dst1_User", dst1)
cv2.imshow("dst2_OpenCV", dst2)

cv2.imshow("hist_img", hist_img)
cv2.imshow("User_hist", hist_img1)
cv2.imshow("OpenCV_hist", hist_img2)

cv2.waitKey(0)
cv2.destroyAllWindows()

