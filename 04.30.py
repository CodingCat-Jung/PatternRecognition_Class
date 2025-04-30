#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2

# 이미지 읽기
image = cv2.imread("C:/Users/SAMSUNG/Desktop/Pattern_Recognition_Class/read_color.jpg", cv2.IMREAD_COLOR)
if image is None:
    raise Exception("영상파일 읽기 오류 발생")  # 예외 처리

# 이미지 변형
x_axis = cv2.flip(image, 0)          # x축 기준 상하 뒤집기
y_axis = cv2.flip(image, 1)          # y축 기준 좌우 뒤집기
xy_axis = cv2.flip(image, -1)        # x축, y축 기준 상하좌우 뒤집기
rep_image = cv2.repeat(image, 2, 2)  # 반복 복사
trans_image = cv2.transpose(image)   # 행렬 전치

# 각 행렬을 영상으로 표시
titles = ['image', 'x_axis', 'y_axis', 'xy_axis', 'rep_image', 'trans_image']
for title in titles:
    cv2.imshow(title, eval(title))

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import cv2

# 이미지 읽기
image = cv2.imread("C:/Users/SAMSUNG/Desktop/Pattern_Recognition_Class/read_color.jpg", cv2.IMREAD_COLOR)
if image is None:
    raise Exception("영상파일 읽기 오류")  # 예외 처리
if image.ndim != 3:
    raise Exception("컬러 영상 아님")  # 예외 처리 - 컬러 영상 확인

# B, G, R 채널 분리
bgr = cv2.split(image)  # bgr[0]: Blue, bgr[1]: Green, bgr[2]: Red

# 자료형 확인
print("bgr 자료형:", type(bgr), type(bgr[0]), type(bgr[0][0]), type(bgr[0][0][0]))
print("bgr 원소개수:", len(bgr))

# 각 채널을 윈도우에 띄우기
cv2.imshow("image", image)                  # 원본 이미지
cv2.imshow("Blue channel", bgr[0])          # Blue 채널
cv2.imshow("Green channel", bgr[1])         # Green 채널
cv2.imshow("Red channel", bgr[2])           # Red 채널

# 아래는 넘파이 인덱싱 방식 참고용 (위와 같은 결과)
# cv2.imshow("Blue channel", image[:, :, 0])
# cv2.imshow("Green channel", image[:, :, 1])
# cv2.imshow("Red channel", image[:, :, 2])

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import numpy as np
import cv2

# 50x512 영상 생성
image1 = np.zeros((50, 512), np.uint8)
image2 = np.zeros((50, 512), np.uint8)

# 행과 열 정보 추출
rows, cols = image1.shape[:2]

# 각 픽셀에 대해 값 설정
for i in range(rows):
    for j in range(cols):
        image1.itemset((i, j), j // 2)         # 화소값 점진적 증가
        image2.itemset((i, j), j // (20 * 10))  # 계단 현상 증가

# 영상 출력
cv2.imshow("image1", image1)
cv2.imshow("image2", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[1]:


import cv2

# 영상 읽기
image = cv2.imread("C:/Users/SAMSUNG/Desktop/Pattern_Recognition_Class/bright.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")

# ROI 좌표 지정 및 추출 (x, y), (w, h)
(x, y), (w, h) = (180, 37), (15, 10)
roi_img = image[y:y+h, x:x+w]  # 행렬 접근은 [y:y+h, x:x+w]

# ROI 픽셀 값 출력
print("[roi_img] =")
for row in roi_img:
    for p in row:
        print("%4d" % p, end="")  # 픽셀 값 하나씩 출력
    print()

# 관심 영역 사각형 표시
cv2.rectangle(image, (x, y, w, h), 255, 1)

# 영상 출력
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import numpy as np
import cv2

# 영상 읽기 (흑백 모드)
image1 = cv2.imread("C:/Users/SAMSUNG/Desktop/Pattern_Recognition_Class/add1.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("C:/Users/SAMSUNG/Desktop/Pattern_Recognition_Class/add2.jpg", cv2.IMREAD_GRAYSCALE)
if image1 is None or image2 is None:
    raise Exception("영상파일 읽기 오류")

## 영상 합성 방법
alpha, beta = 0.7, 0.9  # 곱셈 비율

# 단순 더하기
add_img1 = cv2.add(image1, image2)

# 비율을 곱해서 더하기
add_img2 = cv2.add(image1 * alpha, image2 * beta)
add_img2 = np.clip(add_img2, 0, 255).astype('uint8')  # saturation 처리

# addWeighted 함수 사용
add_img3 = cv2.addWeighted(image1, alpha, image2, beta, 0)

# 영상 표시
titles = ['image1', 'image2', 'add_img1', 'add_img2', 'add_img3']
for t in titles:
    cv2.imshow(t, eval(t))

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import numpy as np
import cv2

# 흑백 이미지 읽기
image = cv2.imread("C:/Users/SAMSUNG/Desktop/Pattern_Recognition_Class/contrast.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("영상파일 읽기 오류")

# 더미 영상 및 평균값 생성
noimage = np.zeros(image.shape[:2], image.dtype)
avg = cv2.mean(image)[0] / 2.0  # 영상 평균값의 절반

# 명암 대비 조절
dst1 = cv2.scaleAdd(image, 0.5, noimage)            # 명암 대비 감소
dst2 = cv2.scaleAdd(image, 2.0, noimage)            # 명암 대비 증가
dst3 = cv2.addWeighted(image, 0.5, noimage, 0, avg)  # 평균 기준 대비 감소
dst4 = cv2.addWeighted(image, 2.0, noimage, 0, -avg) # 평균 기준 대비 증가

# 결과 영상 표시
cv2.imshow("image", image)                                # 원본
cv2.imshow("dst1 - decrease contrast", dst1)              # 대비 감소
cv2.imshow("dst2 - increase contrast", dst2)              # 대비 증가
cv2.imshow("dst3 - decrease contrast using average", dst3)  # 평균 기준 대비 감소
cv2.imshow("dst4 - increase contrast using average", dst4)  # 평균 기준 대비 증가

cv2.waitKey(0)
cv2.destroyAllWindows()

