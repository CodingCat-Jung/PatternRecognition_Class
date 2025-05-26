# 2025-1_Pattern_Recognition_Class

# 🛠️ Jupyter Notebook에서 OpenCV(cv2) 모듈 인식 오류 해결기

Jupyter Notebook에서 OpenCV를 사용하려고 할 때 `cv2` 모듈을 찾지 못하는 오류를 해결한 과정을 정리한 문서입니다.

---

## 🐛 문제 상황

Jupyter Notebook에서 `import cv2`를 실행하면 다음과 같은 오류가 발생:
ModuleNotFoundError: No module named 'cv2'

---

## 🎯 원인 분석

- `pip install opencv-python`으로 설치는 되어 있었지만,
- Jupyter Notebook이 사용하는 Python 환경과 OpenCV가 설치된 환경이 **다르기 때문에** 발생한 문제.

---

## ✅ 해결 방법

### 1. Conda 가상 환경 생성 및 활성화 - Anaconda Prompt

```bash
conda create -n test01 python=3.7 -y
conda activate test01

- 필요한 패키지 설치
pip install opencv-python
pip install jupyter

- Jupyter Notebook 실행
jupyter notebook

- 가상 환경 종료 명령어
conda deactivate

- 가상환경을 Jupyter 커널로 등록
python -m ipykernel install --user --name test01 --display-name "Python (test01)"
