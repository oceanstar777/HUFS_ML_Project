# 얼굴 표정 인식의 범용성 향상 : FER2013 과 한국인 얼굴 데이터셋을 이용한 VGGNet 모델의 최적화 및 평가
-----
## 프로젝트 개요 및 목표
이 프로젝트의 주요 목표는 기존의 FER2013 표정 인식 데이터셋을 사용하여, 서양인과 한국인 얼굴 표정 인식의 차이를 극복하는 것입니다. </br>
FER2013 데이터셋은 주로 서양인 얼굴을 기반으로 하고 있어, 이목구비가 덜 뚜렷하고 표정이 풍부하지 않은 한국인 얼굴 표정 인식률이 낮은 문제점이 있습니다.</br>
이를 해결하고자, 본 프로젝트에서는 FER2013 데이터셋으로 학습된 VGGNet 모델에 한국인의 얼굴 데이터를 전이 학습시켜 정확도를 높이려 합니다.</br>
특히, 기쁨, 분노, 슬픔, 중립, 당황 등의 감정을 비교 분석하고, 이를 통해 모델의 정확도를 평가하는 것이 핵심입니다.


## 데이터셋
본 프로젝트는 두 가지 데이터셋을 사용합니다. </br>
첫 번째는 FER2013 데이터셋으로, 주로 서양인 얼굴이 포함되어 있습니다.</br>
이 데이터셋의 이미지들은 VGGNet 의 표준 입력 사이즈인 224 x 224 로 조정되었으며, 비정상적인 데이터는 수작업으로 제거됩니다.</br>
두 번째 데이터셋은 [한국인 감정 인식을 위한 복합 영상](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=82)데이터셋입니다.</br>
이 데이터는 [AI Hub](https://www.aihub.or.kr/) 에서 제공되며, [OpenCV 의 Haar Cascade Face Detector](https://docs.opencv.org/4.x/d2/d99/tutorial_js_face_detection.html)를 이용하여 얼굴 부분만을 추출하고,</br>
이를 224 x 224 사이즈로 재조정하고 흑백으로 변환하여 사용합니다. </br>
이 방법은 이미지에서 얼굴 부분을 빠르고 효율적으로 검출하기 위해 하르 특징을 기반으로 하는 케스케이드 분류기를 사용하며, 이 데이터셋도 마찬가지로 이상 데이터는 제거하여 품질을 보장합니다.


## 파일 설명

### `fer_resize.ipynb`

fer2013데이터셋의 표정 인식을 위한 데이터 리사이징 작업을 수행하는 코드.

### `facecrop.ipynb`

얼굴 크롭을 위한 코드.

### `face_inference_demo.ipynb`

얼굴 인식 모델의 추론 데모.

### `renaming.ipynb`

opencv의 한국어 인식 에러를 방지하기 위한 이미지 파일명 변경 작업을 위한 코드.

### `val_face_crop.ipynb`

검증 데이터셋의 얼굴 크롭 작업을 위한 코드.

### `val_rename.ipynb`

opencv의 한국어 인식 에러를 방지하기 위한 이미지 파일명 변경 작업을 위한 코드.

### `vggnet_fer.ipynb`

[fer2013](https://www.kaggle.com/datasets/msambare/fer2013) 데이터셋을 이용한 표정 인식을 위한 커스텀 VGGNet 모델 훈련 코드.

### `vggnet_korean.ipynb`

[한국인 감정인식을 위한 복합 영상](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=82) 데이터셋을 이용한 표정 인식을 위한 커스텀 VGGNet 모델 훈련 코드.

## 폴더트리
```
📦HUFS_ML_Project
 ┣ 📂dataset
 ┃ ┣ 📂test
 ┃ ┃ ┗ 📂angry
 ┃ ┃ ┃ ┣ 📜korean faces.jpg
 ┃ ┃ ┃       ...
 ┃ ┃ ┗ 📂embarrassed
 ┃ ┃ ┃ ┣ 📜korean faces.jpg
 ┃ ┃ ┃      ...
 ┃ ┃ ┗ 📂happy
 ┃ ┃ ┃ ┣ 📜korean faces.jpg
 ┃ ┃ ┃      ...
 ┃ ┃ ┗ 📂neutral
 ┃ ┃ ┃ ┣ 📜korean faces.jpg
 ┃ ┃ ┃      ...
 ┃ ┃ ┗ 📂sad
 ┃ ┃ ┃ ┣ 📜korean faces.jpg
 ┃ ┃ ┃     ...
 ┃ ┃ ┗ 📂csvs
 ┃ ┃ ┃ ┣ 📜angry.csv
 ┃ ┃ ┃ ┣ 📜embarrassed.csv
 ┃ ┃ ┃ ┣ 📜happy.csv
 ┃ ┃ ┃ ┣ 📜merged.csv
 ┃ ┃ ┃ ┣ 📜neutral.csv
 ┃ ┃ ┃ ┗ 📜sad.csv
 ┃ ┗ 📂train
 ┃ ┃ ┗ 📂angry
 ┃ ┃ ┃ ┣ 📜korean faces.jpg
 ┃ ┃ ┃       ...
 ┃ ┃ ┗ 📂embarrassed
 ┃ ┃ ┃ ┣ 📜korean faces.jpg
 ┃ ┃ ┃      ...
 ┃ ┃ ┗ 📂happy
 ┃ ┃ ┃ ┣ 📜korean faces.jpg
 ┃ ┃ ┃      ...
 ┃ ┃ ┗ 📂neutral
 ┃ ┃ ┃ ┣ 📜korean faces.jpg
 ┃ ┃ ┃      ...
 ┃ ┃ ┗ 📂sad</br>
 ┃ ┃ ┃ ┣ 📜korean faces.jpg
 ┃ ┃ ┃     ...
 ┃ ┃ ┗ 📂csvs
 ┃ ┃ ┃ ┣ 📜angry.csv
 ┃ ┃ ┃ ┣ 📜embarrassed.csv
 ┃ ┃ ┃ ┣ 📜happy.csv
 ┃ ┃ ┃ ┣ 📜merged.csv
 ┃ ┃ ┃ ┣ 📜neutral.csv
 ┃ ┃ ┃ ┗ 📜sad.csv
 ┣ 📂fer2013
 ┃ ┗ 📂resize
 ┃ ┃ ┣ 📂train
 ┃ ┃ ┃ ┗ 📂angry
 ┃ ┃ ┃ ┃ ┣ 📜fer2013 faces.jpg
 ┃ ┃ ┃ ┃       ...
 ┃ ┃ ┃ ┗ 📂embarrassed
 ┃ ┃ ┃ ┃ ┣ 📜fer2013 faces.jpg
 ┃ ┃ ┃ ┃      ...
 ┃ ┃ ┃ ┗ 📂happy
 ┃ ┃ ┃ ┃ ┣ 📜fer2013 faces.jpg
 ┃ ┃ ┃ ┃      ...
 ┃ ┃ ┃ ┗ 📂neutral
 ┃ ┃ ┃ ┣ 📜fer2013 faces.jpg
 ┃ ┃ ┃ ┃      ...
 ┃ ┃ ┃ ┗ 📂sad
 ┃ ┃ ┃ ┃ ┣ 📜fer2013 faces.jpg
 ┃ ┃ ┃     ...
 ┃ ┃ ┃ ┗ 📂csvs
 ┃ ┃ ┃ ┃ ┣ 📜angry.csv
 ┃ ┃ ┃ ┃ ┣ 📜embarrassed.csv
 ┃ ┃ ┃ ┃ ┣ 📜happy.csv
 ┃ ┃ ┃ ┃ ┣ 📜merged.csv
 ┃ ┃ ┃ ┃ ┣ 📜neutral.csv
 ┃ ┃ ┃ ┃ ┗ 📜sad.csv
 ┃ ┃ ┗ 📂validation
 ┃ ┃ ┃ ┗ 📂angry
 ┃ ┃ ┃ ┃ ┣ 📜fer2013 faces.jpg
 ┃ ┃ ┃ ┃       ...
 ┃ ┃ ┃ ┗ 📂embarrassed
 ┃ ┃ ┃ ┃ ┣ 📜fer2013 faces.jpg
 ┃ ┃ ┃ ┃      ...
 ┃ ┃ ┃ ┗ 📂happy
 ┃ ┃ ┃ ┃ ┣ 📜fer2013 faces.jpg
 ┃ ┃ ┃ ┃      ...
 ┃ ┃ ┃ ┗ 📂neutral
 ┃ ┃ ┃ ┃ ┣ 📜fer2013 faces.jpg
 ┃ ┃ ┃ ┃      ...
 ┃ ┃ ┃ ┗ 📂sad
 ┃ ┃ ┃ ┃ ┣ 📜fer2013 faces.jpg
 ┃ ┃ ┃     ...
 ┃ ┃ ┃ ┣ 📂csvs
 ┃ ┃ ┃ ┃ ┣ 📜angry.csv
 ┃ ┃ ┃ ┃ ┣ 📜embarrassed.csv
 ┃ ┃ ┃ ┃ ┣ 📜happy.csv
 ┃ ┃ ┃ ┃ ┣ 📜merged.csv
 ┃ ┃ ┃ ┃ ┣ 📜neutral.csv
 ┃ ┃ ┃ ┃ ┗ 📜sad.csv
 ┣ 📜face_inference_demo.ipynb
 ┣ 📜facecrop.ipynb
 ┣ 📜fer_resize.ipynb
 ┣ 📜renaming.ipynb
 ┣ 📜val_face_crop.ipynb
 ┣ 📜val_rename.ipynb
 ┣ 📜vggnet_fer.ipynb
 ┗ 📜vggnet_korean.ipynb
```

## 결론
본 프로젝트를 통해 한국인과 서양인의 얼굴 표정 인식에 있어서의 차이를 분석하고, 이를 극복하기 위한 방안을 모색하였습니다.</br>
특히 감정 분류(기쁨, 분노, 슬픔, 중립, 당황)에 초점을 맞춰, 각각의 감정에 대한 인식률을 비교 분석함으로써 모델의 성능을 평가하였습니다.</br>
이를 통해 표정 인식 기술의 범용성과 정확도를 높이는 방향으로 발전시킬 수 있는 기반을 마련하였습니다.
