
# OCR model = keras_ocr
# pip install keras-ocr

import keras_ocr
import matplotlib.pyplot as plt

# OCR 모델 로드
pipeline = keras_ocr.pipeline.Pipeline()

# OCR을 수행할 이미지 경로
image_path = "./test/test_img/image.png"

# 이미지 불러오기
image = keras_ocr.tools.read(image_path)

# OCR 실행 (텍스트 감지 + 인식)
prediction_groups = pipeline.recognize([image])

# OCR 결과 출력
print("OCR 결과:")
for text, box in prediction_groups[0]:
    print(f"텍스트: {text}")

# 결과 시각화
plt.imshow(image)
for text, box in prediction_groups[0]:
    x_values = [point[0] for point in box]
    y_values = [point[1] for point in box]
    plt.plot(x_values + [x_values[0]], y_values + [y_values[0]], 'r-')
    plt.text(box[0][0], box[0][1], text, fontsize=12, color='blue', bbox=dict(facecolor='white', alpha=0.6))
plt.axis("off")
plt.show()


