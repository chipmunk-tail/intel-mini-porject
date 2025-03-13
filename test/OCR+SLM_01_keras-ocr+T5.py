import keras_ocr
import subprocess
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Ollama로 번역하는 함수
def use_ollama(prompt):
    # Ollama 모델을 Docker 컨테이너에서 호출
    result = subprocess.run(['ollama', 'run', 'phi4:14b-q8_0', '--text', prompt], capture_output=True, text=True)
    return result.stdout.strip()  # 번역된 텍스트

# Keras OCR pipeline 초기화
pipeline = keras_ocr.pipeline.Pipeline()

# 이미지 로드 및 OCR 수행
image_name = "exam02"
image_format = ".png"
image_folder = "./test/test_img/"
image_path = image_folder + image_name + image_format

output_path = image_folder + "keras+phi4_" + image_name + "png"

# 이미지 읽기
img = keras_ocr.tools.read(image_path)

# OCR 처리 (이미지에서 텍스트 추출)
prediction_groups = pipeline.recognize([img])

# OCR 결과 추출 (text와 box가 각각 2개의 값으로 반환됨)
ocr_text = "\n".join([str(text) for text, _ in prediction_groups[0]])  # OCR로 추출한 텍스트

# Ollama를 사용하여 번역 (OCR 결과를 한국어로 번역)
translated_text = use_ollama(f"해당 문장을 한국어로 번역해줘!, 오타가 있다면 수정해주고, 번역본만 출력해줘: {ocr_text}")

# 이미지 복사 (OCR 결과 표시용)
img_with_ocr = img.copy()

# OCR 결과 표시 (이미지에 텍스트 및 박스 그리기)
for text, box in prediction_groups[0]:
    pts = np.array(box, dtype=np.int32)  # 좌표를 int32로 변환
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img_with_ocr, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(img_with_ocr, str(text), tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# 번역된 텍스트 이미지 (한국어 번역 출력)
img_with_translation = img.copy()

# 번역된 텍스트 삽입
cv2.putText(img_with_translation, translated_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

# Subplot으로 결과 출력
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 원본 이미지 표시
axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[0].axis('off')

# OCR 결과 표시
axes[1].imshow(img_with_ocr)
axes[1].set_title("OCR Result")
axes[1].axis('off')

# 번역된 텍스트 표시
axes[2].imshow(img_with_translation)
axes[2].set_title("Translated Text")
axes[2].axis('off')

# 결과 출력
plt.tight_layout()

# 결과 저장
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
