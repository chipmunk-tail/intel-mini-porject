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

# 원본 이미지에 OCR 텍스트 표시
img_with_ocr = img.copy()
for item in prediction_groups[0]:
    text = item[0]  # OCR 텍스트
    box = item[1]   # OCR 텍스트의 경계 박스

    # 박스가 경계선 좌표인지 확인
    try:
        pts = np.array(box, dtype=np.int32)  # 좌표를 int32로 변환
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img_with_ocr, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(img_with_ocr, str(text), tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    except ValueError as e:
        print(f"Skipping invalid box data for text: {text}. Error: {e}")

# 번역된 텍스트 이미지를 추가 (한국어 번역 출력)
img_with_translation = img.copy()

# 텍스트가 잘 보이도록 큰 글씨로 번역된 텍스트 삽입
cv2.putText(img_with_translation, translated_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

# Subplot으로 결과 출력
plt.figure(figsize=(15, 5))

# 원본 이미지 (BGR 그대로 출력)
plt.subplot(1, 3, 1)
plt.imshow(img)  # BGR 그대로 출력
plt.title("Original Image")
plt.axis('off')

# OCR 텍스트가 포함된 이미지 (RGB로 변환 후 출력)
plt.subplot(1, 3, 2)
# plt.imshow(cv2.cvtColor(img_with_ocr, cv2.COLOR_BGR2RGB))  # BGR을 RGB로 변환
plt.imshow(img)
plt.title("OCR Result")
plt.axis('off')

# 번역된 텍스트가 포함된 이미지 (RGB로 변환 후 출력)
plt.subplot(1, 3, 3)
# plt.imshow(cv2.cvtColor(img_with_translation, cv2.COLOR_BGR2RGB))  # BGR을 RGB로 변환
plt.imshow(img)
plt.title("Translated Text")
plt.axis('off')

# 결과 출력
plt.tight_layout()
# 결과 저장
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
