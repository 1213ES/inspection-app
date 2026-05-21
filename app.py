"""
app.py  ─  가죽 이상 탐지 Streamlit 웹 앱
infer_keras.py 의 모델 로드 / 전처리 / 추론 로직을 그대로 유지하고
Streamlit UI 를 추가한 버전
"""

import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow import keras

# ── 설정 ─────────────────────────────────────────────────────────
MODEL_PATH     = "./weights/leather_model.keras"
INPUT_IMG_SIZE = (224, 224)
CLASSES        = ["정상", "불량"]

# ─────────────────────────────────────────────────────────────────
# 1. 페이지 설정
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="가죽 품질 검사 AI",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 가죽 품질 검사 AI")
st.caption("VGG16 전이학습 모델로 가죽 이미지의 정상 / 불량을 판별합니다.")
st.divider()

# ─────────────────────────────────────────────────────────────────
# 2. 모델 로드  (@st.cache_resource → 재실행 시 재로드 방지)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

if model is None:
    st.error(f"모델 파일을 찾을 수 없습니다: `{MODEL_PATH}`  \n`weights/` 폴더에 `leather_model.keras` 파일을 넣어주세요.")
    st.stop()

st.success(f"모델 로드 완료 ✅  (`{MODEL_PATH}`)")
st.divider()

# ─────────────────────────────────────────────────────────────────
# 3. 이미지 입력  (파일 업로드 / 카메라 촬영)
# ─────────────────────────────────────────────────────────────────
input_mode = st.radio(
    "입력 방식 선택",
    ["📁 파일 업로드", "📷 카메라 촬영"],
    horizontal=True,
)

pil_img = None

if input_mode == "📁 파일 업로드":
    uploaded = st.file_uploader(
        "가죽 이미지를 업로드하세요",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")

else:  # 카메라 촬영
    captured = st.camera_input("카메라로 가죽을 촬영하세요")
    if captured:
        pil_img = Image.open(captured).convert("RGB")

# 미리보기
if pil_img:
    st.image(pil_img, caption="입력 이미지", use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────
# 4. 전처리 / 추론 함수  (infer_keras.py 로직 그대로)
# ─────────────────────────────────────────────────────────────────
def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = keras.applications.vgg16.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict(model, pil_img):
    arr  = preprocess(pil_img)
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = CLASSES[1 if prob > 0.5 else 0]
    return label, prob

# ─────────────────────────────────────────────────────────────────
# 5. 검사 실행 버튼
# ─────────────────────────────────────────────────────────────────
if st.button("🔍 검사 시작", use_container_width=True):
    if pil_img is None:
        st.warning("이미지를 먼저 업로드하거나 촬영해주세요.")
    else:
        with st.spinner("추론 중..."):
            label, prob = predict(model, pil_img)

        prob_good = 1 - prob  # 정상 확률
        prob_bad  = prob      # 불량 확률

        st.divider()

        # ── 결과 메시지
        if label == "정상":
            st.success(f"## ✅ 정상  (정상 확률: {prob_good:.1%})")
        else:
            st.error(f"## ❌ 불량  (불량 확률: {prob_bad:.1%})")

        # ── 확률 수치 (나란히)
        col1, col2 = st.columns(2)
        col1.metric("정상 확률", f"{prob_good:.1%}")
        col2.metric("불량 확률", f"{prob_bad:.1%}")

        # ── 불량 확률 막대
        st.caption("불량 확률")
        st.progress(float(prob_bad))
