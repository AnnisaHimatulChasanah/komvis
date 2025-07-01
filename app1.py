import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

st.set_page_config(
    page_title="Deteksi Real-time dengan InceptionV3",
    page_icon="🍅",
    layout="centered"
)

st.title("Deteksi Penyakit Daun Tomat (Realtime)")
st.write("Aplikasi ini mendeteksi penyakit daun tomat dari kamera secara real-time.")

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("model_inceptionv3_best.h5")
        st.success("✅ Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

model = load_model()

CLASS_NAMES = [
    'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot',
    'Spider_mites Two-spotted_spider_mite', 'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_mosaic_virus', 'healthy', 'powdery_mildew'
]
INPUT_SHAPE = (299, 299)

def preprocess_image(image_array, target_size):
    image_pil = Image.fromarray(image_array)
    image_resized = image_pil.resize(target_size)
    image_array_resized = img_to_array(image_resized)
    image_array_normalized = image_array_resized / 255.0
    image_array_expanded = np.expand_dims(image_array_normalized, axis=0)
    return image_array_expanded

def predict_frame(frame, model, class_names, input_shape):
    preprocessed = preprocess_image(frame, input_shape)
    predictions = model.predict(preprocessed)
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    predicted_class_name = class_names[predicted_index]
    return predicted_class_name, confidence

# --- Kamera Live ---
if model:
    st.subheader("Deteksi Kamera Langsung")
    start_button = st.button("Mulai Deteksi")
    stop_button = st.button("Hentikan Deteksi")

    if start_button:
        st.session_state.run_camera = True
    if stop_button:
        st.session_state.run_camera = False

    if 'run_camera' not in st.session_state:
        st.session_state.run_camera = False

    if st.session_state.run_camera:
        class VideoProcessor(VideoTransformerBase):
            def __init__(self):
                self.model = model
                self.class_names = CLASS_NAMES
                self.input_shape = INPUT_SHAPE
                self.result_text = "Menunggu deteksi..."

            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                predicted_class_name, confidence = predict_frame(
                    img_rgb, self.model, self.class_names, self.input_shape
                )
                self.result_text = f"{predicted_class_name} ({confidence:.2f}%)"
                cv2.putText(img, self.result_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                return img

        st.info("Mengaktifkan kamera. Mohon izinkan akses kamera.")
        ctx = webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            video_transformer_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
        )

        if ctx.state.playing and ctx.video_transformer:
            st.success(ctx.video_transformer.result_text)
    else:
        st.warning("Klik 'Mulai Deteksi' untuk menjalankan kamera.")

st.markdown("---")
st.caption("© 2025 Aplikasi Deteksi Daun Tomat - UAS Komputer Visi")
