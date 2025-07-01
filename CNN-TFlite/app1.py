import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import tensorflow as tf

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Penyakit Daun Tomat (TFLite)",
    page_icon="🍅",
    layout="centered"
)

st.title("Deteksi Penyakit Daun Tomat (Realtime)")
st.write("Aplikasi ini mendeteksi penyakit daun tomat dari kamera secara real-time menggunakan model TFLite.")

# Load Model TFLite
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="model_inceptionv3_best.tflite")
        interpreter.allocate_tensors()
        st.success("✅ Model TFLite berhasil dimuat!")
        return interpreter
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

interpreter = load_model()

CLASS_NAMES = [
    'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot',
    'Spider_mites Two-spotted_spider_mite', 'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_mosaic_virus', 'healthy', 'powdery_mildew'
]
INPUT_SHAPE = (299, 299)

def preprocess_image(image_array, target_size):
    image_pil = Image.fromarray(image_array)
    image_resized = image_pil.resize(target_size)
    image_array_resized = np.array(image_resized) / 255.0
    image_array_expanded = np.expand_dims(image_array_resized, axis=0).astype(np.float32)
    return image_array_expanded

def predict_tflite(image, interpreter, class_names):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = np.argmax(output_data[0])
    confidence = np.max(output_data[0]) * 100
    predicted_class = class_names[predicted_index]

    return predicted_class, confidence

# Kamera Deteksi Real-time
if interpreter:
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
                self.interpreter = interpreter
                self.class_names = CLASS_NAMES
                self.result_text = "Menunggu deteksi..."

            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                preprocessed = preprocess_image(img_rgb, INPUT_SHAPE)

                predicted_class, confidence = predict_tflite(preprocessed, self.interpreter, self.class_names)
                self.result_text = f"{predicted_class} ({confidence:.2f}%)"

                cv2.putText(img, self.result_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                return img

        st.info("Mengaktifkan kamera. Mohon izinkan akses kamera.")

        ctx = webrtc_streamer(
            key="tomat-stream",
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
st.caption("© 2025 Aplikasi Deteksi Daun Tomat - UAS Komputer Vision")
